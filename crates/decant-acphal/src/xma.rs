//! The XMA engine: decodes the enabled XMA contexts into their output rings, as the SHAPE hardware does.
//!
//! An XMA context is 16 words in the context array that the title and the engine share. The fields, as titles write them
//! (bit ranges inclusive):
//!
//! | Word | Bits | Field |
//! |---|---|---|
//! | 0 | 0 to 11 | size of input buffer 0 in 2048 byte packets |
//! | 0 | 12 to 19 | loop count, 255 loops for ever |
//! | 0 | 20, 21 | input buffer 0, 1 valid (the engine clears a bit when it used the buffer up) |
//! | 0 | 22 to 26 | size of the output ring in 256 byte blocks |
//! | 0 | 27 to 31 | write block (the engine moves it) |
//! | 1 | 0 to 11 | size of input buffer 1 in packets |
//! | 1 | 12, 13 | subframe of the loop end frame that ends the loop (0 to 3, each 128 samples) |
//! | 1 | 17 to 19 | subframes to skip after the jump to the loop start |
//! | 1 | 24 to 26 | subframes to skip at the start |
//! | 1 | 27, 28 | sample rate code: 24, 32, 44.1, 48 kHz |
//! | 1 | 29 | stereo |
//! | 1 | 31 | output writable (cleared when the ring is full, set again when the title takes blocks) |
//! | 2 | 0 to 25 | read position in bits inside the current input buffer |
//! | 2 | 26 to 30 | error status, 4 = the input ran out |
//! | 2 | 31 | error |
//! | 3 | 0 to 25 | loop start in bits |
//! | 4 | 0 to 25 | loop end in bits |
//! | 4 | 31 | current input buffer |
//! | 5, 6 | | APU addresses of input buffers 0 and 1 |
//! | 7 | | APU address of the output ring |
//! | 9 | 0 to 4 | read block (the title moves it with command 0x12) |
//!
//! The engine fills a ring while it is writable: 16 bit samples (interleaved for stereo) in blocks of 256 bytes. When both
//! input buffers are used up it sets error status 4 and stops. It goes on when the title marks a buffer valid again.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

use decant_log::trace;
use decant_xma::{Decoder, FRAME_SAMPLES, HEADER_BITS, PACKET_BYTES, RATES, SUBFRAME_SAMPLES, first_frame, read_frame};

use crate::heap::with_heaps;
use crate::monitor;

/// Bytes of one output block.
pub const BLOCK_BYTES: usize = 256;
/// Error status: the input ran out.
pub const ERROR_INPUT_ENDED: u32 = 4;

const BLOCK_SAMPLES: usize = BLOCK_BYTES / 2;
const WORDS_PER_CONTEXT: usize = 16;
/// The loop count that never runs out.
const LOOP_FOREVER: u32 = 255;

/// A bit field inside one word of a context.
#[derive(Clone, Copy)]
struct Field {
    word: usize,
    shift: u32,
    width: u32,
}

impl Field {
    const fn new(word: usize, shift: u32, width: u32) -> Self {
        Self { word, shift, width }
    }

    const fn mask(self) -> u32 {
        ((1u32 << self.width) - 1) << self.shift
    }

    /// The same field moved `n` bits up, used for the second of a pair of flags.
    const fn plus_bit(self, n: u32) -> Self {
        Self { shift: self.shift + n, ..self }
    }
}

const INPUT0_PACKETS: Field = Field::new(0, 0, 12);
const LOOP_COUNT: Field = Field::new(0, 12, 8);
const INPUT_VALID: Field = Field::new(0, 20, 1);
const RING_BLOCKS: Field = Field::new(0, 22, 5);
const WRITE_BLOCK: Field = Field::new(0, 27, 5);
const INPUT1_PACKETS: Field = Field::new(1, 0, 12);
const LOOP_END_SUBFRAME: Field = Field::new(1, 12, 2);
const LOOP_SKIP: Field = Field::new(1, 17, 3);
const START_SKIP: Field = Field::new(1, 24, 3);
const RATE_CODE: Field = Field::new(1, 27, 2);
const STEREO: Field = Field::new(1, 29, 1);
const WRITABLE: Field = Field::new(1, 31, 1);
const READ_BITS: Field = Field::new(2, 0, 26);
const ERROR_STATUS: Field = Field::new(2, 26, 5);
const ERROR_AND_STATUS: Field = Field::new(2, 26, 6);
const ERROR_FLAG: Field = Field::new(2, 31, 1);
const LOOP_START_BITS: Field = Field::new(3, 0, 26);
const LOOP_END_BITS: Field = Field::new(4, 0, 26);
const CURRENT_INPUT: Field = Field::new(4, 31, 1);
const INPUT0_ADDRESS: usize = 5;
const INPUT1_ADDRESS: usize = 6;
const RING_ADDRESS: usize = 7;
const READ_BLOCK: Field = Field::new(9, 0, 5);

/// One context of the shared array, accessed with volatile reads and writes because the title uses the same words.
#[derive(Clone, Copy)]
struct Context(*mut u32);

impl Context {
    fn word(self, index: usize) -> u32 {
        // SAFETY: inside the context array, which the title writes at the same time, as with the hardware.
        unsafe { self.0.add(index).read_volatile() }
    }

    fn put_word(self, index: usize, value: u32) {
        // SAFETY: as in `word`.
        unsafe { self.0.add(index).write_volatile(value) }
    }

    fn read(self, f: Field) -> u32 {
        (self.word(f.word) & f.mask()) >> f.shift
    }

    fn write(self, f: Field, value: u32) {
        let kept = self.word(f.word) & !f.mask();
        self.put_word(f.word, kept | ((value << f.shift) & f.mask()));
    }

    fn flag(self, f: Field) -> bool {
        self.read(f) != 0
    }

    fn channels(self) -> usize {
        if self.flag(STEREO) { 2 } else { 1 }
    }

    fn rate(self) -> u32 {
        RATES[self.read(RATE_CODE) as usize]
    }

    /// The valid flag of input buffer `which` (0 or 1).
    fn valid_flag(which: u32) -> Field {
        INPUT_VALID.plus_bit(which)
    }

    /// The APU address and the size in packets of input buffer `which`.
    fn input(self, which: u32) -> (u32, u32) {
        match which {
            0 => (self.word(INPUT0_ADDRESS), self.read(INPUT0_PACKETS)),
            _ => (self.word(INPUT1_ADDRESS), self.read(INPUT1_PACKETS)),
        }
    }

    /// Marks input buffer `which` as used up.
    fn retire(self, which: u32) {
        self.write(Self::valid_flag(which), 0);
    }
}

/// The state the engine keeps while a context is enabled.
#[derive(Default)]
struct Voice {
    decoder: Option<Decoder>,
    /// Decoded interleaved samples that have not reached the ring yet.
    pending: VecDeque<i16>,
    /// Samples per channel still to be dropped, from the start skip or a loop skip.
    to_skip: usize,
    /// Both inputs ran out and the error status is set.
    starved: bool,
    frames: u64,
    bad_frames: u64,
    blocks_written: u64,
    blocks_taken: u64,
}

struct Engine {
    /// Host address of the context array.
    array: usize,
    voices: Vec<Option<Voice>>,
}

// SAFETY: `array` is an address in the APU heap, only used while the engine lock is held.
unsafe impl Send for Engine {}

impl Engine {
    fn context(&self, index: u32) -> Option<Context> {
        let slot = index as usize;
        (slot < self.voices.len()).then(|| Context((self.array + slot * WORDS_PER_CONTEXT * 4) as *mut u32))
    }
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

fn lock() -> MutexGuard<'static, Option<Engine>> {
    ENGINE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The context array exists: `count` contexts at host address `array`.
pub fn attach(array: usize, count: u32) {
    let engine = (array != 0 && count != 0).then(|| Engine { array, voices: (0..count).map(|_| None).collect() });
    *lock() = engine;
}

/// The context array is gone.
pub fn detach() {
    let Some(engine) = lock().take() else { return };
    for index in 0..engine.voices.len() as u32 {
        monitor::end(index);
    }
}

/// Command 5 (and 6 per bit): the context starts from its words, the error field is cleared.
pub fn enable(index: u32) {
    let mut guard = lock();
    let Some(engine) = guard.as_mut() else { return };
    let Some(context) = engine.context(index) else { return };
    context.write(ERROR_AND_STATUS, 0);
    let voice = Voice { to_skip: context.read(START_SKIP) as usize * SUBFRAME_SAMPLES, ..Voice::default() };
    let slot = &mut engine.voices[index as usize];
    if slot.is_none() {
        let words: Vec<String> = (0..10).map(|w| format!("{:08x}", context.word(w))).collect();
        trace(&format!("AcpHal: XMA context {index} enabled: words {}", words.join(" ")));
    }
    *slot = Some(voice);
}

/// Command 7 (and 8 per bit): the context stops.
pub fn disable(index: u32) {
    let mut guard = lock();
    let Some(engine) = guard.as_mut() else { return };
    let Some(voice) = engine.voices.get_mut(index as usize).and_then(Option::take) else { return };
    monitor::end(index);
    trace(&format!(
        "AcpHal: XMA context {index} disabled: {} frames ({} bad), {} blocks written, {} taken",
        voice.frames, voice.bad_frames, voice.blocks_written, voice.blocks_taken
    ));
}

/// Command 0x12: the title took `blocks` output blocks, the read block moves on and the ring is writable again.
pub fn output_consumed(index: u32, blocks: u32) {
    let mut guard = lock();
    let Some(engine) = guard.as_mut() else { return };
    let Some(context) = engine.context(index) else { return };
    let ring = context.read(RING_BLOCKS).max(1);
    context.write(READ_BLOCK, (context.read(READ_BLOCK) + blocks % ring) % ring);
    if blocks > 0 {
        context.write(WRITABLE, 1);
    }
    if let Some(voice) = engine.voices[index as usize].as_mut() {
        voice.blocks_taken += u64::from(blocks);
    }
}

/// What the engine did for context `index` while it is enabled: frames decoded, frames that did not decode, blocks written
/// and blocks the title took.
pub fn stats(index: u32) -> Option<(u64, u64, u64, u64)> {
    let guard = lock();
    let voice = guard.as_ref()?.voices.get(index as usize)?.as_ref()?;
    Some((voice.frames, voice.bad_frames, voice.blocks_written, voice.blocks_taken))
}

/// Whether context `index` is enabled.
pub fn is_enabled(index: u32) -> bool {
    lock().as_ref().and_then(|engine| engine.voices.get(index as usize)).is_some_and(Option::is_some)
}

/// Fills the ring of every enabled context as far as it has room. Called by the service thread every 2 ms and after the
/// commands that make room or start a context.
///
/// The lock is taken again for every context instead of once for all of them. The title sends its commands from a
/// time-critical mixer thread under the same lock, and it must not wait for the decoding of every enabled context.
pub fn pump() {
    let count = match lock().as_ref() {
        Some(engine) => engine.voices.len() as u32,
        None => return,
    };
    for index in 0..count {
        let mut guard = lock();
        let Some(engine) = guard.as_mut() else { return };
        let Some(context) = engine.context(index) else { continue };
        if let Some(voice) = engine.voices.get_mut(index as usize).and_then(Option::as_mut) {
            fill_ring(index, context, voice);
        }
    }
}

/// `len` bytes of APU memory at `apu`, when one live allocation holds them.
fn apu_bytes<'a>(apu: u32, len: usize) -> Option<&'a mut [u8]> {
    let ptr = with_heaps(|h| h.map_apu_range(apu, u32::try_from(len).ok()?)).ok().flatten()?;
    // SAFETY: inside one live allocation, which the title and the engine share as with the hardware.
    Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

/// Writes whole blocks into the ring while the context is writable and samples are available.
fn fill_ring(index: u32, context: Context, voice: &mut Voice) {
    let ring = context.read(RING_BLOCKS);
    if ring == 0 {
        return;
    }
    while context.flag(WRITABLE) {
        while voice.pending.len() < BLOCK_SAMPLES && decode_next_frame(index, context, voice) {}
        if voice.pending.is_empty() {
            return;
        }
        let position = context.read(WRITE_BLOCK);
        let destination = context.word(RING_ADDRESS).wrapping_add(position * BLOCK_BYTES as u32);
        let Some(bytes) = apu_bytes(destination, BLOCK_BYTES) else { return };
        let mut block = [0i16; BLOCK_SAMPLES];
        for (sample, slot) in block.iter_mut().zip(bytes.as_chunks_mut::<2>().0.iter_mut()) {
            *sample = voice.pending.pop_front().unwrap_or(0);
            slot.copy_from_slice(&sample.to_le_bytes());
        }
        monitor::feed(index, context.rate(), context.channels(), &block);
        voice.blocks_written += 1;
        let next = (position + 1) % ring;
        context.write(WRITE_BLOCK, next);
        if next == context.read(READ_BLOCK) {
            // Full until the title takes blocks.
            context.write(WRITABLE, 0);
        }
    }
}

/// Picks the input buffer to read from and switches to the other one when the current one is used up. None when both are
/// used up, which also raises the error status once.
fn select_input(context: Context, voice: &mut Voice) -> Option<u32> {
    let mut current = context.read(CURRENT_INPUT);
    if !context.flag(Context::valid_flag(current)) {
        if !context.flag(Context::valid_flag(1 - current)) {
            if !voice.starved {
                voice.starved = true;
                context.write(ERROR_STATUS, ERROR_INPUT_ENDED);
                context.write(ERROR_FLAG, 1);
            }
            return None;
        }
        current = 1 - current;
        context.write(CURRENT_INPUT, current);
        context.write(READ_BITS, 0);
    }
    if voice.starved {
        // The title supplied new input.
        voice.starved = false;
        context.write(ERROR_AND_STATUS, 0);
    }
    Some(current)
}

/// Decodes the next frame into `voice.pending`. False when there is nothing to decode right now.
fn decode_next_frame(index: u32, context: Context, voice: &mut Voice) -> bool {
    let channels = context.channels();
    let Some(current) = select_input(context, voice) else { return false };
    let (address, packets) = context.input(current);
    let Some(data) = apu_bytes(address & !(PACKET_BYTES as u32 - 1), packets as usize * PACKET_BYTES) else {
        context.retire(current);
        return true;
    };
    let mut at = context.read(READ_BITS) as usize;
    if at < HEADER_BITS {
        match first_frame(data) {
            Some(first) => at = first,
            None => {
                context.retire(current);
                return true;
            }
        }
    }
    let Some(frame) = read_frame(data, at) else {
        context.retire(current);
        context.write(READ_BITS, 0);
        return true;
    };
    let decoder = voice.decoder.get_or_insert_with(|| Decoder::new(channels, context.rate()));
    let mut samples = Vec::with_capacity(FRAME_SAMPLES * channels);
    voice.frames += 1;
    if let Err(e) = decoder.decode(&frame, &mut samples) {
        voice.bad_frames += 1;
        if voice.bad_frames <= 2 {
            trace(&format!("AcpHal: XMA context {index}: the frame at bit {at} does not decode ({e})"));
        }
    }
    let mut next = frame.next;
    let loops = context.read(LOOP_COUNT);
    let loop_end = context.read(LOOP_END_BITS) as usize;
    let closes_loop = loops > 0 && loop_end != 0 && frame.start <= loop_end && next.is_none_or(|n| loop_end < n);
    if closes_loop {
        samples.truncate((context.read(LOOP_END_SUBFRAME) as usize + 1) * SUBFRAME_SAMPLES * channels);
        next = Some(context.read(LOOP_START_BITS) as usize);
        voice.to_skip = context.read(LOOP_SKIP) as usize * SUBFRAME_SAMPLES;
        if loops != LOOP_FOREVER {
            context.write(LOOP_COUNT, loops - 1);
        }
    }
    let dropped = (voice.to_skip * channels).min(samples.len());
    voice.to_skip -= dropped / channels;
    voice.pending.extend(&samples[dropped..]);
    match next {
        Some(bit) => context.write(READ_BITS, bit as u32),
        None => {
            context.retire(current);
            context.write(READ_BITS, 0);
        }
    }
    true
}
