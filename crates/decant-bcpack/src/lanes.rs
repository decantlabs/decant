//! The entropy layer: 12 or 16 rANS lanes that share one byte stream.
//!
//! Each lane is a run of fragments. A fragment is a header byte, a length (one byte, or two for a
//! long fragment) and a payload, decoded with one of the 128 models:
//!
//! - header: bits 0 to 3 the model's shape, bit 4 clears the state at the fragment's end, bit 5
//!   marks a long length, bits 6 and 7 narrow the lane's alphabet (group = lane group - bits 6..7);
//! - length: `byte + 1` payload bytes, or for a long fragment `(high << 8 | low) + 257`;
//! - payload: the lane takes a byte while its state is below 2^24 and decodes a symbol while it is
//!   above. The fragment ends once its payload is used up and the state is at or below 2^24.
//!
//! The schedule: in each round the lanes step in order, then a lane that wants a byte takes the
//! next one of the stream, at most [`ROUND_BYTES`] bytes per round between all lanes. A lane that
//! found the round's bytes used up waits one round. The lanes hand over their symbols in lockstep:
//! one symbol per lane for each position of the record, and a lane that has handed over its symbol
//! for the current position waits until all have. A block takes `per` positions.
//!
//! The stream has no length of its own: bytes after the last one the lanes need are not read.

use crate::Error;
use crate::model::{MODELS, Model};

/// rANS lower bound: decode above it, read below it.
pub const LOW: u32 = 1 << 24;
/// Bytes the lanes may take between them in one round.
pub const ROUND_BYTES: u32 = 8;

/// The lanes of a format: each lane's group (symbol bits - 1 at the full alphabet) and the symbols
/// each lane gives per block.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub groups: &'static [u8],
    pub per: usize,
}

impl Layout {
    /// Symbols of one block's record: lane `j`'s symbol `k` is at `j * per + k`.
    pub fn stride(&self) -> usize {
        self.groups.len() * self.per
    }
}

/// Where the lanes take their bytes from: the stream, or the encoder's queues.
pub trait Feed {
    /// The next byte, which goes to `lane`; `None` past the end.
    fn next(&mut self, lane: usize) -> Option<u8>;

    /// The next byte for `lane` if `ask`: whether one came, and the byte (any value when none).
    #[inline(always)]
    fn take(&mut self, lane: usize, ask: bool) -> (bool, u8) {
        if !ask {
            return (false, 0);
        }
        self.next(lane).map_or((false, 0), |byte| (true, byte))
    }
}

/// The stream as a slice.
pub struct SliceFeed<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> SliceFeed<'a> {
    pub fn new(data: &'a [u8]) -> SliceFeed<'a> {
        SliceFeed { data, at: 0 }
    }

    /// Bytes taken so far.
    pub fn taken(&self) -> usize {
        self.at
    }
}

impl Feed for SliceFeed<'_> {
    #[inline(always)]
    fn next(&mut self, _lane: usize) -> Option<u8> {
        let byte = *self.data.get(self.at)?;
        self.at += 1;
        Some(byte)
    }

    /// Without a branch: whether a lane asks is hard to predict.
    #[inline(always)]
    fn take(&mut self, _lane: usize, ask: bool) -> (bool, u8) {
        let len = self.data.len();
        let got = ask & (self.at < len);
        let byte = self.data.get(self.at.min(len.wrapping_sub(1))).copied().unwrap_or(0);
        self.at += usize::from(got);
        (got, byte)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Stage {
    #[default]
    Header,
    Length,
    LengthHigh,
    Payload,
}

#[derive(Clone, Copy)]
struct Lane {
    group: u8,
    stage: Stage,
    long: bool,
    clear: bool,
    /// The model of the current fragment.
    decoder: &'static Model,
    /// The lane asks for a byte this round.
    wants: bool,
    latched: bool,
    byte: u8,
    /// A decoded symbol not handed over yet.
    holding: bool,
    symbol: u8,
    /// Skips its next step: it waits for the other lanes or for the next round's bytes.
    paused: bool,
    x: u32,
    left: u32,
}

impl Lane {
    /// One step: uses the latched byte (header, length or payload) and decodes at most one symbol.
    #[inline(always)]
    fn step(&mut self, models: &'static [Model; MODELS]) {
        if self.stage != Stage::Payload {
            self.prefix(models);
            return;
        }
        // Take the latched byte while the state is low (without a branch: it is hard to predict).
        let read = self.x < LOW && self.latched;
        let x = if read { self.x << 8 | u32::from(self.byte) } else { self.x };
        self.left = self.left.wrapping_sub(u32::from(read));
        self.latched &= !read;
        self.wants = self.left != 0 && !self.latched;
        if x > LOW {
            let (symbol, next) = self.decoder.decode(x);
            self.x = next;
            self.symbol = symbol;
            self.holding = true;
        } else if self.left != 0 {
            // Exactly 2^24 with payload left parks the lane: it neither reads nor decodes again.
            self.x = x;
        } else {
            // The fragment ends. A byte taken in this step does not reach the state carried on.
            self.stage = Stage::Header;
            self.wants = !self.latched;
            if self.clear {
                self.x = 0;
            }
        }
    }

    /// A step before the payload: the header byte, then the length.
    #[cold]
    #[inline(never)]
    fn prefix(&mut self, models: &'static [Model; MODELS]) {
        if !self.latched {
            return;
        }
        self.latched = false;
        let b = self.byte;
        self.wants = true;
        match self.stage {
            Stage::Header => {
                // The groups of every layout are 3 or more, so the narrowing never underflows.
                self.decoder = &models[usize::from((self.group - (b >> 6)) * 16 + (b & 15))];
                self.clear = b & 0x10 != 0;
                self.long = b & 0x20 != 0;
                self.left = 0;
                self.stage = Stage::Length;
            }
            Stage::Length if self.long => {
                self.left = u32::from(b);
                self.stage = Stage::LengthHigh;
            }
            Stage::Length => {
                self.left = u32::from(b) + 1;
                self.stage = Stage::Payload;
            }
            _ => {
                self.left = (u32::from(b) << 8 | self.left) + 257;
                self.stage = Stage::Payload;
            }
        }
    }
}

/// The lanes of one stream, resumable: [`Lanes::run`] fills records batch after batch.
pub struct Lanes {
    layout: Layout,
    lanes: [Lane; 16],
    /// Lanes that handed over their symbol for the current position.
    given: u32,
    /// Position inside the current block, `0..per`.
    position: usize,
    models: &'static [Model; MODELS],
}

impl Lanes {
    pub fn new(layout: Layout) -> Lanes {
        let models = crate::model::models();
        let idle = Lane { group: 0, stage: Stage::Header, long: false, clear: false, decoder: &models[0], wants: true, latched: false, byte: 0, holding: false, symbol: 0, paused: false, x: 0, left: 0 };
        let mut lanes = [idle; 16];
        for (lane, &group) in lanes.iter_mut().zip(layout.groups) {
            lane.group = group;
        }
        Lanes { layout, lanes, given: 0, position: 0, models }
    }

    /// Decodes whole records into `records` (a multiple of the layout's stride).
    pub fn run<F: Feed>(&mut self, feed: &mut F, records: &mut [u8]) -> Result<(), Error> {
        let stride = self.layout.stride();
        let per = self.layout.per;
        let count = self.layout.groups.len();
        let all = (1u32 << count) - 1;
        let blocks = records.len() / stride;
        let mut block = 0;
        let mut idle = 0;
        let mut given = self.given;
        while block < blocks {
            let mut taken = 0u32;
            let mut moved = false;
            let base = block * stride + self.position;
            for (j, lane) in self.lanes[..count].iter_mut().enumerate() {
                let bit = 1u32 << j;
                let before = taken;
                if !lane.paused {
                    lane.step(self.models);
                }
                let waiting = lane.holding & (given & bit != 0);
                // Past the end nothing arrives; in a whole stream only finished lanes ask there.
                let (got, byte) = feed.take(j, lane.wants & !waiting & (taken < ROUND_BYTES));
                lane.byte = if got { byte } else { lane.byte };
                lane.latched |= got;
                taken += u32::from(got);
                lane.paused = (lane.wants & (before >= ROUND_BYTES)) | waiting;
                // Hand the symbol over, without a branch.
                let hand = !lane.paused & lane.holding;
                let slot = &mut records[base + j * per];
                *slot = if hand { lane.symbol } else { *slot };
                lane.holding &= !hand;
                given |= u32::from(hand) << j;
                moved |= hand;
            }
            if given == all {
                given = 0;
                self.position += 1;
                if self.position == per {
                    self.position = 0;
                    block += 1;
                }
                moved = true;
            }
            // A few rounds with no byte taken and no symbol given: no lane can go on.
            idle = if taken > 0 || moved { 0 } else { idle + 1 };
            if idle > 3 {
                return Err(Error::Truncated);
            }
        }
        self.given = given;
        Ok(())
    }
}
