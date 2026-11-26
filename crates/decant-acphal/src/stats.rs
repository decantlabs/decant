//! `DECANT_AUDIO_STATS=1`: what the XMA engine and the monitor do, for a run to be read afterwards.
//!
//! Per voice (one XMA context): blocks written into its ring, blocks the title took, the times the ring was full, the times the
//! decoder had nothing to write while the ring had room, and the frames the monitor had to play while the voice had none
//! (an underrun). Per device: the periods the monitor mixed and its frames. Every second a line with a timestamp (milliseconds
//! since the first audio call) gives the device and up to eight voices that are running. A voice that stops gets a last line.
//!
//! Off unless the value is exactly `1`. Nothing here changes what the engine or the monitor does. The counters are only read.
//! An underrun is counted when a voice that is still running has no frame to play for a device period. It is only seen while
//! the monitor plays (`DECANT_AUDIO_RAW_MONITOR=1`).
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use decant_log::trace;

/// How many running voices a periodic line names.
const LISTED_VOICES: usize = 8;

/// Counters of one voice (one XMA context).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VoiceStats {
    /// Blocks written into the output ring.
    pub blocks_written: u64,
    /// Blocks the title took (command 0x12).
    pub blocks_taken: u64,
    /// Times the engine found the ring full (writable flag cleared).
    pub ring_full: u64,
    /// Times the ring had room but the decoder had no sample to write.
    pub decoder_empty: u64,
    /// Frames the monitor had to play for this voice while it had none.
    pub underrun_frames: u64,
    /// Device periods in which this voice ran short.
    pub underrun_periods: u64,
}

/// Counters of the monitor's device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceStats {
    /// Mix calls (one per device period).
    pub periods: u64,
    /// Frames mixed.
    pub frames: u64,
    /// Frames missing across all running voices.
    pub underrun_frames: u64,
}

/// The ring of a context at the time of a line: blocks, read and write positions, and the writable flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    pub blocks: u32,
    pub read: u32,
    pub write: u32,
    pub writable: bool,
}

impl Ring {
    /// Blocks between the read and the write position (the ring is full when `writable` is off).
    pub fn occupied(self) -> u32 {
        if self.blocks == 0 {
            return 0;
        }
        if !self.writable {
            return self.blocks;
        }
        (self.write + self.blocks - self.read % self.blocks) % self.blocks
    }
}

struct State {
    voices: BTreeMap<u32, VoiceStats>,
    device: DeviceStats,
    last_report_ms: u64,
}

static STATE: Mutex<State> = Mutex::new(State { voices: BTreeMap::new(), device: DeviceStats { periods: 0, frames: 0, underrun_frames: 0 }, last_report_ms: 0 });
static START: OnceLock<Instant> = OnceLock::new();

/// Whether `DECANT_AUDIO_STATS` turns the statistics on. It is off unless the value is exactly `1`.
pub fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether the statistics are on for this process (read once).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| enabled_value(std::env::var("DECANT_AUDIO_STATS").ok().as_deref()))
}

/// Milliseconds since the first call of this function.
pub fn now_ms() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn voice(index: u32, update: impl FnOnce(&mut VoiceStats)) {
    if enabled() {
        update(state().voices.entry(index).or_default());
    }
}

/// The engine wrote one block of context `index` into its ring.
pub fn block_written(index: u32) {
    voice(index, |v| v.blocks_written += 1);
}

/// The title took `blocks` blocks of context `index`.
pub fn blocks_taken(index: u32, blocks: u32) {
    voice(index, |v| v.blocks_taken += u64::from(blocks));
}

/// The ring of context `index` became full.
pub fn ring_full(index: u32) {
    voice(index, |v| v.ring_full += 1);
}

/// The decoder of context `index` had nothing to write while the ring had room.
pub fn decoder_empty(index: u32) {
    voice(index, |v| v.decoder_empty += 1);
}

/// The monitor mixed a period of `frames` frames and context `index` had only `missing` of them (it is still running).
pub fn underrun(index: u32, missing: u64) {
    if !enabled() || missing == 0 {
        return;
    }
    let mut s = state();
    s.device.underrun_frames += missing;
    let v = s.voices.entry(index).or_default();
    v.underrun_frames += missing;
    v.underrun_periods += 1;
}

/// The monitor mixed one period of `frames` frames.
pub fn device_period(frames: u64) {
    if enabled() {
        let mut s = state();
        s.device.periods += 1;
        s.device.frames += frames;
    }
}

/// The counters of context `index`, if any were kept.
pub fn snapshot(index: u32) -> Option<VoiceStats> {
    state().voices.get(&index).copied()
}

/// The counters of the device.
pub fn device() -> DeviceStats {
    state().device
}

/// Whether a second has passed since the last periodic line. If so, it is recorded as the last one.
pub fn report_due() -> bool {
    if !enabled() {
        return false;
    }
    let t = now_ms();
    let mut s = state();
    if t.saturating_sub(s.last_report_ms) >= 1000 {
        s.last_report_ms = t;
        true
    } else {
        false
    }
}

/// The line of the device counters.
pub fn device_line(t_ms: u64) -> String {
    let d = device();
    format!(
        "AcpHal: AUDIO_STATS t={t_ms}ms device periods {} frames {} underrun_frames {}",
        d.periods, d.frames, d.underrun_frames
    )
}

/// The line of one voice: ring, frames, and counters. `ended` is set for the last line of a voice that stopped.
pub fn voice_line(t_ms: u64, index: u32, ring: Ring, frames: u64, bad_frames: u64, ended: bool) -> String {
    let v = snapshot(index).unwrap_or_default();
    format!(
        "AcpHal: AUDIO_STATS t={t_ms}ms ctx {index}{} ring {}/{} blocks (read {} write {} writable {}) frames {frames} ({bad_frames} bad) written {} taken {} ring_full {} decoder_empty {} underrun {} frames in {} periods",
        if ended { " ended" } else { "" },
        ring.occupied(),
        ring.blocks,
        ring.read,
        ring.write,
        u8::from(ring.writable),
        v.blocks_written,
        v.blocks_taken,
        v.ring_full,
        v.decoder_empty,
        v.underrun_frames,
        v.underrun_periods,
    )
}

/// The voice stopped: its last line is returned and its counters are forgotten.
pub fn voice_end(index: u32, ring: Ring, frames: u64, bad_frames: u64) -> Option<String> {
    if !enabled() {
        return None;
    }
    let line = voice_line(now_ms(), index, ring, frames, bad_frames, true);
    state().voices.remove(&index);
    Some(line)
}

/// Writes the periodic lines: the device, and up to eight voices that are running, given as (index, ring, frames, bad frames).
pub fn report(running: &[(u32, Ring, u64, u64)]) {
    if !enabled() {
        return;
    }
    let t = now_ms();
    trace(&device_line(t));
    for &(index, ring, frames, bad) in running.iter().take(LISTED_VOICES) {
        trace(&voice_line(t, index, ring, frames, bad, false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_require_exact_opt_in() {
        assert!(!enabled_value(None));
        assert!(!enabled_value(Some("")));
        assert!(!enabled_value(Some("0")));
        assert!(!enabled_value(Some("true")));
        assert!(enabled_value(Some("1")));
    }

    #[test]
    fn ring_occupancy_wraps_and_full_ring_is_all_blocks() {
        let r = |read, write, writable| Ring { blocks: 23, read, write, writable };
        assert_eq!(r(0, 0, true).occupied(), 0);
        assert_eq!(r(0, 5, true).occupied(), 5);
        assert_eq!(r(20, 3, true).occupied(), 6);
        assert_eq!(r(0, 0, false).occupied(), 23);
        assert_eq!(Ring { blocks: 0, read: 0, write: 0, writable: true }.occupied(), 0);
    }

    #[test]
    fn underrun_and_device_counters_accumulate_only_when_on() {
        // Off by default in tests (no DECANT_AUDIO_STATS): nothing is kept.
        if enabled() {
            return;
        }
        underrun(7001, 100);
        device_period(480);
        assert_eq!(snapshot(7001), None);
        assert_eq!(device(), DeviceStats::default());
    }

    #[test]
    fn a_voice_line_names_ring_and_counters() {
        let line = voice_line(12, 3, Ring { blocks: 23, read: 1, write: 4, writable: true }, 96, 0, false);
        assert!(line.starts_with("AcpHal: AUDIO_STATS t=12ms ctx 3 ring 3/23 blocks (read 1 write 4 writable 1)"), "{line}");
        assert!(line.contains("frames 96 (0 bad)"), "{line}");
        let ended = voice_line(13, 3, Ring { blocks: 23, read: 1, write: 4, writable: true }, 96, 0, true);
        assert!(ended.contains("ctx 3 ended ring"), "{ended}");
    }
}
