//! The counters returned by `DecantXa2GetStats`, the Rust form of the former `decant_xa2.h`.

use std::sync::atomic::{AtomicU32, Ordering};

/// `DecantXa2Stats`, plain C layout. A caller sets `struct_size` before it asks.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecantXa2Stats {
    pub struct_size: u32,
    /// `XAudio2Create` calls that returned an engine.
    pub engines_created: u32,
    /// `XAudio2Create` calls with flag bits the PC engine does not define.
    pub create_flags_unknown: u32,
    /// ... of which needed a retry with the unknown bits removed.
    pub create_retried: u32,
    /// `XAudio2Create` calls with a processor mask other than "use default".
    pub processors_nondefault: u32,
    /// `QueryInterface` calls for an IID the PC engine does not answer.
    pub unknown_iids: u32,
    /// `CreateSourceVoice` calls with an XMA or XMA2 format and no decoder available.
    pub xma_rejected: u32,
    /// `CreateSourceVoice` calls with another format tag the PC engine lacks.
    pub formats_unsupported: u32,
    pub mastering_flags_unknown: u32,
    pub mastering_retried: u32,
    /// Calls passed through untouched.
    pub forwarded_calls: u32,
    /// Threads on which the layer had to call `CoInitializeEx`.
    pub com_initialised: u32,
}

/// Size of `DecantXa2Stats`, the value of `struct_size`.
pub const STATS_SIZE: u32 = size_of::<DecantXa2Stats>() as u32;

const FIELDS: usize = 12;
const _: () = assert!(size_of::<DecantXa2Stats>() == FIELDS * 4);

/// Index of each counter in `DecantXa2Stats`, counted in `u32` fields.
#[derive(Clone, Copy)]
pub enum Counter {
    EnginesCreated = 1,
    CreateFlagsUnknown,
    CreateRetried,
    ProcessorsNondefault,
    UnknownIids,
    XmaRejected,
    FormatsUnsupported,
    MasteringFlagsUnknown,
    MasteringRetried,
    ForwardedCalls,
    ComInitialised,
}

static COUNTERS: [AtomicU32; FIELDS] = [const { AtomicU32::new(0) }; FIELDS];

/// Adds one to a counter.
pub fn bump(counter: Counter) {
    COUNTERS[counter as usize].fetch_add(1, Ordering::SeqCst);
}

/// The counters now.
pub fn snapshot() -> DecantXa2Stats {
    let v: [u32; FIELDS] = std::array::from_fn(|i| if i == 0 { STATS_SIZE } else { COUNTERS[i].load(Ordering::SeqCst) });
    DecantXa2Stats {
        struct_size: v[0],
        engines_created: v[1],
        create_flags_unknown: v[2],
        create_retried: v[3],
        processors_nondefault: v[4],
        unknown_iids: v[5],
        xma_rejected: v[6],
        formats_unsupported: v[7],
        mastering_flags_unknown: v[8],
        mastering_retried: v[9],
        forwarded_calls: v[10],
        com_initialised: v[11],
    }
}

/// Sets every counter back to zero.
pub fn reset() {
    for counter in &COUNTERS[1..] {
        counter.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_count_and_reset() {
        reset();
        bump(Counter::EnginesCreated);
        bump(Counter::ComInitialised);
        bump(Counter::ComInitialised);
        let s = snapshot();
        assert_eq!((s.struct_size, s.engines_created, s.com_initialised, s.xma_rejected), (48, 1, 2, 0));
        reset();
        assert_eq!(snapshot(), DecantXa2Stats { struct_size: 48, ..Default::default() });
    }
}
