//! Plays what the XMA engine writes into the output rings on the default Windows audio device (WASAPI, shared mode, 48 kHz
//! stereo), so a run can be listened to. The console mixes the rings through SHAPE units the engine does not run, so without
//! this nothing of the decoded audio reaches the speakers.
//!
//! `DECANT_AUDIO_RAW_MONITOR=1` turns it on. Each context is converted to 48 kHz by linear interpolation and mixed. The device
//! opens at the first decoded block.
#![allow(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Mutex, MutexGuard, OnceLock};

use decant_log::trace;

/// The device rate.
pub const RATE: u32 = 48_000;
/// At most this many source frames are kept per context (two seconds at 48 kHz). Older ones are dropped.
const MAX_QUEUED: usize = 96_000;

struct Stream {
    rate: u32,
    /// Stereo frames at `rate`.
    frames: VecDeque<(f32, f32)>,
    /// Position between `frames[0]` and `frames[1]`.
    phase: f64,
    ended: bool,
}

static STREAMS: Mutex<BTreeMap<u32, Stream>> = Mutex::new(BTreeMap::new());

fn streams() -> MutexGuard<'static, BTreeMap<u32, Stream>> {
    STREAMS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the raw XMA monitor is enabled. It is off unless the value is exactly `1`.
pub fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| enabled_value(std::env::var("DECANT_AUDIO_RAW_MONITOR").ok().as_deref()))
}

/// Samples (interleaved, `channels` 1 or 2) that context `index` wrote at `rate`.
pub fn feed(index: u32, rate: u32, channels: usize, samples: &[i16]) {
    if !enabled() {
        return;
    }
    start_device();
    let mut s = streams();
    let stream = s.entry(index).or_insert_with(|| Stream { rate, frames: VecDeque::new(), phase: 0.0, ended: false });
    stream.rate = rate;
    stream.ended = false;
    let scale = 1.0 / 32768.0;
    if channels == 2 {
        stream.frames.extend(samples.as_chunks::<2>().0.iter().map(|p| (f32::from(p[0]) * scale, f32::from(p[1]) * scale)));
    } else {
        stream.frames.extend(samples.iter().map(|&v| (f32::from(v) * scale, f32::from(v) * scale)));
    }
    let excess = stream.frames.len().saturating_sub(MAX_QUEUED);
    stream.frames.drain(..excess);
}

/// Context `index` stopped: what it queued still plays, then it is forgotten.
pub fn end(index: u32) {
    if let Some(s) = streams().get_mut(&index) {
        s.ended = true;
    }
}

/// Mixes `out.len() / 2` stereo frames at 48 kHz from every context.
pub fn mix(out: &mut [f32]) {
    out.fill(0.0);
    let mut s = streams();
    for stream in s.values_mut() {
        let step = f64::from(stream.rate) / f64::from(RATE);
        for frame in out.as_chunks_mut::<2>().0 {
            if stream.frames.len() < 2 {
                break;
            }
            let (a, b) = (stream.frames[0], stream.frames[1]);
            let t = stream.phase as f32;
            frame[0] += a.0 + (b.0 - a.0) * t;
            frame[1] += a.1 + (b.1 - a.1) * t;
            stream.phase += step;
            while stream.phase >= 1.0 && !stream.frames.is_empty() {
                stream.frames.pop_front();
                stream.phase -= 1.0;
            }
        }
    }
    s.retain(|_, stream| !(stream.ended && stream.frames.len() < 2));
    for v in out.iter_mut() {
        *v = v.clamp(-1.0, 1.0);
    }
}

/// Opens the device and starts the thread that feeds it, once.
fn start_device() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let spawned = std::thread::Builder::new().name("AcpHal XMA monitor".into()).spawn(|| {
            if let Err(e) = device::run() {
                trace(&format!("AcpHal: the XMA monitor has no audio device ({e})"));
            }
        });
        if spawned.is_ok() {
            trace("AcpHal: the raw XMA monitor plays on the default device (DECANT_AUDIO_RAW_MONITOR=1 enables it)");
        }
    });
}

#[cfg(windows)]
mod device {
    use windows::Win32::Media::Audio::{
        AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, IAudioClient, IAudioRenderClient,
        IMMDeviceEnumerator, MMDeviceEnumerator, WAVEFORMATEX, eConsole, eRender,
    };
    use windows::Win32::System::Com::{CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx};

    use decant_log::trace;

    use super::{RATE, mix};

    /// Feeds the device every 10 ms until the process ends.
    pub(super) fn run() -> windows::core::Result<()> {
        // SAFETY: plain COM and WASAPI calls on this thread, with the format and buffers they document.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            trace(&format!("AcpHal: the XMA monitor thread: {}", crate::priority::raise_current_thread()));
            let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
            let format = WAVEFORMATEX {
                wFormatTag: 1,
                nChannels: 2,
                nSamplesPerSec: RATE,
                nAvgBytesPerSec: RATE * 4,
                nBlockAlign: 4,
                wBitsPerSample: 16,
                cbSize: 0,
            };
            // A 100 ms buffer.
            client.Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, 1_000_000, 0, &format, None)?;
            let size = client.GetBufferSize()?;
            let render: IAudioRenderClient = client.GetService()?;
            client.Start()?;
            trace(&format!("AcpHal: the XMA monitor plays on the default device, buffer of {size} frames"));
            let mut mixed = Vec::new();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let free = size - client.GetCurrentPadding()?;
                if free == 0 {
                    continue;
                }
                mixed.resize(free as usize * 2, 0.0);
                mix(&mut mixed);
                let buffer = render.GetBuffer(free)?;
                let out = std::slice::from_raw_parts_mut(buffer.cast::<i16>(), free as usize * 2);
                for (o, v) in out.iter_mut().zip(&mixed) {
                    *o = (v * 32767.0) as i16;
                }
                render.ReleaseBuffer(free, 0)?;
            }
        }
    }
}

#[cfg(not(windows))]
mod device {
    pub(super) fn run() -> Result<(), &'static str> {
        Err("no audio output on this platform")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_monitor_requires_exact_opt_in() {
        assert!(!enabled_value(None));
        assert!(!enabled_value(Some("")));
        assert!(!enabled_value(Some("0")));
        assert!(!enabled_value(Some("true")));
        assert!(enabled_value(Some("1")));
    }

    #[test]
    fn a_24_khz_stream_is_stretched_to_48_khz() {
        let mut s = streams();
        s.insert(9001, Stream { rate: 24_000, frames: (0..100).map(|i| (i as f32, -(i as f32))).collect(), phase: 0.0, ended: true });
        drop(s);
        let mut out = vec![0f32; 8];
        mix(&mut out);
        // Clamped to ±1, so compare the first two frames: 0, then halfway to 1.
        assert_eq!(&out[..4], &[0.0, 0.0, 0.5, -0.5]);
        streams().remove(&9001);
    }
}
