//! `xaudio2test <decant_xaudio2.dll>`: drives the XAudio2 layer the way a console title does, through
//! the exported `XAudio2Create`, then the `IXAudio2` vtable. The port of the former
//! `tests/xaudio2test.cpp`, with the same lines and exit codes. Everything except the last step
//! runs without an audio device (the hosted CI runner has none). The last step plays a synthetic
//! sine through the layer and verifies, with the engine's own sample counter, that the samples
//! were consumed. With no audio endpoint it prints SKIP and still passes (set
//! `DECANT_XA2_REQUIRE_DEVICE=1` to make that a failure on a machine that has speakers).
//!
//! Checks, in order: exports. Engine creation with a console style flag and processor (logged and
//! handled). IUnknown and the extension interface. A made up IID (console only interface)
//! answered E_NOINTERFACE and counted. An XMA2 format rejected before the engine sees it. The sine.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(run::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("xaudio2test runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::c_void;

    use decant_testkit::{Checks, Library};
    use decant_xaudio2::DecantXa2Stats;
    use windows::Win32::Foundation::{E_NOINTERFACE, HMODULE};
    use windows::Win32::Media::Audio::XAudio2::{
        IXAudio2, IXAudio2Extension, IXAudio2MasteringVoice, IXAudio2SourceVoice, Processor1, XAUDIO2_BUFFER, XAUDIO2_DEFAULT_FREQ_RATIO,
        XAUDIO2_VOICE_STATE,
    };
    use windows::Win32::Media::Audio::{AudioCategory_GameEffects, WAVEFORMATEX};
    use windows::Win32::System::LibraryLoader::GetProcAddress;
    use windows::core::{GUID, HRESULT, IUnknown, Interface, PCSTR, PCWSTR};

    type CreateFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32) -> HRESULT;
    type StatsFn = unsafe extern "system" fn(*mut DecantXa2Stats);
    type ResetFn = unsafe extern "system" fn();

    const UNKNOWN_IID: GUID = GUID::from_values(0x5ca1ab1e, 0x0001, 0x4000, [0x80, 0x00, 0x53, 0x49, 0x58, 0x58, 0x41, 0x32]);
    const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

    fn require_device() -> u8 {
        u8::from(std::env::var_os("DECANT_XA2_REQUIRE_DEVICE").is_some())
    }

    pub fn main() -> u8 {
        let args = decant_testkit::args();
        if args.len() != 1 {
            return 2;
        }
        let lib = match Library::open(&args[0]) {
            Ok(lib) => lib,
            Err(error) => {
                println!("load failed {error}");
                return 2;
            }
        };
        let mut c = Checks::new(72, "OK", "FAIL");
        // SAFETY: the signatures of the former decant_xa2.h and xaudio2.h for these names.
        let (create, stats, reset) =
            unsafe { (lib.get::<CreateFn>("XAudio2Create"), lib.get::<StatsFn>("DecantXa2GetStats"), lib.get::<ResetFn>("DecantXa2ResetStats")) };
        c.check(create.is_some() && stats.is_some() && reset.is_some(), "XAudio2Create and the stats exports are present");
        let (Some(create), Some(stats), Some(reset)) = (create, stats, reset) else {
            return 1;
        };
        // Console modules import by ordinal (the same ones as the PC XAudio2_9.dll): the names
        // alone are not enough.
        for (name, ordinal) in [("XAudio2Create", 1u16), ("CreateAudioReverb", 2), ("CreateAudioVolumeMeter", 3), ("XAudio2CreateWithVersionInfo", 9), ("DecantXa2GetStats", 20), ("DecantXa2ResetStats", 21)] {
            // SAFETY: an ordinal lookup passes the ordinal in the low word of the name pointer.
            let by_ordinal = unsafe { GetProcAddress(HMODULE(lib.handle()), PCSTR(usize::from(ordinal) as *const u8)) }.map(|f| f as usize);
            c.check(by_ordinal.is_some() && by_ordinal == Some(lib.address(name) as usize), &format!("{name} is exported at ordinal {ordinal}"));
        }
        let read = || {
            let mut s = DecantXa2Stats { struct_size: size_of::<DecantXa2Stats>() as u32, ..Default::default() };
            // SAFETY: `s` is a stats structure with its size set.
            unsafe { stats(&mut s) };
            s
        };
        // SAFETY: no arguments.
        unsafe { reset() };

        // Console style call: processor mask Processor1 and an unknown flag bit.
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `raw` is a valid output location.
        let hr = unsafe { create(&mut raw, 0x4000 /* not a PC flag */, Processor1) };
        if hr.is_err() || raw.is_null() {
            println!("  XAudio2Create failed 0x{:08X}: no PC XAudio2 on this machine, nothing more to test", hr.0 as u32);
            return require_device();
        }
        // SAFETY: on success the layer returned an IXAudio2 with one reference, owned here.
        let xa = unsafe { IXAudio2::from_raw(raw) };
        let s = read();
        c.check(s.engines_created == 1, "engine created through the layer");
        c.check(s.create_flags_unknown == 1 && s.processors_nondefault == 1, "unknown flag bit and non-default processor logged");

        c.check(xa.cast::<IUnknown>().is_ok(), "QueryInterface(IUnknown)");
        let ext = xa.cast::<IXAudio2Extension>();
        c.check(ext.is_ok(), "QueryInterface(IXAudio2Extension) answered by the PC engine");
        if let Ok(ext) = ext {
            let (mut num, mut den, mut proc) = (0u32, 0u32, 0u32);
            // SAFETY: three valid output locations.
            unsafe {
                ext.GetProcessingQuantum(&mut num, &mut den);
                ext.GetProcessor(&mut proc);
            }
            println!("  quantum {num}/{den} s, processor mask 0x{proc:08X}");
            c.check(den != 0 && proc != 0, "extension returns a non-zero quantum denominator and processor");
        }

        let mut none = std::ptr::dangling_mut::<c_void>();
        // SAFETY: a raw QueryInterface on the live engine with a valid IID and output location.
        let hr = unsafe { (Interface::vtable(&xa).base__.QueryInterface)(xa.as_raw(), &UNKNOWN_IID, &mut none) };
        let s = read();
        c.check(hr == E_NOINTERFACE && none.is_null() && s.unknown_iids == 1, "made-up IID -> E_NOINTERFACE, output NULL, counted");

        // XMA2: format tag 0x166. The layer must refuse before the PC engine is asked (works without an audio device).
        let xma = WAVEFORMATEX { wFormatTag: 0x0166, nChannels: 2, nSamplesPerSec: 48000, nAvgBytesPerSec: 192000, nBlockAlign: 4, wBitsPerSample: 16, cbSize: 0 };
        let mut bad = std::ptr::dangling_mut::<c_void>();
        // SAFETY: a raw call on the live engine, with the defaults of the C++ header.
        let hr = unsafe {
            (Interface::vtable(&xa).CreateSourceVoice)(xa.as_raw(), &mut bad, &xma, 0, XAUDIO2_DEFAULT_FREQ_RATIO, std::ptr::null_mut(), std::ptr::null(), std::ptr::null())
        };
        let s = read();
        c.check(hr.0 == decant_xaudio2::XAUDIO2_E_INVALID_CALL && bad.is_null() && s.xma_rejected == 1, "XMA2 format rejected with XAUDIO2_E_INVALID_CALL, counted");

        // Sine through the mastering voice.
        let mut master: Option<IXAudio2MasteringVoice> = None;
        // SAFETY: a valid output location and the defaults of the C++ header.
        let hr = unsafe { xa.CreateMasteringVoice(&mut master, 2, 48000, 0x40000 /* console only bit */, PCWSTR::null(), None, AudioCategory_GameEffects) };
        let s = read();
        let Some(master) = master.filter(|_| hr.is_ok()) else {
            let code = hr.err().map_or(0, |e| e.code().0 as u32);
            println!("  SKIP sine: no audio endpoint (CreateMasteringVoice 0x{code:08X})");
            drop(xa);
            return require_device() + u8::from(c.failures() != 0);
        };
        c.check(s.mastering_flags_unknown == 1, "console-only mastering flag bit logged");

        let rate = 48000u32;
        let n = rate / 2;
        let pcm: Vec<f32> = (0..n).map(|i| 0.2f32 * (std::f32::consts::TAU * 440.0f32 * i as f32 / rate as f32).sin()).collect();
        let crossings = pcm.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        c.check((438..=442).contains(&crossings), "synthetic sine has 440 Hz (880 zero crossings per second)");

        let fmt = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_IEEE_FLOAT,
            nChannels: 1,
            nSamplesPerSec: rate,
            nAvgBytesPerSec: rate * 4,
            nBlockAlign: 4,
            wBitsPerSample: 32,
            cbSize: 0,
        };
        let mut voice: Option<IXAudio2SourceVoice> = None;
        // SAFETY: a valid output location and format, with the defaults of the C++ header.
        let created = unsafe { xa.CreateSourceVoice(&mut voice, &fmt, 0, XAUDIO2_DEFAULT_FREQ_RATIO, None, None, None) }.is_ok();
        c.check(created && voice.is_some(), "float32 mono 48 kHz source voice created through the layer");
        if let Some(voice) = voice {
            let buf = XAUDIO2_BUFFER { AudioBytes: n * 4, pAudioData: pcm.as_ptr().cast(), ..Default::default() };
            // SAFETY: `buf` and `pcm` live until the voice is destroyed below.
            unsafe {
                c.check(voice.SubmitSourceBuffer(&buf, None).is_ok(), "SubmitSourceBuffer");
                c.check(voice.Start(0, 0).is_ok(), "Start");
                let mut st = XAUDIO2_VOICE_STATE::default();
                let mut most = 0u64; // XAudio2 resets SamplesPlayed at the end of a stream: keep the largest value seen
                for _ in 0..400 {
                    voice.GetState(&mut st, 0);
                    most = most.max(st.SamplesPlayed);
                    if st.BuffersQueued == 0 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                println!("  samples played: {most} of {n}");
                c.check(st.BuffersQueued == 0 && most >= u64::from(n * 9 / 10), "the engine consumed the sine (BuffersQueued 0, SamplesPlayed >= 90%)");
                voice.DestroyVoice();
            }
        }
        // SAFETY: the mastering voice is destroyed once, before the engine is released.
        unsafe { master.DestroyVoice() };
        drop(xa);
        u8::from(c.failures() != 0)
    }
}
