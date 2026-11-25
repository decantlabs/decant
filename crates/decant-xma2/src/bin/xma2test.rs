//! `xma2test <decant_xma2.dll> <file.xma2.wav> [out.raw]`: decodes a RIFF WAVE_FORMAT_XMA2 file
//! through the `DecantXma2Decoder` ABI exactly as a caller does. Open with the fmt chunk, Decode the
//! data chunk in 2048 byte blocks with a small output buffer (so leftovers are carried across
//! calls), then flush. Prints the frame count, peak and RMS, and writes interleaved signed 16 bit
//! samples to `out.raw` when given. Needs no game file: any XMA2 RIFF file works.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(run::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("xma2test runs on Windows only");
}

/// A chunk as its offset and size in the file.
type Chunk = Option<(usize, usize)>;

/// The fmt and data chunks of a RIFF file.
#[cfg_attr(not(windows), allow(dead_code))]
fn chunks(file: &[u8]) -> (Chunk, Chunk) {
    let u32_at = |at: usize| u32::from_le_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]]) as usize;
    let (mut fmt, mut data) = (None, None);
    let mut at = 12usize;
    while at + 8 <= file.len() {
        let n = u32_at(at + 4);
        if at + 8 + n > file.len() {
            break;
        }
        match &file[at..at + 4] {
            b"fmt " => fmt = Some((at + 8, n)),
            b"data" => data = Some((at + 8, n)),
            _ => {}
        }
        at += 8 + n + (n & 1);
    }
    (fmt, data)
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::c_void;
    use std::io::Write;

    use decant_testkit::Library;
    use decant_xma2::abi::{ABI_VERSION, GetDecoderFn, DecantXma2Decoder};

    pub fn main() -> u8 {
        let args = decant_testkit::args();
        if args.len() < 2 {
            println!("usage: xma2test <decant_xma2.dll> <file.xma2.wav> [out.raw]");
            return 2;
        }
        let lib = Library::open(&args[0]).ok();
        // SAFETY: the export of the decoder DLL has the signature of `GetDecoderFn`.
        let get = lib.as_ref().and_then(|l| unsafe { l.get::<GetDecoderFn>("DecantXma2GetDecoder") });
        let mut d = DecantXma2Decoder::default();
        // SAFETY: `d` is a writable decoder table.
        let loaded = get.is_some_and(|get| unsafe { get(&mut d) } == 0 && d.abi_version == ABI_VERSION);
        let (Some(open), Some(decode), Some(close), true) = (d.open, d.decode, d.close, loaded) else {
            println!("FAIL: decoder not loaded");
            return 1;
        };
        let Ok(file) = std::fs::read(&args[1]) else {
            return 2;
        };
        let (fmt, data) = crate::chunks(&file);
        let (Some((fmt_at, fmt_bytes)), Some((data_at, data_bytes)), true) = (fmt, data, file.starts_with(b"RIFF")) else {
            println!("FAIL: not a RIFF file with fmt and data chunks");
            return 1;
        };
        let fmt = &file[fmt_at..fmt_at + fmt_bytes];
        let data = &file[data_at..data_at + data_bytes];
        let channels = u32::from(fmt.get(2).copied().unwrap_or(0)) | u32::from(fmt.get(3).copied().unwrap_or(0)) << 8;
        // SAFETY: the ABI of the decoder table: Open takes the fmt bytes.
        let ctx = unsafe { open(fmt.as_ptr().cast::<c_void>(), fmt_bytes as u32) };
        if ctx.is_null() {
            println!("FAIL: Open refused the format");
            return 1;
        }
        let mut out = args.get(2).and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new);
        let mut pcm = [0i16; 1024 * 8];
        const MAX_FRAMES: u32 = 1024;
        let mut frames = 0u64;
        let mut sum = 0f64;
        let mut peak = 0i32;
        // Each 2048 byte block, then calls with no new data while the small buffer comes back full;
        // then the flush (a null block) until it runs dry.
        let mut at = 0usize;
        loop {
            let flush = at + 2048 > data_bytes;
            let mut first = true;
            loop {
                let mut got = 0u32;
                let block = if flush { std::ptr::null() } else { data[at..].as_ptr() };
                let bytes = if flush || !first { 0 } else { 2048 };
                // SAFETY: the block is 2048 readable bytes of the data chunk, or null for the flush,
                // and the buffer holds MAX_FRAMES frames of up to 8 channels.
                if unsafe { decode(ctx, block, bytes, pcm.as_mut_ptr(), MAX_FRAMES, &mut got) } != 0 {
                    println!("FAIL: Decode returned an error at byte {at}");
                    return 1;
                }
                let samples = &pcm[..got as usize * channels as usize];
                for &v in samples {
                    let v = i32::from(v);
                    sum += f64::from(v) * f64::from(v);
                    peak = peak.max(v.abs());
                }
                if let Some(out) = out.as_mut() {
                    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                    let _ = out.write_all(&bytes);
                }
                frames += u64::from(got);
                if got < MAX_FRAMES {
                    break;
                }
                first = false;
            }
            if flush {
                break;
            }
            at += 2048;
        }
        // SAFETY: `ctx` came from Open and is closed once.
        unsafe { close(ctx) };
        if let Some(mut out) = out {
            let _ = out.flush();
        }
        let rms = if frames > 0 { (sum / (frames * u64::from(channels)) as f64).sqrt() } else { 0.0 };
        println!("channels {channels}, frames {frames}, peak {peak}, rms {rms:.1}");
        println!("{}", if frames > 0 { "XMA2TEST PASS" } else { "XMA2TEST FAIL: nothing decoded" });
        u8::from(frames == 0)
    }
}
