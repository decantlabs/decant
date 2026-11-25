//! Runs `xma2test` against the `decant_xma2.dll` Cargo built next to it (or the one in
//! `DECANT_DLL_DIR`), as the review workflow ran the C test against the C DLL.
//!
//! The decode test needs an LGPL shared FFmpeg (`avcodec-62.dll` and `avutil-60.dll`), found in
//! `DECANT_FFMPEG_DIR` or in `bin` under `DECANT_FFMPEG`. Without one it prints SKIP and passes, as
//! the build never required FFmpeg. The input is a synthetic XMA2 file: packets with a valid
//! header and pseudo random payload. FFmpeg decodes that deterministically, so no game file and
//! no encoder is needed.
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The folder holding the FFmpeg libraries this DLL loads, if there is one.
fn ffmpeg_dir() -> Option<PathBuf> {
    let has_libraries = |dir: &Path| dir.join(decant_xma2::avcodec_dll()).is_file() && dir.join(decant_xma2::avutil_dll()).is_file();
    let direct = std::env::var_os("DECANT_FFMPEG_DIR").map(PathBuf::from);
    let from_root = std::env::var_os("DECANT_FFMPEG").map(|root| Path::new(&root).join("bin"));
    direct.into_iter().chain(from_root).find(|dir| has_libraries(dir))
}

/// A RIFF WAVE_FORMAT_XMA2 file of `packets` 2048 byte packets with random payload.
fn synthetic_xma2(seed: u64, packets: usize, channels: u16) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = move || {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut data = Vec::with_capacity(packets * 2048);
    for _ in 0..packets {
        let mut packet: Vec<u8> = (0..2048).map(|_| next() as u8).collect();
        // Frame count 1 to 3 in the 6 bit field, the first frame right after the 32 bit header.
        let header = ((1 + next() % 3) as u32) << 26 | 32 << 11;
        packet[..4].copy_from_slice(&header.to_be_bytes());
        data.extend(packet);
    }
    let samples = (packets * 512) as u32;
    let rate = 48_000u32;
    let mut format = Vec::new();
    format.extend(0x166u16.to_le_bytes());
    format.extend(channels.to_le_bytes());
    format.extend(rate.to_le_bytes());
    format.extend((rate * 2 * u32::from(channels)).to_le_bytes());
    format.extend(2048u16.to_le_bytes());
    format.extend(16u16.to_le_bytes());
    format.extend(34u16.to_le_bytes());
    format.extend(1u16.to_le_bytes());
    for v in [if channels == 2 { 3u32 } else { 4 }, samples, 2048, 0, samples, 0, 0] {
        format.extend(v.to_le_bytes());
    }
    format.extend([0u8, 2]);
    format.extend((packets as u16).to_le_bytes());
    let mut body = Vec::new();
    body.extend(b"WAVEfmt ");
    body.extend((format.len() as u32).to_le_bytes());
    body.extend(&format);
    body.extend(b"data");
    body.extend((data.len() as u32).to_le_bytes());
    body.extend(&data);
    let mut file = b"RIFF".to_vec();
    file.extend((body.len() as u32).to_le_bytes());
    file.extend(body);
    file
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("decant_xma2-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch folder");
    dir
}

fn xma2test(ffmpeg_dir: Option<&Path>, args: &[&Path]) -> (std::process::ExitStatus, String) {
    let program = env!("CARGO_BIN_EXE_xma2test");
    let mut command = Command::new(program);
    command.arg(decant_testkit::dll_for(program, "decant_xma2.dll")).args(args);
    match ffmpeg_dir {
        Some(dir) => command.env("DECANT_FFMPEG_DIR", dir),
        None => command.env_remove("DECANT_FFMPEG_DIR"),
    };
    let out = command.output().expect("the test program starts");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    print!("{stdout}");
    (out.status, stdout)
}

#[test]
fn xma2test_decodes_a_synthetic_stream() {
    let Some(ffmpeg) = ffmpeg_dir() else {
        println!("SKIP: no LGPL shared FFmpeg (set DECANT_FFMPEG or DECANT_FFMPEG_DIR)");
        return;
    };
    let dir = scratch("decode");
    for (channels, seed) in [(1u16, 5u64), (2, 8)] {
        let input = dir.join(format!("in{channels}.xma2.wav"));
        let raw = dir.join(format!("out{channels}.raw"));
        std::fs::write(&input, synthetic_xma2(seed, 24, channels)).unwrap();
        let (status, stdout) = xma2test(Some(&ffmpeg), &[&input, &raw]);
        assert!(status.success(), "xma2test failed on {channels} channels: {status:?}");
        assert!(stdout.contains(&format!("channels {channels}, frames ")), "{stdout}");
        assert!(stdout.contains("XMA2TEST PASS"), "{stdout}");
        // Whole frames of 16 bit samples reached the output file.
        let written = std::fs::metadata(&raw).unwrap().len();
        assert!(written > 0 && written.is_multiple_of(2 * u64::from(channels)), "{written} bytes");
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_folder_without_ffmpeg_is_refused() {
    let dir = scratch("empty");
    let input = dir.join("in.xma2.wav");
    std::fs::write(&input, synthetic_xma2(5, 2, 1)).unwrap();
    let (status, stdout) = xma2test(Some(&dir), &[&input]);
    assert_eq!(status.code(), Some(1));
    assert!(stdout.contains("FAIL: decoder not loaded"), "{stdout}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn missing_arguments_print_the_usage() {
    let program = env!("CARGO_BIN_EXE_xma2test");
    let out = Command::new(program).output().expect("the test program starts");
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), "usage: xma2test <decant_xma2.dll> <file.xma2.wav> [out.raw]");
}

/// The second export: frames decoded but not yet handed out, with an output buffer too small for
/// one packet. Loads the DLL into this process, so it is the one test that sets the folder here.
#[test]
#[allow(unsafe_code)]
fn pending_frames_are_reported_until_handed_out() {
    use decant_xma2::abi::{GetDecoderFn, DecantXma2Decoder};

    let Some(ffmpeg) = ffmpeg_dir() else {
        println!("SKIP: no LGPL shared FFmpeg (set DECANT_FFMPEG or DECANT_FFMPEG_DIR)");
        return;
    };
    // SAFETY: no other test of this program reads the variable from this process: they start
    // `xma2test` with its own environment.
    unsafe { std::env::set_var("DECANT_FFMPEG_DIR", &ffmpeg) };
    let file = synthetic_xma2(5, 24, 1);
    let program = env!("CARGO_BIN_EXE_xma2test");
    let lib = decant_testkit::Library::open(decant_testkit::dll_for(program, "decant_xma2.dll").as_os_str()).expect("the DLL loads");
    // SAFETY: the exports have the signatures of the decoder ABI.
    let (get, pending) = unsafe {
        (
            lib.get::<GetDecoderFn>("DecantXma2GetDecoder").expect("DecantXma2GetDecoder"),
            lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>("DecantXma2Pending").expect("DecantXma2Pending"),
        )
    };
    let mut table = DecantXma2Decoder::default();
    // SAFETY: a writable table; the format and block are bytes of `file`; the buffer holds one frame.
    unsafe {
        assert_eq!(get(&mut table), 0);
        assert_eq!(pending(std::ptr::null_mut()), 0);
        let (open, decode, close) = (table.open.unwrap(), table.decode.unwrap(), table.close.unwrap());
        let fmt = &file[20..20 + 52];
        let data = &file[20 + 52 + 8..];
        let context = open(fmt.as_ptr().cast(), fmt.len() as u32);
        assert!(!context.is_null());
        let (mut pcm, mut got) = ([0i16; 1], 0u32);
        for block in data.as_chunks::<2048>().0 {
            assert_eq!(decode(context, block.as_ptr(), 2048, pcm.as_mut_ptr(), 1, &mut got), 0);
            assert!(got <= 1);
        }
        // FFmpeg's xma2 decoder may hold its output until the stream ends: flush, then one frame
        // is handed out and the rest stays behind.
        assert_eq!(decode(context, std::ptr::null(), 0, pcm.as_mut_ptr(), 1, &mut got), 0);
        assert_eq!(got, 1);
        let kept = pending(context);
        assert!(kept > 0, "a one frame buffer must leave frames behind");
        // Draining the rest one frame at a time empties the count.
        let mut left = kept;
        while left > 0 {
            assert_eq!(decode(context, std::ptr::null(), 0, pcm.as_mut_ptr(), 1, &mut got), 0);
            assert_eq!(got, 1);
            left -= 1;
            assert_eq!(pending(context), left);
        }
        close(context);
    }
}
