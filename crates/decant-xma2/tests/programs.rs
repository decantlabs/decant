//! Runs `xma2test` against the `decant_xma2.dll` Cargo built next to it (or the one in
//! `DECANT_DLL_DIR`), on synthetic XMA2 files made with the `decant-xma` synth module. No game
//! file, encoder or other library is needed.
#![cfg(windows)]
#![allow(unsafe_code)]

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::{expected, format, interleave, riff, tones};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("decant_xma2-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch folder");
    dir
}

fn xma2test(args: &[&std::path::Path]) -> (std::process::ExitStatus, String) {
    let program = env!("CARGO_BIN_EXE_xma2test");
    let out = Command::new(program)
        .arg(decant_testkit::dll_for(program, "decant_xma2.dll"))
        .args(args)
        .output()
        .expect("the test program starts");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    print!("{stdout}");
    (out.status, stdout)
}

#[test]
fn xma2test_decodes_synthetic_files_to_the_direct_pcm() {
    let dir = scratch("decode");
    let a = tones(2, 0, 40);
    let b = tones(2, 6, 40);
    let cases: Vec<(u16, u16, Vec<u8>, Vec<i16>)> = vec![
        (1, 1, tones(1, 0, 40), expected(&[tones(1, 0, 40)], &[1])),
        (2, 1, a.clone(), expected(std::slice::from_ref(&a), &[2])),
        (
            4,
            2,
            interleave(&[a.clone(), b.clone()]),
            expected(&[a, b], &[2, 2]),
        ),
    ];
    for (channels, streams, data, want) in cases {
        let input = dir.join(format!("in{channels}.xma2.wav"));
        let raw = dir.join(format!("out{channels}.raw"));
        std::fs::write(
            &input,
            riff(&format(channels, streams, 2048, data.len() / 2048), &data),
        )
        .unwrap();
        let (status, stdout) = xma2test(&[&input, &raw]);
        assert!(
            status.success(),
            "xma2test failed on {channels} channels: {status:?}"
        );
        assert!(
            stdout.contains(&format!(
                "channels {channels}, frames {}",
                want.len() / usize::from(channels)
            )),
            "{stdout}"
        );
        assert!(stdout.contains("XMA2TEST PASS"), "{stdout}");
        let written: Vec<i16> = std::fs::read(&raw)
            .unwrap()
            .as_chunks::<2>().0.iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect();
        assert_eq!(written, want, "{channels} channels");
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_format_the_decoder_refuses_is_reported() {
    let dir = scratch("refused");
    let data = tones(2, 0, 4);
    let input = dir.join("in.xma2.wav");
    // Two streams for two channels: not decodable.
    std::fs::write(&input, riff(&format(2, 2, 2048, data.len() / 2048), &data)).unwrap();
    let (status, stdout) = xma2test(&[&input]);
    assert_eq!(status.code(), Some(1));
    assert!(stdout.contains("FAIL: Open refused the format"), "{stdout}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_missing_dll_is_refused() {
    let dir = scratch("nodll");
    let input = dir.join("in.xma2.wav");
    std::fs::write(&input, b"RIFF").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xma2test"))
        .arg(dir.join("none.dll"))
        .arg(&input)
        .output()
        .expect("the test program starts");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stdout).contains("FAIL: decoder not loaded"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn missing_arguments_print_the_usage() {
    let program = env!("CARGO_BIN_EXE_xma2test");
    let out = Command::new(program)
        .output()
        .expect("the test program starts");
    assert_eq!(out.status.code(), Some(2));
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim_end(),
        "usage: xma2test <decant_xma2.dll> <file.xma2.wav> [out.raw]"
    );
}

/// The second export, in the DLL itself: frames decoded but not yet handed out, with an output
/// buffer too small for one packet.
#[test]
fn pending_frames_are_reported_by_the_dll_until_handed_out() {
    use decant_xma2::abi::{DecantXma2Decoder, GetDecoderFn};

    let data = tones(1, 0, 40);
    let fmt = format(1, 1, 2048, data.len() / 2048);
    let program = env!("CARGO_BIN_EXE_xma2test");
    let lib = decant_testkit::Library::open(
        decant_testkit::dll_for(program, "decant_xma2.dll").as_os_str(),
    )
    .expect("the DLL loads");
    // SAFETY: the exports have the signatures of the decoder ABI.
    let (get, pending) = unsafe {
        (
            lib.get::<GetDecoderFn>("DecantXma2GetDecoder")
                .expect("DecantXma2GetDecoder"),
            lib.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> u32>("DecantXma2Pending")
                .expect("DecantXma2Pending"),
        )
    };
    let mut table = DecantXma2Decoder::default();
    // SAFETY: a writable table; the format and block are bytes of this test; the buffer holds one frame.
    unsafe {
        assert_eq!(get(&mut table), 0);
        assert_eq!(pending(std::ptr::null_mut()), 0);
        let (open, decode, close) = (
            table.open.unwrap(),
            table.decode.unwrap(),
            table.close.unwrap(),
        );
        let context = open(fmt.as_ptr().cast(), fmt.len() as u32);
        assert!(!context.is_null());
        let (mut pcm, mut got) = ([0i16; 1], 0u32);
        assert_eq!(
            decode(
                context,
                data.as_ptr(),
                data.len() as u32,
                pcm.as_mut_ptr(),
                1,
                &mut got
            ),
            0
        );
        assert_eq!(got, 1);
        let kept = pending(context);
        assert!(kept > 1000, "a one frame buffer must leave frames behind");
        // A call with no bytes hands out one more frame and leaves one fewer behind.
        for left in (0..kept).rev() {
            assert_eq!(
                decode(context, data.as_ptr(), 0, pcm.as_mut_ptr(), 1, &mut got),
                0
            );
            assert_eq!(got, 1);
            assert_eq!(pending(context), left);
        }
        close(context);
    }
}
