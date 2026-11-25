//! Decodes synthetic XMA2 data through the exported ABI (`DecantXma2GetDecoder`, `Open`, `Decode`,
//! `Close`, `DecantXma2Pending`) and compares the PCM, bit for bit, with `decant-xma` called directly.
#![allow(unsafe_code)]

mod common;

use std::ffi::c_void;

use common::{expected, format, interleave, tones, tones_packets};
use decant_xma2::abi::{ABI_VERSION, DecantXma2Decoder};
use decant_xma2::{DecantXma2GetDecoder, DecantXma2Pending};

fn table() -> DecantXma2Decoder {
    let mut table = DecantXma2Decoder::default();
    // SAFETY: a writable table.
    assert_eq!(unsafe { DecantXma2GetDecoder(&mut table) }, 0);
    assert_eq!(table.abi_version, ABI_VERSION);
    table
}

/// Feeds `data` in calls of `chunk` bytes and drains with a buffer of `buffer` frames after each call, then flushes and drains
/// again. Checks `DecantXma2Pending` against what the calls hand out.
fn decode_all(
    format: &[u8],
    channels: usize,
    data: &[u8],
    chunk: usize,
    buffer: usize,
) -> Vec<i16> {
    let t = table();
    let (open, decode, close) = (t.open.unwrap(), t.decode.unwrap(), t.close.unwrap());
    let mut out = Vec::new();
    // SAFETY: the format and blocks are slices that outlive the calls, and `pcm` holds `buffer` frames.
    unsafe {
        let context = open(format.as_ptr().cast::<c_void>(), format.len() as u32);
        assert!(!context.is_null());
        let mut pcm = vec![0i16; buffer * channels];
        let mut call = |block: *const u8, bytes: usize, out: &mut Vec<i16>| {
            let mut got = 0u32;
            assert_eq!(
                decode(
                    context,
                    block,
                    bytes as u32,
                    pcm.as_mut_ptr(),
                    buffer as u32,
                    &mut got
                ),
                0
            );
            assert!(got as usize <= buffer);
            out.extend_from_slice(&pcm[..got as usize * channels]);
            got as usize
        };
        for block in data.chunks(chunk) {
            let mut got = call(block.as_ptr(), block.len(), &mut out);
            while got == buffer {
                // A call with no bytes only hands out what is kept.
                got = call(block.as_ptr(), 0, &mut out);
            }
            assert_eq!(DecantXma2Pending(context), 0);
        }
        // The flush: a null block, repeated while the buffer comes back full.
        let mut got = call(std::ptr::null(), 0, &mut out);
        while got == buffer {
            got = call(std::ptr::null(), 0, &mut out);
        }
        assert_eq!(DecantXma2Pending(context), 0);
        close(context);
    }
    out
}

#[test]
fn the_table_is_filled_and_a_null_table_is_refused() {
    let t = table();
    assert!(t.open.is_some() && t.decode.is_some() && t.close.is_some());
    // SAFETY: null is the refusal under test.
    assert_eq!(unsafe { DecantXma2GetDecoder(std::ptr::null_mut()) }, -1);
    // SAFETY: null is allowed for the pending count.
    assert_eq!(unsafe { DecantXma2Pending(std::ptr::null_mut()) }, 0);
}

#[test]
fn a_mono_stream_matches_the_direct_decode() {
    let data = tones(1, 0, 900);
    assert!(data.len() >= 3 * 2048, "{} packets", data.len() / 2048);
    let want = expected(std::slice::from_ref(&data), &[1]);
    assert!(want.iter().any(|&s| s != 0));
    let fmt = format(1, 1, 2048, data.len() / 2048);
    assert_eq!(decode_all(&fmt, 1, &data, 2048, 1024), want);
    assert_eq!(decode_all(&fmt, 1, &data, data.len(), 7), want);
}

#[test]
fn a_stereo_stream_matches_the_direct_decode_for_any_chunking_and_buffer() {
    let data = tones(2, 0, 900);
    assert!(data.len() >= 3 * 2048, "{} packets", data.len() / 2048);
    let want = expected(std::slice::from_ref(&data), &[2]);
    assert!(want.iter().any(|&s| s != 0));
    let fmt = format(2, 1, 2048, data.len() / 2048);
    for (chunk, buffer) in [
        (2048, 1024),
        (2048, 1),
        (4096, 300),
        (data.len(), 513),
    ] {
        assert_eq!(
            decode_all(&fmt, 2, &data, chunk, buffer),
            want,
            "chunk {chunk}, buffer {buffer}"
        );
    }
}

#[test]
fn an_open_block_align_and_missing_stream_count_are_accepted() {
    let data = tones(2, 3, 400);
    let want = expected(std::slice::from_ref(&data), &[2]);
    let mut fmt = format(2, 0, 0, data.len() / 2048);
    assert_eq!(decode_all(&fmt, 2, &data, 2048, 64), want);
    // Only the WAVEFORMATEX, no extra bytes at all.
    fmt.truncate(18);
    fmt[16] = 0;
    fmt[17] = 0;
    assert_eq!(decode_all(&fmt, 2, &data, 2048, 64), want);
}

#[test]
fn two_streams_make_four_channels() {
    let a = tones(2, 0, 700);
    let b = tones_packets(2, 5, a.len() / 2048);
    let want = expected(&[a.clone(), b.clone()], &[2, 2]);
    let data = interleave(&[a, b]);
    let fmt = format(4, 2, 2048, data.len() / 2048);
    for (chunk, buffer) in [(2048, 1024), (data.len(), 100), (4096, 3)] {
        let got = decode_all(&fmt, 4, &data, chunk, buffer);
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        assert!(got == want, "chunk {chunk}, buffer {buffer}: {} vs {} samples, first difference at {first:?}, packets {}", got.len(), want.len(), data.len() / 2048);
    }
}

#[test]
fn an_odd_channel_count_ends_in_a_mono_stream() {
    let a = tones(2, 0, 700);
    let b = tones_packets(1, 9, a.len() / 2048);
    let want = expected(&[a.clone(), b.clone()], &[2, 1]);
    let data = interleave(&[a, b]);
    let fmt = format(3, 2, 2048, data.len() / 2048);
    assert_eq!(decode_all(&fmt, 3, &data, 2048, 512), want);
}

#[test]
fn a_stream_can_be_reused_after_the_flush() {
    let data = tones(2, 0, 500);
    let want = expected(std::slice::from_ref(&data), &[2]);
    let fmt = format(2, 1, 2048, data.len() / 2048);
    let t = table();
    // SAFETY: slices outlive the calls and the buffer holds 400000 frames.
    unsafe {
        let context = (t.open.unwrap())(fmt.as_ptr().cast(), fmt.len() as u32);
        let mut pcm = vec![0i16; 400_000 * 2];
        for _ in 0..2 {
            let mut got = 0u32;
            assert_eq!(
                (t.decode.unwrap())(
                    context,
                    data.as_ptr(),
                    data.len() as u32,
                    pcm.as_mut_ptr(),
                    400_000,
                    &mut got
                ),
                0
            );
            let mut out = pcm[..got as usize * 2].to_vec();
            assert_eq!(
                (t.decode.unwrap())(
                    context,
                    std::ptr::null(),
                    0,
                    pcm.as_mut_ptr(),
                    400_000,
                    &mut got
                ),
                0
            );
            out.extend_from_slice(&pcm[..got as usize * 2]);
            assert_eq!(out, want);
        }
        (t.close.unwrap())(context);
    }
}

#[test]
fn formats_that_cannot_be_decoded_are_refused() {
    let t = table();
    let open = |fmt: &[u8]| {
        // SAFETY: the slice is readable for its length.
        let context = unsafe { (t.open.unwrap())(fmt.as_ptr().cast(), fmt.len() as u32) };
        if !context.is_null() {
            // SAFETY: closed once.
            unsafe { (t.close.unwrap())(context) };
        }
        !context.is_null()
    };
    assert!(open(&format(2, 1, 2048, 4)));
    // Another packet size, or a stream count that does not fit the channels.
    assert!(!open(&format(2, 1, 4096, 4)));
    assert!(!open(&format(2, 2, 2048, 4)));
    assert!(!open(&format(4, 1, 2048, 4)));
    // Not XMA2, too many channels, too short.
    let mut other = format(2, 1, 2048, 4);
    other[0] = 0x65;
    assert!(!open(&other));
    assert!(!open(&format(9, 5, 2048, 4)));
    assert!(!open(&format(2, 1, 2048, 4)[..10]));
    // SAFETY: null is the refusal under test.
    assert!(unsafe { (t.open.unwrap())(std::ptr::null(), 52) }.is_null());
}

#[test]
fn bad_arguments_to_decode_are_errors() {
    let t = table();
    let mut got = 9u32;
    let mut pcm = [0i16; 4];
    // SAFETY: null pointers are the refusals under test.
    unsafe {
        let decode = t.decode.unwrap();
        assert_eq!(
            decode(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                pcm.as_mut_ptr(),
                2,
                &mut got
            ),
            -1
        );
        let fmt = format(2, 1, 2048, 4);
        let context = (t.open.unwrap())(fmt.as_ptr().cast(), fmt.len() as u32);
        assert_eq!(
            decode(
                context,
                std::ptr::null(),
                0,
                pcm.as_mut_ptr(),
                2,
                std::ptr::null_mut()
            ),
            -1
        );
        assert_eq!(
            decode(
                context,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                2,
                &mut got
            ),
            -1
        );
        // A block shorter than one packet is ignored.
        let short = [0u8; 100];
        assert_eq!(
            decode(context, short.as_ptr(), 100, pcm.as_mut_ptr(), 2, &mut got),
            0
        );
        assert_eq!(got, 0);
        assert_eq!(DecantXma2Pending(context), 0);
        (t.close.unwrap())(context);
    }
}
