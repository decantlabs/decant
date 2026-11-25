//! The Opus paths: decode (process type 1) and decode then convert (process type 2).
#![allow(unsafe_code)]

mod common;

use common::*;
use xapu::abi::*;

const BUF: u32 = 0x80 * 4 + 1920;
const BUF_STEREO: u32 = 2 * 0x80 * 4 + 2 * 1920;

fn decoder(channels: u32, options: u32) -> Conn {
    let c = Conn::open(params(PROCESS_DECODE_OPUS, 1, 8, 0, 0, options));
    c.start(0, channels, 1.0, 0);
    c
}

/// Decodes `packet` on a fresh connection.
fn decode_once(packet: &[u8], channels: u32, options: u32, capacity: u32) -> (Hresult, XapuResult, Vec<f32>) {
    let c = decoder(channels, options);
    let mut out = vec![9.0f32; (capacity * channels) as usize];
    let (r, res) = c.data(0, 0, 1.0, 1.0, 0, packet, capacity, &mut out);
    assert_eq!(r, S_OK);
    (res.code(), res, out)
}

#[test]
fn a_packet_decodes_to_960_frames_at_48_khz() {
    let _s = serial();
    for (channels, stereo) in [(1u32, false), (2, true)] {
        let packet = opus_packet(stereo, 1);
        let (code, res, out) = decode_once(&packet, channels, 0, 1024);
        assert_eq!(code, S_OK);
        assert_eq!(res.frames_written(), 960);
        assert_eq!(res.bytes_consumed(), packet.len() as u32);
        assert_eq!(res.state() & STATE_OUTPUT_READY, STATE_OUTPUT_READY);
        let n = 960 * channels as usize;
        assert!(out[..n].iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-3));
        assert!(out[..n].iter().any(|v| *v != 0.0), "the packet carries sound");
        assert!(out[n..].iter().all(|v| *v == 9.0), "nothing is written past the frames");
    }
}

#[test]
fn the_output_capacity_must_hold_the_packet() {
    let _s = serial();
    let (code, res, out) = decode_once(&opus_packet(false, 2), 1, 0, 959);
    assert_eq!(code, XAPU_E_BUFFER_TOO_SMALL);
    assert_eq!(res.frames_written(), 0);
    assert!(out.iter().all(|v| *v == 9.0));
}

#[test]
fn deinterleaved_output_is_left_then_right_at_the_capacity() {
    let _s = serial();
    let packet = opus_packet(true, 3);
    let (_, _, inter) = decode_once(&packet, 2, 0, 1000);
    let (code, res, planar) = decode_once(&packet, 2, OPTION_DEINTERLEAVED_OUTPUT, 1000);
    assert_eq!((code, res.frames_written()), (S_OK, 960));
    for f in 0..960 {
        assert_eq!(planar[f], inter[f * 2], "left {f}");
        assert_eq!(planar[1000 + f], inter[f * 2 + 1], "right {f}");
    }
}

#[test]
fn bad_packets_fail_the_result() {
    let _s = serial();
    // No bytes, and a null input.
    let c = decoder(1, 0);
    let mut out = vec![0f32; 1024];
    let (r, res) = c.data(0, 0, 1.0, 1.0, 0, &[], 1024, &mut out);
    assert_eq!((r, res.code()), (S_OK, XAPU_E_BAD_OPUS_PACKET));
    // A code 3 packet that ends before its frame count byte is not a packet.
    let (r, res) = c.data(0, 0, 1.0, 1.0, 0, &[0xFB], 1024, &mut out);
    assert_eq!((r, res.code()), (S_OK, XAPU_E_BAD_OPUS_PACKET));
    // The stream goes on with a good packet.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 0, &opus_packet(false, 4), 1024, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 960));
    // No output pointer.
    let packet = opus_packet(false, 4);
    let cmd = Command::new(COMMAND_DATA, 0, 0).with_data(1.0, 1.0, 0, packet.len() as u32, packet.as_ptr() as u64, 1024, 0);
    assert_eq!(c.enqueue(&cmd).0, S_OK);
    assert_eq!(c.result().code(), E_INVALIDARG);
}

#[test]
fn the_decoder_keeps_its_state_until_a_reset_with_option_4() {
    let _s = serial();
    let c = decoder(1, 0);
    let packet = opus_packet(false, 5);
    let mut first = vec![0f32; 1024];
    let mut second = vec![0f32; 1024];
    let mut third = vec![0f32; 1024];
    c.data(0, 0, 1.0, 1.0, 0, &packet, 1024, &mut first);
    c.data(0, 0, 1.0, 1.0, 0, &packet, 1024, &mut second);
    assert_ne!(first[..960], second[..960], "the second packet continues the first");
    // A reset without option 4 leaves the decoder alone.
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, S_OK);
    c.result();
    c.data(0, 0, 1.0, 1.0, 0, &packet, 1024, &mut third);
    assert_ne!(first[..960], third[..960]);
    // Option 4 resets it: the same packet gives the first output again.
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, COMMAND_OPTION_RESET_DECODER)).0, S_OK);
    c.result();
    c.data(0, 0, 1.0, 1.0, 0, &packet, 1024, &mut third);
    assert_eq!(first[..960], third[..960]);
}

#[test]
fn decode_then_convert_is_the_decode_delayed_by_5_frames() {
    let _s = serial();
    let packet = opus_packet(false, 6);
    let (_, _, plain) = decode_once(&packet, 1, 0, 1024);
    let c = Conn::open(params(PROCESS_DECODE_OPUS_CONVERT, 1, 4, 0, 0x80, 0));
    c.start(0, 1, 1.0, BUF);
    let mut out = vec![9.0f32; 1024];
    let (r, res) = c.data(0, 0, 1.0, 1.0, 0, &packet, 1024, &mut out);
    assert_eq!(r, S_OK);
    // 960 frames make 7 blocks of 128, 64 frames stay.
    assert_eq!((res.code(), res.frames_written()), (S_OK, 896));
    assert_eq!(res.state() & STATE_NEED_MORE_INPUT, STATE_NEED_MORE_INPUT);
    assert_eq!(res.state_data()[0], 64);
    assert_eq!(res.bytes_consumed(), packet.len() as u32);
    for n in 0..896 {
        let want = if n >= 5 { plain[n - 5] } else { 0.0 };
        assert!((out[n] - want).abs() < 1e-4, "frame {n}");
    }
    // A flush brings out the last 64 frames, and the 5 frames of the delay come with them.
    let mut big = vec![0f32; 2048];
    let (_, res) = c.data(0, COMMAND_OPTION_FLUSH, 1.0, 1.0, 0, &packet, 2048, &mut big);
    assert_eq!(res.code(), S_OK);
    assert_eq!(res.frames_written(), 64 + 960 + 5);
}

#[test]
fn decode_then_convert_changes_the_rate_with_the_pitch() {
    let _s = serial();
    let packet = opus_packet(true, 7);
    let c = Conn::open(params(PROCESS_DECODE_OPUS_CONVERT, 1, 4, 0, 0x80, 0));
    c.start(0, 2, 2.0, BUF_STEREO);
    let mut out = vec![0f32; 2048];
    let (_, res) = c.data(0, 0, 1.0, 2.0, 0, &packet, 1024, &mut out);
    // 960 input frames at pitch 2 are 480 output frames: 3 blocks.
    assert_eq!((res.code(), res.frames_written()), (S_OK, 384));
    assert_eq!(res.pitch(), 2.0);
    let bad = c.data(0, 0, 1.0, 2.0, 0, &[], 1024, &mut out).1;
    assert_eq!(bad.code(), XAPU_E_BAD_OPUS_PACKET);
}
