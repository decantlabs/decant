//! Commands, results, sequence numbers and the PCM conversion path, through the exports.
#![allow(unsafe_code)]

mod common;

use common::*;
use xapu::abi::*;
use xapu::exports::*;

const BUF_MONO: u32 = 0x80 * 4 + 1920;
const BUF_STEREO: u32 = 2 * 0x80 * 4 + 2 * 1920;

/// A PCM converting connection of `streams` streams, with the block of 0x80 frames.
fn pcm(streams: u32, depth: u32, options: u32) -> Conn {
    Conn::open(params(PROCESS_CONVERT_PCM, streams, depth, 0, 0x80, options))
}

fn ramp_signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect()
}

#[test]
fn bad_commands_are_refused_before_they_run() {
    let _s = serial();
    let c = pcm(2, 4, 0);
    // SAFETY: a null command is part of the contract.
    assert_eq!(unsafe { XApuEnqueueCommand(c.handle, std::ptr::null(), std::ptr::null_mut()) }, E_INVALIDARG);
    for kind in [0, 4, 6, 100, u32::MAX] {
        assert_eq!(c.enqueue(&Command::new(kind, 0, 0)).0, E_INVALIDARG, "type {kind}");
    }
    // A stream that does not exist, and commands other than start on an inactive stream.
    assert_eq!(c.enqueue(&Command::new(COMMAND_START, 2, 0).with_start(1, 1.0, BUF_MONO)).0, XAPU_E_BAD_STREAM);
    assert_eq!(c.enqueue(&Command::new(COMMAND_START, u32::MAX, 0).with_start(1, 1.0, BUF_MONO)).0, XAPU_E_BAD_STREAM);
    for kind in [COMMAND_STOP, COMMAND_DATA, COMMAND_RESET] {
        assert_eq!(c.enqueue(&Command::new(kind, 0, 0)).0, XAPU_E_BAD_STREAM, "type {kind} on an inactive stream");
    }
    assert_eq!(c.counts(), (0, 0));
    // A second start.
    c.start(0, 1, 1.0, BUF_MONO);
    assert_eq!(c.enqueue(&Command::new(COMMAND_START, 0, 0).with_start(1, 1.0, BUF_MONO)).0, XAPU_E_STREAM_ACTIVE);
    // Stream 1 is independent.
    c.start(1, 2, 1.0, BUF_STEREO);
}

#[test]
fn sequence_numbers_count_per_connection_and_only_accepted_commands() {
    let _s = serial();
    let a = pcm(2, 8, 0);
    let b = pcm(1, 8, 0);
    let start = |s| Command::new(COMMAND_START, s, 0).with_start(1, 1.0, BUF_MONO);
    assert_eq!(a.enqueue(&start(0)), (S_OK, 1));
    assert_eq!(b.enqueue(&start(0)), (S_OK, 1));
    assert_eq!(a.enqueue(&start(1)), (S_OK, 2));
    // A refused command takes no number and leaves the output untouched.
    let (r, seq) = a.enqueue(&start(1));
    assert_eq!((r, seq), (XAPU_E_STREAM_ACTIVE, 0xAAAA));
    assert_eq!(a.enqueue(&Command::new(COMMAND_STOP, 0, 0)), (S_OK, 3));
    assert_eq!(a.enqueue(&Command::new(COMMAND_RESET, 1, 0)), (S_OK, 4));
    // A null sequence pointer is allowed.
    let c = Command::new(COMMAND_RESET, 1, 0);
    // SAFETY: a valid command, a null sequence.
    assert_eq!(unsafe { XApuEnqueueCommand(a.handle, &c, std::ptr::null_mut()) }, S_OK);
    assert_eq!(a.enqueue(&c).1, 6);
}

#[test]
fn results_come_out_in_command_order_across_streams() {
    let _s = serial();
    let c = pcm(3, 8, 0);
    for s in [2, 0, 1] {
        assert_eq!(c.enqueue(&Command::new(COMMAND_START, s, 0).with_start(1, 1.0, BUF_MONO)).0, S_OK);
    }
    for s in [1, 2, 0, 0] {
        assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, s, 0)).0, S_OK);
    }
    assert_eq!(c.counts(), (7, 0));
    let order: Vec<u32> = (0..7).map(|_| c.result().stream()).collect();
    assert_eq!(order, [2, 0, 1, 1, 2, 0, 0]);
    assert_eq!(c.counts(), (0, 7));
}

#[test]
fn dequeue_validates_handle_and_pointer() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    let mut r = XapuResult::default();
    // SAFETY: a writable result, a null result.
    unsafe {
        assert_eq!(XApuDequeueResult(0, &mut r), E_INVALIDARG);
        assert_eq!(XApuDequeueResult(c.handle, std::ptr::null_mut()), E_POINTER);
    }
    assert_eq!(c.dequeue().0, XAPU_E_NO_RESULT);
    assert_eq!(c.counts(), (0, 0));
    let mut dq = 7;
    // SAFETY: a writable counter.
    assert_eq!(unsafe { XApuGetResultCount(0, &mut dq) }, 0, "a bad handle has no results");
    // SAFETY: a null counter pointer is allowed.
    assert_eq!(unsafe { XApuGetResultCount(c.handle, std::ptr::null_mut()) }, 0);
}

#[test]
fn the_queue_holds_depth_commands_not_yet_dequeued() {
    let _s = serial();
    let c = pcm(2, 2, 0);
    let start = |s| Command::new(COMMAND_START, s, 0).with_start(1, 1.0, BUF_MONO);
    assert_eq!(c.enqueue(&start(0)).0, S_OK);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, S_OK);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)), (XAPU_E_QUEUE_FULL, 0xAAAA));
    // The depth is per stream.
    assert_eq!(c.enqueue(&start(1)).0, S_OK);
    // The oldest result frees a place on its stream.
    assert_eq!(c.result().stream(), 0);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, S_OK);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, XAPU_E_QUEUE_FULL);
}

#[test]
fn the_start_command_checks_the_processing_buffer() {
    let _s = serial();
    // One channel: 0x80 * 4 for the block and 1920 for the delay line.
    let c = pcm(4, 4, 0);
    let start = |s, ch, len| c.enqueue(&Command::new(COMMAND_START, s, 0).with_start(ch, 1.0, len)).0;
    assert_eq!(start(0, 1, BUF_MONO - 1), XAPU_E_BUFFER_TOO_SMALL);
    assert_eq!(start(0, 1, 0), XAPU_E_BUFFER_TOO_SMALL);
    assert_eq!(start(0, 1, u32::MAX), XAPU_E_BUFFER_TOO_SMALL, "the difference is signed");
    assert_eq!(start(0, 2, BUF_STEREO - 1), XAPU_E_BUFFER_TOO_SMALL);
    assert_eq!(c.counts(), (0, 0), "a refused start queues nothing and starts nothing");
    assert_eq!(start(0, 1, BUF_MONO), S_OK);
    assert_eq!(start(1, 2, BUF_STEREO), S_OK);
    assert_eq!(start(2, 2, 0x7FFF_FFFF), S_OK);
    // A connection that only decodes does not check it.
    let d = Conn::open(params(PROCESS_DECODE_OPUS, 1, 1, 0, 0, 0));
    assert_eq!(d.enqueue(&Command::new(COMMAND_START, 0, 0).with_start(1, 1.0, 0)).0, S_OK);
}

#[test]
fn a_bad_channel_count_or_pitch_fails_the_result_not_the_call() {
    let _s = serial();
    let c = pcm(3, 8, 0);
    for (stream, channels, pitch) in [(0, 0, 1.0), (1, 3, 1.0), (2, 1, f32::NAN)] {
        let (r, _) = c.enqueue(&Command::new(COMMAND_START, stream, 0).with_start(channels, pitch, 0x10000));
        assert_eq!(r, S_OK);
        assert_eq!(c.result().code(), E_INVALIDARG);
        // The stream is active all the same.
        assert_eq!(c.enqueue(&Command::new(COMMAND_START, stream, 0).with_start(1, 1.0, 0x10000)).0, XAPU_E_STREAM_ACTIVE);
        // Data on a stream that did not start properly fails its result.
        let mut out = [0f32; 128];
        let (r, res) = c.data(stream, 0, 1.0, 1.0, 128, &bytes_of(&[0.0; 128]), 128, &mut out);
        assert_eq!((r, res.code()), (S_OK, E_INVALIDARG));
    }
}

#[test]
fn the_start_pitch_is_clamped_and_reset_restores_it() {
    let _s = serial();
    let c = pcm(3, 8, 0);
    assert_eq!(c.start(0, 1, 100.0, BUF_MONO).pitch(), 8.0);
    assert_eq!(c.start(1, 1, 0.01, BUF_MONO).pitch(), 0.125);
    assert_eq!(c.start(2, 1, 1.5, BUF_MONO).pitch(), 1.5);
    // Move the pitch with a data command, then reset.
    let input = bytes_of(&[0.0; 128]);
    let mut out = vec![0f32; 4096];
    let (_, res) = c.data(2, 0, 8.0, 4.0, 128, &input, 128, &mut out);
    assert_eq!(res.pitch(), 4.0);
    let (r, _) = c.enqueue(&Command::new(COMMAND_RESET, 2, 0));
    assert_eq!(r, S_OK);
    assert_eq!(c.result().pitch(), 1.5, "reset restores the start pitch");
}

#[test]
fn the_result_image_follows_the_layout() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let input = bytes_of(&ramp_signal(128));
    let mut out = vec![0f32; 128];
    let cmd = Command::new(COMMAND_DATA, 0, 0).with_data(1.0, 1.0, 128, input.len() as u32, input.as_ptr() as u64, 128, out.as_mut_ptr() as u64);
    assert_eq!(c.enqueue(&cmd).0, S_OK);
    let r = c.result();
    let b = &r.0;
    assert_eq!(&b[0..0x18], &cmd.0[0..0x18], "the first 0x18 bytes of the command");
    assert_eq!(r.kind(), COMMAND_DATA);
    assert_eq!(r.state() & STATE_OUTPUT_READY, STATE_OUTPUT_READY);
    assert_eq!(r.code(), S_OK);
    assert_eq!(r.frames_written(), 128);
    assert_eq!(r.bytes_consumed(), 128 * 4);
    assert_eq!(&b[0x38..0x40], &cmd.0[0x38..0x40]);
    assert_eq!(&b[0x44..0x48], &cmd.0[0x28..0x2C]);
    assert_eq!(&b[0x48..0x50], &cmd.0[0x2C..0x34]);
    assert_eq!(r.pitch(), 1.0);
    assert_eq!(r.state_data(), [0, 128, 0, 0]);
    // A start command leaves the data only fields of its result at 0.
    let (r, _) = c.enqueue(&Command::new(COMMAND_STOP, 0, 0));
    assert_eq!(r, S_OK);
    let stop = c.result();
    assert_eq!((stop.kind(), stop.code(), stop.frames_written(), stop.state()), (COMMAND_STOP, S_OK, 0, 0));
}

#[test]
fn stop_deactivates_when_its_result_is_dequeued() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    let start = Command::new(COMMAND_START, 0, 0).with_start(1, 1.0, BUF_MONO);
    c.start(0, 1, 1.0, BUF_MONO);
    assert_eq!(c.enqueue(&Command::new(COMMAND_STOP, 0, 0)).0, S_OK);
    assert_eq!(c.enqueue(&start).0, XAPU_E_STREAM_ACTIVE, "still active until the result is dequeued");
    assert_eq!(c.result().kind(), COMMAND_STOP);
    assert_eq!(c.enqueue(&Command::new(COMMAND_DATA, 0, 0)).0, XAPU_E_BAD_STREAM);
    assert_eq!(c.enqueue(&start).0, S_OK, "a stream can start again");
}

#[cfg(windows)]
#[test]
fn every_committed_result_signals_the_event() {
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let _s = serial();
    let c = pcm(1, 4, OPTION_RESULT_EVENT);
    let event = c.region.result_event as *mut std::ffi::c_void;
    // SAFETY: a handle made by XApuConnect.
    unsafe {
        assert_eq!(WaitForSingleObject(event, 0), 0x102, "not signalled yet");
        assert_eq!(c.enqueue(&Command::new(COMMAND_START, 0, 0).with_start(1, 1.0, BUF_MONO)).0, S_OK);
        assert_eq!(WaitForSingleObject(event, 0), 0, "signalled by the commit");
        assert_eq!(WaitForSingleObject(event, 0), 0x102, "auto reset");
    }
}

#[test]
fn pcm_frame_counts_are_validated_in_the_result() {
    let _s = serial();
    let c = pcm(2, 16, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let mut out = vec![0f32; 1024];
    // Mono: 4 to 960 in steps of 4.
    for frames in [0u32, 3, 6, 964, 1000, 1024] {
        let input = bytes_of(&vec![0.0; frames as usize]);
        let (r, res) = c.data(0, 0, 1.0, 1.0, frames, &input, 1024, &mut out);
        assert_eq!((r, res.code()), (S_OK, E_INVALIDARG), "{frames} frames");
    }
    for frames in [4u32, 8, 960] {
        let input = bytes_of(&vec![0.0; frames as usize]);
        let (_, res) = c.data(0, COMMAND_OPTION_FLUSH, 1.0, 1.0, frames, &input, 1024, &mut out);
        assert_eq!(res.code(), S_OK, "{frames} frames");
    }
    // An input shorter than the frame count says.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&[0.0; 100]), 1024, &mut out);
    assert_eq!(res.code(), E_INVALIDARG);
    // Interleaved stereo needs steps of 8, deinterleaved input steps of 4.
    c.start(1, 2, 1.0, BUF_STEREO);
    let (_, res) = c.data(1, 0, 1.0, 1.0, 12, &bytes_of(&[0.0; 24]), 1024, &mut out);
    assert_eq!(res.code(), E_INVALIDARG);
    let (_, res) = c.data(1, COMMAND_OPTION_FLUSH, 1.0, 1.0, 8, &bytes_of(&[0.0; 16]), 1024, &mut out);
    assert_eq!(res.code(), S_OK);
    let d = pcm(1, 4, OPTION_DEINTERLEAVED_INPUT);
    d.start(0, 2, 1.0, BUF_STEREO);
    let (_, res) = d.data(0, COMMAND_OPTION_FLUSH, 1.0, 1.0, 12, &bytes_of(&[0.0; 24]), 1024, &mut out);
    assert_eq!(res.code(), S_OK);
    // The target and the ramp must be numbers, the output must exist.
    let (_, res) = c.data(0, 0, f32::NAN, 1.0, 128, &bytes_of(&[0.0; 128]), 1024, &mut out);
    assert_eq!(res.code(), E_INVALIDARG);
    let cmd = Command::new(COMMAND_DATA, 0, 0).with_data(1.0, 1.0, 128, 512, bytes_of(&[0.0; 128]).as_ptr() as u64, 1024, 0);
    assert_eq!(c.enqueue(&cmd).0, S_OK);
    assert_eq!(c.result().code(), E_INVALIDARG, "null output");
}

#[test]
fn at_pitch_1_a_block_is_the_input_delayed_by_5_frames() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let input = ramp_signal(128);
    let mut out = vec![9.0f32; 128];
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&input), 128, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 128));
    assert_eq!(res.state() & !STATE_SATURATION, STATE_OUTPUT_READY | STATE_NEED_MORE_INPUT);
    for (n, v) in out.iter().enumerate() {
        let want = if n >= 5 { input[n - 5] } else { 0.0 };
        assert!((v - want).abs() < 1e-5, "frame {n}");
    }
}

#[test]
fn only_whole_blocks_come_out_and_a_flush_releases_the_rest() {
    let _s = serial();
    let c = pcm(1, 8, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let input = ramp_signal(64);
    let mut out = vec![9.0f32; 256];
    // 64 frames in, a block of 128 is not complete.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 64, &bytes_of(&input), 256, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 0));
    assert_eq!(res.state(), STATE_NEED_MORE_INPUT);
    assert_eq!(res.state_data()[0], 64, "frames left in the converter");
    assert!(out.iter().all(|v| *v == 9.0), "nothing is written");
    // 64 more complete the block.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 64, &bytes_of(&input), 256, &mut out);
    assert_eq!((res.frames_written(), res.state() & STATE_OUTPUT_READY), (128, STATE_OUTPUT_READY));
    assert_eq!(res.state_data()[0], 0);
    // A partial block comes out on flush: the 64 frames and the 5 of the delay.
    let (_, res) = c.data(0, COMMAND_OPTION_FLUSH, 1.0, 1.0, 64, &bytes_of(&input), 256, &mut out);
    assert_eq!(res.frames_written(), 69);
    assert_eq!(res.state() & STATE_MORE_OUTPUT, 0);
    // The stream is continuous: this command starts at output frame 128, which is input frame 123 of the three inputs.
    for (n, v) in out.iter().take(69).enumerate() {
        assert!((v - input[(123 + n) % 64]).abs() < 1e-5, "frame {n}");
    }
}

#[test]
fn the_capacity_limits_the_output_and_more_stays_available() {
    let _s = serial();
    let c = pcm(1, 8, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let mut out = vec![0f32; 512];
    // Smaller than a block: refused, and the input is not taken.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&[0.0; 128]), 127, &mut out);
    assert_eq!(res.code(), XAPU_E_BUFFER_TOO_SMALL);
    // 512 frames in, room for 2 blocks of the 4: the others stay in the converter.
    let mut out = vec![0f32; 300];
    let (_, res) = c.data(0, 0, 1.0, 1.0, 512, &bytes_of(&[0.0; 512]), 300, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 256));
    assert_eq!(res.state() & (STATE_MORE_OUTPUT | STATE_NEED_MORE_INPUT), STATE_MORE_OUTPUT);
    assert_eq!(res.state_data()[0], 256);
    // The next command drains the 256 and takes in 4 more.
    let (_, res) = c.data(0, 0, 1.0, 1.0, 4, &bytes_of(&[0.0; 4]), 300, &mut out);
    assert_eq!((res.frames_written(), res.state_data()[0]), (256, 4));
    assert_eq!(res.state() & (STATE_MORE_OUTPUT | STATE_NEED_MORE_INPUT), STATE_NEED_MORE_INPUT);
}

#[test]
fn a_pitch_of_2_halves_the_frames() {
    let _s = serial();
    let c = pcm(1, 8, 0);
    c.start(0, 1, 2.0, BUF_MONO);
    let mut out = vec![0f32; 128];
    let input = bytes_of(&vec![0.5; 256]);
    let (_, res) = c.data(0, 0, 1.0, 2.0, 256, &input, 128, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 128));
    assert_eq!(res.pitch(), 2.0);
    assert!(out[20..].iter().all(|v| (v - 0.5).abs() < 1e-4));
}

#[test]
fn a_command_that_leaves_more_than_the_capacity_reports_more_output() {
    let _s = serial();
    let c = pcm(1, 8, 0);
    c.start(0, 1, 0.5, BUF_MONO);
    // Pitch 0.5: 256 input frames give 512 output frames, four blocks, with room for one.
    let mut out = vec![0f32; 128];
    let input = bytes_of(&vec![0.25; 256]);
    let (_, res) = c.data(0, 0, 1.0, 0.5, 256, &input, 128, &mut out);
    assert_eq!((res.code(), res.frames_written()), (S_OK, 128));
    assert_eq!(res.state() & (STATE_OUTPUT_READY | STATE_MORE_OUTPUT | STATE_NEED_MORE_INPUT), STATE_OUTPUT_READY | STATE_MORE_OUTPUT);
    assert_eq!(res.state_data()[0], 384, "the frames left in the converter");
    // The next command drains one more block of the 384 and adds its own 16.
    let (_, res) = c.data(0, 0, 1.0, 0.5, 4, &bytes_of(&[0.25; 4]), 128, &mut out);
    assert_eq!(res.frames_written(), 128);
    assert!(res.state() & STATE_MORE_OUTPUT != 0);
}

#[test]
fn deinterleaved_output_puts_the_channels_one_after_the_other_at_the_capacity() {
    let _s = serial();
    for (options, planar) in [(0, false), (OPTION_DEINTERLEAVED_OUTPUT, true)] {
        let c = pcm(1, 4, options | OPTION_DEINTERLEAVED_INPUT);
        c.start(0, 2, 1.0, BUF_STEREO);
        let mut input = vec![1.0f32; 128];
        input.extend(vec![-1.0f32; 128]);
        let mut out = vec![9.0f32; 256];
        let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&input), 128, &mut out);
        assert_eq!((res.code(), res.frames_written()), (S_OK, 128), "planar {planar}");
        for f in 20..128 {
            let (l, r) = if planar { (out[f], out[128 + f]) } else { (out[f * 2], out[f * 2 + 1]) };
            assert!((l - 1.0).abs() < 1e-5 && (r + 1.0).abs() < 1e-5, "planar {planar}, frame {f}");
        }
    }
}

#[test]
fn saturation_is_reported() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let mut out = vec![0f32; 128];
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&vec![1.5; 128]), 128, &mut out);
    assert_eq!(res.state() & STATE_SATURATION, STATE_SATURATION);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, S_OK);
    c.result();
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&vec![0.5; 128]), 128, &mut out);
    assert_eq!(res.state() & STATE_SATURATION, 0);
}

#[test]
fn reset_clears_the_converter() {
    let _s = serial();
    let c = pcm(1, 8, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let mut out = vec![0f32; 128];
    let (_, res) = c.data(0, 0, 1.0, 1.0, 64, &bytes_of(&[1.0; 64]), 128, &mut out);
    assert_eq!(res.state_data()[0], 64);
    assert_eq!(c.enqueue(&Command::new(COMMAND_RESET, 0, 0)).0, S_OK);
    c.result();
    let (_, res) = c.data(0, 0, 1.0, 1.0, 64, &bytes_of(&[1.0; 64]), 128, &mut out);
    assert_eq!(res.state_data()[0], 64, "the 64 frames before the reset are gone");
}

#[test]
fn execution_time_is_set_and_failures_leave_a_result() {
    let _s = serial();
    let c = pcm(1, 4, 0);
    c.start(0, 1, 1.0, BUF_MONO);
    let mut out = vec![0f32; 128];
    let (_, res) = c.data(0, 0, 1.0, 1.0, 128, &bytes_of(&[0.0; 128]), 128, &mut out);
    assert!(res.execution_time_us() < 10_000_000);
    assert_eq!(res.options(), 0);
}
