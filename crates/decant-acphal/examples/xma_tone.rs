//! A listening check of the XMA engine: a synthetic stream (a 1 kHz tone for one second, then a 500 Hz tone for one second)
//! goes through an XMA context as a title drives it, and the title side takes the output blocks at the real rate. The monitor
//! plays what the engine writes on the default device. `cargo run -p decant-acphal --example xma_tone`. With
//! `DECANT_XMA_MUTE=1` nothing is heard and the run only prints the positions.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use AcpHal::abi::*;
use AcpHal::exports::*;

fn main() {
    // 94 frames of 512 samples are one second at 48 kHz.
    let mut frames = vec![Some(decant_xma::synth::tone_frame(&[Some((21, 1))], 48_000, 26)); 94];
    frames.extend(vec![Some(decant_xma::synth::tone_frame(&[Some((10, 1))], 48_000, 26)); 94]);
    let stream = decant_xma::synth::pack(&frames);
    let ring = 16u32;
    // SAFETY: the exports driven as a title does, with buffers of the sizes they are given.
    unsafe {
        let mut c = ShapeContexts { counts: [0, 0, 0, 0, 1, 0], ..Default::default() };
        assert_eq!(AcpHalAllocateShapeContexts(&mut c), S_OK);
        let words = c.pointers[4] as *mut u32;
        let (mut input, mut input_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut input, &mut input_apu, stream.len() as u32, 2048, 1), S_OK);
        std::ptr::copy_nonoverlapping(stream.as_ptr(), input as *mut u8, stream.len());
        let (mut output, mut output_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut output, &mut output_apu, ring * 256, 256, 1), S_OK);
        words.write_volatile((stream.len() / 2048) as u32 | 1 << 20 | ring << 22);
        words.add(1).write_volatile(1 << 31 | 3 << 27);
        words.add(5).write_volatile(input_apu);
        words.add(7).write_volatile(output_apu);
        let mut hal = std::ptr::null_mut();
        assert_eq!(AcpHalCreate(&mut hal), S_OK);
        let vt = (*hal).vtbl;
        assert_eq!((vt.connect)(hal, 64, 64), S_OK);
        let submit = |kind: u32, w: &[u32]| assert_eq!((vt.submit_command)(hal, kind, 0, 0, w.as_ptr() as *const c_void, 0), S_OK);
        submit(CMD_XMA_ENABLE, &[0]);
        let start = Instant::now();
        let mut taken = 0u64;
        while start.elapsed() < Duration::from_secs(3) {
            // One block of 128 samples every 2.667 ms, as a mixer that plays the ring would take them.
            let due = (start.elapsed().as_secs_f64() * 48_000.0 / 128.0) as u64;
            let write = words.read_volatile() >> 27 & 31;
            let read = words.add(9).read_volatile() & 31;
            let full = words.add(1).read_volatile() >> 31 == 0;
            let available = if full { ring } else { (write + ring - read) % ring };
            let take = (due.saturating_sub(taken) as u32).min(available);
            if take > 0 {
                submit(CMD_XMA_OUTPUT_CONSUMED, &[0, take]);
                taken += u64::from(take);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let w2 = words.add(2).read_volatile();
        println!("{taken} blocks taken in 3 s (375 per second), read position {} bits, error status {}", w2 & 0x3FF_FFFF, w2 >> 26 & 31);
        // Let the device play what is queued.
        std::thread::sleep(Duration::from_millis(300));
    }
}
