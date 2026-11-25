//! A C style client of the `chad` exports.
#![allow(unsafe_code, dead_code)]

use std::sync::{Mutex, MutexGuard};

use chad::abi::*;
use chad::exports::*;
use chad::fft::forward_packed;

static SERIAL: Mutex<()> = Mutex::new(());

/// Clients and activations are process wide: tests run one after another.
pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Floats of client memory in a rig.
pub const MEM_FLOATS: usize = 0x20000;

pub fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0) as f32
        })
        .collect()
}

pub fn params(process_type: u32, flags: u32, block: u32, channels: u32, ir: &[f32]) -> ActivationParameters {
    ActivationParameters { process_type, flags, block, channels, ir_length: ir.len() as u32, ir_pointer: ir.as_ptr() as u64 }
}

/// The impulse response format of the specification: per block of the response, the packed spectrum of the block padded to twice its length.
pub fn ir_spectra(h: &[f32], block: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for part in h.chunks(block) {
        let mut x: Vec<f64> = part.iter().map(|v| f64::from(*v)).collect();
        x.resize(2 * block, 0.0);
        out.extend(forward_packed(&x).iter().map(|v| *v as f32));
    }
    out
}

/// A client with one activated context and 0x20000 floats of memory.
pub struct Rig {
    pub client: *mut XdspClient,
    pub context: *mut XdspContext,
    pub status: *mut u32,
    pub mem: Vec<f32>,
}

impl Rig {
    pub fn new(p: &ActivationParameters) -> Rig {
        let mut mem = vec![0f32; MEM_FLOATS];
        let mut client = std::ptr::null_mut();
        // SAFETY: memory of the size given and a writable out pointer.
        assert_eq!(unsafe { XDspConnect(mem.as_mut_ptr() as *mut _, (MEM_FLOATS * 4) as u32, 4, &mut client) }, S_OK);
        let (mut status, mut context) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: a client, parameters and writable out pointers.
        assert_eq!(unsafe { XDspActivate(client, &p.to_bytes(), &mut status, &mut context) }, S_OK, "{p:?}");
        Rig { client, context, status, mem }
    }

    pub fn addr(&self, float_index: usize) -> u64 {
        self.mem.as_ptr() as u64 + 4 * float_index as u64
    }

    pub fn write(&mut self, at: usize, data: &[f32]) {
        self.mem[at..at + data.len()].copy_from_slice(data);
    }

    pub fn read(&self, at: usize, n: usize) -> Vec<f32> {
        self.mem[at..at + n].to_vec()
    }

    pub fn command(&self, input_gain: f32, output_gain: f32, input: usize, output: usize, multiplier: Option<usize>) -> XdspCommand {
        XdspCommand { input_gain, output_gain, input: self.addr(input), output: self.addr(output), multiplier: multiplier.map_or(0, |m| self.addr(m)) }
    }

    pub fn submit(&self, c: &XdspCommand) -> (Hresult, u32) {
        let mut seq = 0xAAAA;
        // SAFETY: a live context and a command.
        let r = unsafe { XDspSubmitCommand(self.context, c, &mut seq) };
        (r, seq)
    }

    pub fn submit_envelope(&self, c: &XdspCommand, envelope: u64) -> (Hresult, u32) {
        let mut seq = 0xAAAA;
        // SAFETY: a live context, a command and an address of gains.
        let r = unsafe { XDspSubmitCommandWithEnvelope(self.context, c, envelope as *const f32, &mut seq) };
        (r, seq)
    }

    pub fn status(&self) -> [u32; 4] {
        let mut out = [0u32; 4];
        for (i, v) in out.iter_mut().enumerate() {
            // SAFETY: the status array has 4 words.
            *v = unsafe { self.status.add(i).read_volatile() };
        }
        out
    }

    pub fn set_status(&self, index: usize, value: u32) {
        // SAFETY: the status array has 4 words.
        unsafe { self.status.add(index).write_volatile(value) };
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        assert_eq!(XDspDeactivate(self.context), S_OK);
        assert_eq!(XDspDisconnect(self.client), S_OK);
    }
}
