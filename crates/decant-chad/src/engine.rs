//! Clients, contexts and the DSP work of `chad.dll` (`docs/CONSOLE_LIBS.md` 4.3).
//!
//! Work completes synchronously inside `submit`: the status array shows completed equal to submitted as soon as the call returns.
#![allow(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use decant_log::trace;

use crate::abi::*;
use crate::fft::{forward_packed, inverse_packed, multiply_accumulate};

struct Client {
    memory: usize,
    size: usize,
    max_activations: u32,
    /// Stored, and logged at connection: the specification gives it no further effect.
    _max_streams: u32,
    _priority: u32,
    active: u32,
    /// The allocation whose address is the client handle.
    _token: Box<u64>,
}

struct Context {
    client: usize,
    params: ActivationParameters,
    partitions: usize,
    /// `ir[j][p]`: the packed spectrum of partition `p` of impulse response `j`.
    ir: Vec<Vec<Vec<f64>>>,
    /// Per channel: the previous input block.
    previous: Vec<Vec<f64>>,
    /// Per channel: the spectra of the last `partitions` input blocks, newest first.
    delay: Vec<VecDeque<Vec<f64>>>,
    envelope: Option<Vec<f32>>,
    memory: usize,
    size: usize,
    status: Box<[AtomicU32; STATUS_WORDS]>,
    _token: Box<u64>,
}

#[derive(Default)]
struct State {
    clients: HashMap<usize, Client>,
    contexts: HashMap<usize, Arc<Mutex<Context>>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The argument checks of `XDspConnectWithMaximumStreamLimit` that come before the null check of `client`.
pub fn connect_args_valid(memory: usize, size: u32, max_activations: u32, max_streams: u32, priority: u32) -> bool {
    memory != 0 && (MEMORY_MIN..=MEMORY_MAX).contains(&size) && max_activations <= MAX_ACTIVATIONS && max_streams <= MAX_STREAMS && priority <= MAX_PRIORITY
}

pub fn connect(memory: usize, size: u32, max_activations: u32, max_streams: u32, priority: u32) -> Result<*mut XdspClient, Hresult> {
    let mut guard = state();
    let st = guard.get_or_insert_with(State::default);
    let total: u32 = st.clients.values().map(|c| c.max_activations).sum();
    if total + max_activations > MAX_ACTIVATIONS {
        return Err(XDSP_E_TOO_MANY_ACTIVATIONS);
    }
    if st.clients.len() >= MAX_CLIENTS {
        return Err(XDSP_E_TOO_MANY_CLIENTS);
    }
    let token = Box::new(0u64);
    let handle = &*token as *const u64 as usize;
    st.clients.insert(handle, Client { memory, size: size as usize, max_activations, _max_streams: max_streams, _priority: priority, active: 0, _token: token });
    trace(&format!("chad: client connected, {size:#x} bytes of memory, {max_activations} activations, {max_streams} streams, priority {priority}"));
    Ok(handle as *mut XdspClient)
}

pub fn disconnect(client: usize) -> Hresult {
    let mut guard = state();
    let Some(st) = guard.as_mut() else { return E_INVALIDARG };
    match st.clients.get(&client) {
        None => E_INVALIDARG,
        Some(c) if c.active > 0 => XDSP_E_CONTEXTS_ACTIVE,
        Some(_) => {
            st.clients.remove(&client);
            S_OK
        }
    }
}

/// Checks the activation parameters (not the impulse response).
pub fn params_valid(p: &ActivationParameters) -> bool {
    if !(PROCESS_CONVOLUTION..=PROCESS_INVERSE_FFT).contains(&p.process_type) || !(1..=2).contains(&p.channels) {
        return false;
    }
    if p.channels == 1 && p.flags & (FLAG_DEINTERLEAVED | FLAG_STEREO_IR) != 0 {
        return false;
    }
    if p.process_type == PROCESS_CONVOLUTION {
        p.flags & FLAG_POLAR == 0 && matches!(p.block, 0x200 | 0x400)
    } else {
        matches!(p.block, 0x200 | 0x400 | 0x800) && p.channels == 1
    }
}

/// Activated context: its handle and the status array.
pub struct Activated {
    pub context: *mut XdspContext,
    pub status: *mut u32,
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

pub fn activate(client: usize, p: &ActivationParameters) -> Result<Activated, Hresult> {
    if !params_valid(p) {
        return Err(E_INVALIDARG);
    }
    let mut guard = state();
    let st = guard.as_mut().ok_or(E_INVALIDARG)?;
    let (memory, size, max_activations, active) = {
        let c = st.clients.get(&client).ok_or(E_INVALIDARG)?;
        (c.memory, c.size, c.max_activations, c.active)
    };
    let total_contexts: u32 = st.clients.values().map(|c| c.active).sum();
    if active >= max_activations || total_contexts >= MAX_CONTEXTS {
        return Err(XDSP_E_TOO_MANY_CONTEXTS);
    }
    let block = p.block as usize;
    let channels = p.channels as usize;
    let mut partitions = 0;
    let mut ir = Vec::new();
    if p.process_type == PROCESS_CONVOLUTION {
        partitions = p.ir_length as usize / 2 / block;
        if partitions < MIN_PARTITIONS || round_up(partitions, 8) > MAX_PARTITIONS_ROUNDED {
            return Err(XDSP_E_IR_SIZE);
        }
        if p.ir_pointer == 0 {
            return Err(E_INVALIDARG);
        }
        let responses = if p.flags & FLAG_STEREO_IR != 0 { 2 } else { 1 };
        let per_ir = partitions * 2 * block;
        let base = p.ir_pointer as usize as *const f32;
        for j in 0..responses {
            let mut list = Vec::with_capacity(partitions);
            for part in 0..partitions {
                let offset = j * per_ir + part * 2 * block;
                // SAFETY: the caller's impulse response holds `responses * per_ir` floats, at any alignment.
                let floats: Vec<f32> = (0..2 * block).map(|i| unsafe { std::ptr::read_unaligned(base.add(offset + i)) }).collect();
                if floats[0].to_bits() == METADATA_MARKER {
                    list.push(vec![0.0; 2 * block]);
                } else {
                    list.push(floats.iter().map(|v| if v.is_finite() { f64::from(*v) } else { 0.0 }).collect());
                }
            }
            ir.push(list);
        }
    }
    let token = Box::new(0u64);
    let handle = &*token as *const u64 as usize;
    let fft = p.process_type != PROCESS_CONVOLUTION;
    let ctx = Context {
        client,
        params: *p,
        partitions,
        ir,
        previous: vec![vec![0.0; block]; if fft { 0 } else { channels }],
        delay: vec![VecDeque::new(); if fft { 0 } else { channels }],
        envelope: None,
        memory,
        size,
        status: Box::new(std::array::from_fn(|_| AtomicU32::new(0))),
        _token: token,
    };
    let status = ctx.status.as_ptr() as *mut u32;
    st.contexts.insert(handle, Arc::new(Mutex::new(ctx)));
    if let Some(c) = st.clients.get_mut(&client) {
        c.active += 1;
    }
    Ok(Activated { context: handle as *mut XdspContext, status })
}

pub fn deactivate(context: usize) -> Hresult {
    let mut guard = state();
    let Some(st) = guard.as_mut() else { return E_INVALIDARG };
    let Some(ctx) = st.contexts.remove(&context) else { return E_INVALIDARG };
    let client = ctx.lock().unwrap_or_else(|e| e.into_inner()).client;
    if let Some(c) = st.clients.get_mut(&client) {
        c.active -= 1;
    }
    S_OK
}

/// Whether `[address, address + bytes)` lies inside the client memory of the context.
fn inside(ctx: &Context, address: u64, bytes: usize) -> bool {
    let (start, end) = (ctx.memory as u128, (ctx.memory + ctx.size) as u128);
    let (a, b) = (u128::from(address), u128::from(address) + bytes as u128);
    a >= start && b <= end
}

fn read_floats(address: u64, count: usize) -> Vec<f32> {
    let p = address as usize as *const f32;
    // SAFETY: the caller checked that the range lies inside the client memory.
    (0..count).map(|i| unsafe { std::ptr::read_unaligned(p.add(i)) }).collect()
}

fn write_floats(address: u64, values: &[f64]) {
    let p = address as usize as *mut f32;
    for (i, v) in values.iter().enumerate() {
        let v = if v.is_finite() { *v as f32 } else { 0.0 };
        // SAFETY: the caller checked that the range lies inside the client memory.
        unsafe { std::ptr::write_unaligned(p.add(i), v) };
    }
}

fn finite(v: f32) -> f64 {
    if v.is_finite() { f64::from(v) } else { 0.0 }
}

/// Submits `command` (with an envelope address, 0 for none) and returns the sequence number.
pub fn submit(context: usize, command: &XdspCommand, envelope: u64) -> Result<u32, Hresult> {
    let ctx = {
        let guard = state();
        guard.as_ref().and_then(|s| s.contexts.get(&context).cloned()).ok_or(E_INVALIDARG)?
    };
    let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
    let sticky = ctx.status[3].load(Ordering::SeqCst) as i32;
    if sticky < 0 {
        return Err(sticky);
    }
    let submitted = ctx.status[0].load(Ordering::SeqCst);
    let completed = ctx.status[1].load(Ordering::SeqCst);
    if submitted.wrapping_sub(completed) >= MAX_OUTSTANDING {
        return Err(XDSP_E_NO_SLOT);
    }
    let p = ctx.params;
    let block = p.block as usize;
    let channels = p.channels as usize;
    if p.process_type == PROCESS_CONVOLUTION {
        if !inside(&ctx, command.input, block * channels * 4) || !inside(&ctx, command.output, block * channels * 4) {
            return Err(E_INVALIDARG);
        }
        if envelope != 0 {
            let floats = if p.flags & FLAG_STEREO_IR != 0 { round_up(ctx.partitions, 4) + ctx.partitions } else { ctx.partitions };
            if !inside(&ctx, envelope, floats * 4) {
                return Err(E_INVALIDARG);
            }
        }
        if submitted.is_multiple_of(ENVELOPE_GROUP) {
            ctx.envelope = if envelope != 0 {
                let floats = if p.flags & FLAG_STEREO_IR != 0 { round_up(ctx.partitions, 4) + ctx.partitions } else { ctx.partitions };
                Some(read_floats(envelope, floats))
            } else {
                None
            };
        }
        convolve(&mut ctx, command);
    } else {
        if !inside(&ctx, command.input, block * 4) || !inside(&ctx, command.output, block * 4) || (command.multiplier != 0 && !inside(&ctx, command.multiplier, block * 4)) {
            return Err(E_INVALIDARG);
        }
        transform(&ctx, command);
    }
    let sequence = submitted.wrapping_add(1);
    ctx.status[0].store(sequence, Ordering::SeqCst);
    ctx.status[1].store(sequence, Ordering::SeqCst);
    Ok(sequence)
}

/// One block of partitioned convolution by overlap save, as described in the specification.
fn convolve(ctx: &mut Context, command: &XdspCommand) {
    let p = ctx.params;
    let (block, channels) = (p.block as usize, p.channels as usize);
    let input = read_floats(command.input, block * channels);
    let deinterleaved = p.flags & FLAG_DEINTERLEAVED != 0;
    let stereo_ir = p.flags & FLAG_STEREO_IR != 0;
    let right_start = round_up(ctx.partitions, 4);
    let in_gain = finite(command.input_gain);
    let out_gain = finite(command.output_gain);
    for c in 0..channels {
        let current: Vec<f64> = (0..block).map(|i| in_gain * finite(if deinterleaved { input[c * block + i] } else { input[i * channels + c] })).collect();
        let mut window = ctx.previous[c].clone();
        window.extend_from_slice(&current);
        let spectrum = forward_packed(&window);
        ctx.delay[c].push_front(spectrum);
        ctx.delay[c].truncate(ctx.partitions);
        ctx.previous[c] = current;
        let ir = &ctx.ir[if stereo_ir { c } else { 0 }];
        let mut acc = vec![0.0; 2 * block];
        for (part, x) in ctx.delay[c].iter().enumerate() {
            let gain = ctx.envelope.as_ref().map_or(1.0, |e| f64::from(e[if stereo_ir && c == 1 { right_start + part } else { part }]));
            multiply_accumulate(&mut acc, x, &ir[part], gain);
        }
        let y = inverse_packed(&acc);
        let out: Vec<f64> = y[block..].iter().map(|v| v * out_gain).collect();
        write_floats(command.output + (c * block * 4) as u64, &out);
    }
}

/// Forward or inverse FFT of one block.
fn transform(ctx: &Context, command: &XdspCommand) {
    let p = ctx.params;
    let block = p.block as usize;
    let polar = p.flags & FLAG_POLAR != 0;
    let in_gain = finite(command.input_gain);
    let out_gain = finite(command.output_gain);
    let input = read_floats(command.input, block);
    if p.process_type == PROCESS_FFT {
        let multiplier = if command.multiplier != 0 { Some(read_floats(command.multiplier, block)) } else { None };
        let x: Vec<f64> = (0..block).map(|i| in_gain * finite(input[i]) * multiplier.as_ref().map_or(1.0, |m| finite(m[i]))).collect();
        let mut spectrum = forward_packed(&x);
        // The gain scales the bins before the polar conversion, so a phase is never scaled.
        spectrum.iter_mut().for_each(|v| *v *= out_gain);
        if polar {
            for k in 1..block / 2 {
                let (re, im) = (spectrum[2 * k], spectrum[2 * k + 1]);
                spectrum[2 * k] = re.hypot(im);
                spectrum[2 * k + 1] = im.atan2(re);
            }
        }
        write_floats(command.output, &spectrum);
    } else {
        let mut spectrum: Vec<f64> = input.iter().map(|v| finite(*v)).collect();
        if polar {
            for k in 1..block / 2 {
                let (mag, phase) = (spectrum[2 * k], spectrum[2 * k + 1]);
                spectrum[2 * k] = mag * phase.cos();
                spectrum[2 * k + 1] = mag * phase.sin();
            }
        }
        spectrum.iter_mut().for_each(|v| *v *= in_gain);
        let x: Vec<f64> = inverse_packed(&spectrum).iter().map(|v| v * out_gain).collect();
        write_floats(command.output, &x);
    }
}
