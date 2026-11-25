//! Connect, disconnect, activate and deactivate: every error code of the specification.
#![allow(unsafe_code)]

mod common;

use chad::abi::*;
use chad::exports::*;
use common::*;

fn connect_with(memory: &mut [u8], size: u32, activations: u32, streams: u32, priority: u32) -> (Hresult, *mut XdspClient) {
    let mut client = 0x99 as *mut XdspClient;
    // SAFETY: a writable out pointer.
    let r = unsafe { XDspConnectWithMaximumStreamLimit(memory.as_mut_ptr() as *mut _, size, activations, streams, priority, &mut client) };
    (r, client)
}

fn activate(client: *mut XdspClient, p: &ActivationParameters) -> (Hresult, *mut u32, *mut XdspContext) {
    let (mut status, mut context) = (0x99 as *mut u32, 0x99 as *mut XdspContext);
    // SAFETY: writable out pointers.
    let r = unsafe { XDspActivate(client, &p.to_bytes(), &mut status, &mut context) };
    (r, status, context)
}

#[test]
fn connect_checks_its_arguments_in_the_order_of_the_specification() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    for (size, act, streams, prio) in [(0xFFFF, 1, 1, 0), (0x1F40_0001, 1, 1, 0), (0x10000, 0x81, 1, 0), (0x10000, 1, 0x101, 0), (0x10000, 1, 1, 3), (0, 0, 0, 0)] {
        let (r, client) = connect_with(&mut mem, size, act, streams, prio);
        assert_eq!(r, E_INVALIDARG, "{size:#x} {act} {streams} {prio}");
        assert!(client.is_null(), "the client is cleared");
    }
    // A null memory pointer.
    let mut client = std::ptr::null_mut();
    // SAFETY: a null memory pointer is part of the contract.
    assert_eq!(unsafe { XDspConnect(std::ptr::null_mut(), 0x10000, 1, &mut client) }, E_INVALIDARG);
    // A null client: the argument check comes first.
    // SAFETY: a null out pointer is part of the contract.
    unsafe {
        assert_eq!(XDspConnectWithMaximumStreamLimit(mem.as_mut_ptr() as *mut _, 0x10000, 1, 1, 0, std::ptr::null_mut()), E_POINTER);
        assert_eq!(XDspConnectWithMaximumStreamLimit(mem.as_mut_ptr() as *mut _, 0xFFFF, 1, 1, 0, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(XDspConnect(mem.as_mut_ptr() as *mut _, 0x10000, 1, std::ptr::null_mut()), E_POINTER);
    }
    // The edges of the ranges are valid.
    for (size, act, streams, prio) in [(0x10000, 0, 0, 0), (0x1F40_0000, 0x80, 0x100, 2), (0x10000, 1, 1, 1)] {
        let (r, client) = connect_with(&mut mem, size, act, streams, prio);
        assert_eq!(r, S_OK);
        assert!(!client.is_null());
        assert_eq!(XDspDisconnect(client), S_OK);
    }
    // XDspConnect is the same call with 0x100 streams and priority 2: its activation limit is the same.
    // SAFETY: as above.
    assert_eq!(unsafe { XDspConnect(mem.as_mut_ptr() as *mut _, 0x10000, 0x81, &mut client) }, E_INVALIDARG);
    // SAFETY: as above.
    assert_eq!(unsafe { XDspConnect(mem.as_mut_ptr() as *mut _, 0x10000, 0x80, &mut client) }, S_OK);
    assert_eq!(XDspDisconnect(client), S_OK);
}

#[test]
fn activations_and_clients_are_limited_over_all_clients() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    // 8 clients of 0x10 activations use all of the 0x80.
    let clients: Vec<_> = (0..8).map(|_| connect_with(&mut mem, 0x10000, 0x10, 1, 0)).collect();
    assert!(clients.iter().all(|(r, c)| *r == S_OK && !c.is_null()));
    // A ninth client with no activation is refused for the number of clients, with one for the activations.
    assert_eq!(connect_with(&mut mem, 0x10000, 0, 1, 0).0, XDSP_E_TOO_MANY_CLIENTS);
    assert_eq!(connect_with(&mut mem, 0x10000, 1, 1, 0), (XDSP_E_TOO_MANY_ACTIVATIONS, std::ptr::null_mut()));
    for (_, c) in &clients[..4] {
        assert_eq!(XDspDisconnect(*c), S_OK);
    }
    // Room for 0x40 activations is left.
    let (r, big) = connect_with(&mut mem, 0x10000, 0x40, 1, 0);
    assert_eq!(r, S_OK);
    assert_eq!(connect_with(&mut mem, 0x10000, 1, 1, 0).0, XDSP_E_TOO_MANY_ACTIVATIONS);
    assert_eq!(XDspDisconnect(big), S_OK);
    for (_, c) in &clients[4..] {
        assert_eq!(XDspDisconnect(*c), S_OK);
    }
}

#[test]
fn disconnect_needs_a_live_client_without_contexts() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    assert_eq!(XDspDisconnect(std::ptr::null_mut()), E_INVALIDARG);
    assert_eq!(XDspDisconnect(0x1234 as *mut XdspClient), E_INVALIDARG);
    let (_, client) = connect_with(&mut mem, 0x10000, 2, 1, 0);
    let ir = vec![0f32; 9 * 1024];
    let (r, _, context) = activate(client, &params(PROCESS_CONVOLUTION, 0, 0x200, 1, &ir));
    assert_eq!(r, S_OK);
    assert_eq!(XDspDisconnect(client), XDSP_E_CONTEXTS_ACTIVE);
    assert_eq!(XDspDeactivate(context), S_OK);
    assert_eq!(XDspDisconnect(client), S_OK);
    assert_eq!(XDspDisconnect(client), E_INVALIDARG, "a second disconnect");
}

#[test]
fn activate_checks_pointers_then_arguments() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    let (_, client) = connect_with(&mut mem, 0x10000, 4, 1, 0);
    let p = params(PROCESS_FFT, 0, 0x200, 1, &[]);
    let (mut status, mut context) = (0x99 as *mut u32, 0x99 as *mut XdspContext);
    // SAFETY: null pointers are part of the contract.
    unsafe {
        assert_eq!(XDspActivate(client, &p.to_bytes(), std::ptr::null_mut(), &mut context), E_POINTER);
        assert!(context.is_null(), "the other output is cleared");
        assert_eq!(XDspActivate(client, &p.to_bytes(), &mut status, std::ptr::null_mut()), E_POINTER);
        assert!(status.is_null());
        assert_eq!(XDspActivate(std::ptr::null_mut(), &p.to_bytes(), &mut status, &mut context), E_INVALIDARG);
        assert_eq!(XDspActivate(client, std::ptr::null(), &mut status, &mut context), E_INVALIDARG);
        assert_eq!(XDspActivate(0x1234 as *mut XdspClient, &p.to_bytes(), &mut status, &mut context), E_INVALIDARG);
    }
    assert_eq!(XDspDisconnect(client), S_OK);
}

#[test]
fn every_rule_of_the_activation_parameters() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    let (_, client) = connect_with(&mut mem, 0x10000, 40, 1, 0);
    let ir = vec![0f32; 9 * 1024];
    let conv = |flags, block, channels| params(PROCESS_CONVOLUTION, flags, block, channels, &ir);
    let fft = |t, flags, block, channels| params(t, flags, block, channels, &[]);
    let bad = [
        fft(0, 0, 0x200, 1),
        fft(4, 0, 0x200, 1),
        fft(PROCESS_FFT, 0, 0x200, 0),
        fft(PROCESS_FFT, 0, 0x200, 3),
        // Flags 2 and 8 need two channels.
        fft(PROCESS_FFT, FLAG_DEINTERLEAVED, 0x200, 1),
        fft(PROCESS_FFT, FLAG_STEREO_IR, 0x200, 1),
        conv(FLAG_DEINTERLEAVED, 0x200, 1),
        conv(FLAG_STEREO_IR, 0x200, 1),
        // No polar convolution.
        conv(FLAG_POLAR, 0x200, 1),
        conv(FLAG_POLAR | FLAG_DEINTERLEAVED, 0x200, 2),
        // Block sizes.
        conv(0, 0x100, 1),
        conv(0, 0x800, 1),
        conv(0, 0, 1),
        fft(PROCESS_FFT, 0, 0x100, 1),
        fft(PROCESS_FFT, 0, 0x1000, 1),
        fft(PROCESS_INVERSE_FFT, 0, 0x300, 1),
        // No stereo FFT.
        fft(PROCESS_FFT, 0, 0x200, 2),
        fft(PROCESS_INVERSE_FFT, FLAG_DEINTERLEAVED, 0x200, 2),
    ];
    for p in bad {
        let (r, status, context) = activate(client, &p);
        assert_eq!(r, E_INVALIDARG, "{p:?}");
        assert!(status.is_null() && context.is_null(), "outputs are cleared on failure");
    }
    let good = [
        conv(0, 0x200, 1),
        conv(0, 0x400, 2),
        conv(FLAG_DEINTERLEAVED, 0x200, 2),
        conv(FLAG_STEREO_IR | FLAG_DEINTERLEAVED, 0x200, 2),
        fft(PROCESS_FFT, 0, 0x200, 1),
        fft(PROCESS_FFT, FLAG_POLAR, 0x400, 1),
        fft(PROCESS_INVERSE_FFT, FLAG_POLAR, 0x800, 1),
    ];
    let mut contexts = Vec::new();
    for p in good {
        // The stereo impulse response needs twice the floats: this one is zeros, and only its first half is read.
        let ir2 = vec![0f32; 2 * 9 * 1024 * 2];
        let p = if p.flags & FLAG_STEREO_IR != 0 { params(PROCESS_CONVOLUTION, p.flags, p.block, p.channels, &ir2[..9 * 1024]) } else if p.process_type == PROCESS_CONVOLUTION && p.block == 0x400 { params(PROCESS_CONVOLUTION, 0, 0x400, 2, &ir2[..9 * 2048]) } else { p };
        let (r, status, context) = activate(client, &p);
        assert_eq!(r, S_OK, "{p:?}");
        assert!(!status.is_null() && !context.is_null());
        // SAFETY: the status array has 4 words.
        assert_eq!(unsafe { std::slice::from_raw_parts(status, 4) }, [0, 0, 0, 0]);
        contexts.push(context);
    }
    contexts.sort();
    contexts.dedup();
    assert_eq!(contexts.len(), 7, "every context has its own handle");
    for c in contexts {
        assert_eq!(XDspDeactivate(c), S_OK);
    }
    assert_eq!(XDspDisconnect(client), S_OK);
}

#[test]
fn the_impulse_response_size_is_checked() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    let (_, client) = connect_with(&mut mem, 0x10000, 8, 1, 0);
    let block = 0x200usize;
    let floats = |partitions: usize| vec![0f32; partitions * 2 * block];
    let conv = |ir: &[f32], len: usize| ActivationParameters { process_type: PROCESS_CONVOLUTION, flags: 0, block: 0x200, channels: 1, ir_length: len as u32, ir_pointer: ir.as_ptr() as u64 };
    let nine = floats(9);
    // Fewer than 9 partitions, and a length that makes 8.
    for len in [0, 1, 8 * 2 * block, 9 * 2 * block - 1] {
        assert_eq!(activate(client, &conv(&nine, len)).0, XDSP_E_IR_SIZE, "length {len}");
    }
    // The size check comes before the null check, which gives E_INVALIDARG.
    let null_ir = ActivationParameters { ir_pointer: 0, ..conv(&nine, 9 * 2 * block) };
    assert_eq!(activate(client, &null_ir).0, E_INVALIDARG);
    assert_eq!(activate(client, &ActivationParameters { ir_pointer: 0, ..conv(&nine, 5) }).0, XDSP_E_IR_SIZE);
    // 9 partitions are the smallest. Lengths are floored to whole partitions.
    let (r, _, c) = activate(client, &conv(&nine, 9 * 2 * block + 1000));
    assert_eq!(r, S_OK);
    assert_eq!(XDspDeactivate(c), S_OK);
    // Rounded up to 8, the partitions may not exceed 0x800: 0x7F9 does not, 0x801 does.
    let big = floats(0x7F9);
    let (r, _, c) = activate(client, &conv(&big, big.len()));
    assert_eq!(r, S_OK, "0x7F9 partitions round up to 0x800");
    assert_eq!(XDspDeactivate(c), S_OK);
    assert_eq!(activate(client, &conv(&big, 0x801 * 2 * block)).0, XDSP_E_IR_SIZE);
    assert_eq!(activate(client, &conv(&big, 0x808 * 2 * block)).0, XDSP_E_IR_SIZE);
    assert_eq!(XDspDisconnect(client), S_OK);
}

#[test]
fn contexts_are_limited_by_the_activations_of_the_client() {
    let _s = serial();
    let mut mem = vec![0u8; 0x10000];
    let (_, client) = connect_with(&mut mem, 0x10000, 2, 1, 0);
    let p = params(PROCESS_FFT, 0, 0x200, 1, &[]);
    let (r1, _, c1) = activate(client, &p);
    let (r2, _, c2) = activate(client, &p);
    assert_eq!((r1, r2), (S_OK, S_OK));
    let (r3, status, context) = activate(client, &p);
    assert_eq!(r3, XDSP_E_TOO_MANY_CONTEXTS);
    assert!(status.is_null() && context.is_null());
    assert_eq!(XDspDeactivate(c1), S_OK);
    let (r4, _, c4) = activate(client, &p);
    assert_eq!(r4, S_OK, "a deactivated context frees its place");
    // A client without activations has no place at all.
    let (_, none) = connect_with(&mut mem, 0x10000, 0, 1, 0);
    assert_eq!(activate(none, &p).0, XDSP_E_TOO_MANY_CONTEXTS);
    assert_eq!(XDspDisconnect(none), S_OK);
    for c in [c2, c4] {
        assert_eq!(XDspDeactivate(c), S_OK);
    }
    assert_eq!(XDspDisconnect(client), S_OK);
}

#[test]
fn deactivate_validates_the_context() {
    let _s = serial();
    assert_eq!(XDspDeactivate(std::ptr::null_mut()), E_INVALIDARG);
    assert_eq!(XDspDeactivate(0x1234 as *mut XdspContext), E_INVALIDARG);
    let rig = Rig::new(&params(PROCESS_FFT, 0, 0x200, 1, &[]));
    let c = rig.context;
    drop(rig);
    assert_eq!(XDspDeactivate(c), E_INVALIDARG, "a second deactivate");
}
