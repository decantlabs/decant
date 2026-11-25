//! `XApuConnect`, `XApuDisconnect` and `XApuGetUtilization`.
#![allow(unsafe_code)]

mod common;

use common::*;
use xapu::abi::*;
use xapu::exports::*;

#[test]
fn null_pointers_are_refused() {
    let _s = serial();
    let p = params(1, 1, 1, 0, 0, 0);
    let mut region = ConnectionRegion { memory: std::ptr::null_mut(), result_event: 0 };
    let mut handle: Handle = 5;
    // SAFETY: null pointers are part of the contract.
    unsafe {
        assert_eq!(XApuConnect(&p, std::ptr::null_mut(), &mut handle), E_POINTER);
        assert_eq!(handle, 5, "nothing is cleared when a pointer is null");
        assert_eq!(XApuConnect(&p, &mut region, std::ptr::null_mut()), E_POINTER);
        // A null parameter block clears the outputs first, then fails.
        region.result_event = 9;
        assert_eq!(XApuConnect(std::ptr::null(), &mut region, &mut handle), E_INVALIDARG);
    }
    assert_eq!((handle, region.result_event), (0, 0));
}

#[test]
fn every_range_of_the_parameters_is_checked() {
    let _s = serial();
    let bad = [
        params(0, 1, 1, 0, 0x80, 0),
        params(4, 1, 1, 0, 0x80, 0),
        params(1, 0, 1, 0, 0, 0),
        params(1, 501, 1, 0, 0, 0),
        params(1, 1, 0, 0, 0, 0),
        params(1, 1, 17, 0, 0, 0),
        params(1, 1, 1, 0, 0, 0x10),
        params(1, 1, 1, 0, 0, 0x100),
        // Types 2 and 3 need an even frame count of 0x80 to 0x400.
        params(2, 1, 1, 0, 0x7E, 0),
        params(3, 1, 1, 0, 0x81, 0),
        params(3, 1, 1, 0, 0x402, 0),
        params(2, 1, 1, 0, 0, 0),
        params(3, 1, 1, 0, 0xFFFF_FFFE, 0),
    ];
    for p in bad {
        let (r, handle, region) = try_connect(&p);
        assert_eq!(r, E_INVALIDARG, "{p:?}");
        assert_eq!((handle, region.memory.is_null(), region.result_event), (0, true, 0), "outputs are cleared on failure");
    }
    // The edges are valid.
    for p in [params(1, 1, 1, 0, 0, 0), params(1, 500, 16, 0, 0x1234, 0xF), params(2, 1, 1, 0, 0x80, 0), params(3, 1, 1, 0, 0x400, 0), params(3, 1, 1, 0, 0x180, 0)] {
        drop(Conn::open(p));
    }
}

#[test]
fn the_region_and_the_event() {
    let _s = serial();
    let (r, h, region) = try_connect(&params(1, 1, 1, 0x20000000 + 1, 0, 0));
    assert_eq!((r, h, region.memory.is_null()), (E_OUTOFMEMORY, 0, true));
    let c = Conn::open(params(1, 1, 1, 0, 0, 0));
    assert!(c.region.memory.is_null(), "no region for a size of 0");
    assert_eq!(c.region.result_event, 0, "no event without option 8");
    drop(c);
    let c = Conn::open(params(1, 1, 1, 0x30000, 0, 0));
    assert!(!c.region.memory.is_null());
    assert_eq!(c.region.memory as usize % 0x10000, 0, "aligned to 64 KiB");
    // SAFETY: the region is 0x30000 bytes.
    assert!(unsafe { std::slice::from_raw_parts(c.region.memory, 0x30000) }.iter().all(|b| *b == 0));
    assert_eq!(c.region.result_event, 0);
    drop(c);
    let c = Conn::open(params(1, 1, 1, 0, 0, 8));
    assert_ne!(c.region.result_event, 0, "option 8 makes an event");
}

#[test]
fn the_stream_budget_is_shared_by_all_connections() {
    let _s = serial();
    let mut free = [0u32; 2];
    // SAFETY: two writable UINT32.
    unsafe { assert_eq!(XApuGetUtilization(1, 1, free.as_mut_ptr()), S_OK) };
    assert_eq!(free, [500, 500]);
    let a = Conn::open(params(1, 250, 1, 0, 0, 0));
    let (r, h, _) = try_connect(&params(1, 251, 1, 0, 0, 0));
    assert_eq!((r, h), (XAPU_E_NO_STREAMS, 0));
    let b = Conn::open(params(1, 250, 1, 0, 0, 0));
    // SAFETY: two writable UINT32.
    unsafe { assert_eq!(XApuGetUtilization(3, 2, free.as_mut_ptr()), S_OK) };
    assert_eq!(free, [0, 500]);
    assert_eq!(try_connect(&params(1, 1, 1, 0, 0, 0)).0, XAPU_E_NO_STREAMS);
    drop(a);
    // SAFETY: as above.
    unsafe { assert_eq!(XApuGetUtilization(2, 1, free.as_mut_ptr()), S_OK) };
    assert_eq!(free, [250, 500]);
    drop(Conn::open(params(1, 250, 1, 0, 0, 0)));
    drop(b);
    // SAFETY: as above.
    unsafe { assert_eq!(XApuGetUtilization(2, 1, free.as_mut_ptr()), S_OK) };
    assert_eq!(free, [500, 500]);
}

#[test]
fn disconnect_validates_the_handle() {
    let _s = serial();
    assert_eq!(XApuDisconnect(0), E_INVALIDARG);
    assert_eq!(XApuDisconnect(0x1234_5678), E_INVALIDARG);
    let (r, handle, _) = try_connect(&params(1, 2, 1, 0, 0, 0));
    assert_eq!(r, S_OK);
    assert_eq!(XApuDisconnect(handle), S_OK);
    assert_eq!(XApuDisconnect(handle), E_INVALIDARG, "a second disconnect");
    // A disconnected handle is no handle.
    let mut dq = 0;
    // SAFETY: a writable counter.
    assert_eq!(unsafe { XApuGetResultCount(handle, &mut dq) }, 0);
    let c = Command::new(COMMAND_START, 0, 0);
    // SAFETY: a valid command.
    assert_eq!(unsafe { XApuEnqueueCommand(handle, &c, std::ptr::null_mut()) }, E_INVALIDARG);
}

#[test]
fn utilization_validates_its_arguments() {
    let _s = serial();
    // SAFETY: a null pointer is part of the contract.
    assert_eq!(unsafe { XApuGetUtilization(1, 1, std::ptr::null_mut()) }, E_POINTER);
    for (t, k) in [(0, 1), (4, 1), (1, 0), (1, 3), (0, 0)] {
        let mut out = [7u32, 7];
        // SAFETY: two writable UINT32.
        assert_eq!(unsafe { XApuGetUtilization(t, k, out.as_mut_ptr()) }, E_INVALIDARG, "{t} {k}");
        assert_eq!(out, [0, 0], "cleared first");
    }
}
