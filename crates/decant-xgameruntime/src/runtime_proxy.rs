//! A logging proxy over every table the title gets from the PC runtime (`System32\xgameruntime.dll`
//! through the `QueryApiImpl` pass through). Without it the calls on those tables are invisible.
//!
//! The title gets a proxy object instead of the PC object: `{ vtable, inner, info }`. The vtable
//! has [`SLOTS`] entries, each a small thunk (`mov eax, slot` and a jump) into one common
//! forwarder written in assembly. The forwarder saves the four argument registers and the four
//! argument XMM registers, lets [`proxy_before`] swap `this` for the PC object and pick the PC
//! slot, copies [`STACK_ARGS`] stack arguments of the caller into its own outgoing area at the
//! same offsets, restores the registers and calls the PC slot. RAX and XMM0 of the result are
//! kept while [`proxy_after`] logs, then returned. So a slot of any arity up to `4 + STACK_ARGS`
//! integer or floating point arguments reaches the PC runtime with the values the title passed.
//! The vtable length of a PC table is not known, which is why the thunks cover more slots than
//! any known table has.
//!
//! Log: per slot the first call (a line before and a line after the call), then at most one
//! line per slot per [`RATE_MS`] with the number of calls since the last line. `DECANT_XGR_TRACE=1`
//! logs every call before and after with eight arguments. Each line names the table, the slot
//! (by name when `tables` knows it), the first arguments, the result and the calling module with
//! its RVA. `DECANT_XGR_PROXY=0` hands the PC objects out unwrapped.
//!
//! The async blocks the title starts through a PC table are watched by `runtime_proxy_async`.
//!
//! Our own code.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};

use crate::tables::Guid;

/// Number of slots the proxy vtable has.
pub(crate) const SLOTS: usize = 512;
/// Stack arguments (after the four register arguments) handed on to the PC slot.
const STACK_ARGS: usize = 16;
/// After the first call, a slot logs at most one line per this many milliseconds.
pub(crate) const RATE_MS: u64 = 1000;

/// What the forwarder keeps on its stack for one call. The offsets are fixed by the assembly.
#[repr(C)]
pub(crate) struct Frame {
    pub(crate) rcx: usize,
    pub(crate) rdx: usize,
    pub(crate) r8: usize,
    pub(crate) r9: usize,
    xmm: [[u64; 2]; 4],
    pub(crate) slot: usize,
    /// The return address into the title.
    pub(crate) ret: usize,
    /// The caller's first stack argument (its fifth argument).
    pub(crate) stack: *const usize,
    /// The PC slot to call, written by [`proxy_before`].
    target: usize,
    /// Handed from [`proxy_before`] to [`proxy_after`]: 0 or a boxed [`Call`].
    cookie: usize,
    pub(crate) rax: usize,
    xmm0_out: [u64; 2],
    /// The proxy the title called, kept by [`proxy_before`] (the padding word of the forwarder).
    proxy: usize,
}

const _: () = {
    assert!(std::mem::offset_of!(Frame, xmm) == 0x20);
    assert!(std::mem::offset_of!(Frame, slot) == 0x60);
    assert!(std::mem::offset_of!(Frame, ret) == 0x68);
    assert!(std::mem::offset_of!(Frame, stack) == 0x70);
    assert!(std::mem::offset_of!(Frame, target) == 0x78);
    assert!(std::mem::offset_of!(Frame, cookie) == 0x80);
    assert!(std::mem::offset_of!(Frame, rax) == 0x88);
    assert!(size_of::<Frame>() == 0xA8);
};

// The forwarder. Entry: RCX = proxy object, EAX = slot, [RSP] = return address into the title.
// Stack: 0x00 to 0x20 home area of the calls it makes, 0x20 to 0xA0 the copied stack arguments,
// 0xA0 to 0x148 the `Frame` (its last word only Rust touches), so that RSP stays 16 byte aligned
// at each call.
// The caller's fifth argument is at 0x148 (return address) + 8 + 0x20 = 0x170.
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    ".globl decant_xgr_proxy_common",
    ".seh_proc decant_xgr_proxy_common",
    "decant_xgr_proxy_common:",
    "subq $0x148, %rsp",
    ".seh_stackalloc 0x148",
    ".seh_endprologue",
    "movq %rcx, 0xA0(%rsp)",
    "movq %rdx, 0xA8(%rsp)",
    "movq %r8, 0xB0(%rsp)",
    "movq %r9, 0xB8(%rsp)",
    "movdqu %xmm0, 0xC0(%rsp)",
    "movdqu %xmm1, 0xD0(%rsp)",
    "movdqu %xmm2, 0xE0(%rsp)",
    "movdqu %xmm3, 0xF0(%rsp)",
    "movq %rax, 0x100(%rsp)",
    "movq 0x148(%rsp), %rax",
    "movq %rax, 0x108(%rsp)",
    "leaq 0x170(%rsp), %rax",
    "movq %rax, 0x110(%rsp)",
    "leaq 0xA0(%rsp), %rcx",
    "callq {before}",
    "xorl %eax, %eax",
    "2:",
    "movq 0x170(%rsp,%rax,8), %r10",
    "movq %r10, 0x20(%rsp,%rax,8)",
    "incl %eax",
    "cmpl ${stack_args}, %eax",
    "jb 2b",
    "movq 0xA0(%rsp), %rcx",
    "movq 0xA8(%rsp), %rdx",
    "movq 0xB0(%rsp), %r8",
    "movq 0xB8(%rsp), %r9",
    "movdqu 0xC0(%rsp), %xmm0",
    "movdqu 0xD0(%rsp), %xmm1",
    "movdqu 0xE0(%rsp), %xmm2",
    "movdqu 0xF0(%rsp), %xmm3",
    "callq *0x118(%rsp)",
    "movq %rax, 0x128(%rsp)",
    "movdqu %xmm0, 0x130(%rsp)",
    "leaq 0xA0(%rsp), %rcx",
    "callq {after}",
    "movq 0x128(%rsp), %rax",
    "movdqu 0x130(%rsp), %xmm0",
    "addq $0x148, %rsp",
    "retq",
    ".seh_endproc",
    // One thunk per slot, 16 bytes apart: slot N is at `decant_xgr_proxy_thunks + 16 * N`.
    ".p2align 4",
    ".globl decant_xgr_proxy_thunks",
    "decant_xgr_proxy_thunks:",
    ".set decant_xgr_proxy_slot, 0",
    ".rept {slots}",
    ".p2align 4",
    "movl $decant_xgr_proxy_slot, %eax",
    "jmp decant_xgr_proxy_common",
    ".set decant_xgr_proxy_slot, decant_xgr_proxy_slot + 1",
    ".endr",
    before = sym proxy_before,
    after = sym proxy_after,
    stack_args = const STACK_ARGS,
    slots = const SLOTS,
    options(att_syntax)
);

unsafe extern "C" {
    static decant_xgr_proxy_thunks: u8;
}

/// The vtable every proxy object shares: the addresses of the thunks.
fn vtable() -> &'static [usize; SLOTS] {
    static VTABLE: OnceLock<Box<[usize; SLOTS]>> = OnceLock::new();
    VTABLE.get_or_init(|| {
        let base = (&raw const decant_xgr_proxy_thunks) as usize;
        Box::new(std::array::from_fn(|slot| base + 16 * slot))
    })
}

/// The proxy object the title gets.
#[repr(C)]
pub(crate) struct Proxy {
    vtbl: *const usize,
    inner: *mut c_void,
    info: &'static Info,
}

/// What the log knows about one proxied table.
pub(crate) struct Info {
    /// The table as the log names it.
    pub(crate) table: String,
    /// Slot names by index, from `tables`.
    slots: &'static [Option<&'static str>],
    trace: bool,
    stats: Box<[SlotStat]>,
}

#[derive(Default)]
struct SlotStat {
    calls: AtomicU64,
    /// Milliseconds since [`epoch`] of the last line logged, plus one (0: none yet).
    last_line: AtomicU64,
    /// The call number of the last line logged.
    last_logged: AtomicU64,
}

impl Info {
    pub(crate) fn new(table: String, slots: &'static [Option<&'static str>], trace: bool) -> &'static Info {
        Box::leak(Box::new(Info { table, slots, trace, stats: (0..SLOTS).map(|_| SlotStat::default()).collect() }))
    }

    fn slot_name(&self, slot: usize) -> &'static str {
        match slot {
            0 => "QueryInterface",
            1 => "AddRef",
            2 => "Release",
            _ => self.slots.get(slot).copied().flatten().unwrap_or("?"),
        }
    }
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// `DECANT_XGR_TRACE=1`: every call is logged in full.
fn trace_env() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("DECANT_XGR_TRACE").is_some_and(|v| v == "1"))
}

/// `DECANT_XGR_PROXY=0`: the PC objects are handed out unwrapped.
fn proxy_off() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("DECANT_XGR_PROXY").is_some_and(|v| v == "0"))
}

/// Writes one line to the log (and to the capture of the tests).
pub(crate) fn emit(line: String) {
    #[cfg(test)]
    tests_capture::push(&line);
    crate::exports::log(&line);
}

/// The proxy for `inner`, made once per object and table.
pub(crate) fn proxy_for(inner: *mut c_void, make_info: impl FnOnce() -> &'static Info) -> *mut c_void {
    static PROXIES: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
    let mut map = PROXIES.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    let proxy = *map.entry(inner as usize).or_insert_with(|| {
        let proxy = Box::leak(Box::new(Proxy { vtbl: vtable().as_ptr(), inner, info: make_info() }));
        (proxy as *mut Proxy) as usize
    });
    proxy as *mut c_void
}

/// The table name the log uses for a couple.
fn table_name(api: Option<&Guid>, riid: Option<&Guid>) -> (String, &'static [Option<&'static str>]) {
    let riid_text = riid.map_or_else(|| "-".to_string(), |g| format!("{:08X}", g.data1));
    match api.and_then(|api| crate::tables::known(api, riid)) {
        Some((table, _)) => (format!("{} (riid {riid_text})", table.name), table.slots),
        None => (format!("{} (riid {riid_text})", api.map_or_else(|| "-".to_string(), crate::guid_text)), &[]),
    }
}

/// After a successful `QueryApiImpl` forwarded to the PC runtime: replaces the object in `*out`
/// with its proxy. `out` was written by the PC runtime, so it is a writable pointer location.
///
/// # Safety
/// `out` is null or points to the object pointer the PC runtime just wrote.
pub(crate) unsafe fn wrap_out(api: Option<&Guid>, riid: Option<&Guid>, out: *mut *mut c_void) {
    if proxy_off() || !crate::exports::readable(out.cast(), size_of::<usize>()) {
        return;
    }
    // SAFETY: readable, and the PC runtime wrote it as the caller's out pointer.
    let inner = unsafe { *out };
    if inner.is_null() {
        return;
    }
    let proxy = proxy_for(inner, || {
        let (table, slots) = table_name(api, riid);
        Info::new(table, slots, trace_env())
    });
    // SAFETY: see above.
    unsafe { *out = proxy };
    emit(format!("proxy: table {:016X} of the PC runtime handed out as {:016X}", inner as usize, proxy as usize));
}

/// The state of one call kept from [`proxy_before`] to [`proxy_after`].
struct Call {
    n: u64,
    log_after: bool,
    caller: String,
    blocks: Vec<crate::runtime_proxy_async::Candidate>,
}

/// The caller as `module+0xRVA`, or the bare address.
pub(crate) fn caller_text(address: usize) -> String {
    static NAMES: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
    let mut module = std::ptr::null_mut();
    // SAFETY: the address is only looked up; the reference count is left unchanged.
    let found = unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, address as *const u16, &mut module) } != 0;
    if !found || module.is_null() {
        return format!("0x{address:x}");
    }
    let base = module as usize;
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let name = names.get_or_insert_with(HashMap::new).entry(base).or_insert_with(|| {
        let mut buf = [0u16; 520];
        // SAFETY: `buf` holds the length passed.
        let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        let path = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
        path.rsplit(['\\', '/']).next().unwrap_or("?").to_string()
    });
    format!("{name}+0x{:x}", address - base)
}

/// The arguments after `this` the log shows: three registers, then stack arguments.
fn args(frame: &Frame, count: usize) -> Vec<usize> {
    let mut args = vec![frame.rdx, frame.r8, frame.r9];
    for i in 0..count.saturating_sub(3) {
        // SAFETY: the forwarder copies the same `STACK_ARGS` words, so they are readable.
        args.push(unsafe { *frame.stack.add(i) });
    }
    args
}

fn hex_list(values: &[usize]) -> String {
    values.iter().map(|v| format!("0x{v:x}")).collect::<Vec<_>>().join(", ")
}

/// Which lines a call gets: `(before, after)`.
fn lines_for(stat: &SlotStat, n: u64, trace: bool, now_ms: u64) -> (bool, bool) {
    if trace || n == 1 {
        stat.last_line.store(now_ms + 1, Ordering::Relaxed);
        stat.last_logged.store(n, Ordering::Relaxed);
        return (true, true);
    }
    let last = stat.last_line.load(Ordering::Relaxed);
    if now_ms + 1 >= last + RATE_MS && stat.last_line.compare_exchange(last, now_ms + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        return (false, true);
    }
    (false, false)
}

/// Called by the forwarder before the PC slot: swaps `this`, picks the PC slot, logs.
///
/// # Safety
/// `frame` is the forwarder's frame; its `rcx` is a [`Proxy`] handed out by [`proxy_for`].
unsafe extern "C" fn proxy_before(frame: *mut Frame) {
    // SAFETY: the caller's contract.
    let frame = unsafe { &mut *frame };
    // SAFETY: the forwarder is only reached through a proxy's vtable, with the proxy as `this`.
    let proxy = unsafe { &*(frame.rcx as *const Proxy) };
    let slot = frame.slot;
    // SAFETY: the PC object starts with its vtable; the title only calls slots the table has.
    frame.target = unsafe { *(*(proxy.inner as *const *const usize)).add(slot) };
    frame.proxy = frame.rcx;
    frame.rcx = proxy.inner as usize;
    let info = proxy.info;
    let stat = &info.stats[slot];
    let n = stat.calls.fetch_add(1, Ordering::Relaxed) + 1;
    let (before, after) = lines_for(stat, n, info.trace, epoch().elapsed().as_millis() as u64);
    // An async block is mostly the last argument: eight are searched, four shown (eight traced).
    let all = args(frame, 8);
    let blocks = crate::runtime_proxy_async::candidates(&all);
    let shown = &all[..if info.trace { 8 } else { 4 }];
    if !before && !after && blocks.is_empty() {
        frame.cookie = 0;
        return;
    }
    let caller = caller_text(frame.ret);
    if before {
        emit(format!("windows {} slot {slot} {}({}) from {caller} [call {n}]", info.table, info.slot_name(slot), hex_list(shown)));
    }
    frame.cookie = Box::into_raw(Box::new(Call { n, log_after: after, caller, blocks })) as usize;
}

/// Called by the forwarder after the PC slot returned: logs the result, wraps an object handed
/// out by QueryInterface, registers the async blocks started.
///
/// # Safety
/// As [`proxy_before`], after it ran on the same frame.
unsafe extern "C" fn proxy_after(frame: *mut Frame) {
    // SAFETY: the caller's contract.
    let frame = unsafe { &mut *frame };
    let slot = frame.slot;
    let hr = frame.rax as u32 as i32;
    // SAFETY: `proxy_before` kept the proxy the title called.
    let info = unsafe { &*(frame.proxy as *const Proxy) }.info;
    if slot == 0 && hr >= 0 {
        // QueryInterface handed out another PC object of the same table: wrap it as well.
        let out = frame.r8 as *mut *mut c_void;
        if crate::exports::readable(out.cast(), size_of::<usize>()) {
            // SAFETY: the PC runtime just wrote the object there.
            let object = unsafe { *out };
            if !object.is_null() {
                // SAFETY: as above.
                unsafe { *out = proxy_for(object, || Info::new(info.table.clone(), info.slots, info.trace)) };
            }
        }
    }
    if frame.cookie == 0 {
        return;
    }
    // SAFETY: `proxy_before` put a boxed `Call` there, taken back once.
    let call = unsafe { Box::from_raw(frame.cookie as *mut Call) };
    frame.cookie = 0;
    if call.log_after {
        let stat = &info.stats[slot];
        let since = call.n.saturating_sub(stat.last_logged.swap(call.n, Ordering::Relaxed));
        let wide = if frame.rax >> 32 != 0 && info.trace { format!(" (rax 0x{:x})", frame.rax) } else { String::new() };
        let skipped = if since > 1 { format!(", {} calls since the last line", since) } else { String::new() };
        emit(format!("windows {} slot {slot} {} -> 0x{:08x}{wide} from {} [call {}{skipped}]", info.table, info.slot_name(slot), hr as u32, call.caller, call.n));
    }
    if hr >= 0 {
        crate::runtime_proxy_async::started(call.blocks, &info.table, slot, info.slot_name(slot), &call.caller, info.trace);
    }
}

#[cfg(test)]
pub(crate) mod tests_capture {
    use std::sync::Mutex;

    static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub(crate) fn push(line: &str) {
        LINES.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_string());
    }

    /// The captured lines that contain `needle`.
    pub(crate) fn lines_with(needle: &str) -> Vec<String> {
        LINES.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|l| l.contains(needle)).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_OK: i32 = 0;

    /// A fake PC object: its vtable and a value the slots check `this` by.
    #[repr(C)]
    struct Fake {
        vtbl: &'static [usize; 8],
        tag: u64,
    }

    unsafe extern "system" fn qi(this: *const Fake, _riid: *const Guid, out: *mut *mut c_void) -> i32 {
        // SAFETY: the test passes a writable out pointer.
        unsafe { *out = (&raw const fakes()[1]).cast_mut().cast() };
        let _ = this;
        S_OK
    }
    unsafe extern "system" fn add_ref(_this: *const Fake) -> u32 {
        1
    }
    unsafe extern "system" fn one(this: *const Fake, a: u32) -> i32 {
        // SAFETY: `this` is the fake object (the proxy swapped it in).
        assert_eq!(unsafe { (*this).tag }, 0xF00D);
        a as i32 + 1
    }
    #[allow(clippy::too_many_arguments)]
    unsafe extern "system" fn seven(this: *const Fake, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64) -> u64 {
        // SAFETY: as above.
        assert_eq!(unsafe { (*this).tag }, 0xF00D);
        a + 10 * b + 100 * c + 1000 * d + 10_000 * e + 100_000 * f + 1_000_000 * g
    }
    unsafe extern "system" fn floats(_this: *const Fake, x: f64, y: f32, z: f64) -> f64 {
        x * f64::from(y) + z
    }
    #[allow(clippy::too_many_arguments)]
    unsafe extern "system" fn many(_this: *const Fake, out: *mut u64, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64, i: u64, j: u64, k: u64, l: u64, m: u64) -> i32 {
        let all = [a, b, c, d, e, f, g, h, i, j, k, l, m];
        // SAFETY: the test passes a writable out pointer.
        unsafe { *out = all.iter().enumerate().map(|(n, v)| v * (n as u64 + 1)).sum() };
        S_OK
    }
    unsafe extern "system" fn start(_this: *const Fake, _flags: u32, block: *mut crate::xasync::XAsyncBlock) -> i32 {
        // As a runtime does: its own state in the block's internal bytes.
        // SAFETY: the test passes a writable block.
        unsafe { (*block).internal = [0x5A; 32] };
        S_OK
    }

    /// Two fake objects sharing one vtable, made once.
    fn fakes() -> &'static [Fake; 2] {
        static FAKES: OnceLock<[Fake; 2]> = OnceLock::new();
        FAKES.get_or_init(|| {
            let vtbl: &'static [usize; 8] =
                Box::leak(Box::new([qi as *const () as usize, add_ref as *const () as usize, add_ref as *const () as usize, one as *const () as usize, seven as *const () as usize, floats as *const () as usize, many as *const () as usize, start as *const () as usize]));
            [Fake { vtbl, tag: 0xF00D }, Fake { vtbl, tag: 0xF00D }]
        })
    }

    fn wrapped(table: &str, trace: bool) -> *mut c_void {
        let inner = (&raw const fakes()[0]).cast_mut().cast::<c_void>();
        // A fresh proxy per test (`proxy_for` keeps one per object).
        let info = Info::new(table.to_string(), &[], trace);
        Box::leak(Box::new(Proxy { vtbl: vtable().as_ptr(), inner, info })) as *mut Proxy as *mut c_void
    }

    fn slot<T: Copy>(proxy: *mut c_void, n: usize) -> T {
        // SAFETY: the proxy's first field is the thunk vtable; `T` is a function pointer type.
        unsafe { std::mem::transmute_copy(&*(*(proxy as *const *const usize)).add(n)) }
    }

    #[test]
    fn calls_through_the_proxy_keep_arguments_and_results() {
        let p = wrapped("fake windows table A", false);
        let f: unsafe extern "system" fn(*mut c_void, u32) -> i32 = slot(p, 3);
        // SAFETY: the fake slot takes these arguments.
        assert_eq!(unsafe { f(p, 41) }, 42);
        let f: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64, u64, u64, u64) -> u64 = slot(p, 4);
        // SAFETY: as above.
        assert_eq!(unsafe { f(p, 1, 2, 3, 4, 5, 6, 7) }, 7_654_321);
        let f: unsafe extern "system" fn(*mut c_void, f64, f32, f64) -> f64 = slot(p, 5);
        // SAFETY: as above.
        assert_eq!(unsafe { f(p, 1.5, 4.0, 0.25) }, 6.25);
        let f: unsafe extern "system" fn(*mut c_void, *mut u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> i32 = slot(p, 6);
        let mut out = 0u64;
        // SAFETY: as above.
        assert_eq!(unsafe { f(p, &mut out, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13) }, S_OK);
        assert_eq!(out, (1..=13u64).map(|v| v * v).sum::<u64>());

        let lines = tests_capture::lines_with("windows fake windows table A ");
        assert!(lines.iter().any(|l| l.starts_with("windows fake windows table A slot 3 ?(0x29, ") && l.ends_with("[call 1]")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("windows fake windows table A slot 3 ? -> 0x0000002a from ") && l.contains("[call 1]")), "{lines:?}");
        assert!(lines.iter().any(|l| l.starts_with("windows fake windows table A slot 4 ?(0x1, 0x2, 0x3, 0x4) from ")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("slot 6 ? -> 0x00000000 from ")), "{lines:?}");
    }

    #[test]
    fn later_calls_are_rate_limited_and_trace_logs_all() {
        let p = wrapped("fake windows table B", false);
        let f: unsafe extern "system" fn(*mut c_void, u32) -> i32 = slot(p, 3);
        for i in 0..100 {
            // SAFETY: the fake slot takes these arguments.
            assert_eq!(unsafe { f(p, i) }, i as i32 + 1);
        }
        let lines = tests_capture::lines_with("windows fake windows table B slot 3");
        assert!(lines.len() <= 3, "{lines:?}");

        let p = wrapped("fake windows table C", true);
        let f: unsafe extern "system" fn(*mut c_void, u32) -> i32 = slot(p, 3);
        for i in 0..10 {
            // SAFETY: as above.
            unsafe { f(p, i) };
        }
        assert_eq!(tests_capture::lines_with("windows fake windows table C slot 3").len(), 20);
    }

    #[test]
    fn the_rate_limit_lets_one_line_through_per_period() {
        let stat = SlotStat::default();
        assert_eq!(lines_for(&stat, 1, false, 0), (true, true));
        assert_eq!(lines_for(&stat, 2, false, 10), (false, false));
        assert_eq!(lines_for(&stat, 3, false, RATE_MS), (false, true));
        assert_eq!(lines_for(&stat, 4, false, RATE_MS + 1), (false, false));
        assert_eq!(lines_for(&stat, 5, true, RATE_MS + 2), (true, true));
    }

    #[test]
    fn query_interface_hands_out_a_proxy_too() {
        let p = wrapped("fake windows table D", false);
        let f: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32 = slot(p, 0);
        let mut out = std::ptr::null_mut();
        // SAFETY: the fake slot takes these arguments.
        assert_eq!(unsafe { f(p, std::ptr::null(), &mut out) }, S_OK);
        assert_ne!(out, (&raw const fakes()[1]).cast_mut().cast());
        let g: unsafe extern "system" fn(*mut c_void, u32) -> i32 = slot(out, 3);
        // SAFETY: as above.
        assert_eq!(unsafe { g(out, 1) }, 2);
    }

    #[test]
    fn an_async_block_started_through_a_windows_table_is_watched() {
        unsafe extern "system" fn done(_b: *mut crate::xasync::XAsyncBlock) {}
        let p = wrapped("fake windows table E", false);
        let f: unsafe extern "system" fn(*mut c_void, u32, *mut crate::xasync::XAsyncBlock) -> i32 = slot(p, 7);
        let block = Box::leak(Box::new(crate::xasync::XAsyncBlock { queue: std::ptr::null_mut(), context: std::ptr::null_mut(), callback: done as *const () as usize, internal: [0; 32] }));
        let address = block as *mut _ as usize;
        // SAFETY: the fake slot takes these arguments.
        assert_eq!(unsafe { f(p, 0, block) }, S_OK);
        let later = Instant::now() + std::time::Duration::from_secs(6);
        let pending = |b: usize| if b == address { crate::xasync::E_PENDING } else { crate::taskqueue::E_INVALIDARG };
        crate::runtime_proxy_async::sweep(later, &pending);
        let lines = tests_capture::lines_with(&format!("async block 0x{address:x}"));
        assert!(lines.iter().any(|l| l.starts_with("windows fake windows table E slot 7 ?: async block") && l.contains("not completed after 5 s")), "{lines:?}");
        crate::runtime_proxy_async::sweep(later, &|_| S_OK);
        let lines = tests_capture::lines_with(&format!("async block 0x{address:x}"));
        assert!(lines.iter().any(|l| l.contains("completed late") && l.contains("-> 0x00000000")), "{lines:?}");
    }
}
