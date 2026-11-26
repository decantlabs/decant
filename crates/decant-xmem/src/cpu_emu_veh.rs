//! In-process vectored exception handler for AMD-only instructions (`CLZERO`, `MONITORX`,
//! `MWAITX`, `RDPRU`, `EXTRQ`, `INSERTQ`) when running without an out-of-process debugger or
//! when `DECANT_INPROC_CPU_EMU=1`. Also installs an unhandled exception logger when no debugger
//! is attached.
#![allow(unsafe_code)]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, Once, PoisonError};

use windows_sys::Win32::Foundation::NTSTATUS;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS, FlushInstructionCache, IsDebuggerPresent, SetUnhandledExceptionFilter,
};
use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

unsafe extern "system" {
    fn QueryPerformanceCounter(lpPerformanceCount: *mut i64) -> i32;
}

const EXCEPTION_ILLEGAL_INSTRUCTION: NTSTATUS = 0xC000_001Du32 as i32;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

const CF: u32 = 0x1;
const PF: u32 = 0x4;
const AF: u32 = 0x10;
const ZF: u32 = 0x40;
const SF: u32 = 0x80;
const OF: u32 = 0x800;
const ARITH: u32 = CF | PF | AF | ZF | SF | OF;
const MAX_LOOP_BYTES: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Immediate { length: u8, index: u8 },
    Register(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insn {
    Clzero { addr32: bool },
    Monitorx,
    Mwaitx,
    Rdpru,
    Extrq { dst: u8, field: Field },
    Insertq { dst: u8, src: u8, field: Field },
}

struct Bytes<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn next(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.at)?;
        self.at += 1;
        Some(byte)
    }

    fn next_if(&mut self, wanted: impl Fn(u8) -> bool) -> Option<u8> {
        let byte = *self.data.get(self.at)?;
        wanted(byte).then(|| {
            self.at += 1;
            byte
        })
    }

    fn expect(&mut self, byte: u8) -> Option<()> {
        self.next_if(|b| b == byte).map(|_| ())
    }
}

pub fn field_mask(length: u8) -> u64 {
    match length % 64 {
        0 => !0,
        n => !0u64 >> (64 - u32::from(n)),
    }
}

pub fn extrq(value: u64, length: u8, index: u8) -> u64 {
    (value >> (index % 64)) & field_mask(length)
}

pub fn insertq(value: u64, source: u64, length: u8, index: u8) -> u64 {
    let shift = index % 64;
    let mask = field_mask(length) << shift;
    value ^ ((value ^ (source << shift)) & mask)
}

pub fn decode(bytes: &[u8]) -> Option<(Insn, usize)> {
    if let Some(found) = decode_sse4a(bytes) {
        return Some(found);
    }
    let mut cursor = Bytes::new(bytes);
    let addr32 = cursor.next_if(|b| b == 0x67).is_some();
    cursor.expect(0x0F)?;
    cursor.expect(0x01)?;
    let insn = match cursor.next()? {
        0xFC => Insn::Clzero { addr32 },
        0xFA => Insn::Monitorx,
        0xFB => Insn::Mwaitx,
        0xFD => Insn::Rdpru,
        _ => return None,
    };
    Some((insn, cursor.at))
}

fn decode_sse4a(bytes: &[u8]) -> Option<(Insn, usize)> {
    let mut cursor = Bytes::new(bytes);
    let prefix = cursor.next_if(|b| b == 0x66 || b == 0xF2)?;
    let rex = cursor.next_if(|b| b & 0xF0 == 0x40).unwrap_or(0);
    cursor.expect(0x0F)?;
    let immediate = match cursor.next()? {
        0x78 => true,
        0x79 => false,
        _ => return None,
    };
    let modrm = cursor.next()?;
    if modrm >> 6 != 0b11 {
        return None;
    }
    let reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    let rm = (modrm & 7) | ((rex & 1) << 3);
    let field = if immediate {
        let length = cursor.next()?;
        let index = cursor.next()?;
        Field::Immediate { length, index }
    } else {
        Field::Register(rm)
    };
    let insn = match (prefix, immediate) {
        (0x66, true) if (modrm >> 3) & 7 == 0 => Insn::Extrq { dst: rm, field },
        (0x66, true) => return None,
        (0x66, false) => Insn::Extrq { dst: reg, field },
        _ => Insn::Insertq { dst: reg, src: rm, field },
    };
    Some((insn, cursor.at))
}

pub fn cmp_rax_with(bytes: &[u8]) -> Option<u8> {
    let &[rex, opcode, modrm, ..] = bytes else { return None };
    let register_direct = modrm >> 6 == 0b11;
    let reg_field = (modrm >> 3) & 7;
    let rm_field = modrm & 7;
    let other = match opcode {
        0x39 if rex & 0xFB == 0x48 && register_direct && rm_field == 0 => reg_field | ((rex & 4) << 1),
        0x3B if rex & 0xFE == 0x48 && register_direct && reg_field == 0 => rm_field | ((rex & 1) << 3),
        _ => return None,
    };
    (other != 0 && other != 4).then_some(other)
}

pub fn match_loop(bytes: &[u8]) -> Option<(usize, u8, [u8; 3])> {
    const CLZERO: [u8; 3] = [0x0F, 0x01, 0xFC];
    const ADDS: [&[u8]; 3] = [&[0x48, 0x05, 0x40, 0, 0, 0], &[0x48, 0x83, 0xC0, 0x40], &[0x48, 0x8D, 0x40, 0x40]];
    if !bytes.starts_with(&CLZERO) {
        return None;
    }
    let mut at = CLZERO.len();
    at += ADDS.iter().find(|add| bytes[at..].starts_with(add))?.len();
    let limit = cmp_rax_with(&bytes[at..])?;
    let cmp_bytes = <[u8; 3]>::try_from(bytes.get(at..at + 3)?).ok()?;
    at += 3;
    let (size, offset) = match bytes[at..] {
        [0x72, rel, ..] => (2, i64::from(rel as i8)),
        [0x0F, 0x82, a, b, c, d, ..] => (6, i64::from(i32::from_le_bytes([a, b, c, d]))),
        _ => return None,
    };
    let end = at + size;
    (end as i64 + offset == 0).then_some((end, limit, cmp_bytes))
}

pub fn sub_flags(a: u64, b: u64) -> u32 {
    let (diff, borrow) = a.overflowing_sub(b);
    let (_, signed_overflow) = (a as i64).overflowing_sub(b as i64);
    let conditions = [
        (CF, borrow),
        (PF, (diff as u8).count_ones().is_multiple_of(2)),
        (AF, (a ^ b ^ diff) & 0x10 != 0),
        (ZF, diff == 0),
        (SF, (diff as i64) < 0),
        (OF, signed_overflow),
    ];
    conditions.iter().filter(|(_, set)| *set).fold(0, |flags, (bit, _)| flags | bit)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopEffect {
    pub start: u64,
    pub lines: u64,
    pub rax: u64,
    pub flags: u32,
}

pub fn loop_effect(rax: u64, limit: u64) -> Option<LoopEffect> {
    let lines = limit.saturating_sub(rax).div_ceil(64).max(1);
    let bytes = lines.checked_mul(64).filter(|&n| n <= MAX_LOOP_BYTES)?;
    let end = rax.checked_add(bytes)?;
    Some(LoopEffect { start: rax & !63, lines, rax: end, flags: sub_flags(end, limit) })
}

pub fn build_clzero_loop_stub(stub_addr: usize, return_addr: usize, cmp_bytes: [u8; 3], limit_reg: u8) -> Option<Vec<u8>> {
    let scratch: u8 = if limit_reg == 1 { 2 } else { 1 };
    let mut code = Vec::with_capacity(86);
    code.push(0x50 | scratch);
    let loop_top = code.len();
    code.extend([0x48, 0x89, 0xC0 | scratch]);
    code.extend([0x48, 0x83, 0xE0 | scratch, 0xC0]);
    code.extend([0x48, 0xC7, scratch, 0, 0, 0, 0]);
    for off in [8u8, 16, 24, 32, 40, 48, 56] {
        code.extend([0x48, 0xC7, 0x40 | scratch, off, 0, 0, 0, 0]);
    }
    code.extend([0x48, 0x83, 0xC0, 0x40]);
    code.extend(cmp_bytes);
    let after_jb = code.len() + 2;
    let rel8 = i8::try_from(loop_top as isize - after_jb as isize).ok()?;
    code.extend([0x72, rel8 as u8]);
    code.push(0x58 | scratch);
    let after_jmp = stub_addr.checked_add(code.len() + 5)?;
    let rel32 = i32::try_from(return_addr as i128 - after_jmp as i128).ok()?;
    code.push(0xE9);
    code.extend(rel32.to_le_bytes());
    Some(code)
}

/// Returns whether the in-process CPU emulation VEH should be installed.
pub fn should_install_inproc_emu(inproc_env: Option<&str>, emu_env: Option<&str>, debugger_present: bool) -> bool {
    let emu_off = emu_env.map(|v| v.trim().to_ascii_lowercase()).is_some_and(|v| ["0", "off", "false", "no"].contains(&v.as_str()));
    if emu_off {
        return false;
    }
    let forced = inproc_env.map(|v| v.trim().to_ascii_lowercase()).is_some_and(|v| ["1", "on", "true", "yes"].contains(&v.as_str()));
    forced || !debugger_present
}

fn rdpru_next(ecx: u32, raw: u64) -> u64 {
    static LAST: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let Some(slot) = LAST.get(ecx as usize) else { return 0 };
    let mut prev = slot.load(Ordering::Relaxed);
    loop {
        let next = raw.max(prev.saturating_add(1));
        match slot.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(actual) => prev = actual,
        }
    }
}

#[derive(Default)]
struct PatchState {
    slabs: Vec<(usize, usize, usize)>,
    patched: HashSet<usize>,
}

static PATCH_STATE: Mutex<Option<PatchState>> = Mutex::new(None);
static INSTALLED: Once = Once::new();

fn patch_code(address: usize, bytes: &[u8]) -> bool {
    let mut old = 0u32;
    // SAFETY: changes protection on the target code page(s) in this process to write `bytes`.
    let protected = unsafe { VirtualProtect(address as *const c_void, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) } != 0;
    if !protected {
        return false;
    }
    // SAFETY: `address` is now writable for `bytes.len()` bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
    let mut dummy = 0u32;
    // SAFETY: restores the original page protection and flushes the instruction cache.
    unsafe {
        VirtualProtect(address as *const c_void, bytes.len(), old, &mut dummy);
        FlushInstructionCache(GetCurrentProcess(), address as *const c_void, bytes.len());
    }
    true
}

fn alloc_stub(state: &mut PatchState, near: usize, size: usize) -> Option<usize> {
    const MAX_REL: usize = 0x7FFF_0000;
    for &mut (base, cap, ref mut used) in &mut state.slabs {
        if cap - *used >= size {
            let addr = base + *used;
            if addr.abs_diff(near) < MAX_REL {
                *used += size;
                return Some(addr);
            }
        }
    }
    const SLAB_SIZE: usize = 0x10000;
    let align = near & !0xFFFF;
    let step = 0x100_0000usize;
    let max_off = 0x7000_0000usize;
    let mut off = step;
    while off <= max_off {
        for candidate in [align.checked_add(off), align.checked_sub(off)].into_iter().flatten() {
            if candidate < 0x1_0000 {
                continue;
            }
            // SAFETY: reserves and commits an executable slab near `near` in this process.
            let base = unsafe { VirtualAlloc(candidate as *const c_void, SLAB_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) } as usize;
            if base != 0 && base.abs_diff(near) < MAX_REL {
                state.slabs.push((base, SLAB_SIZE, size));
                return Some(base);
            }
        }
        off += step;
    }
    None
}

fn try_patch_clzero_loop(rip: usize, total: usize, cmp_bytes: [u8; 3], limit_reg: u8) {
    if total < 5 {
        return;
    }
    let mut guard = PATCH_STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let state = guard.get_or_insert_with(PatchState::default);
    if !state.patched.insert(rip) {
        return;
    }
    let return_addr = rip + total;
    let Some(stub_addr) = alloc_stub(state, rip, 96) else { return };
    let Some(stub) = build_clzero_loop_stub(stub_addr, return_addr, cmp_bytes, limit_reg) else {
        return;
    };
    // SAFETY: `stub_addr` was just allocated as PAGE_EXECUTE_READWRITE with at least 96 bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(stub.as_ptr(), stub_addr as *mut u8, stub.len());
        FlushInstructionCache(GetCurrentProcess(), stub_addr as *const c_void, stub.len());
    }
    let Ok(rel32) = i32::try_from(stub_addr as i128 - (rip + 5) as i128) else {
        return;
    };
    let mut patch = vec![0x90u8; total];
    patch[0] = 0xE9;
    patch[1..5].copy_from_slice(&rel32.to_le_bytes());
    let _ = patch_code(rip, &patch);
}

fn try_patch_nop3(rip: usize) {
    let mut guard = PATCH_STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let state = guard.get_or_insert_with(PatchState::default);
    if state.patched.insert(rip) {
        let _ = patch_code(rip, &[0x0F, 0x1F, 0x00]);
    }
}

fn register(c: &CONTEXT, reg: u8) -> u64 {
    [c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13, c.R14, c.R15][usize::from(reg & 15)]
}

unsafe extern "system" fn illegal_insn_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows passes valid `EXCEPTION_POINTERS`, `ExceptionRecord`, and `ContextRecord`.
    let record = unsafe { &*(*info).ExceptionRecord };
    if record.ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: `ContextRecord` is valid for the duration of the vectored handler.
    let context = unsafe { &mut *(*info).ContextRecord };
    let rip = context.Rip as usize;
    let room = (0x1000 - (rip & 0xFFF)).clamp(1, 15);
    let mut bytes = [0u8; 15];
    // SAFETY: `rip` is the faulting instruction pointer inside this process's executable memory.
    unsafe { std::ptr::copy_nonoverlapping(rip as *const u8, bytes.as_mut_ptr(), room) };
    let Some((insn, len)) = decode(&bytes[..room]) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    match insn {
        Insn::Clzero { addr32 } => {
            if let Some((total, reg, cmp_bytes)) = match_loop(&bytes[..room]).filter(|_| !addr32)
                && let Some(e) = loop_effect(context.Rax, register(context, reg))
            {
                // SAFETY: zeroes `e.lines * 64` bytes at `e.start` in the current process.
                unsafe { std::ptr::write_bytes(e.start as usize as *mut u8, 0, (e.lines * 64) as usize) };
                context.Rax = e.rax;
                context.EFlags = (context.EFlags & !ARITH) | e.flags;
                context.Rip += total as u64;
                try_patch_clzero_loop(rip, total, cmp_bytes, reg);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            let address = (if addr32 { context.Rax & 0xFFFF_FFFF } else { context.Rax }) & !63;
            // SAFETY: zeroes the 64-byte cache line addressed by `CLZERO`.
            unsafe { std::ptr::write_bytes(address as usize as *mut u8, 0, 64) };
            context.Rip += len as u64;
        }
        Insn::Monitorx | Insn::Mwaitx => {
            context.Rip += len as u64;
            if len == 3 {
                try_patch_nop3(rip);
            }
        }
        Insn::Rdpru => {
            let mut counter = 0i64;
            // SAFETY: fills `counter` from `QueryPerformanceCounter`.
            unsafe { QueryPerformanceCounter(&mut counter) };
            let value = rdpru_next(context.Rcx as u32, counter as u64);
            context.Rax = value & 0xFFFF_FFFF;
            context.Rdx = value >> 32;
            context.Rip += len as u64;
        }
        Insn::Extrq { dst, field } => {
            // SAFETY: `ContextRecord` includes floating-point XMM registers on x86_64 Windows.
            let xmm = unsafe { &mut context.Anonymous.FltSave.XmmRegisters };
            let (length, index) = match field {
                Field::Immediate { length, index } => (length, index),
                Field::Register(r) => (xmm[usize::from(r)].Low as u8, (xmm[usize::from(r)].Low >> 8) as u8),
            };
            xmm[usize::from(dst)].Low = extrq(xmm[usize::from(dst)].Low, length, index);
            context.Rip += len as u64;
        }
        Insn::Insertq { dst, src, field } => {
            // SAFETY: as above.
            let xmm = unsafe { &mut context.Anonymous.FltSave.XmmRegisters };
            let (length, index) = match field {
                Field::Immediate { length, index } => (length, index),
                Field::Register(r) => (xmm[usize::from(r)].High as u8, (xmm[usize::from(r)].High >> 8) as u8),
            };
            xmm[usize::from(dst)].Low = insertq(xmm[usize::from(dst)].Low, xmm[usize::from(src)].Low, length, index);
            context.Rip += len as u64;
        }
    }
    EXCEPTION_CONTINUE_EXECUTION
}

unsafe extern "system" fn unhandled_exception_filter(info: *const EXCEPTION_POINTERS) -> i32 {
    if !info.is_null() {
        // SAFETY: Windows passes a valid `EXCEPTION_POINTERS` to the unhandled filter.
        let record = unsafe { &*(*info).ExceptionRecord };
        let code = record.ExceptionCode as u32;
        let addr = record.ExceptionAddress as usize;
        decant_log::trace(&format!("[xmem] unhandled exception {code:#010x} at {addr:#014x}\r\n"));
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Installs the in-process CPU emulation VEH (when enabled) and unhandled exception logger
/// (when no debugger is attached).
pub fn install() {
    INSTALLED.call_once(|| {
        // SAFETY: `IsDebuggerPresent` has no preconditions.
        let debugger = unsafe { IsDebuggerPresent() } != 0;
        let inproc_env = std::env::var("DECANT_INPROC_CPU_EMU").ok();
        let emu_env = std::env::var("DECANT_CPU_EMULATION").ok();
        if should_install_inproc_emu(inproc_env.as_deref(), emu_env.as_deref(), debugger) {
            // SAFETY: registers a pure process-wide vectored exception handler.
            unsafe { AddVectoredExceptionHandler(1, Some(illegal_insn_handler)) };
        }
        if !debugger {
            // SAFETY: registers an unhandled exception filter that logs and continues search.
            unsafe { SetUnhandledExceptionFilter(Some(unhandled_exception_filter)) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_amd_and_sse4a_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((Insn::Clzero { addr32: false }, 3)));
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((Insn::Clzero { addr32: true }, 4)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((Insn::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((Insn::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((Insn::Rdpru, 3)));
        assert_eq!(
            decode(&[0xF2, 0x0F, 0x78, 0xD1, 0x08, 0x10]),
            Some((Insn::Insertq { dst: 2, src: 1, field: Field::Immediate { length: 8, index: 16 } }, 6))
        );
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xC6]), Some((Insn::Extrq { dst: 0, field: Field::Register(6) }, 4)));
    }

    #[test]
    fn match_loop_and_stub_build_cleanly() {
        let code = [0x0F, 0x01, 0xFC, 0x48, 0x8D, 0x40, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4];
        let (total, reg, cmp_bytes) = match_loop(&code).unwrap();
        assert_eq!((total, reg, cmp_bytes), (12, 10, [0x4C, 0x39, 0xD0]));
        let stub = build_clzero_loop_stub(0x1_0000_0000, 0x1_0000_1000, cmp_bytes, reg).unwrap();
        assert_eq!(stub.len(), 86);
        assert_eq!(&stub[78..80], &[0x72, 0xB1]);
    }

    #[test]
    fn should_install_inproc_emu_respects_env_and_debugger() {
        assert!(should_install_inproc_emu(None, None, false));
        assert!(!should_install_inproc_emu(None, None, true));
        assert!(should_install_inproc_emu(Some("1"), None, true));
        assert!(!should_install_inproc_emu(Some("1"), Some("0"), false));
    }

    #[test]
    fn rdpru_next_is_strictly_increasing_and_zero_for_other_ecx() {
        let a = rdpru_next(0, 100);
        let b = rdpru_next(0, 50);
        assert!(b > a);
        assert_eq!(rdpru_next(2, 999), 0);
    }
}
