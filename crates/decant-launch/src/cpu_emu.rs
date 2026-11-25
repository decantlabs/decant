//! Software versions of instructions that only AMD CPUs have. A title built for a Zen 2 console uses
//! CLZERO, MONITORX, MWAITX, RDPRU and the SSE4a bit field moves EXTRQ and INSERTQ. On an Intel CPU
//! each of them raises `0xc000001d` (illegal instruction). The debugger reads the bytes at the faulting
//! address, performs the effect itself and moves the thread past the instruction.
//!
//! Decoding and the arithmetic are pure and tested on every host. Everything that touches the debuggee
//! lives in [`platform`].
//!
//! * CLZERO (`0f 01 fc`, optional `67` prefix that makes the address EAX) zeroes the 64 byte line that
//!   holds the address in RAX. The loop `clzero; add rax,0x40; cmp rax,<reg>; jb back` is recognised
//!   and done in a single step ([`match_loop`]). A retail title clears a block of tens of KiB with five
//!   such loops in a row, each against a different limit register.
//! * MONITORX (`0f 01 fa`) and MWAITX (`0f 01 fb`) do nothing, which the architecture allows for MWAITX.
//! * RDPRU (`0f 01 fd`): ECX 0 reads MPERF and ECX 1 reads APERF. Both are answered from
//!   `QueryPerformanceCounter` as a strictly increasing 64 bit value in EDX:EAX. Any other ECX gives 0.
//!   RDPRU leaves every flag alone (AMD APM vol. 3).
//! * EXTRQ (`66 0f 78 /0 ib ib`, `66 0f 79 /r`) and INSERTQ (`f2 0f 78 /r ib ib`, `f2 0f 79 /r`) are
//!   SSE4a (AMD APM vol. 4). They cut a field of `length` bits (0 means 64) at bit `index` out of the
//!   low quadword of an XMM register, or write one in from the low bits of another register. Only
//!   register operands are handled. The high quadword of the destination is undefined, so it stays.
//!   A retail title runs `insertq xmm1,xmm1,8,16` once it leaves its first screen.

use std::collections::HashSet;
use std::time::Duration;

/// Environment variable: `0` (or `off`, `false`, `no`) turns the emulation off.
pub const ENV: &str = "DECANT_CPU_EMULATION";

/// The exception code of an illegal instruction.
pub const ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;

/// Whether the emulation is on for this value of [`ENV`]. An unset variable means on.
pub fn enabled(value: Option<&str>) -> bool {
    let off = value.map(|v| v.trim().to_ascii_lowercase()).is_some_and(|v| ["0", "off", "false", "no"].contains(&v.as_str()));
    !off
}

/// An instruction that is emulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insn {
    /// `addr32` is true when a `67` prefix makes the line address EAX rather than RAX.
    Clzero { addr32: bool },
    Monitorx,
    Mwaitx,
    Rdpru,
    /// The field of `dst` moves down to bit 0. `field` says where its length and index come from.
    Extrq { dst: u8, field: Field },
    /// The low bits of `src` go into the field of `dst`.
    Insertq { dst: u8, src: u8, field: Field },
}

/// The source of the length and the index of a bit field. They are either bytes of the instruction, or
/// bits 0 to 5 (length) and 8 to 13 (index) of one quadword of an XMM register: the low one for EXTRQ,
/// the high one for INSERTQ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Immediate { length: u8, index: u8 },
    Register(u8),
}

/// Names in the order of [`Insn::slot`].
const NAMES: [&str; 6] = ["clzero", "monitorx", "mwaitx", "rdpru", "extrq", "insertq"];

impl Insn {
    /// The lower case mnemonic.
    pub fn name(self) -> &'static str {
        NAMES[self.slot()]
    }

    /// The index of this kind of instruction in the counters of [`Stats`].
    fn slot(self) -> usize {
        match self {
            Insn::Clzero { .. } => 0,
            Insn::Monitorx => 1,
            Insn::Mwaitx => 2,
            Insn::Rdpru => 3,
            Insn::Extrq { .. } => 4,
            Insn::Insertq { .. } => 5,
        }
    }
}

/// A mask of the low `length` bits, where only the low six bits of `length` count and 0 stands for 64.
pub fn field_mask(length: u8) -> u64 {
    match length % 64 {
        0 => !0,
        n => !0u64 >> (64 - u32::from(n)),
    }
}

/// EXTRQ: the `length` bits of `value` that start at bit `index`, moved to bit 0 with zeros above.
pub fn extrq(value: u64, length: u8, index: u8) -> u64 {
    (value >> (index % 64)) & field_mask(length)
}

/// INSERTQ: `value` with the `length` bits at bit `index` replaced by the low bits of `source`.
pub fn insertq(value: u64, source: u64, length: u8, index: u8) -> u64 {
    let shift = index % 64;
    let mask = field_mask(length) << shift;
    value ^ ((value ^ (source << shift)) & mask)
}

/// A cursor over instruction bytes.
struct Bytes<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bytes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bytes { data, at: 0 }
    }

    fn next(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.at)?;
        self.at += 1;
        Some(byte)
    }

    /// Takes the next byte only when `wanted` accepts it.
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

/// The instruction at the start of `bytes` and its length, or `None` when it is none of ours.
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

/// EXTRQ or INSERTQ at the start of `bytes` and its length. The bytes are the mandatory prefix (`66` for
/// EXTRQ, `f2` for INSERTQ), an optional REX prefix that reaches XMM8 to XMM15, `0f 78` (two immediate
/// bytes follow) or `0f 79` (register form), and a ModRM byte that names two registers.
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
        // The immediate EXTRQ names its register in r/m and has /0 in the reg field.
        (0x66, true) if (modrm >> 3) & 7 == 0 => Insn::Extrq { dst: rm, field },
        (0x66, true) => return None,
        (0x66, false) => Insn::Extrq { dst: reg, field },
        _ => Insn::Insertq { dst: reg, src: rm, field },
    };
    Some((insn, cursor.at))
}

/// The register that a `cmp rax,<reg>` at the start of `bytes` compares RAX with, as an x86 register
/// number (8 is r8). Both encodings are understood: `cmp r/m64,r64` (opcode 39, RAX in r/m) and
/// `cmp r64,r/m64` (opcode 3b, RAX in reg). RAX itself and RSP are refused.
pub fn cmp_rax_with(bytes: &[u8]) -> Option<u8> {
    let &[rex, opcode, modrm, ..] = bytes else { return None };
    let register_direct = modrm >> 6 == 0b11;
    let reg_field = (modrm >> 3) & 7;
    let rm_field = modrm & 7;
    let other = match opcode {
        // REX.W set, no REX.X and no REX.B, REX.R extends the register in reg.
        0x39 if rex & 0xFB == 0x48 && register_direct && rm_field == 0 => reg_field | ((rex & 4) << 1),
        // REX.W set, no REX.R and no REX.X, REX.B extends the register in r/m.
        0x3B if rex & 0xFE == 0x48 && register_direct && reg_field == 0 => rm_field | ((rex & 1) << 3),
        _ => return None,
    };
    (other != 0 && other != 4).then_some(other)
}

/// The length of the loop `clzero; add rax,0x40; cmp rax,<reg>; jb <clzero>` at the start of `bytes`
/// and its limit register, with any encoding of the add, the cmp and the jb. `None` for everything
/// else. The clzero carries no prefix, so it works on RAX, and the jump lands exactly on it.
pub fn match_loop(bytes: &[u8]) -> Option<(usize, u8)> {
    const CLZERO: [u8; 3] = [0x0F, 0x01, 0xFC];
    const ADDS: [&[u8]; 2] = [&[0x48, 0x05, 0x40, 0, 0, 0], &[0x48, 0x83, 0xC0, 0x40]];
    if !bytes.starts_with(&CLZERO) {
        return None;
    }
    let mut at = CLZERO.len();
    at += ADDS.iter().find(|add| bytes[at..].starts_with(add))?.len();
    let limit = cmp_rax_with(&bytes[at..])?;
    at += 3;
    // `jb` is the jump if below, which is the jump if carry. A short form has a signed byte, the near
    // form a signed 32 bit offset, both counted from the end of the jump.
    let (size, offset) = match bytes[at..] {
        [0x72, rel, ..] => (2, i64::from(rel as i8)),
        [0x0F, 0x82, a, b, c, d, ..] => (6, i64::from(i32::from_le_bytes([a, b, c, d]))),
        _ => return None,
    };
    let end = at + size;
    (end as i64 + offset == 0).then_some((end, limit))
}

/// The most that the loop emulation zeroes in one step. A larger request goes line by line.
pub const MAX_LOOP_BYTES: u64 = 1 << 30;

/// The outcome of a whole loop for the register values on entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopEffect {
    /// First byte zeroed, which is `rax` rounded down to a line.
    pub start: u64,
    /// The number of 64 byte lines zeroed.
    pub lines: u64,
    /// RAX when the loop falls through.
    pub rax: u64,
    /// The arithmetic flags that the last `cmp` left.
    pub flags: u32,
}

pub const CF: u32 = 0x1;
pub const PF: u32 = 0x4;
pub const AF: u32 = 0x10;
pub const ZF: u32 = 0x40;
pub const SF: u32 = 0x80;
pub const OF: u32 = 0x800;
/// The six arithmetic flags that a `cmp` writes.
pub const ARITH: u32 = CF | PF | AF | ZF | SF | OF;

/// The flags that `cmp a,b` leaves, which are the flags of `a - b`.
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

/// The effect of the loop as a do while: one line is zeroed before the first test, and it repeats while
/// `rax` stays below `limit` (unsigned) after adding 64. `None` when the arithmetic would wrap or the
/// block is larger than [`MAX_LOOP_BYTES`].
pub fn loop_effect(rax: u64, limit: u64) -> Option<LoopEffect> {
    let lines = limit.saturating_sub(rax).div_ceil(64).max(1);
    let bytes = lines.checked_mul(64).filter(|&n| n <= MAX_LOOP_BYTES)?;
    let end = rax.checked_add(bytes)?;
    Some(LoopEffect { start: rax & !63, lines, rax: end, flags: sub_flags(end, limit) })
}

/// The counters of one run and the set of instructions seen so far.
#[derive(Default)]
pub struct Stats {
    /// Emulated instructions per kind, indexed like [`NAMES`]. The loop fast path adds one per line.
    pub counts: [u64; 6],
    pub spent: Duration,
    seen: HashSet<(usize, usize)>,
    last_rdpru: [u64; 2],
}

impl Stats {
    /// True the first time this kind of instruction is seen at `address`.
    pub fn first(&mut self, insn: Insn, address: usize) -> bool {
        self.seen.insert((insn.slot(), address))
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    pub fn add(&mut self, insn: Insn, n: u64) {
        self.counts[insn.slot()] += n;
    }

    /// A strictly increasing value for MPERF (ECX 0) or APERF (ECX 1) based on the counter `raw`. Any
    /// other `ecx` reads as 0.
    pub fn rdpru_value(&mut self, ecx: u32, raw: u64) -> u64 {
        match self.last_rdpru.get_mut(ecx as usize) {
            Some(last) => {
                *last = raw.max(last.saturating_add(1));
                *last
            }
            None => 0,
        }
    }

    pub fn summary(&self) -> String {
        let counted: Vec<String> = NAMES.iter().zip(self.counts).map(|(name, n)| format!("{name} {n}")).collect();
        format!("emulated {}, time {} ms", counted.join(", "), self.spent.as_millis())
    }
}

/// Whether this CPU has CLZERO itself (CPUID 0x80000008, EBX bit 0).
#[cfg(target_arch = "x86_64")]
pub fn cpu_has_clzero() -> bool {
    use std::arch::x86_64::__cpuid;
    // Every x86_64 CPU has CPUID. Leaf 0x80000008 may be past the highest extended leaf.
    let highest_extended = __cpuid(0x8000_0000).eax;
    highest_extended >= 0x8000_0008 && __cpuid(0x8000_0008).ebx & 1 == 1
}

#[cfg(not(target_arch = "x86_64"))]
pub fn cpu_has_clzero() -> bool {
    false
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub mod platform {
    use super::*;
    use std::ffi::c_void;
    use std::time::Instant;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_FLOATING_POINT_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, ReadProcessMemory, SetThreadContext, WriteProcessMemory};
    use windows_sys::Win32::System::Performance::QueryPerformanceCounter;
    use windows_sys::Win32::System::Threading::{OpenThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT};

    /// `CONTEXT` must sit on a 16 byte boundary, which windows-sys does not declare.
    #[repr(C, align(16))]
    struct Aligned(CONTEXT);

    /// What one emulation did.
    pub struct Emulated {
        pub insn: Insn,
        pub address: usize,
        /// True the first time this instruction kind was seen at this address.
        pub first: bool,
    }

    fn read(process: HANDLE, address: usize, buffer: &mut [u8]) -> usize {
        let mut read = 0usize;
        // SAFETY: reads at most the buffer's length into it.
        unsafe { ReadProcessMemory(process, address as *const c_void, buffer.as_mut_ptr().cast(), buffer.len(), &mut read) };
        read
    }

    fn zero(process: HANDLE, start: u64, len: u64) -> bool {
        static ZEROS: [u8; 1 << 16] = [0; 1 << 16];
        let (mut at, mut left) = (start, len);
        while left > 0 {
            let n = left.min(ZEROS.len() as u64) as usize;
            let mut written = 0usize;
            // SAFETY: writes `n` bytes from a live static slice.
            let ok = unsafe { WriteProcessMemory(process, at as usize as *const c_void, ZEROS.as_ptr().cast(), n, &mut written) } != 0 && written == n;
            if !ok {
                return false;
            }
            at += n as u64;
            left -= n as u64;
        }
        true
    }

    fn counter() -> u64 {
        let mut value = 0i64;
        // SAFETY: fills one integer.
        unsafe { QueryPerformanceCounter(&mut value) };
        value as u64
    }

    /// Emulates the instruction `thread` faulted on, when it is one of ours. `None` leaves the
    /// exception to the normal path (not ours, or the debuggee memory could not be read or written).
    pub fn emulate(process: HANDLE, thread: u32, stats: &mut Stats) -> Option<Emulated> {
        let began = Instant::now();
        // SAFETY: a plain handle request.
        let handle = unsafe { OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, 0, thread) };
        if handle.is_null() {
            return None;
        }
        let done = emulate_on(process, handle, stats);
        // SAFETY: the handle is ours.
        unsafe { CloseHandle(handle) };
        stats.spent += began.elapsed();
        done
    }

    /// The region of the debuggee's memory at `address` as one log fragment (state, protection,
    /// type, allocation base and the region's base and size).
    // `process` is an opaque handle the OS checks, never dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn describe_memory(process: HANDLE, address: usize) -> String {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQueryEx};
        // SAFETY: a plain structure the call fills.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: queries the debuggee's address space into `m`.
        let n = unsafe { VirtualQueryEx(process, address as *const c_void, &mut m, size_of::<MEMORY_BASIC_INFORMATION>()) };
        if n == 0 {
            return format!("no answer ({})", std::io::Error::last_os_error());
        }
        format!(
            "state {:#x} protect {:#x} type {:#x} allocation base {:#x} region {:#x} + {:#x}",
            m.State, m.Protect, m.Type, m.AllocationBase as usize, m.BaseAddress as usize, m.RegionSize
        )
    }

    /// The first `n` stack words of `thread` (from its rsp), for finding return addresses.
    pub fn stack_words(process: HANDLE, thread: u32, n: usize) -> Vec<usize> {
        // SAFETY: a plain handle request.
        let handle = unsafe { OpenThread(THREAD_GET_CONTEXT, 0, thread) };
        if handle.is_null() {
            return Vec::new();
        }
        // SAFETY: a plain structure the call fills; the flags name the parts wanted.
        let mut aligned: Aligned = unsafe { std::mem::zeroed() };
        let c = &mut aligned.0;
        c.ContextFlags = CONTEXT_CONTROL_AMD64;
        // SAFETY: `handle` is open with the rights the call needs.
        let ok = unsafe { GetThreadContext(handle, c) } != 0;
        // SAFETY: the handle is ours.
        unsafe { CloseHandle(handle) };
        if !ok {
            return Vec::new();
        }
        let mut bytes = vec![0u8; n * 8];
        let got = read(process, c.Rsp as usize, &mut bytes);
        bytes[..got].as_chunks::<8>().0.iter().map(|w| u64::from_le_bytes(*w) as usize).collect()
    }

    /// The integer registers of `thread` as one log line, for the report of an access violation.
    pub fn registers(thread: u32) -> Option<String> {
        // SAFETY: a plain handle request.
        let handle = unsafe { OpenThread(THREAD_GET_CONTEXT, 0, thread) };
        if handle.is_null() {
            return None;
        }
        // SAFETY: a plain structure the call fills; the flags name the parts wanted.
        let mut aligned: Aligned = unsafe { std::mem::zeroed() };
        let c = &mut aligned.0;
        c.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
        // SAFETY: `handle` is open with the rights the call needs.
        let ok = unsafe { GetThreadContext(handle, c) } != 0;
        // SAFETY: the handle is ours.
        unsafe { CloseHandle(handle) };
        ok.then(|| {
            format!(
                "rax {:#x} rbx {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x} r8 {:#x} r9 {:#x} rsp {:#x}",
                c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rsi, c.Rdi, c.R8, c.R9, c.Rsp
            )
        })
    }

    /// Every integer register of `thread`, one log line each, with the first 64 bytes at the address
    /// it holds when that address can be read: what the title passed or kept when it gave up.
    // `process` is an opaque handle the OS checks, never dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn registers_pointed(process: HANDLE, thread: u32) -> Vec<String> {
        // SAFETY: a plain handle request.
        let handle = unsafe { OpenThread(THREAD_GET_CONTEXT, 0, thread) };
        if handle.is_null() {
            return Vec::new();
        }
        // SAFETY: a plain structure the call fills; the flags name the parts wanted.
        let mut aligned: Aligned = unsafe { std::mem::zeroed() };
        let c = &mut aligned.0;
        c.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
        // SAFETY: `handle` is open with the rights the call needs.
        let ok = unsafe { GetThreadContext(handle, c) } != 0;
        // SAFETY: the handle is ours.
        unsafe { CloseHandle(handle) };
        if !ok {
            return Vec::new();
        }
        const NAMES: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
        (0..16u8)
            .map(|reg| {
                let value = register(c, reg);
                let mut bytes = [0u8; 64];
                let got = if value >= 0x10000 { read(process, value as usize, &mut bytes) } else { 0 };
                let words: Vec<String> = bytes[..got].as_chunks::<8>().0.iter().map(|w| format!("{:016x}", u64::from_le_bytes(*w))).collect();
                if words.is_empty() { format!("{} {value:#x}", NAMES[usize::from(reg)]) } else { format!("{} {value:#x}: {}", NAMES[usize::from(reg)], words.join(" ")) }
            })
            .collect()
    }

    /// The value of general purpose register `reg` (x86 number: 0 rax, 1 rcx, 2 rdx, 3 rbx, 4 rsp,
    /// 5 rbp, 6 rsi, 7 rdi, 8 to 15 r8 to r15).
    fn register(c: &CONTEXT, reg: u8) -> u64 {
        [c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13, c.R14, c.R15][usize::from(reg & 15)]
    }

    fn emulate_on(process: HANDLE, handle: HANDLE, stats: &mut Stats) -> Option<Emulated> {
        // SAFETY: a plain structure the call fills; the flags name the parts wanted.
        let mut aligned: Aligned = unsafe { std::mem::zeroed() };
        let context = &mut aligned.0;
        context.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64 | CONTEXT_FLOATING_POINT_AMD64;
        // SAFETY: `handle` is open with the rights the call needs.
        if unsafe { GetThreadContext(handle, context) } == 0 {
            return None;
        }
        let rip = context.Rip as usize;
        // Up to 15 bytes, fewer when the instruction sits at the end of the mapped memory.
        let mut bytes = [0u8; 15];
        let mut n = read(process, rip, &mut bytes);
        if n == 0 {
            let room = (0x1000 - (rip & 0xFFF)).min(15);
            n = read(process, rip, &mut bytes[..room]);
        }
        let (insn, len) = decode(&bytes[..n])?;
        match insn {
            Insn::Clzero { addr32 } => {
                let looped = match_loop(&bytes[..n]).filter(|_| !addr32).and_then(|(total, reg)| loop_effect(context.Rax, register(context, reg)).map(|e| (total, e)));
                match looped {
                    Some((total, e)) if zero(process, e.start, e.lines * 64) => {
                        context.Rax = e.rax;
                        context.EFlags = (context.EFlags & !ARITH) | e.flags;
                        context.Rip += total as u64;
                        stats.add(insn, e.lines);
                    }
                    _ => {
                        let address = if addr32 { context.Rax & 0xFFFF_FFFF } else { context.Rax } & !63;
                        if !zero(process, address, 64) {
                            return None;
                        }
                        context.Rip += len as u64;
                        stats.add(insn, 1);
                    }
                }
            }
            Insn::Monitorx | Insn::Mwaitx => {
                context.Rip += len as u64;
                stats.add(insn, 1);
            }
            Insn::Rdpru => {
                let value = stats.rdpru_value(context.Rcx as u32, counter());
                context.Rax = value & 0xFFFF_FFFF;
                context.Rdx = value >> 32;
                context.Rip += len as u64;
                stats.add(insn, 1);
            }
            Insn::Extrq { dst, field } => {
                // SAFETY: the floating point part of the context was asked for, the XMM registers are in it.
                let xmm = unsafe { &mut context.Anonymous.FltSave.XmmRegisters };
                let (length, index) = match field {
                    Field::Immediate { length, index } => (length, index),
                    Field::Register(r) => (xmm[usize::from(r)].Low as u8, (xmm[usize::from(r)].Low >> 8) as u8),
                };
                xmm[usize::from(dst)].Low = extrq(xmm[usize::from(dst)].Low, length, index);
                context.Rip += len as u64;
                stats.add(insn, 1);
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
                stats.add(insn, 1);
            }
        }
        // SAFETY: `handle` is open with the rights the call needs.
        if unsafe { SetThreadContext(handle, context) } == 0 {
            return None;
        }
        Some(Emulated { insn, address: rip, first: stats.first(insn, rip) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLZERO: [u8; 3] = [0x0F, 0x01, 0xFC];
    const ADD_SHORT: [u8; 4] = [0x48, 0x83, 0xC0, 0x40];
    const ADD_LONG: [u8; 6] = [0x48, 0x05, 0x40, 0, 0, 0];
    /// `cmp rax,r10` in the two encodings.
    const CMP_R10: [[u8; 3]; 2] = [[0x4C, 0x39, 0xD0], [0x49, 0x3B, 0xC2]];

    /// A jump back to the start of the code, as a function of the number of bytes in front of it.
    type Jump = fn(usize) -> Vec<u8>;

    fn jb_short(before: usize) -> Vec<u8> {
        vec![0x72, (-((before + 2) as i64)) as u8]
    }

    fn jb_near(before: usize) -> Vec<u8> {
        let mut jump = vec![0x0F, 0x82];
        jump.extend((-((before + 6) as i32)).to_le_bytes());
        jump
    }

    fn assemble(add: &[u8], cmp: &[u8], jump: Jump) -> Vec<u8> {
        let mut code = [&CLZERO[..], add, cmp].concat();
        code.extend(jump(code.len()));
        code
    }

    #[test]
    fn the_environment_switch_defaults_to_on() {
        assert!(enabled(None));
        assert!(enabled(Some("1")));
        assert!(enabled(Some("")));
        for off in ["0", "OFF", "False", "  no  "] {
            assert!(!enabled(Some(off)), "{off}");
        }
    }

    #[test]
    fn each_plain_instruction_decodes_with_its_length() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x90]), Some((Insn::Clzero { addr32: false }, 3)));
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((Insn::Clzero { addr32: true }, 4)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((Insn::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB, 0xC3]), Some((Insn::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((Insn::Rdpru, 3)));
        assert_eq!(Insn::Rdpru.name(), "rdpru");
        assert_eq!(Insn::Insertq { dst: 0, src: 0, field: Field::Register(0) }.name(), "insertq");
    }

    #[test]
    fn other_bytes_are_not_decoded() {
        let others: [&[u8]; 8] = [&[0x0F, 0x01, 0xF9], &[0x0F, 0x0B], &[0x0F, 0x01], &[0x90, 0x0F, 0x01, 0xFC], &[0x66, 0x0F, 0x01, 0xFC], &[], &[0x67], &[0x67, 0x67, 0x0F, 0x01, 0xFD]];
        for bytes in others {
            assert_eq!(decode(bytes), None, "{bytes:02x?}");
        }
    }

    #[test]
    fn sse4a_forms_decode_with_their_registers() {
        let insertq = decode(&[0xF2, 0x0F, 0x78, 0xD1, 0x10, 0x04]);
        assert_eq!(insertq, Some((Insn::Insertq { dst: 2, src: 1, field: Field::Immediate { length: 0x10, index: 4 } }, 6)));
        // REX.R reaches xmm10 as the destination, REX.B xmm13 as the source.
        let high = decode(&[0xF2, 0x45, 0x0F, 0x79, 0xD5]);
        assert_eq!(high, Some((Insn::Insertq { dst: 10, src: 13, field: Field::Register(13) }, 5)));
        let extrq = decode(&[0x66, 0x0F, 0x78, 0xC5, 0x03, 0x09]);
        assert_eq!(extrq, Some((Insn::Extrq { dst: 5, field: Field::Immediate { length: 3, index: 9 } }, 6)));
        let extrq_register = decode(&[0x66, 0x44, 0x0F, 0x79, 0xC6]);
        assert_eq!(extrq_register, Some((Insn::Extrq { dst: 8, field: Field::Register(6) }, 5)));
    }

    #[test]
    fn sse4a_forms_that_are_not_register_to_register_are_refused() {
        let refused: [&[u8]; 7] = [
            &[0xF2, 0x0F, 0x79, 0x11],                   // a memory operand
            &[0xF2, 0x0F, 0x78, 0xD1, 0x10],             // the index byte is missing
            &[0xF2, 0x0F, 0x78, 0xD1],                   // both immediates are missing
            &[0x66, 0x0F, 0x78, 0xC8, 1, 2],             // extrq with something other than /0 in reg
            &[0xF3, 0x0F, 0x78, 0xD1, 8, 16],            // not one of the two prefixes
            &[0x66, 0x0F, 0x7A, 0xC0],                   // another opcode
            &[0x66],
        ];
        for bytes in refused {
            assert_eq!(decode(bytes), None, "{bytes:02x?}");
        }
    }

    #[test]
    fn bit_field_masks_use_six_bits_and_zero_means_all() {
        assert_eq!(field_mask(0), u64::MAX);
        assert_eq!(field_mask(64), u64::MAX);
        assert_eq!(field_mask(1), 1);
        assert_eq!(field_mask(63), u64::MAX >> 1);
        assert_eq!(field_mask(70), 0x3F);
    }

    #[test]
    fn extrq_and_insertq_move_bit_fields() {
        assert_eq!(extrq(0xFEDC_BA98_7654_3210, 12, 20), 0x765);
        assert_eq!(extrq(0xFEDC_BA98_7654_3210, 0, 0), 0xFEDC_BA98_7654_3210);
        assert_eq!(extrq(1 << 63, 1, 63), 1);
        assert_eq!(insertq(u64::MAX, 0, 8, 8), 0xFFFF_FFFF_FFFF_00FF);
        assert_eq!(insertq(0, 0xFFFF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0xAAAA, 0x5555_5555, 0, 0), 0x5555_5555);
        assert_eq!(insertq(0, 0xDEAD_BEEF, 16, 0), 0xBEEF);
    }

    #[test]
    fn the_loop_is_found_in_all_eight_encodings() {
        for add in [&ADD_SHORT[..], &ADD_LONG[..]] {
            for cmp in CMP_R10 {
                for jump in [jb_short as Jump, jb_near as Jump] {
                    let code = assemble(add, &cmp, jump);
                    let mut followed = code.clone();
                    followed.extend([0x90, 0xC3]);
                    assert_eq!(match_loop(&followed), Some((code.len(), 10)), "{code:02x?}");
                }
            }
        }
    }

    #[test]
    fn the_limit_register_is_read_from_the_cmp() {
        let cases: [([u8; 3], u8); 5] = [([0x4C, 0x39, 0xC8], 9), ([0x4C, 0x39, 0xC0 | (3 << 3)], 11), ([0x48, 0x39, 0xD0], 2), ([0x48, 0x3B, 0xC1], 1), ([0x49, 0x3B, 0xC7], 15)];
        for (cmp, reg) in cases {
            let code = assemble(&ADD_SHORT, &cmp, jb_short);
            assert_eq!(match_loop(&code), Some((code.len(), reg)), "{cmp:02x?}");
        }
    }

    #[test]
    fn the_cmp_must_have_rax_on_one_side_and_another_register_on_the_other() {
        assert_eq!(cmp_rax_with(&[0x48, 0x39, 0xE0]), None, "rsp is not taken");
        assert_eq!(cmp_rax_with(&[0x48, 0x39, 0xC0]), None, "rax with itself");
        assert_eq!(cmp_rax_with(&[0x48, 0x3B, 0xC8]), None, "rcx is on the left");
        assert_eq!(cmp_rax_with(&[0x48, 0x39, 0x00]), None, "a memory operand");
        assert_eq!(cmp_rax_with(&[0x48, 0x89, 0xD0]), None, "a mov");
        assert_eq!(cmp_rax_with(&[0x48, 0x39]), None);
    }

    #[test]
    fn near_misses_of_the_loop_do_not_match() {
        let good = assemble(&ADD_SHORT, &CMP_R10[0], jb_short);
        assert!(match_loop(&good).is_some());
        let mut wrong_step = ADD_SHORT;
        wrong_step[3] = 0x20;
        let mut jae = jb_short(good.len() - 2);
        jae[0] = 0x73;
        let mut lands_late = jb_short(good.len() - 2);
        lands_late[1] = lands_late[1].wrapping_add(1);
        let mut padded = [&CLZERO[..], &ADD_SHORT, &[0x90]].concat();
        padded.extend(CMP_R10[0]);
        padded.extend(jb_short(padded.len()));
        let mut prefixed = vec![0x67];
        prefixed.extend(&good);
        let bad: Vec<Vec<u8>> = vec![
            assemble(&wrong_step, &CMP_R10[0], jb_short),
            assemble(&ADD_SHORT, &[0x4C, 0x39, 0xD1], jb_short),
            [&CLZERO[..], &ADD_SHORT, &CMP_R10[0], &jae].concat(),
            [&CLZERO[..], &ADD_SHORT, &CMP_R10[0], &lands_late].concat(),
            padded,
            prefixed,
            good[..good.len() - 1].to_vec(),
            good[..CLZERO.len()].to_vec(),
            Vec::new(),
        ];
        for code in bad {
            assert_eq!(match_loop(&code), None, "{code:02x?}");
        }
    }

    #[test]
    fn a_loop_runs_its_body_once_before_it_tests() {
        let aligned = loop_effect(0x4000, 0x4200).unwrap();
        assert_eq!((aligned.start, aligned.lines, aligned.rax), (0x4000, 8, 0x4200));
        assert_eq!(aligned.flags & (CF | ZF), ZF);
        let ragged = loop_effect(0x4000, 0x4201).unwrap();
        assert_eq!((ragged.lines, ragged.rax), (9, 0x4240));
        assert_eq!(ragged.flags & (CF | ZF), 0);
        let unaligned = loop_effect(0x4030, 0x4070).unwrap();
        assert_eq!((unaligned.start, unaligned.lines, unaligned.rax), (0x4000, 1, 0x4070));
        let past = loop_effect(0x9000, 0x1000).unwrap();
        assert_eq!((past.lines, past.rax), (1, 0x9040));
        let equal = loop_effect(0x500, 0x500).unwrap();
        assert_eq!((equal.lines, equal.rax), (1, 0x540));
    }

    #[test]
    fn a_loop_that_is_too_big_or_would_wrap_is_left_to_single_steps() {
        assert_eq!(loop_effect(0, MAX_LOOP_BYTES + 64), None);
        assert!(loop_effect(0, MAX_LOOP_BYTES).is_some());
        assert_eq!(loop_effect(u64::MAX - 63, u64::MAX), None);
    }

    #[test]
    fn subtraction_flags_follow_cmp() {
        assert_eq!(sub_flags(9, 9) & ARITH, ZF | PF);
        assert_eq!(sub_flags(3, 4) & (CF | SF), CF | SF);
        assert_eq!(sub_flags(1 << 63, 1) & (OF | SF | CF), OF);
        assert_eq!(sub_flags(0, 1 << 63) & (OF | CF), OF | CF);
        assert_eq!(sub_flags(0x50, 1) & AF, AF);
        assert_eq!(sub_flags(0x80, 0) & ARITH, 0);
        assert_eq!(sub_flags(7, 0) & PF, 0);
    }

    #[test]
    fn rdpru_counters_only_go_up_and_unknown_registers_read_zero() {
        let mut stats = Stats::default();
        let first = stats.rdpru_value(0, 500);
        let same_counter = stats.rdpru_value(0, 500);
        let counter_went_back = stats.rdpru_value(0, 10);
        assert!(first < same_counter && same_counter < counter_went_back);
        assert_eq!(stats.rdpru_value(1, 7), 7, "APERF counts apart from MPERF");
        assert_eq!(stats.rdpru_value(2, 700), 0);
        assert_eq!(stats.rdpru_value(u32::MAX, 700), 0);
    }

    #[test]
    fn stats_count_by_kind_and_remember_what_they_saw() {
        let mut stats = Stats::default();
        assert!(stats.first(Insn::Mwaitx, 0x40));
        assert!(!stats.first(Insn::Mwaitx, 0x40));
        assert!(stats.first(Insn::Mwaitx, 0x41));
        assert!(stats.first(Insn::Monitorx, 0x40));
        stats.add(Insn::Clzero { addr32: true }, 4);
        stats.add(Insn::Clzero { addr32: false }, 2);
        stats.add(Insn::Extrq { dst: 0, field: Field::Register(1) }, 1);
        assert_eq!(stats.total(), 7);
        assert_eq!(stats.summary(), "emulated clzero 6, monitorx 0, mwaitx 0, rdpru 0, extrq 1, insertq 0, time 0 ms");
    }
}
