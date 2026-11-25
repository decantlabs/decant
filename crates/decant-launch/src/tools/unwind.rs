//! Stack walking of another process with the x64 unwind tables, written for the debugging switches:
//! the launcher stops a title thread, reads its registers and walks its stack frame by frame with
//! the `.pdata` / `.xdata` of the module each frame is in, as the system unwinder would. Each frame
//! is named `module+rva`, and `module!Export+off` too when the frame's function is an export (the
//! wait functions of `ntdll.dll` and `kernelbase.dll` are, which is how a wait is recognised).
//!
//! Everything here reads memory through [`Memory`], so the unwinder is tested on this process.

use std::cell::RefCell;
use std::collections::HashMap;

/// Reads the other process's memory.
pub trait Memory {
    /// Fills as much of `buf` as can be read at `address`, returns how many bytes that was.
    fn read(&self, address: usize, buf: &mut [u8]) -> usize;

    fn u8_at(&self, a: usize) -> Option<u8> {
        let mut b = [0u8; 1];
        (self.read(a, &mut b) == 1).then_some(b[0])
    }
    fn u16_at(&self, a: usize) -> Option<u16> {
        let mut b = [0u8; 2];
        (self.read(a, &mut b) == 2).then(|| u16::from_le_bytes(b))
    }
    fn u32_at(&self, a: usize) -> Option<u32> {
        let mut b = [0u8; 4];
        (self.read(a, &mut b) == 4).then(|| u32::from_le_bytes(b))
    }
    fn u64_at(&self, a: usize) -> Option<u64> {
        let mut b = [0u8; 8];
        (self.read(a, &mut b) == 8).then(|| u64::from_le_bytes(b))
    }
}

/// The integer registers in x86 order: 0 rax, 1 rcx, 2 rdx, 3 rbx, 4 rsp, 5 rbp, 6 rsi, 7 rdi,
/// 8 to 15 r8 to r15, and the instruction pointer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Regs {
    pub gpr: [u64; 16],
    pub rip: u64,
}

const RSP: usize = 4;

/// One function of a module's `.pdata`, RVAs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Function {
    begin: u32,
    end: u32,
    unwind: u32,
}

/// A loaded module with its function table and exports, read once when first needed.
struct Module {
    base: usize,
    size: usize,
    name: String,
    functions: Option<Vec<Function>>,
    exports: Option<HashMap<u32, String>>,
}

/// The modules of the process, as the debug events announce them.
#[derive(Default)]
pub struct Modules {
    list: RefCell<Vec<Module>>,
}

impl Modules {
    /// A module mapped at `base`: its size is read from its PE header.
    pub fn add(&self, mem: &dyn Memory, base: usize, name: &str) {
        let size = mem.u32_at(base + 0x3C).and_then(|pe| mem.u32_at(base + pe as usize + 0x50)).unwrap_or(0) as usize;
        let mut list = self.list.borrow_mut();
        list.retain(|m| m.base != base);
        list.push(Module { base, size, name: name.to_owned(), functions: None, exports: None });
    }

    pub fn remove(&self, base: usize) {
        self.list.borrow_mut().retain(|m| m.base != base);
    }

    /// The module holding `address`: its base and name.
    pub fn find(&self, address: usize) -> Option<(usize, String)> {
        self.list.borrow().iter().find(|m| (m.base..m.base + m.size).contains(&address)).map(|m| (m.base, m.name.clone()))
    }

    /// The base of the module named `name` (case ignored), the first one loaded.
    pub fn base_of(&self, name: &str) -> Option<usize> {
        self.list.borrow().iter().find(|m| m.name.eq_ignore_ascii_case(name)).map(|m| m.base)
    }

    /// The first module, the executable.
    pub fn first(&self) -> Option<(usize, String)> {
        self.list.borrow().first().map(|m| (m.base, m.name.clone()))
    }

    /// `module+rva` for `address`, `0x...` outside every module.
    pub fn name(&self, address: usize) -> String {
        match self.find(address) {
            Some((base, name)) => format!("{name}+{:#x}", address - base),
            None => format!("{address:#x}"),
        }
    }

    /// `module!Export+off` when the function holding `address` starts at an exported name.
    pub fn symbol(&self, mem: &dyn Memory, address: usize) -> Option<String> {
        let mut list = self.list.borrow_mut();
        let m = list.iter_mut().find(|m| (m.base..m.base + m.size).contains(&address))?;
        let rva = (address - m.base) as u32;
        let f = function_of(m, mem, rva)?;
        if m.exports.is_none() {
            m.exports = Some(read_exports(mem, m.base));
        }
        let name = m.exports.as_ref()?.get(&f.begin)?;
        Some(format!("{}!{name}+{:#x}", m.name, rva - f.begin))
    }

    fn function(&self, mem: &dyn Memory, address: usize) -> Option<(usize, Function)> {
        let mut list = self.list.borrow_mut();
        let m = list.iter_mut().find(|m| (m.base..m.base + m.size).contains(&address))?;
        let f = function_of(m, mem, (address - m.base) as u32)?;
        Some((m.base, f))
    }

    fn known(&self, address: usize) -> bool {
        self.find(address).is_some()
    }
}

fn function_of(m: &mut Module, mem: &dyn Memory, rva: u32) -> Option<Function> {
    if m.functions.is_none() {
        m.functions = Some(read_functions(mem, m.base));
    }
    let fs = m.functions.as_ref()?;
    let i = fs.partition_point(|f| f.begin <= rva);
    let f = *fs.get(i.checked_sub(1)?)?;
    (rva < f.end).then_some(f)
}

/// The data directory `index` of the PE image at `base`: (rva, size).
fn directory(mem: &dyn Memory, base: usize, index: usize) -> Option<(u32, u32)> {
    let pe = mem.u32_at(base + 0x3C)? as usize;
    // PE32+ optional header: the data directories start at 0x70 of it, the header at pe + 0x18.
    let at = base + pe + 0x18 + 0x70 + index * 8;
    Some((mem.u32_at(at)?, mem.u32_at(at + 4)?))
}

fn read_functions(mem: &dyn Memory, base: usize) -> Vec<Function> {
    let Some((rva, size)) = directory(mem, base, 3) else { return Vec::new() };
    if rva == 0 || size < 12 {
        return Vec::new();
    }
    let mut bytes = vec![0u8; size as usize / 12 * 12];
    let got = mem.read(base + rva as usize, &mut bytes);
    let mut out: Vec<Function> = bytes[..got / 12 * 12]
        .as_chunks::<12>()
        .0
        .iter()
        .map(|c| Function { begin: u32::from_le_bytes([c[0], c[1], c[2], c[3]]), end: u32::from_le_bytes([c[4], c[5], c[6], c[7]]), unwind: u32::from_le_bytes([c[8], c[9], c[10], c[11]]) })
        .filter(|f| f.begin < f.end)
        .collect();
    out.sort_by_key(|f| f.begin);
    out
}

/// Export RVAs to names (forwarders and ordinal only exports are left out).
fn read_exports(mem: &dyn Memory, base: usize) -> HashMap<u32, String> {
    let mut out = HashMap::new();
    let Some((rva, _)) = directory(mem, base, 0) else { return out };
    if rva == 0 {
        return out;
    }
    let dir = base + rva as usize;
    let (Some(names), Some(functions), Some(name_table), Some(ordinals)) = (mem.u32_at(dir + 0x18), mem.u32_at(dir + 0x1C), mem.u32_at(dir + 0x20), mem.u32_at(dir + 0x24)) else {
        return out;
    };
    for i in 0..names.min(65536) as usize {
        let (Some(name_rva), Some(ordinal)) = (mem.u32_at(base + name_table as usize + i * 4), mem.u16_at(base + ordinals as usize + i * 2)) else { continue };
        let Some(function) = mem.u32_at(base + functions as usize + usize::from(ordinal) * 4) else { continue };
        let mut name = [0u8; 128];
        let n = mem.read(base + name_rva as usize, &mut name);
        let end = name[..n].iter().position(|&b| b == 0).unwrap_or(n);
        out.entry(function).or_insert_with(|| String::from_utf8_lossy(&name[..end]).into_owned());
    }
    out
}

/// Undoes the prolog of the function `f` of the module at `base` for a frame at `regs`. Returns
/// true when a machine frame already set rip and rsp.
fn unwind_function(mem: &dyn Memory, base: usize, f: Function, regs: &mut Regs) -> Option<bool> {
    let offset = (regs.rip as usize).checked_sub(base + f.begin as usize)? as u32;
    let mut info_rva = f.unwind;
    let mut chained = false;
    for _ in 0..32 {
        // An unwind entry with the low bit set points to another RUNTIME_FUNCTION (an alias).
        if info_rva & 1 != 0 {
            info_rva = mem.u32_at(base + (info_rva & !1) as usize + 8)?;
            continue;
        }
        let at = base + info_rva as usize;
        let head = mem.u32_at(at)?;
        let (version, flags) = (head & 7, (head >> 3) & 0x1F);
        let prolog = (head >> 8) & 0xFF;
        let count = ((head >> 16) & 0xFF) as usize;
        let (frame_reg, frame_off) = (((head >> 24) & 0xF) as usize, u64::from((head >> 28) & 0xF) * 16);
        let mut i = 0;
        while i < count {
            let code = mem.u16_at(at + 4 + i * 2)?;
            let (code_offset, op, info) = (u32::from(code & 0xFF), (code >> 8) & 0xF, usize::from(code >> 12));
            let slots = match op {
                1 if info == 0 => 2,
                1 => 3,
                4 | 8 => 2,
                5 | 9 => 3,
                6 => 2,
                7 => 3,
                _ => 1,
            };
            let applies = chained || offset >= prolog || code_offset <= offset;
            if applies && !(version >= 2 && op == 6) {
                let next16 = || mem.u16_at(at + 4 + (i + 1) * 2).map(u64::from);
                let next32 = || mem.u32_at(at + 4 + (i + 1) * 2).map(u64::from);
                match op {
                    0 => {
                        regs.gpr[info] = mem.u64_at(regs.gpr[RSP] as usize)?;
                        regs.gpr[RSP] += 8;
                    }
                    1 => regs.gpr[RSP] += if info == 0 { next16()? * 8 } else { next32()? },
                    2 => regs.gpr[RSP] += info as u64 * 8 + 8,
                    3 => regs.gpr[RSP] = regs.gpr[frame_reg].wrapping_sub(frame_off),
                    4 => regs.gpr[info] = mem.u64_at((regs.gpr[RSP] + next16()? * 8) as usize)?,
                    5 => regs.gpr[info] = mem.u64_at((regs.gpr[RSP] + next32()?) as usize)?,
                    10 => {
                        let sp = regs.gpr[RSP] as usize + if info == 1 { 8 } else { 0 };
                        regs.rip = mem.u64_at(sp)?;
                        regs.gpr[RSP] = mem.u64_at(sp + 24)?;
                        return Some(true);
                    }
                    _ => {}
                }
            }
            i += slots;
        }
        if flags & 4 == 0 {
            break;
        }
        // The chained RUNTIME_FUNCTION follows the codes, which are padded to an even count.
        let chain = at + 4 + count.next_multiple_of(2) * 2;
        info_rva = mem.u32_at(chain + 8)?;
        chained = true;
    }
    Some(false)
}

/// One step: the caller's registers, `None` at the end of the stack.
pub fn step(mem: &dyn Memory, modules: &Modules, regs: &Regs) -> Option<Regs> {
    let mut next = *regs;
    let rip = regs.rip as usize;
    if !modules.known(rip) {
        return None;
    }
    let machine = match modules.function(mem, rip) {
        // At a `ret` (the end of an epilog) the prolog is already undone.
        Some(_) if mem.u8_at(rip) == Some(0xC3) => false,
        Some((base, f)) => unwind_function(mem, base, f, &mut next)?,
        // A leaf function: no frame of its own.
        None => false,
    };
    if !machine {
        next.rip = mem.u64_at(next.gpr[RSP] as usize)?;
        next.gpr[RSP] += 8;
    }
    if next.rip == 0 || next.gpr[RSP] <= regs.gpr[RSP] && !machine {
        return None;
    }
    Some(next)
}

/// The instruction pointers of up to `max` frames starting at `regs`.
pub fn walk(mem: &dyn Memory, modules: &Modules, regs: Regs, max: usize) -> Vec<usize> {
    walk_frames(mem, modules, regs, max).iter().map(|r| r.rip as usize).collect()
}

/// The registers of up to `max` frames starting at `regs`.
pub fn walk_frames(mem: &dyn Memory, modules: &Modules, regs: Regs, max: usize) -> Vec<Regs> {
    let mut out = vec![regs];
    while out.len() < max {
        match step(mem, modules, &out[out.len() - 1]) {
            Some(next) => out.push(next),
            None => break,
        }
    }
    out
}

/// The frames as one text per frame: `module+rva`, with ` module!Export+off` when known.
pub fn describe(mem: &dyn Memory, modules: &Modules, frames: &[usize]) -> Vec<String> {
    frames
        .iter()
        .map(|&f| match modules.symbol(mem, f) {
            Some(s) => format!("{} ({s})", modules.name(f)),
            None => modules.name(f),
        })
        .collect()
}

#[cfg(windows)]
#[allow(unsafe_code)]
pub mod platform {
    use super::*;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, ReadProcessMemory};

    /// The memory of a process opened for reading.
    pub struct Process(pub HANDLE);

    impl Memory for Process {
        fn read(&self, address: usize, buf: &mut [u8]) -> usize {
            let mut got = 0usize;
            // SAFETY: reads at most the buffer's length into it.
            let ok = unsafe { ReadProcessMemory(self.0, address as *const std::ffi::c_void, buf.as_mut_ptr().cast(), buf.len(), &mut got) } != 0;
            if ok || got > 0 {
                return got;
            }
            // A read across the end of a mapping fails whole: read what is left in the first page.
            let room = 0x1000 - (address & 0xFFF);
            if room < buf.len() {
                // SAFETY: as above, a shorter read.
                unsafe { ReadProcessMemory(self.0, address as *const std::ffi::c_void, buf.as_mut_ptr().cast(), room, &mut got) };
            }
            got
        }
    }

    /// The registers of a thread context.
    pub fn regs(c: &CONTEXT) -> Regs {
        Regs { gpr: [c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11, c.R12, c.R13, c.R14, c.R15], rip: c.Rip }
    }
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, RtlCaptureContext, RtlCaptureStackBackTrace};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    #[repr(C, align(16))]
    struct Aligned(CONTEXT);

    fn module_name(base: usize) -> String {
        let mut buf = [0u16; 260];
        // SAFETY: the buffer length is passed.
        let n = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW(base as _, buf.as_mut_ptr(), 260) } as usize;
        let p = String::from_utf16_lossy(&buf[..n]);
        p.rsplit('\\').next().unwrap_or(&p).to_owned()
    }

    #[inline(never)]
    fn both() -> (Vec<usize>, Vec<usize>) {
        let mem = platform::Process(unsafe { GetCurrentProcess() });
        let modules = Modules::default();
        // SAFETY: plain calls on this process.
        let exe = unsafe { GetModuleHandleW(std::ptr::null()) } as usize;
        modules.add(&mem, exe, &module_name(exe));
        for name in ["ntdll.dll", "kernel32.dll", "kernelbase.dll"] {
            let w: Vec<u16> = name.encode_utf16().chain([0]).collect();
            // SAFETY: a NUL terminated name.
            let base = unsafe { GetModuleHandleW(w.as_ptr()) } as usize;
            modules.add(&mem, base, name);
        }
        let mut ctx: Aligned = unsafe { std::mem::zeroed() };
        let mut system = [std::ptr::null_mut(); 64];
        // SAFETY: both fill the buffers they are given.
        let n = unsafe {
            RtlCaptureContext(&mut ctx.0);
            RtlCaptureStackBackTrace(0, 64, system.as_mut_ptr(), std::ptr::null_mut())
        } as usize;
        let ours = walk(&mem, &modules, platform::regs(&ctx.0), 64);
        (ours, system[..n].iter().map(|p| *p as usize).collect())
    }

    #[test]
    fn the_walk_matches_the_system_unwinder() {
        let (ours, system) = both();
        // Both start inside this function (at two nearby calls), then the callers must be the same.
        assert!(ours.len() >= 4, "{ours:x?}");
        let tail = &system[1..];
        let ours_tail = &ours[1..];
        let common = tail.iter().zip(ours_tail).take_while(|(a, b)| a == b).count();
        assert!(common >= 3, "ours {ours_tail:x?}\nsystem {tail:x?}");
    }

    #[test]
    fn exports_name_the_function_they_start() {
        let mem = platform::Process(unsafe { GetCurrentProcess() });
        let modules = Modules::default();
        let w: Vec<u16> = "kernelbase.dll".encode_utf16().chain([0]).collect();
        // SAFETY: a NUL terminated name, then the export of a loaded module.
        let (base, f) = unsafe {
            let base = GetModuleHandleW(w.as_ptr());
            (base as usize, windows_sys::Win32::System::LibraryLoader::GetProcAddress(base, c"WaitForSingleObjectEx".as_ptr().cast()).unwrap() as usize)
        };
        modules.add(&mem, base, "KERNELBASE.dll");
        assert_eq!(modules.symbol(&mem, f + 3).as_deref(), Some("KERNELBASE.dll!WaitForSingleObjectEx+0x3"));
        assert_eq!(modules.name(f), format!("KERNELBASE.dll+{:#x}", f - base));
    }
}
