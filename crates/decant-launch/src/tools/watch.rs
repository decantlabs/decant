//! `DECANT_WATCH` and `DECANT_WATCH_READ`: hardware data breakpoints on title values. The debugger puts
//! the addresses into the debug registers of every thread of the title (each new thread too, at its
//! creation event), and the processor stops a thread right after it wrote (or read) one of them.
//! The debugger then logs the old and the new value, the thread and its stack, and lets it go on.
//! The title never sees these stops. At most four addresses in all, the number of debug registers.

/// Debug registers of an x64 processor.
pub const SLOTS: usize = 4;

/// Where a watched address lies, as given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    /// An offset inside the module of that name (`exe` is the title executable).
    Module(String, usize),
    /// A plain address.
    Absolute(usize),
}

/// One watch, as parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub place: Place,
    /// 1, 2, 4 or 8 bytes; `None` picks the widest the address alignment allows (at most 8).
    pub len: Option<usize>,
    /// True for DECANT_WATCH_READ (reads and writes), false for DECANT_WATCH (writes).
    pub read: bool,
}

fn number(text: &str) -> Option<usize> {
    let t = text.trim();
    match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(hex) => usize::from_str_radix(hex, 16).ok(),
        None if t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty() => t.parse().ok(),
        // Bare hex digits as an address printed by the log (`7ff6a1b2c3d4`).
        None => usize::from_str_radix(t, 16).ok(),
    }
}

/// Parses one list: `exe+0x1234:4`, `game.dll+0x10`, `0x1234` (an offset in the executable when
/// below 4 GiB, an address otherwise), each with an optional `:len`.
pub fn parse(value: &str, read: bool) -> Result<Vec<Spec>, String> {
    let mut out = Vec::new();
    for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (place, len) = match item.rsplit_once(':') {
            Some((p, l)) => (p, Some(l.trim().parse::<usize>().ok().filter(|l| matches!(l, 1 | 2 | 4 | 8)).ok_or(format!("{item}: the length must be 1, 2, 4 or 8"))?)),
            None => (item, None),
        };
        let place = match place.rsplit_once('+') {
            Some((module, offset)) => Place::Module(module.trim().to_owned(), number(offset).ok_or(format!("{item}: not a number after +"))?),
            None => {
                let n = number(place).ok_or(format!("{item}: not an address"))?;
                if n < 1 << 32 { Place::Module("exe".into(), n) } else { Place::Absolute(n) }
            }
        };
        out.push(Spec { place, len, read });
    }
    Ok(out)
}

/// The length used at `address`: the one asked for, else the widest the alignment allows. An
/// address not aligned to the length asked for is refused (the processor would watch the aligned
/// block around it).
pub fn length(address: usize, asked: Option<usize>) -> Result<usize, String> {
    match asked {
        Some(l) if !address.is_multiple_of(l) => Err(format!("{address:#x} is not aligned to {l} bytes")),
        Some(l) => Ok(l),
        None => Ok([8, 4, 2, 1].into_iter().find(|l| address.is_multiple_of(*l)).unwrap_or(1)),
    }
}

/// The DR7 bits that enable `slot` for `len` bytes, on writes or on any access.
pub fn dr7_bits(slot: usize, len: usize, read: bool) -> u64 {
    let rw: u64 = if read { 0b11 } else { 0b01 };
    let ln: u64 = match len {
        1 => 0b00,
        2 => 0b01,
        8 => 0b10,
        _ => 0b11,
    };
    (1 << (2 * slot)) | (rw << (16 + 4 * slot)) | (ln << (18 + 4 * slot))
}

/// The DR7 mask of everything that belongs to `slot`.
pub fn dr7_mask(slot: usize) -> u64 {
    (0b11 << (2 * slot)) | (0b1111 << (16 + 4 * slot))
}

#[cfg(all(windows, target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub mod platform {
    use super::*;
    use crate::tools::Threads;
    use crate::tools::unwind::{self, Memory, Modules};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_DEBUG_REGISTERS_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, SetThreadContext};

    #[repr(C, align(16))]
    struct Aligned(CONTEXT);

    /// Lines logged per watch before only the count goes on.
    const LOG_LIMIT: u64 = 2000;
    const FRAMES: usize = 16;

    struct Armed {
        spec: Spec,
        address: usize,
        len: usize,
        last: u64,
        hits: u64,
        changes: u64,
    }

    pub struct Watch {
        pending: Vec<Spec>,
        armed: Vec<Option<Armed>>,
        exe: Option<String>,
    }

    fn read_value(mem: &dyn Memory, address: usize, len: usize) -> Option<u64> {
        let mut b = [0u8; 8];
        (mem.read(address, &mut b[..len]) == len).then(|| u64::from_le_bytes(b))
    }

    fn context(handle: HANDLE, flags: u32) -> Option<Aligned> {
        // SAFETY: a plain structure the call fills, the flags name the parts wanted.
        let mut c: Aligned = unsafe { std::mem::zeroed() };
        c.0.ContextFlags = flags;
        // SAFETY: a debuggee thread handle with THREAD_GET_CONTEXT.
        (unsafe { GetThreadContext(handle, &mut c.0) } != 0).then_some(c)
    }

    impl Watch {
        /// `None` when neither switch is set. A list that does not parse is logged and dropped.
        pub fn new(write: Option<String>, read: Option<String>, log: &mut dyn FnMut(&str)) -> Option<Watch> {
            let mut specs = Vec::new();
            for (value, is_read, name) in [(write, false, "DECANT_WATCH"), (read, true, "DECANT_WATCH_READ")] {
                let Some(v) = value.filter(|v| !v.trim().is_empty()) else { continue };
                match parse(&v, is_read) {
                    Ok(s) => specs.extend(s),
                    Err(e) => log(&format!("{name}: {e}, ignored")),
                }
            }
            if specs.is_empty() {
                return None;
            }
            if specs.len() > SLOTS {
                log(&format!("DECANT_WATCH: {} addresses asked, the processor has {SLOTS} debug registers: the first {SLOTS} are watched", specs.len()));
                specs.truncate(SLOTS);
            }
            Some(Watch { pending: specs, armed: (0..SLOTS).map(|_| None).collect(), exe: None })
        }

        /// A module was mapped: the watches in it get their address, and every thread its registers.
        pub fn module(&mut self, mem: &dyn Memory, base: usize, name: &str, threads: &Threads, log: &mut dyn FnMut(&str)) {
            if self.exe.is_none() {
                self.exe = Some(name.to_owned());
            }
            let exe = self.exe.clone().unwrap_or_default();
            let mut changed = false;
            let mut keep = Vec::new();
            for spec in std::mem::take(&mut self.pending) {
                let address = match &spec.place {
                    Place::Module(m, off) if m.eq_ignore_ascii_case(name) || (m.eq_ignore_ascii_case("exe") && name == exe) => base + off,
                    Place::Absolute(a) => *a,
                    Place::Module(..) => {
                        keep.push(spec);
                        continue;
                    }
                };
                let len = match length(address, spec.len) {
                    Ok(l) => l,
                    Err(e) => {
                        log(&format!("DECANT_WATCH: {e}, not watched"));
                        continue;
                    }
                };
                let Some(slot) = self.armed.iter().position(Option::is_none) else { continue };
                let last = read_value(mem, address, len).unwrap_or(0);
                log(&format!("DECANT_WATCH: #{slot} {} at {address:#x}, {len} bytes, on {}, now {last:#x}", describe_place(&spec.place), if spec.read { "reads and writes" } else { "writes" }));
                self.armed[slot] = Some(Armed { spec, address, len, last, hits: 0, changes: 0 });
                changed = true;
            }
            self.pending = keep;
            if changed {
                for (_, t) in threads.iter() {
                    self.arm(t.handle);
                }
            }
        }

        /// Puts the armed watches into the debug registers of the thread.
        // handle is an opaque handle the OS checks, never dereferenced here.
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        pub fn arm(&self, handle: HANDLE) {
            let Some(mut c) = context(handle, CONTEXT_DEBUG_REGISTERS_AMD64) else { return };
            let ctx = &mut c.0;
            for (slot, a) in self.armed.iter().enumerate() {
                let Some(a) = a else { continue };
                match slot {
                    0 => ctx.Dr0 = a.address as u64,
                    1 => ctx.Dr1 = a.address as u64,
                    2 => ctx.Dr2 = a.address as u64,
                    _ => ctx.Dr3 = a.address as u64,
                }
                ctx.Dr7 = (ctx.Dr7 & !dr7_mask(slot)) | dr7_bits(slot, a.len, a.spec.read);
            }
            ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
            // SAFETY: a debuggee thread handle with THREAD_SET_CONTEXT, a context just read.
            unsafe { SetThreadContext(handle, ctx) };
        }

        /// A single step of thread `tid`: true when one of the watches fired (then it is logged and
        /// the thread goes on).
        pub fn single_step(&mut self, mem: &dyn Memory, modules: &Modules, threads: &Threads, tid: u32, log: &mut dyn FnMut(&str)) -> bool {
            let Some(t) = threads.get(tid) else { return false };
            let Some(mut c) = context(t.handle, CONTEXT_DEBUG_REGISTERS_AMD64 | CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64) else { return false };
            let fired: Vec<usize> = (0..SLOTS).filter(|&s| c.0.Dr6 & (1 << s) != 0 && self.armed[s].is_some()).collect();
            if fired.is_empty() {
                return false;
            }
            let mut stack: Option<String> = None;
            let who = threads.label(tid);
            for slot in fired {
                let Some(a) = self.armed[slot].as_mut() else { continue };
                a.hits += 1;
                let now = read_value(mem, a.address, a.len).unwrap_or(a.last);
                let changed = now != a.last;
                if changed {
                    a.changes += 1;
                }
                // A write that leaves the value as it was is only counted.
                let worth = changed || a.spec.read;
                if worth && a.hits <= LOG_LIMIT {
                    let frames = stack.get_or_insert_with(|| {
                        let walked = unwind::walk(mem, modules, unwind::platform::regs(&c.0), FRAMES);
                        unwind::describe(mem, modules, &walked).join(" < ")
                    });
                    let what = if changed { format!("{:#x} -> {now:#x}", a.last) } else { format!("{now:#x} (unchanged)") };
                    log(&format!("DECANT_WATCH: #{slot} {} {what} by {who}: {frames}", describe_place(&a.spec.place)));
                    if a.hits == LOG_LIMIT {
                        log(&format!("DECANT_WATCH: #{slot} logged {LOG_LIMIT} times, only counted from now on"));
                    }
                }
                a.last = now;
            }
            c.0.Dr6 = 0;
            c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
            // SAFETY: as in `arm`.
            unsafe { SetThreadContext(t.handle, &c.0) };
            true
        }

        pub fn exit(&self, log: &mut dyn FnMut(&str)) {
            for (slot, a) in self.armed.iter().enumerate() {
                if let Some(a) = a {
                    log(&format!("DECANT_WATCH: #{slot} {}: {} hits, {} changes, last value {:#x}", describe_place(&a.spec.place), a.hits, a.changes, a.last));
                }
            }
            for spec in &self.pending {
                log(&format!("DECANT_WATCH: {} never armed (its module was not loaded)", describe_place(&spec.place)));
            }
        }
    }
}

fn describe_place(p: &Place) -> String {
    match p {
        Place::Module(m, off) => format!("{m}+{off:#x}"),
        Place::Absolute(a) => format!("{a:#x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_parse_into_places_and_lengths() {
        assert_eq!(
            parse("0x1234:4, exe+0x10, d3d12_xs.dll+20:8, 0x7ff6a0001000", false).unwrap(),
            vec![
                Spec { place: Place::Module("exe".into(), 0x1234), len: Some(4), read: false },
                Spec { place: Place::Module("exe".into(), 0x10), len: None, read: false },
                Spec { place: Place::Module("d3d12_xs.dll".into(), 20), len: Some(8), read: false },
                Spec { place: Place::Absolute(0x7ff6_a000_1000), len: None, read: false },
            ]
        );
        assert_eq!(parse("7ff6a0001000", true).unwrap()[0], Spec { place: Place::Absolute(0x7ff6_a000_1000), len: None, read: true });
        assert!(parse("0x10:3", false).is_err());
        assert!(parse("exe+zz", false).is_err());
        assert_eq!(parse(" , ", false).unwrap(), vec![]);
    }

    #[test]
    fn the_length_follows_the_alignment() {
        assert_eq!(length(0x1000, None), Ok(8));
        assert_eq!(length(0x1004, None), Ok(4));
        assert_eq!(length(0x1006, None), Ok(2));
        assert_eq!(length(0x1007, None), Ok(1));
        assert_eq!(length(0x1004, Some(4)), Ok(4));
        assert!(length(0x1004, Some(8)).is_err());
    }

    #[test]
    fn dr7_has_the_enable_rw_and_len_bits() {
        assert_eq!(dr7_bits(0, 4, false), 0b1 | (0b01 << 16) | (0b11 << 18));
        assert_eq!(dr7_bits(3, 8, true), (1 << 6) | (0b11 << 28) | (0b10 << 30));
        assert_eq!(dr7_bits(1, 1, false) & !dr7_mask(1), 0);
    }
}
