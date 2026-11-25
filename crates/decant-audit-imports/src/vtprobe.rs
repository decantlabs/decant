//! Measures the console D3D12 ABI from code compiled against it: which vtable slots a module
//! calls on the device, the command lists and other COM objects, with which interface IDs.
//!
//! A module of the package that takes an `ID3D12Device11*` or `ID3D12GraphicsCommandList4*` in a
//! mangled C++ export (XboxBvhBuilder.dll does) gives typed entry points. From there a small
//! forward data flow over the machine code follows where those pointers go (registers, stack
//! slots, members of `this`), and every `call [vtable+N]` on them is a measured slot. Calls on
//! objects of unknown type are still recorded when an argument is a known interface ID (a
//! `riid`), which tells the method family. Nothing is executed; files are read as data.
//!
//! The PC side comes from the public Windows SDK header `d3d12.h`: interface IDs and the method
//! order of each `...Vtbl`, so a console slot can be compared with the PC slot of the same name.

use std::collections::{BTreeMap, HashMap, HashSet};

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, OpKind, Register};
use serde::Serialize;

use crate::pe::Pe;

/// The PC reference parsed from `d3d12.h`: method names per interface, and interface IDs.
#[derive(Default)]
pub struct PcHeader {
    pub methods: HashMap<String, Vec<String>>,
    pub iids: HashMap<String, String>,
}

impl PcHeader {
    /// Parses `MIDL_INTERFACE("guid")` declarations and the C `...Vtbl` structures.
    pub fn parse(text: &str) -> PcHeader {
        let mut h = PcHeader::default();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            if let Some(rest) = line.strip_prefix("MIDL_INTERFACE(\"") {
                let guid = rest.split('"').next().unwrap_or("").to_ascii_lowercase();
                if let Some(next) = lines.get(i + 1)
                    && let Some(name) = next.trim().split([' ', ':']).next().filter(|n| !n.is_empty()) {
                        h.iids.insert(guid, name.to_string());
                    }
            } else if let Some(rest) = line.strip_prefix("typedef struct ")
                && let Some(iface) = rest.strip_suffix("Vtbl").map(str::trim) {
                    let mut names = Vec::new();
                    let mut j = i + 1;
                    while j < lines.len() && !lines[j].trim_start().starts_with('}') {
                        // A method returning a struct is declared twice, one per `#if` branch of
                        // the C ABI: it is one slot.
                        if let Some(name) = vtbl_entry(lines[j]).filter(|n| names.last() != Some(n)) {
                            names.push(name);
                        }
                        j += 1;
                    }
                    if !names.is_empty() {
                        h.methods.insert(iface.to_string(), names);
                    }
                    i = j;
                }
            i += 1;
        }
        h
    }

    pub fn iid_name(&self, guid: &str) -> Option<&str> {
        self.iids.get(guid).map(String::as_str)
    }

    pub fn pc_method(&self, iface: &str, slot: u32) -> Option<&str> {
        self.methods.get(iface)?.get(slot as usize).map(String::as_str)
    }
}

/// `( STDMETHODCALLTYPE *Name )(` in a C vtable, or `*Name )(` written over several lines.
fn vtbl_entry(line: &str) -> Option<String> {
    let star = line.find('*')?;
    let rest = &line[star + 1..];
    let close = rest.find(')')?;
    let name = rest[..close].trim();
    (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && rest[close..].trim_start_matches(')').trim_start().starts_with('(')).then(|| name.to_string())
}

/// What a register or a memory slot is known to hold.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Tag {
    /// A pointer to a COM object of this interface.
    Obj(String),
    /// The vtable of such an object.
    Vt(String),
    /// A method pointer read from that vtable.
    Meth(String, u32),
    /// `this` of a C++ method of this class.
    This(String),
    /// An address into the stack frame, as an offset from the stack pointer at entry.
    Stack(i64),
    /// A rip relative address (an interface ID candidate), as a virtual address.
    Rip(u64),
}

/// One measured vtable call.
#[derive(Clone, Debug, Serialize)]
pub struct Site {
    pub module: String,
    pub function: String,
    pub rva: u32,
    /// Interface of the object, `?` when only an argument identifies the method family.
    pub iface: String,
    pub slot: u32,
    /// Interface IDs passed as arguments: (argument, guid).
    pub iids: Vec<(String, String)>,
    /// `rdx` points into the caller's stack frame: a hidden return buffer or an out parameter.
    pub rdx_stack: bool,
}

#[derive(Default)]
struct Ctx {
    sites: Vec<Site>,
    members: HashMap<(String, i64), Tag>,
    seen: HashSet<(u64, String)>,
}

const ARG_REGS: [Register; 4] = [Register::RCX, Register::RDX, Register::R8, Register::R9];
const VOLATILE: [Register; 7] = [Register::RAX, Register::RCX, Register::RDX, Register::R8, Register::R9, Register::R10, Register::R11];

/// Measures every typed export of a module (two passes, so members stored by the constructor are
/// known in the methods), then returns the call sites.
pub fn probe(pe: &Pe, module: &str) -> Vec<Site> {
    let mut ctx = Ctx::default();
    let exports = pe.named_exports();
    for pass in 0..2 {
        ctx.sites.clear();
        ctx.seen.clear();
        for (name, rva) in &exports {
            let Some((class, params)) = demangle_params(name) else { continue };
            let mut entry: [Option<Tag>; 4] = Default::default();
            let mut slot = 0;
            if let Some(class) = &class {
                entry[0] = Some(Tag::This(class.clone()));
                slot = 1;
            }
            for p in params {
                if slot >= 4 {
                    break;
                }
                if let Some(iface) = p.filter(|t| t.starts_with("ID3D12") || t.starts_with("IDXGI")) {
                    entry[slot] = Some(Tag::Obj(iface));
                }
                slot += 1;
            }
            if pass == 0 && entry.iter().all(Option::is_none) {
                continue;
            }
            walk(pe, module, name, u64::from(*rva), entry, &mut ctx, 0);
        }
    }
    // Calls of unknown type that pass an interface ID are worth keeping only once each.
    let mut uniq = HashSet::new();
    ctx.sites.retain(|s| uniq.insert((s.rva, s.iface.clone(), s.slot)));
    ctx.sites.sort_by_key(|s| s.rva);
    ctx.sites
}

/// Scans every function of a module for calls through a vtable that pass a known interface ID,
/// whatever the object (modules with C exports have no typed entry point).
pub fn probe_untyped(pe: &Pe, module: &str, known: &PcHeader) -> Vec<Site> {
    let mut ctx = Ctx::default();
    for s in pe.sections.iter().filter(|s| s.chars & 0x2000_0000 != 0) {
        let Some(bytes) = pe.bytes_at(u64::from(s.va), s.rsize as usize) else { continue };
        let ip = pe.image_base() + u64::from(s.va);
        let mut dec = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE);
        let mut state = State::default();
        let mut ins = Instruction::default();
        while dec.can_decode() {
            dec.decode_out(&mut ins);
            if ins.mnemonic() == Mnemonic::Int3 {
                state = State::default();
                continue;
            }
            if ins.flow_control() == FlowControl::IndirectCall && ins.op0_kind() == OpKind::Memory && ins.memory_base() != Register::RIP {
                let iids = state.arg_iids(pe, &ins);
                if iids.iter().any(|(_, g)| known.iid_name(g).is_some()) {
                    let slot = (ins.memory_displacement64() / 8) as u32;
                    let rva = (ins.ip() - pe.image_base()) as u32;
                    ctx.sites.push(Site { module: module.into(), function: "?".into(), rva, iface: "?".into(), slot, iids, rdx_stack: state.is_stack(Register::RDX) });
                }
            }
            state.step(&ins, &mut ctx.members, None);
        }
    }
    ctx.sites
}

/// Receives (interface, slot) for each vtable call on a typed object.
type SiteSink<'a> = &'a mut dyn FnMut(&State, String, u32);

#[derive(Default, Clone)]
struct State {
    regs: HashMap<Register, Tag>,
    stack: HashMap<i64, Tag>,
    /// Bytes pushed or reserved since entry.
    sp: i64,
}

impl State {
    fn get(&self, r: Register) -> Option<&Tag> {
        self.regs.get(&r.full_register())
    }

    fn set(&mut self, r: Register, t: Option<Tag>) {
        let r = r.full_register();
        match t {
            Some(t) => {
                self.regs.insert(r, t);
            }
            None => {
                self.regs.remove(&r);
            }
        }
    }

    fn is_stack(&self, r: Register) -> bool {
        matches!(self.get(r), Some(Tag::Stack(_)))
    }

    /// The stack slot a memory operand designates, as an offset from the entry stack pointer.
    fn slot(&self, ins: &Instruction) -> Option<i64> {
        if ins.memory_index() != Register::None {
            return None;
        }
        let disp = ins.memory_displacement64() as i64;
        match ins.memory_base() {
            Register::RSP => Some(disp - self.sp),
            b => match self.get(b) {
                Some(Tag::Stack(o)) => Some(o + disp),
                _ => None,
            },
        }
    }

    /// Interface IDs among the arguments of a call: `rcx`..`r9` and the first stack arguments.
    fn arg_iids(&self, pe: &Pe, _ins: &Instruction) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (i, r) in ARG_REGS.iter().enumerate() {
            if let Some(Tag::Rip(va)) = self.get(*r)
                && let Some(g) = guid_at(pe, *va) {
                    out.push((["rcx", "rdx", "r8", "r9"][i].to_string(), g));
                }
        }
        for k in 0..4 {
            if let Some(Tag::Rip(va)) = self.stack.get(&(0x20 + 8 * k - self.sp))
                && let Some(g) = guid_at(pe, *va) {
                    out.push((format!("arg{}", 5 + k), g));
                }
        }
        out
    }

    /// Advances the data flow over one instruction. `site` receives vtable calls on typed objects.
    fn step(&mut self, ins: &Instruction, members: &mut HashMap<(String, i64), Tag>, site: Option<SiteSink<'_>>) {
        let m = ins.mnemonic();
        match m {
            Mnemonic::Push => self.sp += 8,
            Mnemonic::Pop => {
                self.sp -= 8;
                if ins.op0_kind() == OpKind::Register {
                    self.set(ins.op0_register(), None);
                }
            }
            Mnemonic::Sub | Mnemonic::Add if ins.op0_kind() == OpKind::Register && ins.op0_register() == Register::RSP => {
                if matches!(ins.op1_kind(), OpKind::Immediate8to64 | OpKind::Immediate32to64 | OpKind::Immediate8 | OpKind::Immediate32) {
                    let v = ins.immediate(1) as i64;
                    self.sp += if m == Mnemonic::Sub { v } else { -v };
                }
            }
            Mnemonic::Call => {
                let mut target = None;
                if ins.op0_kind() == OpKind::Memory && ins.memory_index() == Register::None {
                    if let Some(Tag::Vt(t)) = self.get(ins.memory_base()) {
                        target = Some((t.clone(), (ins.memory_displacement64() / 8) as u32));
                    }
                } else if ins.op0_kind() == OpKind::Register
                    && let Some(Tag::Meth(t, s)) = self.get(ins.op0_register()) {
                        target = Some((t.clone(), *s));
                    }
                if let (Some((t, s)), Some(f)) = (target, site) {
                    f(self, t, s);
                }
                for r in VOLATILE {
                    self.set(r, None);
                }
                // The outgoing argument area is rewritten before the next call: forget it, so a
                // stale argument of an earlier call is never read as an argument of a later one.
                for k in 0..8 {
                    self.stack.remove(&(0x20 + 8 * k - self.sp));
                }
            }
            Mnemonic::Mov => self.mov(ins, members),
            Mnemonic::Lea => {
                let dst = ins.op0_register();
                let tag = if ins.is_ip_rel_memory_operand() {
                    Some(Tag::Rip(ins.ip_rel_memory_address()))
                } else {
                    self.slot(ins).map(Tag::Stack)
                };
                self.set(dst, tag);
            }
            Mnemonic::Cmp | Mnemonic::Test | Mnemonic::Nop => {}
            _ => {
                if ins.op_count() > 0 && ins.op0_kind() == OpKind::Register {
                    self.set(ins.op0_register(), None);
                }
            }
        }
    }

    fn mov(&mut self, ins: &Instruction, members: &mut HashMap<(String, i64), Tag>) {
        match (ins.op0_kind(), ins.op1_kind()) {
            (OpKind::Register, OpKind::Register) => {
                let src = ins.op1_register();
                let t = if src == Register::RSP { Some(Tag::Stack(-self.sp)) } else { self.get(src).cloned() };
                self.set(ins.op0_register(), t);
            }
            (OpKind::Register, OpKind::Memory) => {
                let dst = ins.op0_register();
                if dst.size() != 8 {
                    self.set(dst, None);
                    return;
                }
                let t = if let Some(s) = self.slot(ins) {
                    self.stack.get(&s).cloned()
                } else if ins.memory_index() == Register::None && !ins.is_ip_rel_memory_operand() {
                    let disp = ins.memory_displacement64() as i64;
                    match self.get(ins.memory_base()) {
                        Some(Tag::Obj(t)) if disp == 0 => Some(Tag::Vt(t.clone())),
                        Some(Tag::Vt(t)) if disp >= 0 && disp % 8 == 0 => Some(Tag::Meth(t.clone(), (disp / 8) as u32)),
                        Some(Tag::This(c)) => members.get(&(c.clone(), disp)).cloned(),
                        _ => None,
                    }
                } else {
                    None
                };
                self.set(dst, t);
            }
            (OpKind::Memory, OpKind::Register) => {
                let src = self.get(ins.op1_register()).cloned();
                if let Some(s) = self.slot(ins) {
                    match src {
                        Some(t) => {
                            self.stack.insert(s, t);
                        }
                        None => {
                            self.stack.remove(&s);
                        }
                    }
                } else if ins.memory_index() == Register::None
                    && let (Some(Tag::This(c)), Some(t @ Tag::Obj(_))) = (self.get(ins.memory_base()).cloned(), src) {
                        members.insert((c, ins.memory_displacement64() as i64), t);
                    }
            }
            (OpKind::Register, _) => self.set(ins.op0_register(), None),
            _ => {}
        }
    }
}

fn guid_at(pe: &Pe, va: u64) -> Option<String> {
    let b = pe.bytes_at(va.checked_sub(pe.image_base())?, 16)?;
    if b.len() < 16 || b.iter().all(|&x| x == 0) || b.iter().all(|&x| x == 0xFF) {
        return None;
    }
    let d1 = u32::from_le_bytes(b[0..4].try_into().ok()?);
    let d2 = u16::from_le_bytes(b[4..6].try_into().ok()?);
    let d3 = u16::from_le_bytes(b[6..8].try_into().ok()?);
    Some(format!("{d1:08x}-{d2:04x}-{d3:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]))
}

/// Follows one function from `rva` with the given argument tags; recurses into direct calls that
/// receive a typed argument (depth limited, each (function, arguments) once).
fn walk(pe: &Pe, module: &str, name: &str, rva: u64, entry: [Option<Tag>; 4], ctx: &mut Ctx, depth: u32) {
    let key = format!("{entry:?}");
    if !ctx.seen.insert((rva, key)) || depth > 4 {
        return;
    }
    let Some(bytes) = pe.bytes_at(rva, 0x20000) else { return };
    let base = pe.image_base();
    let mut dec = Decoder::with_ip(64, bytes, base + rva, DecoderOptions::NONE);
    let mut st = State::default();
    for (r, t) in ARG_REGS.iter().zip(entry) {
        st.set(*r, t);
    }
    let mut ins = Instruction::default();
    let mut count = 0;
    let mut pending: Vec<(u64, [Option<Tag>; 4])> = Vec::new();
    while dec.can_decode() && count < 20_000 {
        dec.decode_out(&mut ins);
        count += 1;
        if ins.mnemonic() == Mnemonic::Int3 {
            break;
        }
        if ins.mnemonic() == Mnemonic::Call {
            let iids = st.arg_iids(pe, &ins);
            let rdx_stack = st.is_stack(Register::RDX);
            let site_rva = (ins.ip() - base) as u32;
            if ins.op0_kind() == OpKind::NearBranch64 {
                let t = ins.near_branch_target();
                let args: [Option<Tag>; 4] = ARG_REGS.map(|r| st.get(r).cloned().filter(|t| matches!(t, Tag::Obj(_) | Tag::This(_))));
                if t >= base && pe.is_code(t - base) && args.iter().any(Option::is_some) {
                    pending.push((t - base, args));
                }
            }
            let mut found = Vec::new();
            st.step(&ins, &mut ctx.members, Some(&mut |_: &State, t: String, s: u32| found.push((t, s))));
            for (iface, slot) in found {
                ctx.sites.push(Site { module: module.into(), function: name.into(), rva: site_rva, iface, slot, iids: iids.clone(), rdx_stack });
            }
            continue;
        }
        let tail = matches!(ins.flow_control(), FlowControl::Return) || (ins.flow_control() == FlowControl::UnconditionalBranch && ins.op0_kind() == OpKind::NearBranch64);
        st.step(&ins, &mut ctx.members, None);
        if tail {
            let next = bytes.get((ins.next_ip() - base - rva) as usize).copied();
            if next.is_none() || next == Some(0xCC) {
                break;
            }
        }
    }
    for (t, args) in pending {
        walk(pe, module, &format!("{name} > sub_{t:x}"), t, args, ctx, depth + 1);
    }
}

/// Class (for member functions) and parameter types of an MSVC mangled function name. Only the
/// forms these modules use are understood; anything else returns `None`.
pub fn demangle_params(name: &str) -> Option<(Option<String>, Vec<Option<String>>)> {
    let s = name.as_bytes();
    if s.first() != Some(&b'?') {
        return None;
    }
    let mut p = 1;
    let mut names: Vec<String> = Vec::new();
    // Qualified name: fragments ending with '@', closed by an extra '@'. "?0" / "?1" are ctor/dtor.
    let mut frags = Vec::new();
    if s.get(p) == Some(&b'?') {
        p += 2;
    }
    while p < s.len() && s[p] != b'@' {
        let end = p + s[p..].iter().position(|&c| c == b'@')?;
        frags.push(String::from_utf8_lossy(&s[p..end]).into_owned());
        names.push(frags.last().unwrap().clone());
        p = end + 1;
    }
    p += 1;
    let code = *s.get(p)?;
    p += 1;
    let has_this = b"ABEFIJMNQRUV".contains(&code);
    let class = if has_this { frags.first().cloned() } else { None };
    if b"ABCDEFIJKLMNQRSTUV".contains(&code) {
        if has_this {
            if s.get(p) == Some(&b'E') {
                p += 1;
            }
            p += 1; // cv qualifier of this
        }
    } else if code != b'Y' {
        return None;
    }
    p += 1; // calling convention
    // Return type (or '@' for constructors and destructors).
    if s.get(p) == Some(&b'@') {
        p += 1;
    } else {
        parse_type(s, &mut p, &mut names)?;
    }
    let mut params = Vec::new();
    let mut backrefs: Vec<Option<String>> = Vec::new();
    if s.get(p) == Some(&b'X') {
        return Some((class, params));
    }
    while p < s.len() && s[p] != b'@' && s[p] != b'Z' {
        if s[p].is_ascii_digit() {
            let i = usize::from(s[p] - b'0');
            params.push(backrefs.get(i).cloned().flatten());
            p += 1;
            continue;
        }
        let start = p;
        let t = parse_type(s, &mut p, &mut names)?;
        if p - start > 1 {
            backrefs.push(t.clone());
        }
        params.push(t);
    }
    Some((class, params))
}

/// Parses one type; returns the interface name for pointers to structs or classes.
fn parse_type(s: &[u8], p: &mut usize, names: &mut Vec<String>) -> Option<Option<String>> {
    let c = *s.get(*p)?;
    *p += 1;
    match c {
        b'_' => {
            *p += 1;
            Some(None)
        }
        b'P' | b'Q' | b'R' | b'S' | b'A' | b'B' => {
            if s.get(*p) == Some(&b'E') {
                *p += 1;
            }
            *p += 1; // cv of the pointee
            let inner = parse_type(s, p, names)?;
            Some(inner)
        }
        b'U' | b'V' | b'T' => {
            let n = parse_name(s, p, names)?;
            Some(Some(n))
        }
        b'W' => {
            *p += 1; // enum underlying type
            parse_name(s, p, names)?;
            Some(None)
        }
        b'$' => None,
        _ if c.is_ascii_uppercase() => Some(None),
        _ => None,
    }
}

fn parse_name(s: &[u8], p: &mut usize, names: &mut Vec<String>) -> Option<String> {
    let mut first = None;
    loop {
        let c = *s.get(*p)?;
        if c == b'@' {
            *p += 1;
            return first;
        }
        if c.is_ascii_digit() {
            let n = names.get(usize::from(c - b'0')).cloned();
            *p += 1;
            first = first.or(n);
            continue;
        }
        let end = *p + s[*p..].iter().position(|&c| c == b'@')?;
        let frag = String::from_utf8_lossy(&s[*p..end]).into_owned();
        names.push(frag.clone());
        first = first.or(Some(frag));
        *p = end + 1;
    }
}

/// Prints the instructions around a call site (decoding starts 0x60 bytes earlier and resyncs).
pub fn dump(pe: &Pe, rva: u64) {
    use iced_x86::{Formatter, IntelFormatter};
    let start = rva.saturating_sub(0x60);
    let Some(bytes) = pe.bytes_at(start, 0x70) else { return };
    let mut dec = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
    let mut f = IntelFormatter::new();
    let mut ins = Instruction::default();
    let mut text = String::new();
    while dec.can_decode() {
        dec.decode_out(&mut ins);
        text.clear();
        f.format(&ins, &mut text);
        println!("{}{:08x}  {text}", if ins.ip() == rva { ">" } else { " " }, ins.ip());
    }
}

/// Aggregated result per (interface, slot), for the report.
#[derive(Serialize)]
pub struct SlotRow {
    pub iface: String,
    pub slot: u32,
    pub calls: usize,
    pub pc_method: Option<String>,
    pub iids: Vec<String>,
    pub rdx_stack: bool,
    pub sites: Vec<String>,
}

pub fn aggregate(sites: &[Site], pc: &PcHeader) -> Vec<SlotRow> {
    let mut map: BTreeMap<(String, u32), SlotRow> = BTreeMap::new();
    for s in sites {
        let row = map.entry((s.iface.clone(), s.slot)).or_insert_with(|| SlotRow {
            iface: s.iface.clone(),
            slot: s.slot,
            calls: 0,
            pc_method: pc.pc_method(&s.iface, s.slot).map(String::from),
            iids: Vec::new(),
            rdx_stack: false,
            sites: Vec::new(),
        });
        row.calls += 1;
        row.rdx_stack |= s.rdx_stack;
        for (arg, g) in &s.iids {
            let label = format!("{arg}={}", pc.iid_name(g).map_or_else(|| g.clone(), String::from));
            if !row.iids.contains(&label) {
                row.iids.push(label);
            }
        }
        if row.sites.len() < 6 {
            row.sites.push(format!("{}+0x{:x}", s.module, s.rva));
        }
    }
    map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demangles_the_member_functions_of_the_witness() {
        let (c, p) = demangle_params("??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z").unwrap();
        assert_eq!(c.as_deref(), Some("XboxBvhBuilder"));
        assert_eq!(p, vec![Some("ID3D12Device11".to_string()), None]);
        let (_, p) = demangle_params("?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z").unwrap();
        assert_eq!(p[0].as_deref(), Some("ID3D12GraphicsCommandList4"));
        assert_eq!(p.len(), 4);
        let (_, p) = demangle_params("?RehydrateRaytracingAccelerationStructureX@XboxBvhBuilder@@QEAAJPEAUID3D12Device11@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z").unwrap();
        assert_eq!(p[0].as_deref(), Some("ID3D12Device11"));
    }

    #[test]
    fn parses_vtables_and_iids_of_a_header() {
        let h = PcHeader::parse(
            "MIDL_INTERFACE(\"189819F1-1DB6-4B57-BE54-1821339B85F7\")\n    ID3D12Device : public ID3D12Object\n\
             typedef struct ID3D12DeviceVtbl\n{\n    HRESULT ( STDMETHODCALLTYPE *QueryInterface )(\n    ULONG ( STDMETHODCALLTYPE *AddRef )(\n\
             #if !defined(_WIN32)\n    INFO ( STDMETHODCALLTYPE *GetInfo )(\n#else\n    INFO *( STDMETHODCALLTYPE *GetInfo )(\n#endif\n\
                 HRESULT ( STDMETHODCALLTYPE *Next )(\n} ID3D12DeviceVtbl;\n",
        );
        assert_eq!(h.iid_name("189819f1-1db6-4b57-be54-1821339b85f7"), Some("ID3D12Device"));
        assert_eq!(h.pc_method("ID3D12Device", 1), Some("AddRef"));
        assert_eq!(h.pc_method("ID3D12Device", 3), Some("Next"));
    }

    #[test]
    fn follows_a_device_into_a_vtable_call() {
        // mov rax,[rcx] ; call qword ptr [rax+0x1F0]  (slot 62)
        let code = [0x48, 0x8B, 0x01, 0xFF, 0x90, 0xF0, 0x01, 0x00, 0x00];
        let mut dec = Decoder::with_ip(64, &code, 0x1000, DecoderOptions::NONE);
        let mut st = State::default();
        st.set(Register::RCX, Some(Tag::Obj("ID3D12Device11".into())));
        let mut members = HashMap::new();
        let mut hits = Vec::new();
        let mut ins = Instruction::default();
        while dec.can_decode() {
            dec.decode_out(&mut ins);
            st.step(&ins, &mut members, Some(&mut |_: &State, t: String, s: u32| hits.push((t, s))));
        }
        assert_eq!(hits, vec![("ID3D12Device11".to_string(), 62)]);
    }
}
