//! `--dstorage-probe`: finds, in an x64 PE, what a title expects from DirectStorage, to fill the
//! console ABI spec of `dstorage_xs.dll` (`crates/decant-dstorage/abi`, format in `docs/DSTORAGE_ABI.md`).
//!
//! It reads the file as data, never runs it. For every import from a DirectStorage DLL
//! (`dstorage_xs`, `xcrdapi`, `dstorage`) it looks for the `call [rip+disp32]` instructions that go
//! through that import's address table entry. Around each call to `DStorageGetFactory`:
//! - before it, the last `lea rcx, [rip+disp32]`: the first argument, the interface ID asked for,
//!   whose 16 bytes are read where it points;
//! - after it, the indirect calls `call [reg+disp]`: on the returned object they call vtable slot
//!   `disp / 8`.
//!
//! This is a linear byte scan, not a disassembly: the GUIDs and slots it reports are candidates to
//! check by hand on the executable, and the draft spec it writes says so.

use crate::pe::{Func, Pe};
use std::collections::BTreeMap;
use std::fmt::Write as _;

const DLLS: &[&str] = &["dstorage_xs", "xcrdapi", "dstorage"];
const BACK: usize = 64;
const AHEAD: usize = 512;

pub struct Site {
    pub call_rva: u32,
    pub function: String,
    /// RVA and bytes of the GUID loaded into `rcx` before the call.
    pub iid: Option<(u32, [u8; 16])>,
    /// Vtable slots called after the call, in order of appearance, with the call's RVA.
    pub slots: Vec<(u32, u32)>,
}

pub struct Probe {
    /// `(dll, function, IAT entry RVA)` for every DirectStorage import.
    pub imports: Vec<(String, String, u32)>,
    pub sites: Vec<Site>,
}

fn is_storage_dll(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let stem = lower.strip_suffix(".dll").unwrap_or(&lower);
    DLLS.contains(&stem)
}

fn i32_at(b: &[u8], at: usize) -> Option<i32> {
    b.get(at..at + 4).map(|s| i32::from_le_bytes(s.try_into().unwrap()))
}

/// The RVA `disp` bytes after the instruction that ends at `next`.
fn rip_target(next: u64, disp: i32) -> Option<u32> {
    u32::try_from(next as i64 + i64::from(disp)).ok()
}

/// `{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}` from the in-memory layout of a Windows GUID.
pub fn guid_text(g: &[u8; 16]) -> String {
    let d1 = u32::from_le_bytes(g[0..4].try_into().unwrap());
    let d2 = u16::from_le_bytes(g[4..6].try_into().unwrap());
    let d3 = u16::from_le_bytes(g[6..8].try_into().unwrap());
    let tail: String = g[10..].iter().map(|b| format!("{b:02x}")).collect();
    format!("{{{d1:08x}-{d2:04x}-{d3:04x}-{:02x}{:02x}-{tail}}}", g[8], g[9])
}

/// The vtable slot of an indirect `call [reg+disp]` at `at` (optional REX prefix), with its length.
fn indirect_call(code: &[u8], at: usize) -> Option<(u32, usize)> {
    let op = if (0x40..=0x4F).contains(code.get(at)?) && *code.get(at + 1)? == 0xFF { at + 1 } else { at };
    if *code.get(op)? != 0xFF {
        return None;
    }
    let modrm = *code.get(op + 1)?;
    let (md, reg, rm) = (modrm >> 6, (modrm >> 3) & 7, modrm & 7);
    if reg != 2 || md == 3 || (md == 0 && rm == 5) {
        return None; // not `call r/m`, a register call, or a rip-relative (import) call
    }
    let mut p = op + 2;
    if rm == 4 {
        p += 1; // SIB byte
    }
    let disp = match md {
        0 => 0i64,
        1 => i64::from(*code.get(p)? as i8),
        _ => i64::from(i32_at(code, p)?),
    };
    let len = p + match md {
        0 => 0,
        1 => 1,
        _ => 4,
    } - at;
    (disp >= 0 && disp % 8 == 0 && disp / 8 < 256).then_some(((disp / 8) as u32, len))
}

pub fn probe(pe: &Pe) -> Probe {
    let mut imports = Vec::new();
    let mut by_slot: BTreeMap<u32, String> = BTreeMap::new();
    for (dll, func, iat) in pe.iat_slots() {
        if is_storage_dll(&dll) {
            let name = match &func {
                Func::Name(n) => n.clone(),
                Func::Ordinal(o) => format!("#{o}"),
            };
            by_slot.insert(iat, name.clone());
            imports.push((dll, name, iat));
        }
    }
    let mut sites = Vec::new();
    if !pe.is64() {
        return Probe { imports, sites }; // the instruction patterns below are x64 ones
    }
    let bytes = pe.bytes();
    let code_sections: Vec<(u64, &[u8])> = pe
        .sections
        .iter()
        .filter(|s| s.chars & 0x2000_0020 != 0)
        .filter_map(|s| {
            let start = s.rptr as usize;
            let len = (s.rsize as usize).min(bytes.len().saturating_sub(start));
            bytes.get(start..start + len).map(|c| (u64::from(s.va), c))
        })
        .collect();
    // Without `dllimport` a compiler calls an import through a linker thunk, `jmp [rip+disp32]` to
    // the IAT entry, reached by `call rel32`.
    let mut thunks: BTreeMap<u32, String> = BTreeMap::new();
    for (va, code) in &code_sections {
        for i in 0..code.len().saturating_sub(6) {
            if code[i] != 0xFF || code[i + 1] != 0x25 {
                continue;
            }
            let Some(target) = i32_at(code, i + 2).and_then(|d| rip_target(va + i as u64 + 6, d)) else { continue };
            if let Some(name) = by_slot.get(&target) {
                thunks.insert((va + i as u64) as u32, name.clone());
            }
        }
    }
    for (va, code) in code_sections {
        let mut calls = Vec::new();
        let mut i = 0;
        while i + 5 <= code.len() {
            if code[i] == 0xFF && code.get(i + 1) == Some(&0x15) {
                let target = i32_at(code, i + 2).and_then(|d| rip_target(va + i as u64 + 6, d));
                if let Some(name) = target.and_then(|t| by_slot.get(&t)) {
                    calls.push((i, 6, name.clone()));
                }
            } else if code[i] == 0xE8 {
                let target = i32_at(code, i + 1).and_then(|d| rip_target(va + i as u64 + 5, d));
                if let Some(name) = target.and_then(|t| thunks.get(&t)) {
                    calls.push((i, 5, name.clone()));
                }
            }
            i += 1;
        }
        for (n, (i, call_len, function)) in calls.iter().enumerate() {
            let mut site = Site { call_rva: (va + *i as u64) as u32, function: function.clone(), iid: None, slots: Vec::new() };
            for j in (i.saturating_sub(BACK)..*i).rev() {
                if code.get(j..j + 3) == Some(&[0x48, 0x8D, 0x0D]) {
                    if let Some(target) = i32_at(code, j + 3).and_then(|d| rip_target(va + j as u64 + 7, d)) {
                        let guid = pe.file_offset(u64::from(target)).and_then(|o| bytes.get(o..o + 16));
                        if let Some(g) = guid {
                            site.iid = Some((target, g.try_into().unwrap()));
                        }
                    }
                    break;
                }
            }
            let end = calls.get(n + 1).map_or(code.len(), |(next, _, _)| *next).min(i + call_len + AHEAD);
            let mut k = i + call_len;
            while k < end {
                match indirect_call(code, k) {
                    Some((slot, len)) => {
                        site.slots.push((slot, (va + k as u64) as u32));
                        k += len;
                    }
                    None => k += 1,
                }
            }
            sites.push(site);
        }
    }
    Probe { imports, sites }
}

pub fn markdown(p: &Probe, file: &str) -> String {
    let mut out = format!("# DirectStorage probe: {file}\n\nCandidates from a byte scan, to check on the executable.\n\n");
    if p.imports.is_empty() {
        out.push_str("No import from a DirectStorage DLL (dstorage_xs, xcrdapi, dstorage).\n");
        return out;
    }
    out.push_str("## Imports\n\n| DLL | Function | IAT entry |\n|---|---|---|\n");
    for (dll, f, iat) in &p.imports {
        let _ = writeln!(out, "| {dll} | {f} | 0x{iat:08x} |");
    }
    let _ = writeln!(out, "\n## Call sites ({})\n", p.sites.len());
    for s in &p.sites {
        let iid = s.iid.map_or("none found".to_owned(), |(rva, g)| format!("{} (at 0x{rva:08x})", guid_text(&g)));
        let slots: Vec<String> = s.slots.iter().map(|(slot, at)| format!("{slot} @0x{at:08x}")).collect();
        let _ = writeln!(out, "- 0x{:08x} `{}`: IID {iid}; vtable slots after it: {}", s.call_rva, s.function,
            if slots.is_empty() { "none".to_owned() } else { slots.join(", ") });
    }
    out
}

/// A draft spec for the factory interface: the IIDs passed to `DStorageGetFactory` and the slots
/// called after it, all as "?" for a person to map to engine handlers.
pub fn draft_spec(p: &Probe, file: &str) -> String {
    let factory: Vec<&Site> = p.sites.iter().filter(|s| s.function == "DStorageGetFactory").collect();
    let mut iids: Vec<String> = Vec::new();
    let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
    for s in &factory {
        if let Some((_, g)) = s.iid {
            let t = guid_text(&g);
            if !iids.contains(&t) {
                iids.push(t);
            }
        }
        for (slot, _) in &s.slots {
            *seen.entry(*slot).or_default() += 1;
        }
    }
    let mut out = format!(
        "abi 1\n# DRAFT written by audit_xbox_imports --dstorage-probe from {file}.\n\
         # Candidates from a byte scan: check every IID and slot on the executable, then map each\n\
         # slot to an engine handler (see docs/DSTORAGE_ABI.md and abi/console_hypothesis.abi).\n\
         # {} call(s) to DStorageGetFactory.\n\ninterface factory draft_factory\n  status draft\n",
        factory.len()
    );
    if iids.is_empty() {
        out.push_str("  iid unknown\n");
    }
    for t in &iids {
        let _ = writeln!(out, "  iid {t}");
    }
    for (slot, count) in seen.iter().filter(|(slot, _)| **slot >= 3 && **slot < 64) {
        let _ = writeln!(out, "  method {slot} ? seen_{count}x");
    }
    out.push_str("end\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACTORY: [u8; 16] = [0x0c, 0xea, 0x24, 0x69, 0xcd, 0xc3, 0x26, 0x48, 0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1];

    fn put(b: &mut [u8], at: usize, v: &[u8]) {
        b[at..at + v.len()].copy_from_slice(v);
    }

    /// A minimal PE32+: `.text` at RVA 0x1000 (file 0x200), `.rdata` at 0x2000 (file 0x400) holding
    /// an import of XCRDAPI.dll!DStorageGetFactory (IAT entry 0x2060) and a GUID at 0x2100.
    fn synthetic_pe() -> Vec<u8> {
        let mut b = vec![0u8; 0x600];
        put(&mut b, 0, b"MZ");
        put(&mut b, 0x3C, &0x40u32.to_le_bytes());
        put(&mut b, 0x40, b"PE\0\0");
        put(&mut b, 0x44, &0x8664u16.to_le_bytes());
        put(&mut b, 0x46, &2u16.to_le_bytes());
        put(&mut b, 0x54, &0xF0u16.to_le_bytes());
        let opt = 0x58;
        put(&mut b, opt, &0x20Bu16.to_le_bytes());
        put(&mut b, opt + 24, &0x1_4000_0000u64.to_le_bytes());
        put(&mut b, opt + 108, &16u32.to_le_bytes());
        put(&mut b, opt + 112 + 8, &0x2000u32.to_le_bytes()); // import directory
        put(&mut b, opt + 112 + 12, &40u32.to_le_bytes());
        let sec = opt + 0xF0;
        for (n, (name, va, rptr, chars)) in
            [(&b".text"[..], 0x1000u32, 0x200u32, 0x6000_0020u32), (&b".rdata"[..], 0x2000, 0x400, 0x4000_0040)].iter().enumerate()
        {
            let h = sec + 40 * n;
            put(&mut b, h, name);
            put(&mut b, h + 8, &0x200u32.to_le_bytes());
            put(&mut b, h + 12, &va.to_le_bytes());
            put(&mut b, h + 16, &0x200u32.to_le_bytes());
            put(&mut b, h + 20, &rptr.to_le_bytes());
            put(&mut b, h + 36, &chars.to_le_bytes());
        }
        let rd = |rva: usize| rva - 0x2000 + 0x400;
        put(&mut b, rd(0x2000), &0x2040u32.to_le_bytes()); // original first thunk
        put(&mut b, rd(0x2000) + 12, &0x2080u32.to_le_bytes()); // name
        put(&mut b, rd(0x2000) + 16, &0x2060u32.to_le_bytes()); // first thunk (IAT)
        put(&mut b, rd(0x2040), &0x20A0u64.to_le_bytes());
        put(&mut b, rd(0x2060), &0x20A0u64.to_le_bytes());
        put(&mut b, rd(0x2080), b"XCRDAPI.dll\0");
        put(&mut b, rd(0x20A2), b"DStorageGetFactory\0");
        put(&mut b, rd(0x2100), &FACTORY);
        let code: &[u8] = &[
            0x48, 0x8D, 0x0D, 0xF9, 0x10, 0x00, 0x00, // 0x1000 lea rcx, [rip+0x10F9] -> 0x2100
            0x48, 0x8D, 0x54, 0x24, 0x20, // 0x1007 lea rdx, [rsp+0x20]
            0xFF, 0x15, 0x4E, 0x10, 0x00, 0x00, // 0x100C call [rip+0x104E] -> IAT 0x2060
            0x48, 0x8B, 0x4C, 0x24, 0x20, // 0x1012 mov rcx, [rsp+0x20]
            0x48, 0x8B, 0x01, // 0x1017 mov rax, [rcx]
            0xFF, 0x50, 0x18, // 0x101A call [rax+0x18]  slot 3
            0xFF, 0x50, 0x28, // 0x101D call [rax+0x28]  slot 5
            0x41, 0xFF, 0x92, 0x00, 0x01, 0x00, 0x00, // 0x1020 call [r10+0x100] slot 32
            0xFF, 0x50, 0x13, // 0x1027 call [rax+0x13] not a slot
            0xC3, // 0x102A
        ];
        put(&mut b, 0x200, code);
        // The same import through a linker thunk, as without `dllimport`.
        put(&mut b, 0x230, &[0xFF, 0x25, 0x2A, 0x10, 0x00, 0x00]); // 0x1030 jmp [rip+0x102A] -> IAT 0x2060
        put(
            &mut b,
            0x240,
            &[
                0x48, 0x8D, 0x0D, 0xB9, 0x10, 0x00, 0x00, // 0x1040 lea rcx, [rip+0x10B9] -> 0x2100
                0xE8, 0xE4, 0xFF, 0xFF, 0xFF, // 0x1047 call 0x1030 (the thunk)
                0x48, 0x8B, 0x01, // 0x104C mov rax, [rcx]
                0xFF, 0x50, 0x20, // 0x104F call [rax+0x20] slot 4
                0xC3,
            ],
        );
        b
    }

    #[test]
    fn finds_the_factory_iid_and_the_slots_called_after_it() {
        let path = std::env::temp_dir().join(format!("decant_storage_probe_{}.exe", std::process::id()));
        std::fs::write(&path, synthetic_pe()).unwrap();
        let pe = Pe::open(&path).unwrap();
        let p = probe(&pe);
        assert_eq!(p.imports, vec![("XCRDAPI.dll".to_owned(), "DStorageGetFactory".to_owned(), 0x2060)]);
        assert_eq!(p.sites.len(), 2);
        let s = &p.sites[0];
        assert_eq!(s.call_rva, 0x100C);
        assert_eq!(s.iid, Some((0x2100, FACTORY)));
        assert_eq!(s.slots.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(), vec![3, 5, 32]);
        let t = &p.sites[1]; // through the thunk
        assert_eq!((t.call_rva, t.iid), (0x1047, Some((0x2100, FACTORY))));
        assert_eq!(t.slots.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(), vec![4]);
        let spec = draft_spec(&p, "synthetic.exe");
        assert!(spec.starts_with("abi 1\n"));
        assert_eq!(spec.matches("  iid {6924ea0c-c3cd-4826-b10a-f64f4ed927c1}\n").count(), 1);
        assert!(spec.contains("  method 3 ? seen_1x\n") && spec.contains("  method 4 ? seen_1x\n"));
        assert!(spec.contains("  method 5 ? seen_1x\n"));
        assert!(spec.contains("  method 32 ? seen_1x\n") && spec.ends_with("end\n"));
        assert!(markdown(&p, "synthetic.exe").contains("DStorageGetFactory"));
        drop(pe);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn guid_text_uses_the_windows_layout() {
        assert_eq!(guid_text(&FACTORY), "{6924ea0c-c3cd-4826-b10a-f64f4ed927c1}");
    }

    #[test]
    fn a_pe_without_storage_imports_gives_an_empty_report() {
        let p = Probe { imports: vec![], sites: vec![] };
        assert!(markdown(&p, "x").contains("No import from a DirectStorage DLL"));
        assert!(draft_spec(&p, "x").contains("iid unknown"));
    }
}
