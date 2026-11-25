//! Reading a PE file as data: sections, imports (normal and delayed), exports and section entropy.
//! Every bound and limit follows `audit_xbox_imports.py`, so both report the same imports.

use std::collections::HashSet;
use std::path::Path;

/// An imported function: by name, or by ordinal when the name is absent.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Func {
    Name(String),
    Ordinal(u16),
}

impl Func {
    /// `name` or `#ordinal`, as the report writes it.
    pub fn label(&self) -> String {
        match self {
            Func::Name(n) => n.clone(),
            Func::Ordinal(o) => format!("#{o}"),
        }
    }
}

pub struct Import {
    pub dll: String,
    pub delay: bool,
    pub funcs: Vec<Func>,
}

pub struct Section {
    pub name: String,
    pub va: u32,
    pub vsize: u32,
    pub rsize: u32,
    pub rptr: u32,
    pub chars: u32,
}

pub struct SectionRow {
    pub name: String,
    pub size_mib: f64,
    pub executable: bool,
    pub entropy: f64,
}

pub struct Pe {
    data: Vec<u8>,
    pub machine: u16,
    is64: bool,
    dirs: Vec<(u32, u32)>,
    image_base: u64,
    pub sections: Vec<Section>,
}

fn u16_at(d: &[u8], at: usize) -> Result<u16, String> {
    d.get(at..at + 2).map(|b| u16::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| short(at, 2, d.len()))
}

fn u32_at(d: &[u8], at: usize) -> Result<u32, String> {
    d.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| short(at, 4, d.len()))
}

fn u64_at(d: &[u8], at: usize) -> Result<u64, String> {
    d.get(at..at + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).ok_or_else(|| short(at, 8, d.len()))
}

fn short(at: usize, len: usize, size: usize) -> String {
    format!("{len} bytes at offset {at} are past the end of the file ({size} bytes)")
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

impl Pe {
    pub fn open(path: &Path) -> Result<Pe, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let d = &data[..];
        if d.len() < 0x40 || &d[..2] != b"MZ" {
            return Err("pas un fichier PE (en-tete MZ absent)".into());
        }
        let pe = u32_at(d, 0x3C)? as usize;
        if pe + 24 > d.len() || &d[pe..pe + 4] != b"PE\0\0" {
            return Err("signature PE absente".into());
        }
        let (machine, nsec, optsize) = (u16_at(d, pe + 4)?, u16_at(d, pe + 6)?, u16_at(d, pe + 20)?);
        let opt = pe + 24;
        let magic = u16_at(d, opt)?;
        if magic != 0x10B && magic != 0x20B {
            return Err("en-tete optionnel inconnu".into());
        }
        let is64 = magic == 0x20B;
        let ddoff = opt + if is64 { 112 } else { 96 };
        let ndd = u32_at(d, ddoff - 4)?;
        let dirs = (0..ndd.min(16) as usize).map(|i| Ok((u32_at(d, ddoff + 8 * i)?, u32_at(d, ddoff + 8 * i + 4)?))).collect::<Result<Vec<_>, String>>()?;
        let image_base = if is64 { u64_at(d, opt + 24)? } else { u64::from(u32_at(d, opt + 28)?) };
        let secoff = opt + usize::from(optsize);
        let mut sections = Vec::with_capacity(usize::from(nsec));
        for i in 0..usize::from(nsec) {
            let base = secoff + 40 * i;
            let name = d.get(base..base + 8).ok_or_else(|| short(base, 8, d.len()))?;
            let end = name.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
            sections.push(Section {
                name: latin1(&name[..end]),
                vsize: u32_at(d, base + 8)?,
                va: u32_at(d, base + 12)?,
                rsize: u32_at(d, base + 16)?,
                rptr: u32_at(d, base + 20)?,
                chars: u32_at(d, base + 36)?,
            });
        }
        let (machine, dirs) = (machine, dirs);
        Ok(Pe { data, machine, is64, dirs, image_base, sections })
    }

    fn d(&self) -> &[u8] {
        &self.data
    }

    /// The file offset of a relative virtual address.
    fn off(&self, rva: u64) -> Option<usize> {
        for s in &self.sections {
            let (va, span) = (u64::from(s.va), u64::from(s.vsize.max(s.rsize)));
            if va <= rva && rva < va + span {
                let delta = rva - va;
                if delta >= u64::from(s.rsize) {
                    return None;
                }
                return usize::try_from(u64::from(s.rptr) + delta).ok();
            }
        }
        let lowest = self.sections.iter().map(|s| u64::from(s.va)).min()?;
        (rva < lowest).then(|| usize::try_from(rva).ok()).flatten()
    }

    /// A zero terminated Latin 1 string of at most 1024 bytes. `None` when unreadable.
    fn cstr(&self, rva: u64) -> Option<String> {
        let o = self.off(rva)?;
        let d = self.d();
        if o >= d.len() {
            return None;
        }
        let end = memchr0(&d[o..d.len().min(o + 1024)])?;
        Some(latin1(&d[o..o + end]))
    }

    fn thunks(&self, rva: u64) -> Vec<Func> {
        let mut out = Vec::new();
        let Some(mut o) = self.off(rva) else { return out };
        let size = if self.is64 { 8 } else { 4 };
        let flag = 1u64 << if self.is64 { 63 } else { 31 };
        let d = self.d();
        while o + size <= d.len() && out.len() < 200_000 {
            let value = if self.is64 { u64::from_le_bytes(d[o..o + 8].try_into().unwrap()) } else { u64::from(u32::from_le_bytes(d[o..o + 4].try_into().unwrap())) };
            if value == 0 {
                break;
            }
            if value & flag != 0 {
                out.push(Func::Ordinal((value & 0xFFFF) as u16));
            } else {
                let name = self.cstr((value & 0x7FFF_FFFF) + 2).filter(|n| !n.is_empty());
                out.push(Func::Name(name.unwrap_or_else(|| "<nom illisible>".into())));
            }
            o += size;
        }
        out
    }

    pub fn imports(&self) -> Vec<Import> {
        let d = self.d();
        let mut result = Vec::new();
        if let Some(&(rva, _)) = self.dirs.get(1).filter(|(rva, _)| *rva != 0) {
            let mut o = self.off(u64::from(rva));
            while let Some(at) = o.filter(|&at| at + 20 <= d.len() && result.len() < 4096) {
                let w = |i: usize| u32::from_le_bytes(d[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                let (oft, name_rva, ft) = (w(0), w(3), w(4));
                if oft == 0 && name_rva == 0 && ft == 0 {
                    break;
                }
                if let Some(dll) = self.cstr(u64::from(name_rva)).filter(|n| !n.is_empty()) {
                    let first = if oft != 0 { oft } else { ft };
                    result.push(Import { dll, delay: false, funcs: self.thunks(u64::from(first)) });
                }
                o = Some(at + 20);
            }
        }
        if let Some(&(rva, _)) = self.dirs.get(13).filter(|(rva, _)| *rva != 0) {
            let mut o = self.off(u64::from(rva));
            let mut count = 0;
            while let Some(at) = o.filter(|&at| at + 32 <= d.len() && count < 4096) {
                let w = |i: usize| u32::from_le_bytes(d[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                let (attrs, mut name_rva, mut int_rva) = (w(0), u64::from(w(1)), u64::from(w(4)));
                if name_rva == 0 && int_rva == 0 {
                    break;
                }
                if attrs & 1 == 0 {
                    name_rva = name_rva.wrapping_sub(self.image_base) & 0xFFFF_FFFF;
                    int_rva = int_rva.wrapping_sub(self.image_base) & 0xFFFF_FFFF;
                }
                if let Some(dll) = self.cstr(name_rva).filter(|n| !n.is_empty()) {
                    result.push(Import { dll, delay: true, funcs: self.thunks(int_rva) });
                }
                o = Some(at + 32);
                count += 1;
            }
        }
        result
    }

    /// The preferred load address.
    #[allow(dead_code)] // used by d3d12x_vtprobe
    pub fn image_base(&self) -> u64 {
        self.image_base
    }

    /// `len` bytes of the image at a relative virtual address, clipped to the raw data of its
    /// section. `None` when the address is not backed by file data.
    #[allow(dead_code)] // used by d3d12x_vtprobe
    pub fn bytes_at(&self, rva: u64, len: usize) -> Option<&[u8]> {
        let o = self.off(rva)?;
        let sec = self.sections.iter().find(|s| u64::from(s.va) <= rva && rva < u64::from(s.va) + u64::from(s.vsize.max(s.rsize)))?;
        let end_raw = usize::try_from(u64::from(sec.rptr) + u64::from(sec.rsize)).ok()?;
        let end = end_raw.min(self.d().len()).min(o.checked_add(len)?);
        (o < end).then(|| &self.d()[o..end])
    }

    /// Whether a relative virtual address lies in an executable section.
    #[allow(dead_code)] // used by d3d12x_vtprobe
    pub fn is_code(&self, rva: u64) -> bool {
        self.sections.iter().any(|s| s.chars & 0x2000_0000 != 0 && u64::from(s.va) <= rva && rva < u64::from(s.va) + u64::from(s.vsize))
    }

    /// Exported names with the relative virtual address of their code (forwarders skipped).
    #[allow(dead_code)] // used by d3d12x_vtprobe
    pub fn named_exports(&self) -> Vec<(String, u32)> {
        let mut out = Vec::new();
        let d = self.d();
        let Some(&(rva, size)) = self.dirs.first().filter(|(rva, _)| *rva != 0) else { return out };
        let Some(o) = self.off(u64::from(rva)).filter(|&o| o + 40 <= d.len()) else { return out };
        let w = |at: usize| d.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let h = |at: usize| d.get(at..at + 2).map(|b| u16::from_le_bytes(b.try_into().unwrap()));
        let (Some(nnames), Some(afuncs), Some(anames), Some(aords)) = (w(o + 24), w(o + 28), w(o + 32), w(o + 36)) else { return out };
        let (Some(fo), Some(no), Some(oo)) = (self.off(u64::from(afuncs)), self.off(u64::from(anames)), self.off(u64::from(aords))) else { return out };
        for i in 0..(nnames as usize).min(200_000) {
            let (Some(name_rva), Some(ord)) = (w(no + 4 * i), h(oo + 2 * i)) else { break };
            let Some(func) = w(fo + 4 * usize::from(ord)) else { continue };
            if func >= rva && func < rva + size {
                continue; // forwarder string, not code
            }
            if let Some(name) = self.cstr(u64::from(name_rva)).filter(|n| !n.is_empty()) {
                out.push((name, func));
            }
        }
        out
    }

    pub fn is64(&self) -> bool {
        self.is64
    }

    /// The whole file.
    pub fn bytes(&self) -> &[u8] {
        self.d()
    }

    /// The file offset of a relative virtual address backed by file data.
    pub fn file_offset(&self, rva: u64) -> Option<usize> {
        self.off(rva)
    }

    /// Every imported function with the RVA of its import address table entry, for normal and
    /// delayed imports: `(dll, function, IAT entry RVA)`. Code calls an import through that entry.
    pub fn iat_slots(&self) -> Vec<(String, Func, u32)> {
        let d = self.d();
        let size: u32 = if self.is64 { 8 } else { 4 };
        let mut out = Vec::new();
        let mut add = |dll: &str, names: u64, iat: u64| {
            for (i, f) in self.thunks(names).into_iter().enumerate() {
                if let Ok(slot) = u32::try_from(iat + i as u64 * u64::from(size)) {
                    out.push((dll.to_owned(), f, slot));
                }
            }
        };
        if let Some(&(rva, _)) = self.dirs.get(1).filter(|(rva, _)| *rva != 0) {
            let mut o = self.off(u64::from(rva));
            let mut count = 0;
            while let Some(at) = o.filter(|&at| at + 20 <= d.len() && count < 4096) {
                let w = |i: usize| u32::from_le_bytes(d[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                let (oft, name_rva, ft) = (w(0), w(3), w(4));
                if oft == 0 && name_rva == 0 && ft == 0 {
                    break;
                }
                if let Some(dll) = self.cstr(u64::from(name_rva)).filter(|n| !n.is_empty()) {
                    add(&dll, u64::from(if oft != 0 { oft } else { ft }), u64::from(ft));
                }
                o = Some(at + 20);
                count += 1;
            }
        }
        if let Some(&(rva, _)) = self.dirs.get(13).filter(|(rva, _)| *rva != 0) {
            let mut o = self.off(u64::from(rva));
            let mut count = 0;
            while let Some(at) = o.filter(|&at| at + 32 <= d.len() && count < 4096) {
                let w = |i: usize| u32::from_le_bytes(d[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
                let (attrs, mut name_rva, mut iat, mut int_rva) = (w(0), u64::from(w(1)), u64::from(w(3)), u64::from(w(4)));
                if name_rva == 0 && int_rva == 0 {
                    break;
                }
                if attrs & 1 == 0 {
                    name_rva = name_rva.wrapping_sub(self.image_base) & 0xFFFF_FFFF;
                    iat = iat.wrapping_sub(self.image_base) & 0xFFFF_FFFF;
                    int_rva = int_rva.wrapping_sub(self.image_base) & 0xFFFF_FFFF;
                }
                if let Some(dll) = self.cstr(name_rva).filter(|n| !n.is_empty()) {
                    add(&dll, int_rva, iat);
                }
                o = Some(at + 32);
                count += 1;
            }
        }
        out
    }

    /// Exported names and ordinals.
    pub fn exports(&self) -> (HashSet<String>, HashSet<u64>) {
        let (mut names, mut ordinals) = (HashSet::new(), HashSet::new());
        let d = self.d();
        let Some(&(rva, _)) = self.dirs.first().filter(|(rva, _)| *rva != 0) else { return (names, ordinals) };
        let Some(o) = self.off(u64::from(rva)).filter(|&o| o + 40 <= d.len()) else { return (names, ordinals) };
        let w = |at: usize| u32::from_le_bytes(d[at..at + 4].try_into().unwrap());
        let (base, nfuncs, nnames, afuncs, anames) = (w(o + 16), w(o + 20), w(o + 24), w(o + 28), w(o + 32));
        if let Some(fo) = self.off(u64::from(afuncs)) {
            for i in 0..(nfuncs as usize).min(200_000) {
                if fo + 4 * i + 4 > d.len() {
                    break;
                }
                if w(fo + 4 * i) != 0 {
                    ordinals.insert(u64::from(base) + i as u64);
                }
            }
        }
        if let Some(no) = self.off(u64::from(anames)) {
            for i in 0..(nnames as usize).min(200_000) {
                if no + 4 * i + 4 > d.len() {
                    break;
                }
                if let Some(name) = self.cstr(u64::from(w(no + 4 * i))).filter(|n| !n.is_empty()) {
                    names.insert(name);
                }
            }
        }
        (names, ordinals)
    }

    /// Size, code flag and entropy of the first 8 MiB of each section.
    pub fn section_report(&self) -> Vec<SectionRow> {
        let d = self.d();
        self.sections
            .iter()
            .map(|s| {
                let size = (s.rsize as usize).min(8 << 20);
                let start = s.rptr as usize;
                let entropy = if size > 0 && start + size <= d.len() { entropy_in_first_seen_order(&d[start..start + size]) } else { 0.0 };
                SectionRow { name: s.name.clone(), size_mib: s.rsize as f64 / 1_048_576.0, executable: s.chars & 0x2000_0000 != 0, entropy }
            })
            .collect()
    }
}

fn memchr0(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|&b| b == 0)
}

/// Shannon entropy, summed in the order bytes first appear, as Python's `Counter` iterates, so the
/// rounded result matches the Python report.
fn entropy_in_first_seen_order(block: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    let mut order = Vec::with_capacity(256);
    for &b in block {
        if counts[usize::from(b)] == 0 {
            order.push(b);
        }
        counts[usize::from(b)] += 1;
    }
    let size = block.len() as f64;
    -order.iter().map(|&b| counts[usize::from(b)] as f64 / size).map(|p| p * p.log2()).sum::<f64>()
}
