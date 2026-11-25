//! The console shader container: a DXBC container whose DXIL part carries the console GPU code
//! (an XDAT table, then an XBCH part holding a zlib stream with the FSSM block).

/// One DXBC part: its four letter tag, where its body starts and its length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub tag: String,
    pub offset: usize,
    pub size: usize,
}

/// A whole zlib stream, or `None` when it is cut or damaged.
fn zlib_all(input: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(input).read_to_end(&mut out).ok()?;
    Some(out)
}

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

/// The parts of a DXBC container, empty when it is not one or its table is out of bounds.
pub fn chunks(blob: &[u8]) -> Vec<Chunk> {
    if blob.len() < 32 || &blob[..4] != b"DXBC" {
        return Vec::new();
    }
    let count = u32_at(blob, 28).unwrap_or(0) as usize;
    if count > 64 || 32 + 4 * count > blob.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let offset = u32_at(blob, 32 + 4 * i).unwrap_or(u32::MAX) as usize;
        let Some(size) = offset.checked_add(4).and_then(|at| u32_at(blob, at)) else { return Vec::new() };
        let tag = blob[offset..offset + 4].iter().map(|&b| char::from(b)).collect();
        out.push(Chunk { tag, offset: offset + 8, size: size as usize });
    }
    out
}

/// The pipeline stage and shader model of a DXBC container, as `("pixel", 6, 6)`.
pub fn kind(blob: &[u8]) -> (String, u8, u8) {
    for c in chunks(blob) {
        if c.tag == "DXIL" && c.size >= 8
            && let Some(version) = u32_at(blob, c.offset)
        {
            let stage = match version >> 16 {
                0 => "pixel".to_owned(),
                1 => "vertex".to_owned(),
                2 => "geometry".to_owned(),
                3 => "hull".to_owned(),
                4 => "domain".to_owned(),
                5 => "compute".to_owned(),
                6 => "library".to_owned(),
                13 => "mesh".to_owned(),
                14 => "amplification".to_owned(),
                other => format!("type {other}"),
            };
            return (stage, ((version >> 4) & 15) as u8, (version & 15) as u8);
        }
    }
    ("?".to_owned(), 0, 0)
}

/// The FSSM block with the console GPU code, or `None` when the container has none.
pub fn gpu_code(blob: &[u8]) -> Option<Vec<u8>> {
    find_gpu_code(blob).ok()
}

/// Why a container gives no GPU code, from the outside in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuGap {
    /// Not a DXBC container, or its part table is out of bounds.
    NotDxbc,
    /// A container without a DXIL part.
    NoDxil,
    /// A DXIL part without an `XDAT` table.
    NoXdat,
    /// The `XDAT` table is cut short: its header, or an entry the count announces.
    XdatCut,
    /// The `XDAT` entries hold no `XBCH` part.
    NoXbch,
    /// An `XBCH` part whose 24 byte header runs past the DXIL part.
    XbchCut,
    /// The zlib stream after the `XBCH` header is cut or damaged.
    Inflate,
}

/// RDNA2 `s_endpgm`.
pub const S_ENDPGM: u32 = 0xBF81_0000;
/// The `s_code_end` words that pad the end of the code.
const CODE_END: u32 = 0xBF9F_0000;

/// True for any word encoded as `s_setpc_b64` (SOP1, opcode 0x20), whatever its operands.
pub fn is_setpc_b64(word: u32) -> bool {
    word & 0xFF80_FF00 == 0xBE80_2000
}

/// True when the code of an FSSM block ends with `s_endpgm` and never jumps to another program.
/// The test reads raw words, literal constants included:
/// a constant that looks like `s_setpc_b64` can only reject a program, never accept one that jumps.
pub fn fssm_standalone(fssm: &[u8]) -> bool {
    if fssm.len() < 0xD0 || &fssm[..4] != b"FSSM" {
        return false;
    }
    let read = |at: usize| u64::from_le_bytes(fssm[at..at + 8].try_into().unwrap());
    let (offset, size) = (read(0xC0), read(0xC8));
    let end = offset.checked_add(16).and_then(|e| e.checked_add(size));
    if size % 4 != 0 || offset < 0xD0 || end.is_none_or(|e| e > fssm.len() as u64) {
        return false;
    }
    let start = offset as usize + 16;
    let mut words: Vec<u32> = fssm[start..start + size as usize].as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    while words.last() == Some(&CODE_END) {
        words.pop();
    }
    words.last() == Some(&S_ENDPGM) && !words.iter().any(|&w| is_setpc_b64(w))
}

/// The FSSM block with the console GPU code, or where the path to it stops. Every DXIL part is tried in order, but one without an
/// `XDAT` table ends the search.
///
/// A container usually holds one `XBCH` part. A few hold two: the first program writes to LDS and
/// jumps (`s_setpc_b64`) to the geometry shader, the second stands alone and exports its outputs.
/// Only then, when the first is not standalone, the first standalone program after it is returned.
pub fn find_gpu_code(blob: &[u8]) -> Result<Vec<u8>, GpuGap> {
    let parts = chunks(blob);
    if parts.is_empty() {
        return Err(GpuGap::NotDxbc);
    }
    let mut gap = GpuGap::NoDxil;
    for c in parts.iter().filter(|c| c.tag == "DXIL") {
        let end = c.offset.saturating_add(c.size).min(blob.len());
        let body = blob.get(c.offset..end).unwrap_or_default();
        let Some(x) = memchr::memmem::find(body, b"XDAT") else { return Err(GpuGap::NoXdat) };
        if x + 12 > body.len() {
            return Err(GpuGap::XdatCut);
        }
        let count = u32_at(body, x + 4).unwrap_or(0) as usize;
        gap = GpuGap::NoXbch;
        let mut first: Option<Vec<u8>> = None;
        for i in 0..count.min(16) {
            if x + 24 + 12 * i > body.len() {
                if first.is_none() {
                    gap = GpuGap::XdatCut;
                }
                break;
            }
            let entry = x + 12 + 12 * i;
            let (kind, part) = (u32_at(body, entry).unwrap_or(0), u32_at(body, entry + 4).unwrap_or(0) as usize);
            if kind != 0x0D || body.get(part..part + 4) != Some(b"XBCH") {
                continue;
            }
            if part + 24 > body.len() {
                if first.is_none() {
                    gap = GpuGap::XbchCut;
                }
                continue;
            }
            let comp = u64::from_le_bytes(body[part + 8..part + 16].try_into().unwrap()) as usize;
            let end = part.saturating_add(24).saturating_add(comp).min(body.len());
            let inflated = zlib_all(&body[part + 24..end]);
            match (&first, inflated) {
                (None, inflated) => first = Some(inflated.ok_or(GpuGap::Inflate)?),
                (Some(head), _) if fssm_standalone(head) => break,
                (Some(_), Some(code)) if fssm_standalone(&code) => return Ok(code),
                (Some(_), _) => {}
            }
        }
        if let Some(first) = first {
            return Ok(first);
        }
    }
    Err(gap)
}

#[cfg(test)]
#[path = "container_tests.rs"]
mod tests;
