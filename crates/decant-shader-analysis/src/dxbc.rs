//! The DXBC parts the research scripts read: the `ISG1` and `OSG1` signatures, the `PSV0`
//! runtime information and the `SFI0` feature flags. Every bound that would raise in Python
//! (`struct.error`, `KeyError`, `ValueError`) is an error here.

use std::collections::HashMap;

use decant_extract_shaders::container as shader;

use crate::py::Py;

/// A little endian `u32`, or the error Python's `struct.unpack_from` raises.
pub fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    at.checked_add(4)
        .and_then(|end| b.get(at..end))
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .ok_or_else(|| format!("unpack_from: no 4 bytes at {at} in a {} byte buffer", b.len()))
}

pub fn u64_at(b: &[u8], at: usize) -> Result<u64, String> {
    at.checked_add(8)
        .and_then(|end| b.get(at..end))
        .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        .ok_or_else(|| format!("unpack_from: no 8 bytes at {at} in a {} byte buffer", b.len()))
}

pub fn u16_at(b: &[u8], at: usize) -> Result<u16, String> {
    at.checked_add(2)
        .and_then(|end| b.get(at..end))
        .map(|w| u16::from_le_bytes(w.try_into().unwrap()))
        .ok_or_else(|| format!("unpack_from: no 2 bytes at {at} in a {} byte buffer", b.len()))
}

/// `{name: blob[pos:pos + n]}` over the DXBC parts: a later part of the same name wins, and the
/// slice stops at the end of the container.
pub fn blocks(blob: &[u8]) -> HashMap<String, &[u8]> {
    shader::chunks(blob)
        .into_iter()
        .map(|c| {
            let start = c.offset.min(blob.len());
            let end = c.offset.saturating_add(c.size).min(blob.len());
            (c.tag, &blob[start..end.max(start)])
        })
        .collect()
}

/// The part `name`, or the `KeyError` of `blocks[name]`.
pub fn block<'a>(blocks: &HashMap<String, &'a [u8]>, name: &str) -> Result<&'a [u8], String> {
    blocks.get(name).copied().ok_or_else(|| format!("KeyError: '{name}'"))
}

/// One signature element, in the order the scripts list it:
/// `[semantic, index, register, mask, read or written mask, system value, type, stream, precision]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Element {
    pub semantic: String,
    pub index: u32,
    pub register: u32,
    pub mask: u8,
    pub used: u8,
    pub system: u32,
    pub component: u32,
    pub stream: u32,
    pub precision: u32,
}

impl Element {
    pub fn py(&self) -> Py {
        Py::List(vec![
            self.semantic.clone().into(),
            self.index.into(),
            self.register.into(),
            u32::from(self.mask).into(),
            u32::from(self.used).into(),
            self.system.into(),
            self.component.into(),
            self.stream.into(),
            self.precision.into(),
        ])
    }
}

/// An `ISG1` or `OSG1` part: a count, the offset of the first element, then 32 byte elements
/// `<6IBBHI` whose name is a zero ended ASCII string at an offset in the part.
pub fn signature(sig: &[u8]) -> Result<Vec<Element>, String> {
    let count = u32_at(sig, 0)?;
    let pos = u32_at(sig, 4)? as usize;
    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = pos + i * 32;
        if sig.len() < at + 32 {
            return Err(format!("unpack_from: element {i} at {at} runs past a {} byte signature", sig.len()));
        }
        let w = |k: usize| u32_at(sig, at + 4 * k);
        let (stream, name, index, system, component, register) = (w(0)?, w(1)? as usize, w(2)?, w(3)?, w(4)?, w(5)?);
        let (mask, used, precision) = (sig[at + 24], sig[at + 25], w(7)?);
        let end = sig.get(name..).and_then(|rest| rest.iter().position(|&b| b == 0)).ok_or("ValueError: subsection not found")?;
        let bytes = &sig[name..name + end];
        if !bytes.is_ascii() {
            return Err("UnicodeDecodeError: 'ascii' codec can't decode a signature name".to_owned());
        }
        let semantic = bytes.iter().map(|&b| char::from(b)).collect();
        out.push(Element { semantic, index, register, mask, used, system, component, stream, precision });
    }
    Ok(out)
}

/// One `PSV0` resource binding: `[prefix, lower, upper, space, type, shape, flags]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub prefix: &'static str,
    pub lower: u32,
    pub upper: u32,
    pub space: u32,
    pub kind: u32,
    pub shape: u32,
    pub flags: u32,
}

impl Binding {
    pub fn py(&self) -> Py {
        Py::List(vec![
            self.prefix.into(),
            self.lower.into(),
            self.upper.into(),
            self.space.into(),
            self.kind.into(),
            self.shape.into(),
            self.flags.into(),
        ])
    }
}

/// The register letter of a `PSV0` resource type.
pub fn prefix(kind: u32) -> Option<&'static str> {
    match kind {
        1 => Some("s"),
        2 => Some("b"),
        3..=5 => Some("t"),
        6..=9 => Some("u"),
        _ => None,
    }
}

/// The resource bindings after the runtime information of `PSV0`, whose size `ri` is its first word.
pub fn psv0_bindings(p: &[u8], ri: u32) -> Result<Vec<Binding>, String> {
    let mut pos = 4 + ri as usize;
    let count = u32_at(p, pos)?;
    pos += 4;
    let mut out = Vec::new();
    if count == 0 {
        return Ok(out);
    }
    let stride = u32_at(p, pos)? as usize;
    pos += 4;
    if stride < 24 || pos as u64 + u64::from(count) * stride as u64 > p.len() as u64 {
        return Err("AssertionError: PSV0 resource table out of bounds".to_owned());
    }
    for i in 0..count as usize {
        let w = |k: usize| u32_at(p, pos + i * stride + 4 * k);
        let (kind, space, lower, upper, shape, flags) = (w(0)?, w(1)?, w(2)?, w(3)?, w(4)?, w(5)?);
        let Some(prefix) = prefix(kind).filter(|_| upper >= lower) else {
            return Err(format!("AssertionError: PSV0 resource type {kind}, registers {lower} to {upper}"));
        };
        out.push(Binding { prefix, lower, upper, space, kind, shape, flags });
    }
    Ok(out)
}

/// `(int.from_bytes(sfi0, "little") >> 25) & 3`.
pub fn heap_flags(sfi0: &[u8]) -> u32 {
    sfi0.get(3).map_or(0, |b| u32::from(b >> 1) & 3)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An invented signature part with the given elements `(name, register, mask, used, system)`.
    pub fn signature_part(elements: &[(&str, u32, u8, u8, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend((elements.len() as u32).to_le_bytes());
        out.extend(8u32.to_le_bytes());
        let names_at = 8 + 32 * elements.len();
        let mut names = Vec::new();
        for (i, (name, register, mask, used, system)) in elements.iter().enumerate() {
            let words = [i as u32 % 2, (names_at + names.len()) as u32, i as u32, *system, 3, *register];
            out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
            out.extend([*mask, *used, 0, 0]);
            out.extend(7u32.to_le_bytes());
            names.extend(name.as_bytes());
            names.push(0);
        }
        out.extend(names);
        out
    }

    #[test]
    fn signature_reads_elements_in_the_script_order() {
        let part = signature_part(&[("POSITION", 0, 0xF, 0x7, 0), ("SV_VertexID", 1, 1, 1, 6)]);
        let elements = signature(&part).unwrap();
        assert_eq!(elements[0].py().repr(), "['POSITION', 0, 0, 15, 7, 0, 3, 0, 7]");
        assert_eq!(elements[1].py().repr(), "['SV_VertexID', 1, 1, 1, 1, 6, 3, 1, 7]");
        assert!(signature(&part[..40]).is_err(), "a cut element raises");
        let mut unnamed = part.clone();
        unnamed.truncate(part.len() - 1);
        assert!(signature(&unnamed).is_err(), "a name without its zero raises");
    }

    #[test]
    fn heap_flags_are_bits_25_and_26() {
        assert_eq!(heap_flags(&[]), 0);
        assert_eq!(heap_flags(&[0xFF, 0xFF, 0xFF]), 0);
        assert_eq!(heap_flags(&[0, 0, 0, 0b0000_0110, 0xFF]), 3);
        assert_eq!(heap_flags(&[0, 0, 0, 0b0000_0010]), 1);
    }
}
