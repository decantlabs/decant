//! The resource bindings a console shader container declares: the register, register space and kind
//! of every resource, read from its `PSV0` part (the pipeline state validation record of the public
//! DXIL container format: runtime info, then a count and a size, then one record per resource of
//! type, space, lower bound, upper bound, kind and flags).
//!
//! The Direct3D debug layer reads the same records at pipeline creation ("Shader CBV descriptor range
//! (BaseShaderRegister=2, RegisterSpace=1)"), so they are the authority for the space a register of
//! the title's root signature lives in. `identify` knows the register of a descriptor but not its space.

/// One resource record of a container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    /// The register class letter: `t`, `u`, `b` or `s`.
    pub class: char,
    pub space: u32,
    pub lower: u32,
    pub upper: u32,
    /// DXIL `ResourceKind`: 1 to 9 textures, 10 typed buffer, 11 raw buffer, 12 structured buffer, 13 constant buffer, 14 sampler.
    pub kind: u32,
}

/// What the container says about a register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// No record has this class and register.
    Missing,
    /// One space.
    One(u32),
    /// Several spaces fit, the spaces in the order of the records: the register alone is ambiguous.
    Ambiguous(Vec<u32>),
}

fn rd32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// The resource records of the `PSV0` part of a container, `None` when there is none.
pub fn bindings(container: &[u8]) -> Option<Vec<Binding>> {
    let parts = decant_dxil::container::parts(container)?;
    let (_, psv) = parts.into_iter().find(|(f, _)| f == b"PSV0")?;
    let info = rd32(psv, 0)? as usize;
    let mut at = 4 + info;
    let count = rd32(psv, at)? as usize;
    at += 4;
    let mut out = Vec::new();
    if count == 0 {
        return Some(out);
    }
    let size = rd32(psv, at)? as usize;
    at += 4;
    // A record size or count that does not fit the part is not a resource table.
    if size < 16 || count.checked_mul(size).and_then(|n| n.checked_add(at)).is_none_or(|end| end > psv.len()) {
        return None;
    }
    for k in 0..count {
        let r = at + k * size;
        let ty = rd32(psv, r)?;
        let kind = if size >= 20 { rd32(psv, r + 16)? } else { 0 };
        let class = match ty {
            1 => 's',
            2 => 'b',
            3..=5 => 't',
            6..=9 => 'u',
            _ => continue,
        };
        out.push(Binding { class, space: rd32(psv, r + 4)?, lower: rd32(psv, r + 8)?, upper: rd32(psv, r + 12)?, kind });
    }
    Some(out)
}

/// Whether a record of kind `kind` can be what the translator calls `what` (`buffer`, `image`, `sampler`).
fn fits(kind: u32, what: &str) -> bool {
    match what {
        "buffer" => matches!(kind, 10..=13),
        "image" => matches!(kind, 1..=10 | 16),
        "sampler" => kind == 14,
        _ => true,
    }
}

/// The space of register `register` of class `class` for a descriptor the translator calls `what`.
/// Records of the wrong kind only count when none of the right kind has the register.
pub fn lookup(list: &[Binding], class: char, register: u32, what: &str) -> Lookup {
    let hits: Vec<&Binding> = list.iter().filter(|b| b.class == class && b.lower <= register && register <= b.upper).collect();
    let right: Vec<&Binding> = hits.iter().copied().filter(|b| fits(b.kind, what)).collect();
    let chosen = if right.is_empty() { hits } else { right };
    let mut spaces: Vec<u32> = Vec::new();
    for b in chosen {
        if !spaces.contains(&b.space) {
            spaces.push(b.space);
        }
    }
    match spaces.as_slice() {
        [] => Lookup::Missing,
        [one] => Lookup::One(*one),
        _ => Lookup::Ambiguous(spaces),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use decant_dxil::container::{PsvResource, assemble, psv0_compute};

    fn container(resources: &[PsvResource]) -> Vec<u8> {
        assemble(&[(*b"PSV0", psv0_compute([1, 1, 1], None, resources))])
    }

    fn res(ty: u32, space: u32, register: u32, kind: u32) -> PsvResource {
        PsvResource { ty, space, lower: register, upper: register, kind, flags: 0 }
    }

    #[test]
    fn the_records_give_class_register_space_and_kind() {
        let c = container(&[res(2, 1, 27, 13), res(5, 0, 11, 12), res(3, 1, 334, 10), res(6, 1, 28, 10), res(1, 1, 3, 14)]);
        let list = bindings(&c).unwrap();
        assert_eq!(list.len(), 5);
        assert_eq!(list[0], Binding { class: 'b', space: 1, lower: 27, upper: 27, kind: 13 });
        assert_eq!(lookup(&list, 'b', 27, "buffer"), Lookup::One(1));
        assert_eq!(lookup(&list, 't', 11, "buffer"), Lookup::One(0));
        assert_eq!(lookup(&list, 'u', 28, "image"), Lookup::One(1));
        assert_eq!(lookup(&list, 's', 3, "sampler"), Lookup::One(1));
        assert_eq!(lookup(&list, 't', 12, "buffer"), Lookup::Missing);
    }

    #[test]
    fn a_register_in_two_spaces_is_ambiguous_unless_the_kind_decides() {
        let c = container(&[res(5, 0, 7, 12), res(3, 1, 7, 2)]);
        let list = bindings(&c).unwrap();
        assert_eq!(lookup(&list, 't', 7, "buffer"), Lookup::One(0));
        assert_eq!(lookup(&list, 't', 7, "image"), Lookup::One(1));
        let both = bindings(&container(&[res(5, 0, 7, 12), res(5, 2, 7, 12)])).unwrap();
        assert_eq!(lookup(&both, 't', 7, "buffer"), Lookup::Ambiguous(vec![0, 2]));
    }

    #[test]
    fn a_container_without_psv0_has_no_bindings() {
        assert!(bindings(&assemble(&[(*b"HASH", vec![0; 20])])).is_none());
        assert!(bindings(b"not a container").is_none());
    }
}
