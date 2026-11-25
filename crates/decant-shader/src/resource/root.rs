//! The console packing of a root signature into user data registers.
//!
//! Measured by the layer (`d3d12-xs`, `user_mirror.rs`) and on the corpus: the root parameters
//! take scalar registers in their order, a root constant one register each, a root descriptor two
//! (the low and the high half of its address), a descriptor table one (the low half of the table
//! address, the program supplies a zero high half). Compute programs start at `s4`, the graphics
//! stages at `s8`, and the registers end at `s31`.

use crate::api::{RootParam, RootParamKind, RootSignature, Stage};

/// What a user data register holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot<'a> {
    pub param: &'a RootParam,
    /// The dword of the parameter this register is: the constant number, or 0 and 1 for the
    /// halves of a root descriptor.
    pub dword: u32,
}

fn dwords(kind: &RootParamKind) -> u32 {
    match kind {
        RootParamKind::Table(_) => 1,
        RootParamKind::Constants { count, .. } => *count,
        RootParamKind::Descriptor { .. } => 2,
    }
}

/// The parameter behind user data register `sgpr` (a machine register) of a program of `stage`.
pub fn slot(root: &RootSignature, stage: Stage, sgpr: u8) -> Option<Slot<'_>> {
    let registers = stage.user_registers();
    let mut next = u32::from(registers.start);
    for param in &root.params {
        let n = dwords(&param.kind);
        if next + n > u32::from(registers.end) {
            return None;
        }
        if (next..next + n).contains(&u32::from(sgpr)) {
            return Some(Slot {
                param,
                dword: u32::from(sgpr) - next,
            });
        }
        next += n;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::RegisterClass;

    #[test]
    fn parameters_pack_in_order() {
        let root = RootSignature {
            params: vec![
                RootParam {
                    kind: RootParamKind::Table(Vec::new()),
                    visibility: 0,
                },
                RootParam {
                    kind: RootParamKind::Constants {
                        register: 2,
                        space: 0,
                        count: 3,
                    },
                    visibility: 0,
                },
                RootParam {
                    kind: RootParamKind::Descriptor {
                        class: RegisterClass::Cbv,
                        register: 5,
                        space: 0,
                    },
                    visibility: 0,
                },
            ],
            ..RootSignature::default()
        };
        assert!(matches!(
            slot(&root, Stage::Pixel, 8).unwrap().param.kind,
            RootParamKind::Table(_)
        ));
        assert_eq!(slot(&root, Stage::Pixel, 11).unwrap().dword, 2);
        assert_eq!(slot(&root, Stage::Pixel, 13).unwrap().dword, 1);
        assert!(slot(&root, Stage::Pixel, 14).is_none());
        // Compute starts four registers earlier, mesh eight later.
        assert_eq!(slot(&root, Stage::Compute, 9).unwrap().dword, 1);
        assert_eq!(slot(&root, Stage::Mesh, 21).unwrap().dword, 1);
        assert!(slot(&root, Stage::Mesh, 13).is_none());
    }
}
