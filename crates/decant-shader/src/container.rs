//! The console shader container: a DXBC container whose `DXIL` part holds an `XDAT` table with
//! a zlib packed `FSSM` block instead of bitcode. This file turns one into a [`StageProgram`]:
//! the code words, the `ISG1` and `OSG1` signatures and the resource list of `PSV0`.
//!
//! The container walk itself is `decant_extract_shaders::container` and `decant_extract_shaders::fssm`, our own code.

use decant_extract_shaders::container as shader;

use crate::api::{
    DeclaredResource, GeometryInfo, ParamExport, PixelInput, RegisterClass, SigElement, Stage,
    StageProgram, TableBinding, UserDataKind, UserDataSlot,
};
use crate::refusal::{Refusal, RefusalKind};

/// Every program starts with six words that are not code: two instructions that do nothing
/// (`v_nop` in the two word encoding) with a data word after each. Measured on the 26,503
/// programs of the title: word 6 is always the first instruction.
pub const SIGNATURE_WORDS: usize = 6;

/// The hash the layer names override files by: FNV-1a, 64 bit, over the whole container.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Whether `code` starts with the signature.
pub fn has_signature(code: &[u32]) -> bool {
    code.len() > SIGNATURE_WORDS && code[0] >> 16 == 0xD580 && code[3] >> 16 == 0xD580
}

fn refuse(what: &'static str) -> Refusal {
    Refusal::new(RefusalKind::Container(what), "")
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at.checked_add(4)?)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

fn part<'a>(blob: &'a [u8], tag: &str) -> Option<&'a [u8]> {
    // A later part of the same name wins, as in our research tools.
    shader::chunks(blob)
        .into_iter()
        .rfind(|c| c.tag == tag)
        .and_then(|c| blob.get(c.offset..c.offset.checked_add(c.size)?.min(blob.len())))
}

/// The elements of an `ISG1` or `OSG1` part: a count, the offset of the first element, then 32
/// byte elements whose name is a zero ended string at an offset in the part.
pub fn signature(sig: &[u8]) -> Option<Vec<SigElement>> {
    let count = u32_at(sig, 0)? as usize;
    let first = u32_at(sig, 4)? as usize;
    let mut out = Vec::with_capacity(count.min(64));
    for i in 0..count {
        let at = first.checked_add(i.checked_mul(32)?)?;
        let name = u32_at(sig, at + 4)? as usize;
        let rest = sig.get(name..)?;
        let end = rest.iter().position(|&b| b == 0)?;
        out.push(SigElement {
            semantic: rest[..end].iter().map(|&b| char::from(b)).collect(),
            index: u32_at(sig, at + 8)?,
            system_value: u32_at(sig, at + 12)?,
            component_type: u32_at(sig, at + 16)?,
            register: u32_at(sig, at + 20)?,
            mask: *sig.get(at + 24)?,
            rw_mask: *sig.get(at + 25)?,
        });
    }
    Some(out)
}

/// The resource records of a `PSV0` part: after the runtime information (its size is the first
/// word) a count, a record size, then records of type, space, first register, last register and,
/// in the long form, kind and flags.
pub fn psv0_resources(psv: &[u8]) -> Option<Vec<DeclaredResource>> {
    let mut at = 4 + u32_at(psv, 0)? as usize;
    let count = u32_at(psv, at)? as usize;
    at += 4;
    if count == 0 {
        return Some(Vec::new());
    }
    let stride = u32_at(psv, at)? as usize;
    at += 4;
    if stride < 16 {
        return None;
    }
    let mut out = Vec::with_capacity(count.min(256));
    for i in 0..count {
        let rec = at.checked_add(i.checked_mul(stride)?)?;
        let psv_type = u32_at(psv, rec)?;
        let class = match psv_type {
            1 => RegisterClass::Sampler,
            2 => RegisterClass::Cbv,
            3..=5 => RegisterClass::Srv,
            6..=9 => RegisterClass::Uav,
            _ => return None,
        };
        out.push(DeclaredResource {
            class,
            space: u32_at(psv, rec + 4)?,
            lower: u32_at(psv, rec + 8)?,
            upper: u32_at(psv, rec + 12)?,
            psv_type,
            psv_kind: if stride >= 24 {
                u32_at(psv, rec + 16)?
            } else {
                0
            },
        });
    }
    Some(out)
}

/// The binding table of an FSSM block (its table B): the position is at 0xF0 (counted from byte
/// 16 of the block), the record count at 0xFC, a record is five dwords: kind, offset in dwords,
/// register, an extra word and flags with the table register in bits 8 to 15. Kinds 0x300 (`t`),
/// 0x302 (`b`), 0x303 (`u`) and 0x401 (`s`) are identified, records of other kinds are left out.
pub fn binding_table(fssm: &[u8]) -> Vec<TableBinding> {
    let (Some(lo), Some(hi), Some(count)) =
        (u32_at(fssm, 0xF0), u32_at(fssm, 0xF4), u32_at(fssm, 0xFC))
    else {
        return Vec::new();
    };
    let start = (u64::from(hi) << 32) | u64::from(lo);
    if start > fssm.len() as u64 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = start as usize + 16 + 20 * i;
        let (Some(kind), Some(word_offset), Some(register), Some(flags)) = (
            u32_at(fssm, at),
            u32_at(fssm, at + 4),
            u32_at(fssm, at + 8),
            u32_at(fssm, at + 16),
        ) else {
            break;
        };
        let class = match kind {
            0x300 => RegisterClass::Srv,
            0x302 => RegisterClass::Cbv,
            0x303 => RegisterClass::Uav,
            0x401 => RegisterClass::Sampler,
            _ => continue,
        };
        out.push(TableBinding {
            class,
            register,
            sgpr: ((flags >> 8) & 0xFF) as u8,
            word_offset,
        });
    }
    out
}

/// The thread group size in a `PSV0` part: three dwords at bytes 36, 40 and 44 of the runtime
/// information, when it is at least 48 bytes long (all compute containers of the title
/// have 52).
pub fn threads(psv: &[u8]) -> Option<[u32; 3]> {
    if u32_at(psv, 0)? < 48 {
        return None;
    }
    Some([u32_at(psv, 40)?, u32_at(psv, 44)?, u32_at(psv, 48)?])
}

/// The geometry fields of the runtime information of a `PSV0` part (the bytes after its size
/// word): the input primitive at byte 0, the output topology at 4, the stream mask at 8, the
/// largest vertex count as a 16 bit number at 26.
pub fn geometry_info(psv: &[u8]) -> Option<GeometryInfo> {
    if u32_at(psv, 0)? < 28 {
        return None;
    }
    let max = psv.get(4 + 26..4 + 28)?;
    Some(GeometryInfo {
        input_primitive: u32_at(psv, 4)?,
        output_topology: u32_at(psv, 8)?,
        stream_mask: u32_at(psv, 12)?,
        max_vertices: u32::from(u16::from_le_bytes([max[0], max[1]])),
    })
}

/// The root layout of an FSSM block (its table A): the position is at 0xE8 (counted from byte 16
/// of the block), the record count at 0xF8, a record is five dwords. Record kinds, measured on
/// all containers of the title: 3 a descriptor in a table (several records per table), 0 root
/// constants (dword 2 the count, dword 3 the `b` register), 1 a root shader resource view (dword
/// 3 the `t` register), 5 the register with the resource heap base, 6 the one with the sampler
/// heap base. The register is the low byte of dword 4. Other kinds are left out.
pub fn user_data(fssm: &[u8]) -> Vec<UserDataSlot> {
    let (Some(lo), Some(hi), Some(count)) =
        (u32_at(fssm, 0xE8), u32_at(fssm, 0xEC), u32_at(fssm, 0xF8))
    else {
        return Vec::new();
    };
    let start = (u64::from(hi) << 32) | u64::from(lo);
    if start > fssm.len() as u64 {
        return Vec::new();
    }
    let mut out: Vec<UserDataSlot> = Vec::new();
    for i in 0..count as usize {
        let at = start as usize + 16 + 20 * i;
        let (Some(kind), Some(count), Some(register), Some(flags)) = (
            u32_at(fssm, at),
            u32_at(fssm, at + 8),
            u32_at(fssm, at + 12),
            u32_at(fssm, at + 16),
        ) else {
            break;
        };
        let kind = match kind {
            3 => UserDataKind::Table,
            0 => UserDataKind::Constants { register, count },
            1 => UserDataKind::RootSrv { register },
            5 => UserDataKind::ResourceHeap,
            6 => UserDataKind::SamplerHeap,
            _ => continue,
        };
        let slot = UserDataSlot {
            sgpr: (flags & 0xFF) as u8,
            kind,
        };
        // A table has one record per descriptor: one slot is enough.
        if !out.contains(&slot) {
            out.push(slot);
        }
    }
    out
}

/// The parameter export table of an FSSM 1.1 block: records of 6 bytes from byte 0x1DC, up to
/// 32, ended by a record whose kind is not 6 or 7. Bytes: kind, signature register, slot, the
/// register again when it is a 16 bit register, 0, flags.
pub fn param_exports(fssm: &[u8]) -> Vec<ParamExport> {
    let mut out = Vec::new();
    for i in 0..32usize {
        let Some(r) = fssm.get(0x1DC + 6 * i..0x1DC + 6 * i + 6) else {
            break;
        };
        if r[0] != 6 && r[0] != 7 {
            break;
        }
        out.push(ParamExport {
            slot: r[2],
            kind: r[0],
            register: u32::from(r[1]),
            sixteen_bit_register: u32::from(r[3]),
            flags: r[5],
        });
    }
    out
}

/// The pixel input table of an FSSM 1.1 block: 32 records of 8 bytes at byte 0x138. A record
/// whose first byte is 0 is unused. Bytes: kind, signature register, channel mask, attribute
/// number (equal to the record position), 0, the second register of a packed 16 bit pair (equal
/// to the register for a single 16 bit element, else 0), one unknown byte, flags.
pub fn pixel_inputs(fssm: &[u8]) -> Vec<PixelInput> {
    let mut out = Vec::new();
    for slot in 0..32usize {
        let Some(r) = fssm.get(0x138 + 8 * slot..0x138 + 8 * slot + 8) else {
            break;
        };
        if r[0] == 0 {
            continue;
        }
        let sixteen = r[7] & 0x08 != 0;
        out.push(PixelInput {
            attr: r[3],
            kind: r[0],
            register: u32::from(r[1]),
            high_register: (sixteen && r[5] != r[1]).then_some(u32::from(r[5])),
            channels: r[2],
            flags: r[7],
        });
    }
    out
}

/// Reads one console container. `name` is only kept for messages.
pub fn program(blob: &[u8], name: &str) -> Result<StageProgram, Refusal> {
    let stage = match shader::kind(blob).0.as_str() {
        "pixel" => Stage::Pixel,
        "vertex" => Stage::Vertex,
        "geometry" => Stage::Geometry,
        "hull" => Stage::Hull,
        "domain" => Stage::Domain,
        "compute" => Stage::Compute,
        "mesh" => Stage::Mesh,
        "amplification" => Stage::Amplification,
        _ => return Err(refuse("stage")),
    };
    let fssm = shader::find_gpu_code(blob).map_err(|_| refuse("no-fssm"))?;
    let code = decant_extract_shaders::fssm::split(&fssm).map_err(|_| refuse("fssm-bounds"))?;
    let inputs = part(blob, "ISG1")
        .and_then(signature)
        .ok_or_else(|| refuse("isg1"))?;
    let outputs = part(blob, "OSG1")
        .and_then(signature)
        .ok_or_else(|| refuse("osg1"))?;
    let resources = part(blob, "PSV0")
        .and_then(psv0_resources)
        .ok_or_else(|| refuse("psv0"))?;
    Ok(StageProgram {
        stage,
        code: code.words,
        inputs,
        outputs,
        resources,
        binding_table: binding_table(&fssm),
        threads: if matches!(stage, Stage::Compute | Stage::Mesh | Stage::Amplification) {
            part(blob, "PSV0").and_then(threads).unwrap_or([0; 3])
        } else {
            [0; 3]
        },
        // Measured on all compute programs: the dword at 0x10C is not zero in exactly the
        // the programs with local data share instructions, and never below the highest constant
        // offset they use. The five mesh programs with such memory agree with their `PSV0`.
        lds_bytes: u32_at(&fssm, 0x10C).unwrap_or(0),
        geometry: if stage == Stage::Geometry {
            part(blob, "PSV0").and_then(geometry_info)
        } else {
            None
        },
        user_data: user_data(&fssm),
        param_exports: param_exports(&fssm),
        pixel_inputs: if stage == Stage::Pixel {
            pixel_inputs(&fssm)
        } else {
            Vec::new()
        },
        name: name.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_input_records_read_back() {
        let mut fssm = vec![0u8; 0x138 + 32 * 8];
        // Attribute 0: register 1, all channels, perspective. Attribute 2: registers 3 and 4 packed.
        fssm[0x138..0x140].copy_from_slice(&[6, 1, 0xF, 0, 0, 0, 0, 0x02]);
        fssm[0x148..0x150].copy_from_slice(&[6, 3, 0xF, 2, 0, 4, 0, 0x0A]);
        let t = pixel_inputs(&fssm);
        assert_eq!(t.len(), 2);
        assert_eq!(
            (t[0].attr, t[0].register, t[0].high_register, t[0].flags),
            (0, 1, None, 2)
        );
        assert_eq!(
            (t[1].attr, t[1].register, t[1].high_register, t[1].channels),
            (2, 3, Some(4), 0xF)
        );
        assert!(pixel_inputs(&fssm[..0x100]).is_empty());
    }

    #[test]
    fn the_hash_is_fnv1a() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn a_signature_part_reads_back() {
        // One element: TEXCOORD 3 in register 2, components xy, float.
        let mut sig = Vec::new();
        for v in [1u32, 8, 0, 40, 3, 0, 3, 2] {
            sig.extend(v.to_le_bytes());
        }
        sig.extend([0x03, 0x01, 0, 0, 0, 0, 0, 0]);
        sig.extend(b"TEXCOORD\0");
        let e = &signature(&sig).unwrap()[0];
        assert_eq!(
            (
                e.semantic.as_str(),
                e.index,
                e.register,
                e.mask,
                e.rw_mask,
                e.component_type
            ),
            ("TEXCOORD", 3, 2, 3, 1, 3)
        );
    }

    #[test]
    fn psv0_records_read_back() {
        let mut psv = 8u32.to_le_bytes().to_vec();
        psv.extend([0u8; 8]);
        for v in [1u32, 24, 2, 1, 27, 27, 13, 0] {
            psv.extend(v.to_le_bytes());
        }
        let r = psv0_resources(&psv).unwrap();
        assert_eq!(
            r,
            [DeclaredResource {
                class: RegisterClass::Cbv,
                lower: 27,
                upper: 27,
                space: 1,
                psv_type: 2,
                psv_kind: 13
            }]
        );
    }
}
