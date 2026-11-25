//! Root signatures: a synthesized [`Root`] serialized for D3D12, and a given serialized root
//! signature read back into a [`Root`] through the D3D12 deserializer.

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::model::Class;
use crate::plan::{Param, Range, Res, Root, Sampler};

fn blob_bytes(b: &ID3DBlob) -> Vec<u8> {
    // SAFETY: the blob owns its buffer.
    unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()).to_vec() }
}

fn range_type(c: Class) -> D3D12_DESCRIPTOR_RANGE_TYPE {
    match c {
        Class::Srv => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        Class::Uav => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        Class::Cbv => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
        Class::Sampler => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    }
}

/// The static sampler of a synthesized root: comparison samplers compare less or equal, other even
/// registers filter linearly and wrap, odd registers take the nearest texel and clamp.
pub fn static_sampler(s: &Sampler) -> D3D12_STATIC_SAMPLER_DESC {
    let (filter, address) = if s.comparison {
        (D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP)
    } else if s.register.is_multiple_of(2) {
        (D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP)
    } else {
        (D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP)
    };
    D3D12_STATIC_SAMPLER_DESC {
        Filter: filter,
        AddressU: address,
        AddressV: address,
        AddressW: address,
        ComparisonFunc: if s.comparison { D3D12_COMPARISON_FUNC_LESS_EQUAL } else { D3D12_COMPARISON_FUNC_NEVER },
        MaxLOD: f32::MAX,
        MaxAnisotropy: 1,
        ShaderRegister: s.register,
        RegisterSpace: s.space,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    }
}

/// The serialized form of `root`: its given blob, or root signature 1.0 built from its layout.
pub fn serialize(root: &Root) -> Result<Vec<u8>, String> {
    if let Some(b) = &root.blob {
        return Ok(b.clone());
    }
    let tables: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = root
        .params
        .iter()
        .map(|p| match p {
            Param::Table(ranges) => ranges
                .iter()
                .map(|r| D3D12_DESCRIPTOR_RANGE {
                    RangeType: range_type(r.class),
                    NumDescriptors: r.declared,
                    BaseShaderRegister: r.base,
                    RegisterSpace: r.space,
                    OffsetInDescriptorsFromTableStart: r.offset.unwrap_or(D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND),
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect();
    let params: Vec<D3D12_ROOT_PARAMETER> = root
        .params
        .iter()
        .zip(&tables)
        .map(|(p, t)| match p {
            Param::Table(_) => D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: t.len() as u32, pDescriptorRanges: t.as_ptr() } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            Param::Constants { register, space, count } => D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: *register, RegisterSpace: *space, Num32BitValues: *count } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            Param::Root { class, register, space, .. } => D3D12_ROOT_PARAMETER {
                ParameterType: match class {
                    Class::Cbv => D3D12_ROOT_PARAMETER_TYPE_CBV,
                    Class::Srv => D3D12_ROOT_PARAMETER_TYPE_SRV,
                    _ => D3D12_ROOT_PARAMETER_TYPE_UAV,
                },
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: *register, RegisterSpace: *space } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        })
        .collect();
    let samplers: Vec<D3D12_STATIC_SAMPLER_DESC> = root.samplers.iter().map(static_sampler).collect();
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: if params.is_empty() { std::ptr::null() } else { params.as_ptr() },
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: if samplers.is_empty() { std::ptr::null() } else { samplers.as_ptr() },
        Flags: D3D12_ROOT_SIGNATURE_FLAGS(root.flags as i32),
    };
    let mut blob: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    // SAFETY: the description and every array it points at outlive the call.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, Some(&mut err)) }.map_err(|e| format!("serialize root signature: {e} {}", err.as_ref().map(|b| String::from_utf8_lossy(&blob_bytes(b)).into_owned()).unwrap_or_default()))?;
    Ok(blob_bytes(&blob.ok_or("no root signature blob")?))
}

/// Reads a serialized root signature (or a container with an `RTS0` part) into a [`Root`] whose
/// resources are still to be filled. The blob is kept and used as it is.
pub fn deserialize(blob: &[u8]) -> Result<Root, String> {
    // SAFETY: the deserializer copies what it needs. The description it returns lives as long as
    // the deserializer, which outlives every read below.
    unsafe {
        let mut raw: *mut c_void = std::ptr::null_mut();
        D3D12CreateVersionedRootSignatureDeserializer(blob.as_ptr().cast(), blob.len(), &ID3D12VersionedRootSignatureDeserializer::IID, &mut raw).map_err(|e| format!("root signature: {e}"))?;
        let de = ID3D12VersionedRootSignatureDeserializer::from_raw(raw);
        let desc = &*de.GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1).map_err(|e| format!("root signature 1.1: {e}"))?;
        let d = &desc.Anonymous.Desc_1_1;
        let mut params = Vec::new();
        for i in 0..d.NumParameters as usize {
            let p = &*d.pParameters.add(i);
            let class = |t: D3D12_DESCRIPTOR_RANGE_TYPE| match t {
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV => Class::Srv,
                D3D12_DESCRIPTOR_RANGE_TYPE_UAV => Class::Uav,
                D3D12_DESCRIPTOR_RANGE_TYPE_CBV => Class::Cbv,
                _ => Class::Sampler,
            };
            params.push(match p.ParameterType {
                D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE => {
                    let t = &p.Anonymous.DescriptorTable;
                    let ranges = (0..t.NumDescriptorRanges as usize)
                        .map(|k| {
                            let r = &*t.pDescriptorRanges.add(k);
                            let offset = (r.OffsetInDescriptorsFromTableStart != D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND).then_some(r.OffsetInDescriptorsFromTableStart);
                            Range { class: class(r.RangeType), base: r.BaseShaderRegister, space: r.RegisterSpace, declared: r.NumDescriptors, offset, slots: Vec::new() }
                        })
                        .collect();
                    Param::Table(ranges)
                }
                D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => {
                    let c = &p.Anonymous.Constants;
                    Param::Constants { register: c.ShaderRegister, space: c.RegisterSpace, count: c.Num32BitValues }
                }
                t => {
                    let r = &p.Anonymous.Descriptor;
                    let class = match t {
                        D3D12_ROOT_PARAMETER_TYPE_CBV => Class::Cbv,
                        D3D12_ROOT_PARAMETER_TYPE_SRV => Class::Srv,
                        _ => Class::Uav,
                    };
                    Param::Root { class, register: r.ShaderRegister, space: r.RegisterSpace, res: Res::Null }
                }
            });
        }
        Ok(Root { flags: d.Flags.0 as u32, params, samplers: Vec::new(), blob: Some(blob.to_vec()) })
    }
}
