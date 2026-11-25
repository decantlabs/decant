//! DXC through `dxcompiler.dll`: HLSL to a signed DXIL container, and the D3D12 reflection of any
//! DXIL container, which gives the bindings, signatures and thread counts a pair needs.
//!
//! `dxcompiler.dll` is searched in `DECANT_SHADER_DXCOMPILER`, in `Windows Kits\10\bin\<version>\x64`
//! (newest first), in `%VULKAN_SDK%\Bin`, then on the normal DLL search path. It is loaded with its
//! own folder in the search path so that it finds `dxil.dll` there and signs what it compiles.

use std::ffi::c_void;
use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D::Dxc::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
use windows::core::{GUID, HRESULT, HSTRING, Interface, PCSTR, PCWSTR};

use crate::model::{Binding, Class, Component, Element, Reflection, Ret, Shape, Stage};

/// The candidate paths of `dxcompiler.dll`, best first.
pub fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("DECANT_SHADER_DXCOMPILER") {
        out.push(PathBuf::from(p));
    }
    for root in ["C:\\Program Files (x86)\\Windows Kits\\10\\bin", "C:\\Program Files\\Windows Kits\\10\\bin"] {
        let Ok(dir) = std::fs::read_dir(root) else { continue };
        let mut versions: Vec<PathBuf> = dir.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("x64").join("dxcompiler.dll").is_file()).collect();
        versions.sort();
        out.extend(versions.into_iter().rev().map(|p| p.join("x64").join("dxcompiler.dll")));
    }
    if let Some(sdk) = std::env::var_os("VULKAN_SDK") {
        out.push(PathBuf::from(sdk).join("Bin").join("dxcompiler.dll"));
    }
    out.push(PathBuf::from("dxcompiler.dll"));
    out
}

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

/// A loaded DXC.
pub struct Dxc {
    create: CreateInstance,
    utils: IDxcUtils,
    compiler: IDxcCompiler3,
    pub path: PathBuf,
}

impl Dxc {
    /// Loads the first `dxcompiler.dll` that works.
    pub fn load() -> Result<Dxc, String> {
        let mut why = Vec::new();
        for path in candidates() {
            if path.components().count() > 1 && !path.is_file() {
                continue;
            }
            match Self::load_from(&path) {
                Ok(d) => return Ok(d),
                Err(e) => why.push(format!("{}: {e}", path.display())),
            }
        }
        Err(format!("dxcompiler.dll not found or not usable (set DECANT_SHADER_DXCOMPILER): {}", why.join(", ")))
    }

    fn load_from(path: &std::path::Path) -> Result<Dxc, String> {
        // SAFETY: the DLL stays loaded for the life of the process, the entry point has the documented signature.
        unsafe {
            let module = LoadLibraryExW(&HSTRING::from(path.as_os_str()), None, LOAD_WITH_ALTERED_SEARCH_PATH).map_err(|e| e.to_string())?;
            let proc = GetProcAddress(module, PCSTR(c"DxcCreateInstance".as_ptr().cast())).ok_or("DxcCreateInstance is not exported")?;
            let create: CreateInstance = std::mem::transmute(proc);
            let make = |clsid: &GUID, iid: &GUID| -> Result<*mut c_void, String> {
                let mut out: *mut c_void = std::ptr::null_mut();
                create(clsid, iid, &mut out).ok().map_err(|e| e.to_string())?;
                Ok(out)
            };
            let utils = IDxcUtils::from_raw(make(&CLSID_DxcUtils, &IDxcUtils::IID)?);
            let compiler = IDxcCompiler3::from_raw(make(&CLSID_DxcCompiler, &IDxcCompiler3::IID)?);
            Ok(Dxc { create, utils, compiler, path: path.to_path_buf() })
        }
    }

    /// Compiles `source` with entry point `entry` for `profile` (`ps_6_0`...), with extra arguments.
    pub fn compile(&self, source: &str, entry: &str, profile: &str, extra: &[&str]) -> Result<Vec<u8>, String> {
        let mut args: Vec<HSTRING> = vec!["-E".into(), entry.into(), "-T".into(), profile.into()];
        args.extend(extra.iter().map(|a| HSTRING::from(*a)));
        let ptrs: Vec<PCWSTR> = args.iter().map(|a| PCWSTR(a.as_ptr())).collect();
        let buffer = DxcBuffer { Ptr: source.as_ptr().cast(), Size: source.len(), Encoding: DXC_CP_UTF8.0 };
        // SAFETY: the buffer and the arguments outlive the call, outputs are checked.
        unsafe {
            let result: IDxcResult = self.compiler.Compile(&buffer, Some(&ptrs), None).map_err(|e| format!("Compile: {e}"))?;
            let status = result.GetStatus().map_err(|e| e.to_string())?;
            let mut errors: Option<IDxcBlobUtf8> = None;
            let _ = result.GetOutput(DXC_OUT_ERRORS, std::ptr::null_mut(), &mut errors);
            let text = errors.map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetStringPointer().0, e.GetStringLength())).into_owned()).unwrap_or_default();
            if status.is_err() {
                return Err(format!("DXC: {}", text.trim()));
            }
            let mut object: Option<IDxcBlob> = None;
            result.GetOutput(DXC_OUT_OBJECT, std::ptr::null_mut(), &mut object).map_err(|e| format!("no object: {e}"))?;
            let object = object.ok_or("no object")?;
            Ok(std::slice::from_raw_parts(object.GetBufferPointer().cast::<u8>(), object.GetBufferSize()).to_vec())
        }
    }

    /// `container` with every `createHandleFromBinding` index that lies outside its range rewritten
    /// to the register (see [`crate::heapmap::fix_binding_indices`]), reassembled and signed, and
    /// how many it rewrote. `None` when there is nothing to rewrite.
    pub fn repair_binding_indices(&self, container: &[u8]) -> Result<Option<(Vec<u8>, usize)>, String> {
        let (text, n) = crate::heapmap::fix_binding_indices(&self.disassemble(container)?);
        if n == 0 {
            return Ok(None);
        }
        Ok(Some((self.assemble(&text)?, n)))
    }

    /// Assembles DXIL text (LLVM assembly as the disassembler writes it) into a signed container.
    pub fn assemble(&self, text: &str) -> Result<Vec<u8>, String> {
        // SAFETY: the blob owns a copy of the text, every result is checked.
        let mut out = unsafe {
            let assembler: IDxcAssembler = {
                let mut raw: *mut c_void = std::ptr::null_mut();
                (self.create)(&CLSID_DxcAssembler, &IDxcAssembler::IID, &mut raw).ok().map_err(|e| format!("assembler: {e}"))?;
                IDxcAssembler::from_raw(raw)
            };
            let blob = self.utils.CreateBlob(text.as_ptr().cast(), text.len() as u32, DXC_CP_UTF8).map_err(|e| format!("blob: {e}"))?;
            let result = assembler.AssembleToContainer(&blob).map_err(|e| format!("AssembleToContainer: {e}"))?;
            let status = result.GetStatus().map_err(|e| e.to_string())?;
            if status.is_err() {
                let why = result.GetErrorBuffer().ok().map(|b| String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())).into_owned()).unwrap_or_default();
                return Err(format!("assembler: {}", why.trim()));
            }
            let c = result.GetResult().map_err(|e| e.to_string())?;
            std::slice::from_raw_parts(c.GetBufferPointer().cast::<u8>(), c.GetBufferSize()).to_vec()
        };
        decant_dxil::validate::validate_in_place(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// The text disassembly of a DXIL container.
    pub fn disassemble(&self, container: &[u8]) -> Result<String, String> {
        let buffer = DxcBuffer { Ptr: container.as_ptr().cast(), Size: container.len(), Encoding: 0 };
        // SAFETY: the buffer outlives the call, the result is checked.
        unsafe {
            let result: IDxcResult = self.compiler.Disassemble::<IDxcResult>(&buffer).map_err(|e| format!("Disassemble: {e}"))?;
            let mut text: Option<IDxcBlobUtf8> = None;
            result.GetOutput(DXC_OUT_DISASSEMBLY, std::ptr::null_mut(), &mut text).map_err(|e| format!("disassembly: {e}"))?;
            let t = text.ok_or("no disassembly")?;
            Ok(String::from_utf8_lossy(std::slice::from_raw_parts(t.GetStringPointer().0, t.GetStringLength())).into_owned())
        }
    }

    /// The D3D12 reflection of a DXIL container.
    pub fn reflect(&self, container: &[u8]) -> Result<Reflection, String> {
        let buffer = DxcBuffer { Ptr: container.as_ptr().cast(), Size: container.len(), Encoding: 0 };
        // SAFETY: the buffer outlives the call. Every descriptor is filled by the reflection object
        // before it is read, and its strings live as long as that object.
        unsafe {
            let mut raw: *mut c_void = std::ptr::null_mut();
            self.utils.CreateReflection(&buffer, &ID3D12ShaderReflection::IID, &mut raw).map_err(|e| format!("reflection: {e}"))?;
            let r = ID3D12ShaderReflection::from_raw(raw);
            let mut desc = D3D12_SHADER_DESC::default();
            r.GetDesc(&mut desc).map_err(|e| format!("reflection GetDesc: {e}"))?;
            let stage = match desc.Version >> 16 {
                0 => Stage::Pixel,
                1 => Stage::Vertex,
                5 => Stage::Compute,
                other => return Err(format!("unsupported stage (shader version type {other})")),
            };
            let model = ((desc.Version >> 4) & 0xF, desc.Version & 0xF);
            let mut threads = [0u32; 3];
            if stage == Stage::Compute {
                r.GetThreadGroupSize(Some(&mut threads[0]), Some(&mut threads[1]), Some(&mut threads[2]));
            }
            let mut bindings = Vec::new();
            for i in 0..desc.BoundResources {
                let mut b = D3D12_SHADER_INPUT_BIND_DESC::default();
                r.GetResourceBindingDesc(i, &mut b).map_err(|e| format!("binding {i}: {e}"))?;
                let name = if b.Name.is_null() { String::new() } else { b.Name.to_string().unwrap_or_default() };
                let (class, shape) = binding_shape(b.Type.0, b.Dimension.0, b.ReturnType.0, b.NumSamples, b.uFlags);
                let count = if b.BindCount == 0 || b.BindCount == u32::MAX { u32::MAX } else { b.BindCount };
                bindings.push(Binding { class, space: b.Space, lower: b.BindPoint, count, shape, name });
            }
            let mut inputs = Vec::new();
            for i in 0..desc.InputParameters {
                let mut p = D3D12_SIGNATURE_PARAMETER_DESC::default();
                r.GetInputParameterDesc(i, &mut p).map_err(|e| format!("input {i}: {e}"))?;
                inputs.push(element(&p));
            }
            let mut outputs = Vec::new();
            for i in 0..desc.OutputParameters {
                let mut p = D3D12_SIGNATURE_PARAMETER_DESC::default();
                r.GetOutputParameterDesc(i, &mut p).map_err(|e| format!("output {i}: {e}"))?;
                outputs.push(element(&p));
            }
            let requires = r.GetRequiresFlags();
            let resource_heap = requires & u64::from(D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING) != 0;
            let sampler_heap = requires & u64::from(D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING) != 0;
            let heap = if resource_heap || sampler_heap { crate::heapmap::slots(&self.disassemble(container)?) } else { Vec::new() };
            Ok(Reflection {
                stage,
                model,
                threads,
                bindings,
                inputs,
                outputs,
                resource_heap,
                sampler_heap,
                heap,
            })
        }
    }
}

fn element(p: &D3D12_SIGNATURE_PARAMETER_DESC) -> Element {
    // SAFETY: the name points into the reflection object, which outlives this call.
    let semantic = if p.SemanticName.is_null() { String::new() } else { unsafe { p.SemanticName.to_string() }.unwrap_or_default() };
    let component = match p.ComponentType.0 {
        1 | 4 | 7 => Component::Uint,
        2 | 5 | 8 => Component::Sint,
        _ => Component::Float,
    };
    Element { semantic, index: p.SemanticIndex, register: p.Register, mask: p.Mask, system: p.SystemValueType.0 as u32, component }
}

/// The class and shape of a reflected binding from `D3D_SHADER_INPUT_TYPE`, `D3D_SRV_DIMENSION`,
/// `D3D_RESOURCE_RETURN_TYPE`, the sample count (the stride of structured buffers) and the flags.
pub fn binding_shape(ty: i32, dim: i32, ret: i32, samples: u32, flags: u32) -> (Class, Shape) {
    let ret = match ret {
        3 => Ret::Sint,
        4 => Ret::Uint,
        _ => Ret::Float,
    };
    let tex = |dim: i32| match dim {
        1 | 11 => Shape::TypedBuffer(ret),
        2 => Shape::Texture1D(ret),
        3 => Shape::Texture1DArray(ret),
        4 => Shape::Texture2D(ret),
        5 => Shape::Texture2DArray(ret),
        6 | 7 => Shape::Texture2DMS(ret),
        8 => Shape::Texture3D(ret),
        9 => Shape::TextureCube(ret),
        10 => Shape::TextureCubeArray(ret),
        other => Shape::Other(100 + other as u32),
    };
    match ty {
        0 => (Class::Cbv, Shape::ConstantBuffer),
        1 => (Class::Srv, Shape::Other(1)),
        2 => (Class::Srv, tex(dim)),
        3 => (Class::Sampler, Shape::Sampler { comparison: flags & 2 != 0 }),
        4 => (Class::Uav, tex(dim)),
        5 => (Class::Srv, Shape::StructuredBuffer { stride: samples }),
        6 | 9 | 10 | 11 => (Class::Uav, Shape::StructuredBuffer { stride: samples }),
        7 => (Class::Srv, Shape::RawBuffer),
        8 => (Class::Uav, Shape::RawBuffer),
        12 => (Class::Srv, Shape::Other(12)),
        other => (Class::Uav, Shape::Other(other as u32)),
    }
}
