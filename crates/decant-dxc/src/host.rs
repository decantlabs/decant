//! The Windows side of the crate: the loader and the COM calls.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use windows_sys::core::{GUID, HRESULT};
use windows_sys::Win32::System::LibraryLoader::{
    GetModuleFileNameW, GetModuleHandleExW, GetProcAddress, LoadLibraryExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    LOAD_WITH_ALTERED_SEARCH_PATH,
};

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> GUID {
    GUID { data1, data2, data3, data4 }
}

/// `CLSID_DxcCompiler`.
const CLSID_DXC_COMPILER: GUID = guid(0x73e2_2d93, 0xe6ce, 0x47f3, [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0]);
/// `IDxcCompiler3`.
const IID_DXC_COMPILER3: GUID = guid(0x228b_4687, 0x5a6a, 0x4730, [0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54]);
/// `IDxcResult` (derives from `IDxcOperationResult`).
const IID_DXC_RESULT: GUID = guid(0x5834_6cda, 0xdde7, 0x4497, [0x94, 0x61, 0x6f, 0x87, 0xaf, 0x5e, 0x06, 0x59]);
/// `IDxcVersionInfo`.
const IID_DXC_VERSION_INFO: GUID = guid(0xb04f_5b50, 0x2059, 0x4f12, [0xa8, 0xff, 0xa1, 0xe0, 0xcd, 0xe1, 0xcc, 0x7e]);
/// `IDxcVersionInfo2`.
const IID_DXC_VERSION_INFO2: GUID = guid(0xfb69_04c4, 0x42f0, 0x4b62, [0x9c, 0x46, 0x98, 0x3a, 0xf7, 0xda, 0x7c, 0x83]);
/// `DXC_CP_UTF8`.
const CP_UTF8: u32 = 65001;

type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> HRESULT;

/// `DxcBuffer`.
#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}

/// A COM pointer that is released on drop.
struct Com(*mut c_void);

impl Com {
    /// Slot `n` of the vtable as a function of type `F`.
    ///
    /// # Safety
    /// The object is live and slot `n` of its interface has the signature `F`.
    unsafe fn slot<F: Copy>(&self, n: usize) -> F {
        // SAFETY: by the contract above. A vtable is an array of function pointers.
        unsafe {
            let vtbl = *(self.0 as *const *const usize);
            std::mem::transmute_copy(&*vtbl.add(n))
        }
    }

    /// `QueryInterface` (slot 0).
    fn query(&self, iid: &GUID) -> Option<Com> {
        let mut out = std::ptr::null_mut();
        // SAFETY: every COM object has QueryInterface in slot 0.
        let hr = unsafe { self.slot::<unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT>(0)(self.0, iid, &mut out) };
        (hr >= 0 && !out.is_null()).then_some(Com(out))
    }

    /// The bytes of an `IDxcBlob` (slots 3 and 4).
    fn blob_bytes(&self) -> Vec<u8> {
        // SAFETY: the object is an IDxcBlob: GetBufferPointer in slot 3, GetBufferSize in slot 4.
        unsafe {
            let ptr = self.slot::<unsafe extern "system" fn(*mut c_void) -> *const u8>(3)(self.0);
            let len = self.slot::<unsafe extern "system" fn(*mut c_void) -> usize>(4)(self.0);
            if ptr.is_null() || len == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(ptr, len).to_vec()
            }
        }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: every COM object has Release in slot 2.
            unsafe { self.slot::<unsafe extern "system" fn(*mut c_void) -> u32>(2)(self.0) };
        }
    }
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

fn wide_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

/// Loads `path` with its own folder first in the search for its dependencies. 0 when it fails.
fn load(path: &Path) -> usize {
    let w = wide_path(path);
    // SAFETY: `w` is a NUL terminated path.
    unsafe { LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) as usize }
}

/// The folder of the module that holds this code (`d3d12_xs.dll`, or the test program).
fn own_dir() -> Option<PathBuf> {
    let mut module = std::ptr::null_mut();
    let anchor = own_dir as *const () as *const u16;
    // SAFETY: `anchor` is an address inside this module, `module` is writable.
    let ok = unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, anchor, &mut module) };
    if ok == 0 {
        return None;
    }
    let mut buf = vec![0u16; 1024];
    // SAFETY: `buf` is writable for its length.
    let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if n == 0 || n >= buf.len() {
        return None;
    }
    PathBuf::from(String::from_utf16_lossy(&buf[..n])).parent().map(Path::to_path_buf)
}

/// The architecture folder name of the SDK for this build.
const SDK_ARCH: &str = if cfg!(target_arch = "aarch64") { "arm64" } else { "x64" };

/// The SDK bin folders that hold a `dxcompiler.dll`, newest version first.
pub fn sdk_dirs() -> Vec<PathBuf> {
    let roots = [std::env::var_os("ProgramFiles(x86)"), std::env::var_os("ProgramFiles")];
    let mut out = Vec::new();
    for root in roots.into_iter().flatten() {
        let bin = PathBuf::from(root).join("Windows Kits").join("10").join("bin");
        let Ok(entries) = std::fs::read_dir(&bin) else { continue };
        let mut versions: Vec<(Vec<u32>, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let parts: Option<Vec<u32>> = name.split('.').map(|p| p.parse().ok()).collect();
                parts.map(|v| (v, e.path().join(SDK_ARCH)))
            })
            .filter(|(_, dir)| dir.join("dxcompiler.dll").is_file())
            .collect();
        versions.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, dir) in versions {
            if !out.contains(&dir) {
                out.push(dir);
            }
        }
    }
    out
}

/// The folders searched for `dxcompiler.dll`, in order (the loader's own search comes after them).
pub fn candidate_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = std::env::var_os("DECANT_DXC_DIR").filter(|d| !d.is_empty()) {
        out.push(PathBuf::from(dir));
    }
    out.extend(own_dir());
    out.extend(sdk_dirs());
    out
}

/// A loaded DXC.
pub struct Dxc {
    create: CreateInstance,
    version: String,
    /// Where `dxcompiler.dll` came from (empty for the loader's own search).
    pub dir: PathBuf,
    /// Whether `dxil.dll` was found beside it.
    pub validator: bool,
}

// SAFETY: `create` is a free function of a DLL that stays loaded, every call makes its own objects.
unsafe impl Send for Dxc {}
// SAFETY: as above.
unsafe impl Sync for Dxc {}

impl Dxc {
    /// The first DXC of the search order. The error lists what was tried.
    pub fn load() -> Result<Dxc, String> {
        let mut tried = Vec::new();
        for dir in candidate_dirs() {
            match Dxc::load_from(Some(&dir)) {
                Ok(d) => return Ok(d),
                Err(e) => tried.push(e),
            }
        }
        match Dxc::load_from(None) {
            Ok(d) => Ok(d),
            Err(e) => {
                tried.push(e);
                Err(tried.join("; "))
            }
        }
    }

    /// DXC from `dir`, or from the loader's own search when `None`.
    pub fn load_from(dir: Option<&Path>) -> Result<Dxc, String> {
        let at = |name: &str| dir.map_or_else(|| PathBuf::from(name), |d| d.join(name));
        let compiler = at("dxcompiler.dll");
        if dir.is_some() && !compiler.is_file() {
            return Err(format!("{} not found", compiler.display()));
        }
        // dxil.dll first: dxcompiler.dll asks for it by name and then gets this one.
        let validator = (dir.is_none() || at("dxil.dll").is_file()) && load(&at("dxil.dll")) != 0;
        let module = load(&compiler);
        if module == 0 {
            return Err(format!("{} does not load", compiler.display()));
        }
        // SAFETY: `module` is a loaded module, the name is NUL terminated.
        let proc = unsafe { GetProcAddress(module as _, c"DxcCreateInstance".as_ptr().cast()) };
        let Some(proc) = proc else {
            return Err(format!("{} has no DxcCreateInstance", compiler.display()));
        };
        // SAFETY: DxcCreateInstance has this signature (dxcapi.h).
        let create: CreateInstance = unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, CreateInstance>(proc) };
        let mut dxc = Dxc { create, version: String::new(), dir: dir.map(Path::to_path_buf).unwrap_or_default(), validator };
        let instance = dxc.instance(&IID_DXC_COMPILER3).ok_or_else(|| format!("{} gives no IDxcCompiler3", compiler.display()))?;
        dxc.version = version_of(&instance);
        Ok(dxc)
    }

    /// A new compiler object asked for `iid`.
    fn instance(&self, iid: &GUID) -> Option<Com> {
        let mut out = std::ptr::null_mut();
        // SAFETY: DxcCreateInstance with valid GUIDs and a writable out pointer.
        let hr = unsafe { (self.create)(&CLSID_DXC_COMPILER, iid, &mut out) };
        (hr >= 0 && !out.is_null()).then_some(Com(out))
    }

    /// One line for the log: version, folder and whether DXIL gets signed.
    pub fn describe(&self) -> String {
        let dir = if self.dir.as_os_str().is_empty() { "the loader's search".to_string() } else { self.dir.display().to_string() };
        format!("DXC {} from {dir}, {}", self.version, if self.validator { "dxil.dll found, DXIL is signed" } else { "no dxil.dll, DXIL is not signed" })
    }
}

/// `major.minor`, with `+<commit count>.<commit hash>` when the compiler tells (`IDxcVersionInfo2`).
fn version_of(compiler: &Com) -> String {
    let Some(info) = compiler.query(&IID_DXC_VERSION_INFO) else { return "unknown".to_string() };
    let (mut major, mut minor) = (0u32, 0u32);
    // SAFETY: IDxcVersionInfo::GetVersion is slot 3.
    unsafe { info.slot::<unsafe extern "system" fn(*mut c_void, *mut u32, *mut u32) -> HRESULT>(3)(info.0, &mut major, &mut minor) };
    let mut text = format!("{major}.{minor}");
    if let Some(info2) = compiler.query(&IID_DXC_VERSION_INFO2) {
        let (mut count, mut hash) = (0u32, std::ptr::null_mut::<std::ffi::c_char>());
        // SAFETY: IDxcVersionInfo2::GetCommitInfo is slot 5, the hash is a CoTaskMemAlloc string.
        let hr = unsafe { info2.slot::<unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut std::ffi::c_char) -> HRESULT>(5)(info2.0, &mut count, &mut hash) };
        if hr >= 0 && !hash.is_null() {
            // SAFETY: a NUL terminated string from DXC.
            let h = unsafe { std::ffi::CStr::from_ptr(hash) }.to_string_lossy().into_owned();
            text.push_str(&format!("+{count}.{h}"));
            co_task_mem_free(hash.cast());
        }
    }
    text
}

/// `CoTaskMemFree` from ole32, looked up at run time like the rest.
fn co_task_mem_free(ptr: *mut c_void) {
    // SAFETY: ole32 is a system DLL, CoTaskMemFree takes the pointer DXC allocated with its allocator.
    unsafe {
        let module = windows_sys::Win32::System::LibraryLoader::LoadLibraryW(windows_sys::w!("ole32.dll"));
        if module.is_null() {
            return;
        }
        if let Some(f) = GetProcAddress(module, c"CoTaskMemFree".as_ptr().cast()) {
            let free: unsafe extern "system" fn(*mut c_void) = std::mem::transmute::<unsafe extern "system" fn() -> isize, unsafe extern "system" fn(*mut c_void)>(f);
            free(ptr);
        }
    }
}

impl Dxc {
    /// The compiler version for cache keys: `dxc <version>`, with ` unsigned` when the DXIL is
    /// not signed.
    pub fn version(&self) -> String {
        format!("dxc {}{}", self.version, if self.validator { "" } else { " unsigned" })
    }

    /// The DXIL container of `hlsl` for `entry` and `profile`, or the compiler's message. Debug
    /// information and reflection are stripped, `extra` are further DXC arguments, each one item.
    pub fn compile(&self, hlsl: &str, entry: &str, profile: &str, extra: &[String]) -> Result<Vec<u8>, String> {
        let compiler = self.instance(&IID_DXC_COMPILER3).ok_or("no IDxcCompiler3 instance")?;
        let mut args: Vec<&str> = vec!["-E", entry, "-T", profile, "-Qstrip_debug", "-Qstrip_reflect"];
        args.extend(extra.iter().map(String::as_str));
        let wide_args: Vec<Vec<u16>> = args.iter().map(|a| wide(a)).collect();
        let arg_ptrs: Vec<*const u16> = wide_args.iter().map(|a| a.as_ptr()).collect();
        let source = DxcBuffer { ptr: hlsl.as_ptr().cast(), size: hlsl.len(), encoding: CP_UTF8 };
        let mut out = std::ptr::null_mut();
        type Compile = unsafe extern "system" fn(*mut c_void, *const DxcBuffer, *const *const u16, u32, *mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;
        // SAFETY: IDxcCompiler3::Compile is slot 3. The buffer, arguments and out pointer are live.
        let hr = unsafe { compiler.slot::<Compile>(3)(compiler.0, &source, arg_ptrs.as_ptr(), arg_ptrs.len() as u32, std::ptr::null_mut(), &IID_DXC_RESULT, &mut out) };
        if hr < 0 || out.is_null() {
            return Err(format!("IDxcCompiler3::Compile failed, hr=0x{:08X}", hr as u32));
        }
        let result = Com(out);
        let mut status: HRESULT = 0;
        // SAFETY: IDxcOperationResult::GetStatus is slot 3.
        unsafe { result.slot::<unsafe extern "system" fn(*mut c_void, *mut HRESULT) -> HRESULT>(3)(result.0, &mut status) };
        let get = |slot: usize| {
            let mut blob = std::ptr::null_mut();
            // SAFETY: IDxcOperationResult::GetResult (4) and GetErrorBuffer (5) write a blob or null.
            let hr = unsafe { result.slot::<unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> HRESULT>(slot)(result.0, &mut blob) };
            (hr >= 0 && !blob.is_null()).then(|| Com(blob).blob_bytes())
        };
        if status < 0 {
            let text = get(5).map(|b| String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_string()).unwrap_or_default();
            return Err(if text.is_empty() { format!("DXC status 0x{:08X}", status as u32) } else { text });
        }
        let dxil = get(4).unwrap_or_default();
        if dxil.is_empty() {
            return Err("DXC gave no object".to_string());
        }
        Ok(dxil)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dxc() -> Option<Dxc> {
        match Dxc::load() {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipped: no DXC ({e})");
                None
            }
        }
    }

    #[test]
    fn dxc_compiles_fixed_hlsl_to_signed_dxil() {
        let Some(dxc) = dxc() else { return };
        eprintln!("{}", dxc.describe());
        assert!(dxc.version().starts_with("dxc 1."), "{}", dxc.version());
        let stages = [
            ("[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {}
", "cs_6_0"),
            ("float4 main(uint id : SV_VertexID) : SV_Position { return float4(float(id & 1), float(id >> 1), 0, 1); }
", "vs_6_0"),
            ("float4 main() : SV_Target0 { return float4(1, 0, 1, 1); }
", "ps_6_6"),
        ];
        for (hlsl, profile) in stages {
            let dxil = dxc.compile(hlsl, "main", profile, &["-Gis".to_string()]).unwrap_or_else(|e| panic!("{profile}: {e}"));
            assert_eq!(&dxil[..4], b"DXBC");
            assert!(dxil.windows(4).any(|w| w == b"DXIL"), "{profile} has a DXIL part");
            assert_eq!(crate::signed(&dxil), dxc.validator, "{profile} is signed when dxil.dll is there");
        }
    }

    #[test]
    fn a_dxc_error_comes_back_as_its_message() {
        let Some(dxc) = dxc() else { return };
        let err = dxc.compile("float4 main() : SV_Target0 { return undeclared; }", "main", "ps_6_0", &[]).expect_err("does not compile");
        assert!(err.contains("undeclared"), "{err}");
    }
}
