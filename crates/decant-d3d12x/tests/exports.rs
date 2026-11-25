//! The built `d3d12_xs.dll` against the export table of the former C build (`dumpbin /exports`
//! of `cl /LD d3d12_xs.c`, and of the mingw-w64 build with `d3d12_xs.def`), then loaded and called
//! the way a companion DLL or a title would. The device and root signature behaviour is checked by
//! `d3d12xs_test` (see `programs.rs`).
#![cfg(windows)]

use std::path::PathBuf;

/// Ordinal, name, forwarder: the table both C builds produced.
const EXPECTED: [(u32, &str, Option<&str>); 17] = [
    (1, "D3D12ConfigureVirtualMemoryX", None),
    (2, "D3D12CreateDevice", None),
    (3, "D3D12CreateRootSignatureDeserializer", None),
    (4, "D3D12CreateVersionedRootSignatureDeserializer", None),
    (5, "D3D12GetDebugInterface", Some("d3d12.D3D12GetDebugInterface")),
    (6, "D3D12SerializeRootSignature", None),
    (7, "D3D12SerializeVersionedRootSignature", None),
    (8, "D3D12XboxCreateDevice", None),
    (9, "D3D12XboxSetProcessDebugFlags", None),
    (10, "D3D12XsUnwrap", None),
    (11, "DXGIXGetFrameStatistics", None),
    (12, "DXGIXPresentArray", None),
    (13, "DXGIXSetFrameNotification", None),
    (14, "DXGIXSetVLineNotification", None),
    (15, "DecantD3D12XNoteWrite", None),
    (16, "DecantD3D12X_SetPresentHook", None),
    (17, "IID_ID3D12Device", None),
];

/// The DLL cargo built next to this test.
fn dll() -> PathBuf {
    let deps = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let path = deps.join("d3d12_xs.dll");
    assert!(path.is_file(), "{} not built", path.display());
    path
}

/// The export table of a PE32+ image: (ordinal, name, forwarder) sorted by ordinal.
fn exports(image: &[u8]) -> Vec<(u32, String, Option<String>)> {
    let u16_at = |at: usize| u16::from_le_bytes(image[at..at + 2].try_into().unwrap()) as usize;
    let u32_at = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap()) as usize;
    let pe = u32_at(0x3c);
    assert_eq!(&image[pe..pe + 4], b"PE\0\0");
    let sections = u16_at(pe + 6);
    let optional = pe + 24;
    assert_eq!(u16_at(optional), 0x20b, "PE32+");
    let (dir_rva, dir_size) = (u32_at(optional + 112), u32_at(optional + 116));
    let table = optional + u16_at(pe + 20);
    let offset = |rva: usize| -> usize {
        (0..sections)
            .map(|i| table + i * 40)
            .find_map(|s| {
                let (va, size, raw) = (u32_at(s + 12), u32_at(s + 8).max(u32_at(s + 16)), u32_at(s + 20));
                (va..va + size).contains(&rva).then(|| rva - va + raw)
            })
            .unwrap()
    };
    let text = |rva: usize| -> String {
        let at = offset(rva);
        let end = image[at..].iter().position(|&b| b == 0).unwrap();
        String::from_utf8(image[at..at + end].to_vec()).unwrap()
    };
    let dir = offset(dir_rva);
    let base = u32_at(dir + 16) as u32;
    let (functions, names) = (offset(u32_at(dir + 28)), offset(u32_at(dir + 32)));
    let ordinals = offset(u32_at(dir + 36));
    let mut out: Vec<_> = (0..u32_at(dir + 24))
        .map(|i| {
            let index = u16_at(ordinals + 2 * i);
            let rva = u32_at(functions + 4 * index);
            let forward = (dir_rva..dir_rva + dir_size).contains(&rva).then(|| text(rva));
            (base + index as u32, text(u32_at(names + 4 * i)), forward)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn export_table_matches_the_c_build() {
    let image = std::fs::read(dll()).unwrap();
    let got = exports(&image);
    let want: Vec<_> = EXPECTED.iter().map(|&(o, n, f)| (o, n.to_owned(), f.map(str::to_owned))).collect();
    assert_eq!(got, want);
}

#[allow(unsafe_code, non_snake_case)]
mod calls {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::{E_INVALIDARG, E_POINTER, S_OK};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
    use windows_sys::core::{GUID, HRESULT};

    /// What `GetProcAddress` answers, before the cast to the export's own signature.
    type Raw = unsafe extern "system" fn() -> isize;
    type CreateDevice = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
    type SetFlags = unsafe extern "system" fn(u32) -> HRESULT;
    type AnyArgs = unsafe extern "system" fn(u64, u64, u64, u64) -> HRESULT;

    const DXGI_ERROR_UNSUPPORTED: HRESULT = 0x887A_0004_u32 as HRESULT;

    fn wide(text: &std::ffi::OsStr) -> Vec<u16> {
        text.encode_wide().chain([0]).collect()
    }

    #[test]
    fn entry_points_answer_as_the_c_version() {
        let path = wide(super::dll().as_os_str());
        // SAFETY: loads our own DLL, which has no DllMain of its own. Every symbol below is cast to the
        // signature of its definition in src/exports.rs.
        unsafe {
            let module = LoadLibraryW(path.as_ptr());
            assert!(!module.is_null());
            let get = |name: &std::ffi::CStr| GetProcAddress(module, name.as_ptr().cast()).expect("export");

            // No output pointer: refused before any device is created.
            let create = std::mem::transmute::<Raw, CreateDevice>(get(c"D3D12XboxCreateDevice"));
            let iid = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);
            assert_eq!(create(std::ptr::null_mut(), std::ptr::null(), &iid, std::ptr::null_mut()), E_POINTER);

            let flags = std::mem::transmute::<Raw, SetFlags>(get(c"D3D12XboxSetProcessDebugFlags"));
            assert_eq!(flags(0), S_OK);
            assert_eq!(flags(0xFFFF_FFFF), S_OK);

            // The four DXGIX entry points ignore their arguments.
            for name in [c"DXGIXPresentArray", c"DXGIXSetVLineNotification", c"DXGIXSetFrameNotification", c"DXGIXGetFrameStatistics"] {
                let any = std::mem::transmute::<Raw, AnyArgs>(get(name));
                assert_eq!(any(0, 0, 0, 0), DXGI_ERROR_UNSUPPORTED);
                assert_eq!(any(1, 2, 3, 4), DXGI_ERROR_UNSUPPORTED);
            }

            // D3D12ConfigureVirtualMemoryX: one 32-byte description (shape measured on the title executable).
            type Configure = unsafe extern "system" fn(*mut [u64; 4]) -> HRESULT;
            let configure = std::mem::transmute::<Raw, Configure>(get(c"D3D12ConfigureVirtualMemoryX"));
            // flags 0, 4 page-table pages, preserve 0, reserved 0: the title's values. Outputs cleared.
            let mut desc = [4u64 << 32, 0, u64::MAX, u64::MAX];
            assert_eq!(configure(&mut desc), 0x1000_0000);
            assert_eq!((desc[2], desc[3]), (0, 0));
            // A flag other than bit 0, or more than 0x21 pages: E_INVALIDARG, outputs untouched.
            let mut desc = [2 | (4u64 << 32), 0, 7, 7];
            assert_eq!(configure(&mut desc), E_INVALIDARG);
            assert_eq!((desc[2], desc[3]), (7, 7));
            let mut desc = [1 | (0x22u64 << 32), 0, 7, 7];
            assert_eq!(configure(&mut desc), E_INVALIDARG);
            let mut desc = [1 | (0x21u64 << 32), 0, 7, 7];
            assert_eq!(configure(&mut desc), 0x1000_0000);
            assert_eq!(configure(std::ptr::null_mut()), E_POINTER);

            // The IID is data, the public value.
            let data = get(c"IID_ID3D12Device") as usize as *const GUID;
            assert!((*data).data1 == iid.data1 && (*data).data2 == iid.data2 && (*data).data3 == iid.data3 && (*data).data4 == iid.data4);

            // Resolving the forwarder loads d3d12.dll and answers its own function.
            let forwarded = get(c"D3D12GetDebugInterface") as usize;
            let d3d12 = GetModuleHandleW(wide("d3d12.dll".as_ref()).as_ptr());
            assert!(!d3d12.is_null(), "d3d12.dll pulled in by the forwarder");
            let direct = GetProcAddress(d3d12, c"D3D12GetDebugInterface".as_ptr().cast()).expect("d3d12 export") as usize;
            assert_eq!(forwarded, direct);
        }
    }
}
