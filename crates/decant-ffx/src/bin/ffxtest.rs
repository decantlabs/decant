//! `ffxtest <Decant folder> <PC FidelityFX DLL>`: creates an upscaling context through
//! `decant_ffx.dll` with a real PC backend, asks it for the jitter phase count and destroys it. The
//! port of the former `tests/ffxtest.cpp`, with the same lines and exit codes (0 passed, 1
//! failed, 2 cannot run). The FidelityFX structures it needs are declared here from the public
//! FidelityFX SDK headers (`ffx_api.h`, `ffx_upscale.h`, `dx12/ffx_api_dx12.h`), so no SDK is
//! needed to build it. `gpu_diag` runs it from the build folder.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(run::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("ffxtest runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::c_void;
    use std::path::Path;

    use decant_testkit::Library;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
    use windows::core::Interface;

    /// `ffxApiHeader`: a 64 bit structure type and the next structure of the chain.
    #[repr(C)]
    struct Header {
        kind: u64,
        next: *mut Header,
    }

    /// `ffxCreateBackendDX12Desc`.
    #[repr(C)]
    struct BackendDx12 {
        header: Header,
        device: *mut c_void,
    }

    /// `ffxCreateContextDescUpscale`.
    #[repr(C)]
    struct CreateUpscale {
        header: Header,
        flags: u32,
        max_render_size: [u32; 2],
        max_upscale_size: [u32; 2],
        message: *const c_void,
    }

    /// `ffxQueryDescUpscaleGetJitterPhaseCount`.
    #[repr(C)]
    struct JitterPhaseCount {
        header: Header,
        render_width: u32,
        display_width: u32,
        out_phase_count: *mut i32,
    }

    const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: u64 = 0x0000_0002;
    const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x0001_0000;
    const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 = 0x0001_0004;
    const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
    const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;
    const FFX_API_RETURN_OK: u32 = 0;

    type CreateFn = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
    type QueryFn = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;
    type DestroyFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;

    pub fn main() -> u8 {
        let args = decant_testkit::args();
        if args.len() < 2 {
            println!("usage : ffxtest <dossier Decant> <DLL FidelityFX du PC>");
            return 2;
        }
        // SAFETY: the program is single threaded here. Nothing else reads the environment yet.
        unsafe { std::env::set_var("DECANT_FFX_BACKEND", &args[1]) };
        let path = Path::new(&args[0]).join("decant_ffx.dll");
        let full = std::path::absolute(&path).unwrap_or(path);
        let shim = match Library::open_altered(full.as_os_str()) {
            Ok(shim) => shim,
            Err(error) => {
                println!("chargement de notre DLL impossible ({error})");
                return 2;
            }
        };
        // SAFETY: the types are the signatures of ffx_api.h for these names.
        let (create, query, destroy) = unsafe {
            (shim.get::<CreateFn>("ffxCreateContext"), shim.get::<QueryFn>("ffxQuery"), shim.get::<DestroyFn>("ffxDestroyContext"))
        };
        let (Some(create), Some(query), Some(destroy)) = (create, query, destroy) else {
            println!("fonctions ffx introuvables");
            return 2;
        };

        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation on the default adapter.
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() || device.is_none() {
            println!("creation du peripherique PC impossible");
            return 2;
        }
        let device = device.unwrap_or_else(|| unreachable!());

        let mut backend = BackendDx12 { header: Header { kind: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12, next: std::ptr::null_mut() }, device: device.as_raw() };
        let mut upscale = CreateUpscale {
            header: Header { kind: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, next: &mut backend.header },
            flags: FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE,
            max_render_size: [1280, 720],
            max_upscale_size: [2560, 1440],
            message: std::ptr::null(),
        };

        println!("Contexte d'upscaling 1280x720 -> 2560x1440 par notre DLL :");
        let mut context: *mut c_void = std::ptr::null_mut();
        // SAFETY: the descriptor chain lives on this stack frame for the whole call and `device` is alive.
        let mut result = unsafe { create(&mut context, &mut upscale.header, std::ptr::null()) };
        println!("ffxCreateContext : {} (code {result})", if result == FFX_API_RETURN_OK { "reussi" } else { "echec" });
        let mut failures = u32::from(result != FFX_API_RETURN_OK);
        if result == FFX_API_RETURN_OK {
            let mut phases: i32 = 0;
            let mut jitter = JitterPhaseCount {
                header: Header { kind: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT, next: std::ptr::null_mut() },
                render_width: 1280,
                display_width: 2560,
                out_phase_count: &mut phases,
            };
            // SAFETY: `context` was created above and `jitter` points to `phases` on this frame.
            result = unsafe { query(&mut context, &mut jitter.header) };
            println!("ffxQuery (phases de jitter) : {}, {phases} phases", if result == FFX_API_RETURN_OK { "reussi" } else { "echec" });
            failures += u32::from(!(result == FFX_API_RETURN_OK && phases > 0));
            // SAFETY: `context` was created above and is destroyed once.
            result = unsafe { destroy(&mut context, std::ptr::null()) };
            failures += u32::from(result != FFX_API_RETURN_OK);
        }
        println!("{}", if failures == 0 { "ESSAI REUSSI" } else { "ESSAI ECHOUE" });
        u8::from(failures != 0)
    }
}
