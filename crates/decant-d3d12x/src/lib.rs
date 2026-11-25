//! `d3d12_xs.dll`, the Scarlett D3D12.X layer over a PC D3D12 runtime, with the exports,
//! names and ordinals of the console DLL. The behaviour of each part is described in docs/D3D12X_LAYER.md.
//!
//! The PC D3D12 is whatever `d3d12.dll` resolves to: vkd3d-proton (Proton/Wine prefix, or an app
//! local copy on Windows) or Windows' own. Nothing here talks to Vulkan directly.
//!
//! * Root signature helpers are real wrappers. Console only flag bits (0x03000000 on the RTS0 v2
//!   root signatures of `XboxBvhBuilder.dll`) are cleared before the PC runtime sees them, and
//!   logged. Their meaning on the console is unknown, dropping them is an assumption. The
//!   container code is in [`rts0`].
//! * `D3D12XboxCreateDevice` and `D3D12CreateDevice` (titles import both from `d3d12_xs`) create
//!   a PC device and return it inside a console ABI proxy (`proxy`). The device, its command lists,
//!   queues and descriptor heaps get vtables in the console order ([`abi`]). Unknown console slots
//!   are logged and answer `E_NOTIMPL`. Handing a PC vtable to code that expects the console one
//!   would be a silent lie.
//! * `IID_ID3D12Device` is exported as data: console titles import it from `d3d12_xs` (they
//!   do), and a missing data import stops the executable from loading at all.
//! * Present: `PresentX` on the console queue reaches the hook of `DecantD3D12X_SetPresentHook`, else is
//!   logged. The `DXGIX*` entry points answer `DXGI_ERROR_UNSUPPORTED` (0x887A0004), as the console
//!   DLL does without reading its arguments, and are logged. The process debug flags are accepted.
//! * Frame events (device slots 78 to 80) are emulated with a timer (`frame`).
//!
//! Logging: every line goes to `OutputDebugString` (`WINEDEBUG=+debugstr` under Wine) and to the
//! Decant session log (`DECANT_LOG_PATH`, set by the launcher), see `log`.
//!
//! Only public material is used: the public D3D12 headers' layouts, the GDK's public documentation,
//! and what the package's own DLLs show. No console SDK.
//!
//! The test program `d3d12xs_test` (`src/bin/d3d12xs_test.rs`) is the port of the former
//! `tests/d3d12xs_test.c`.

pub mod adapter_choice;
pub mod abi;
pub mod alloc_info;
pub mod compile_service;
pub mod detile_plan;
pub mod direct_cache;
#[cfg(test)]
mod direct_fixture;
pub mod direct_source;
pub mod dxil_override;
pub mod device_cache;
pub mod kind_check;
pub mod zero_null;
pub mod list_sync;
pub mod page_maps;
pub mod mesh_pso;
pub mod pipeline_manifest;
pub mod pso_library;
pub mod present_over;
pub mod rts0;
pub mod resource_args;
pub mod sampler_registry;
pub mod uav_barrier_merge;
pub mod indirect_clamp;
pub mod slot_tags;
pub mod shader_source;
pub mod title_heap;
pub mod rect_list;
pub mod upload_mem;
pub mod user_mirror;
pub mod stream_pso;
pub mod views_at;
pub(crate) mod watchdog_hold;
pub mod window_service;

#[cfg(windows)]
mod cpu_writes;
#[cfg(windows)]
mod capture;
#[cfg(windows)]
mod cbv_copies;
#[cfg(windows)]
mod detile;
#[cfg(windows)]
mod dxgi;
#[cfg(windows)]
mod exports;
#[cfg(windows)]
mod fence_cell;
#[cfg(windows)]
mod flicker_probe;
#[cfg(windows)]
mod diagnose;
#[cfg(windows)]
mod direct_pso;
#[cfg(any(windows, test))]
mod precompile_queue;
#[cfg(windows)]
mod draw_diag;
#[cfg(windows)]
mod fallback;
#[cfg(windows)]
mod present;
#[cfg(windows)]
mod frame;
#[cfg(windows)]
mod lazy_pso;
#[cfg(windows)]
mod log;
#[cfg(windows)]
mod proxy;
#[cfg(windows)]
mod pso_diag;
#[cfg(windows)]
mod retire;
#[cfg(windows)]
mod rt_diag;
#[cfg(windows)]
mod title_mem;
#[cfg(windows)]
mod mem_sample;
pub(crate) mod mem_owners;
#[cfg(windows)]
mod mem_resource;

#[cfg(windows)]
mod view_diag;

#[cfg(windows)]
mod view_heap;
#[cfg(windows)]
mod view_image;
#[cfg(windows)]
mod view_window;
#[cfg(windows)]
mod view_raw;
#[cfg(windows)]
mod view_shadow;
#[cfg(all(test, windows))]
mod view_gpu_tests;
#[cfg(windows)]
mod title_mirror;
#[cfg(windows)]
mod service_mem;

/// The one forwarder, by name to the `d3d12.dll` export of the same name. It is not code of this
/// DLL: the loader resolves it in `d3d12.dll` (vkd3d-proton under Wine, the system DLL on Windows),
/// as the `d3d12_xs.def` of the C version did. The root signature helpers are not forwarders, they
/// are wrappers of this DLL under the public names.
macro_rules! forwarders {
    ($prefix:literal: $($name:literal),*) => {
        concat!($(" ", $prefix, $name, "=d3d12.", $name),*, " ")
    };
}

/// Linker options for the forwarder, in the syntax of the target's linker: link.exe reads
/// `/EXPORT:`, GNU ld and lld in MinGW mode read `-export:`.
#[cfg_attr(not(windows), allow(dead_code))]
const FORWARDER_DIRECTIVES: &str = if cfg!(target_env = "msvc") {
    forwarders!("/EXPORT:": "D3D12GetDebugInterface")
} else {
    forwarders!("-export:": "D3D12GetDebugInterface")
};

/// `text` as a byte array, for a static whose length is fixed at compile time.
#[cfg_attr(not(windows), allow(dead_code))]
const fn bytes<const N: usize>(text: &str) -> [u8; N] {
    let text = text.as_bytes();
    let mut out = [0; N];
    let mut i = 0;
    while i < N {
        out[i] = text[i];
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_forwarder_names_the_d3d12_export_of_the_same_name() {
        let prefix = if cfg!(target_env = "msvc") { "/EXPORT:" } else { "-export:" };
        let options: Vec<&str> = super::FORWARDER_DIRECTIVES.split_whitespace().collect();
        assert_eq!(options, [format!("{prefix}D3D12GetDebugInterface=d3d12.D3D12GetDebugInterface")]);
        assert_eq!(super::bytes::<4>("abcd"), *b"abcd");
    }
}
