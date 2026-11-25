//! `gdk_smoketest`, the port of the former `tests/gdk_smoketest.c`: loads every DLL staged next to
//! the `xmem.dll` and `d3d12_xs.dll` shims (under Wine or Proton, or on Windows), then runs light
//! functional checks.
//!
//! The package's companion DLLs are discovered in the working directory, not named here
//! (repository rule: no per package facts in the tree), and the library checks find their targets
//! by export, never by file name. Run it from a folder holding the companion DLLs of your own
//! package and the shims; `launch_prep test` does this. It needs files that are never committed,
//! so CI does not run it.
//!
//! Prints one RESULT line per check and a SUMMARY line, with the text of the C version. The exit
//! code is the number of failures.

mod report;
#[cfg(windows)]
mod sys;

#[cfg(windows)]
fn main() {
    let mut report = report::Report::new(std::io::stdout().lock());
    run(&mut report);
    report.summary();
    std::process::exit(report.failures);
}

#[cfg(not(windows))]
fn main() {
    eprintln!("gdk_smoketest is a Windows program: build it for x86_64-pc-windows-gnu and run it under Wine");
    std::process::exit(2);
}

#[cfg(windows)]
fn run(report: &mut report::Report<impl std::io::Write>) {
    use report::{hresult, load_label, pointer};
    use sys::{HttpClient, Module, XMem};

    /// The C version kept at most this many modules for the export searches.
    const MAX_MODULES: usize = 128;

    sys::quiet_errors();

    // Discovery: load every *.dll in the working directory.
    let mut modules = Vec::new();
    match sys::local_dlls() {
        None => report.check("discover *.dll in cwd", false, "none found"),
        Some(names) => {
            for name in names {
                let loaded = Module::load(&name);
                report.check(&load_label(&name), loaded.is_ok(), &format!("err={}", loaded.err().unwrap_or(0)));
                if let Ok(module) = loaded
                    && modules.len() < MAX_MODULES
                {
                    modules.push(module);
                }
            }
            // The two shims plus at least one companion DLL are expected.
            report.check("companion DLLs discovered", modules.len() >= 3, &format!("found {}", modules.len()));
        }
    }
    let exporting = |name: &std::ffi::CStr| modules.iter().copied().find(|m| m.export(name).is_some());

    // XMem: heap (memtype 0, alignment 2^8) and graphics (memtype 1) allocations.
    match Module::loaded("xmem.dll").and_then(XMem::find) {
        Some(xmem) => {
            let heap: u64 = 8 << 24; // dwAlignment=8 at bit 24
            let gfx: u64 = (1 << 29) | (1 << 14); // memtype GRAPHICS_1, 64K pages
            let (h, g) = (xmem.alloc(1 << 20, heap), xmem.alloc(16 << 20, gfx));
            report.check("XMemAlloc heap 1MiB align256", h != 0 && h & 255 == 0, &pointer(h));
            report.check("XMemAlloc graphics 16MiB", g != 0, &pointer(g));
            if g != 0 {
                xmem.fill(g, 0xAB, 16 << 20);
            }
            if xmem.has_query()
                && g != 0
                && let Some((ok, info)) = xmem.query(g)
            {
                // MEMORY_BASIC_INFORMATION (x64): RegionSize at 0x18, State at 0x20.
                let region = u64::from_le_bytes(info[0x18..0x20].try_into().unwrap_or_default());
                let state = u32::from_le_bytes(info[0x20..0x24].try_into().unwrap_or_default());
                const MEM_COMMIT: u32 = 0x1000;
                report.check(
                    "XMemVirtualQuery graphics block",
                    ok && state == MEM_COMMIT && info[0x30] & 3 != 0,
                    &format!("size={region}"),
                );
            }
            xmem.free(h, heap);
            xmem.free(g, gfx);
        }
        None => report.check("xmem exports", false, "missing"),
    }

    // d3d12_xs forwarders -> d3d12.dll (vkd3d-proton under Wine).
    let serialize = Module::loaded("d3d12_xs.dll").and_then(|m| m.export(c"D3D12SerializeVersionedRootSignature"));
    report.check(
        "d3d12_xs!D3D12SerializeVersionedRootSignature resolves",
        serialize.is_some(),
        &pointer(serialize.unwrap_or(0)),
    );
    report.check("d3d12.dll pulled in by forwarder", Module::loaded("d3d12.dll").is_some(), "");

    // HTTP client library found by export: version, then init and cleanup (pure library, no
    // network).
    match exporting(c"HCGetLibVersion") {
        Some(module) => {
            let client = HttpClient::find(module);
            let (hr, version) = client.version();
            report.check(
                "HCGetLibVersion",
                hr >= 0 && version.is_some(),
                &format!("hr={} ver={}", hresult(hr), version.as_deref().unwrap_or("-")),
            );
            let hr = client.initialize();
            report.check("HCInitialize(NULL)", hr >= 0, &format!("hr={}", hresult(hr)));
            if hr >= 0 {
                client.cleanup();
            }
        }
        None => report.check("HTTP client library present", false, "no module exports HCGetLibVersion"),
    }

    // CURL wrapper found by export: the curl_version() string.
    let curl = exporting(c"curl_version").and_then(sys::curl_version);
    report.check("curl_version", curl.is_some(), curl.as_deref().unwrap_or("-"));

    // GPU middleware loader found by export: ffxCreateContext needs a real D3D12 device, so only
    // the export is checked.
    let ffx = exporting(c"ffxCreateContext");
    report.check("ffx loader ffxCreateContext export", ffx.is_some(), &pointer(ffx.map_or(0, Module::address)));
}
