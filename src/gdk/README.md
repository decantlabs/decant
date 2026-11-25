# GDK host layer

Pieces for loading a Scarlett (Xbox Series) GDK title's companion DLLs, and the title, under Wine and Proton on Linux or natively on Windows. The GDK API itself comes from the [Ferestre](https://github.com/icex/ferestre) runtime on Linux and from Gaming Services on Windows. This folder covers what they do not: the Scarlett only DLLs and the console family configuration. The architecture is in [docs/LAUNCH_ARCHITECTURE.md](../../docs/LAUNCH_ARCHITECTURE.md).

No title file, executable or runtime is included or downloaded by the build. `launch_prep runtime` downloads the version pinned Ferestre runtime release and verifies the archive against a trust anchor recorded in this repository before extracting, it never trusts a checksum the release itself ships.

| Part | Purpose |
|---|---|
| `crates/decant-d3d12x` | `d3d12_xs.dll`, the D3D12.X layer over the PC `d3d12.dll`. Status in [docs/D3D12X_LAYER.md](../../docs/D3D12X_LAYER.md) |
| `crates/decant-gdk-config` | `MicrosoftGame.config` reader (identity, TitleId, StoreId, MSAAppId, executables with `TargetDeviceFamily`) and the `gdk_config_reader` CLI: `--exec` prints the primary executable, `--pe-check` runs the minimal PE pre flight (DOS and PE headers, AMD64 machine, GUI or CUI subsystem, headers only, sees no encryption). `build.cmd` builds it with cargo |
| `crates/decant-linux` | `launch_prep`, the Linux driver: `runtime`, `stubs`, `prefix`, `stage`, `test`, `launch`, `all`, `status`, `env`. Build it with `cargo build --release -p decant-linux` in the repository root. The config reader is built in |
| `Makefile` | One cargo build for the target `x86_64-pc-windows-gnu` of every GDK DLL (`xmem.dll`, `decant_xaudio2.dll`, `decant_xma2.dll`, `xgameruntime.dll`, `dstorage_xs.dll`, `xcrdapi.dll`, `decant_ffx.dll`, `decant_bvh.dll`, `d3d12_xs.dll`) and of the test programs, copied into `build/gdk/`. The CI job `wine` does the same and runs the programs under Wine ([docs/CI.md](../../docs/CI.md)) |

Tests: `d3d12xs_test` (`cargo test -p decant-d3d12x` runs it, device checks skip on a machine without a D3D12 12_0 GPU), `xmem_abitest` (crate `decant-xmem`), the tests of `decant-gdk-config` (synthetic config, hand crafted PE header buffers) and `gdk_smoketest` of `decant-smoketest` (needs the package's companion DLLs, so it is not run in CI, it discovers them in its working directory instead of naming them).

## Windows launch tool: `decant-launch`

`crates/decant-launch` is the Windows counterpart of `launch_prep` of `decant-linux` (stage, then launch), as a Rust tool:

```bat
decant-launch stage --package D:\titles\package --overlay D:\titles\overlay --shims build [--vkd3d <folder>]
decant-launch run   --overlay D:\titles\overlay [--exe Name.exe] [--log run.log] [--timeout 60] [-- title arguments]
decant-launch all   --package ... --overlay ... --shims ... [--vkd3d <folder>]      # both
```

* **Staging.** The package folder is never written to. The overlay folder gets a junction for each package folder, a hard link for each package file (or a symbolic link when the overlay is on another volume, which needs the right to create them), and a copy of every `.dll` of `--shims`, which replaces a package file of the same name. A manifest (`.decant_launch_stage.txt`) lists what was made, so a new staging removes exactly that: links as links, never their targets. The overlay may not contain the package or sit inside it, and an existing file that no staging made is never overwritten. Windows has no overlay file system: a title that opens a linked file for writing writes the package's file. GDK titles write to their storage folders, not their install folder, but nothing enforces it.
* **Renderer.** `--vkd3d` (or `DECANT_VKD3D_DIR`) names a folder with the `d3d12.dll` and `d3d12core.dll` of vkd3d-proton, which are staged next to the title and logged with their SHA-256. Without it the title keeps the native D3D12.
* **DLL order.** The title starts from the overlay, so the shims are in its application folder. Windows searches that folder for any DLL that is not a known system DLL before anything else, which needs no loader trick. The log shows where each DLL really came from.
* **Launch.** The executable is `--exe`, or else the primary executable of the staged `MicrosoftGame.config` (`decant-gdk-config`: the first that is not dev only), and it must pass the PE pre flight. It runs under the Windows debugging API, with `DECANT_LOG=1` so xmem reports through `OutputDebugString`. The log (`decant_launch.log` in the overlay, and the console) has, with timestamps, every DLL the loader maps with its final path (`[title folder]` for the overlay's), every debug string, every exception with its code and chance, and the exit code with its name (`0xc0000135 DLL not found` when a shim is missing).
* **Game drive.** A title may hardcode the Xbox game drive `G:/` and assert when one of its install files is missing there. For the length of a run, `decant-launch` maps `G:` to the overlay folder with `DefineDosDeviceW`, as `subst` does (no admin rights), and removes it when the run ends, also on error or cancel. A `G:` that already points to the overlay is kept and left alone, one that points elsewhere (a real drive) is not touched: the debugger then rewrites the title's file paths that start with `G:/` or `G:\` (`CreateFileW` and `GetFileAttributesW` in kernelbase, the only calls the title makes for them) to the overlay folder, one log line for the mode and one for each distinct path. `DECANT_GAME_DRIVE_MODE=redirect` forces that mode even when `G:` is free (for tests). `DECANT_GAME_DRIVE=0` turns both off, a letter such as `H` uses that letter instead.
* **Test without a title.** `cargo build -p decant-launch --example stand_in_title` builds a stand in title that imports `xmem.dll` and `d3d12_xs.dll` as a Scarlett title does, calls them and reports through `OutputDebugString`.
* **Stand ins for absent console modules.** `decant_stubgen <imports_report.json> --out <folder> [--build]` (crate `crates/decant-stub`) writes, for each module the report lists as absent, a standalone Rust `cdylib` project (`Cargo.toml`, `build.rs`, `.cargo/config.toml`, `src/lib.rs`, no dependency, static C runtime). Each export logs its name through `OutputDebugString` (first 8 calls) and returns `E_NOTIMPL`. `--build` runs `cargo build --release` in each project and copies the DLL next to the projects. The export lists come from the title, so the generated projects and DLLs stay outside the repository.

### Who answers what on Windows

| Calls | Served by | Notes |
|---|---|---|
| GDK API reached through `QueryApiImpl`: `XGameRuntime*`, `XUser*`, `XStore*`, `XPackage*`, `XThread*`... | Microsoft's Gaming Services (`xgameruntime.dll` in the system folder) | In the public GDK these functions are compiled into the title from the static `xgameruntime.lib`, which calls the runtime through `QueryApiImpl`. The optional `xgameruntime.dll` interposer (below) only logs and forwards: nothing here answers a licence or store call. |
| `xmem.dll` (`XMem*`) | `crates/decant-xmem`, staged in the title folder | |
| `d3d12_xs.dll` | Rust crate `crates/decant-d3d12x`, staged in the title folder | Root signature helpers clean the console flags, then call `d3d12.dll` (app local vkd3d-proton first, else the system one), `D3D12XboxCreateDevice` / `D3D12CreateDevice` give the PC device inside the console ABI proxy ([docs/D3D12X_LAYER.md](../../docs/D3D12X_LAYER.md)) |
| Win32 thread affinity and priority | Windows itself | Plain Win32 calls, nothing to translate |

`decant-launch run` starts its log with the runtime's state: whether `xgameruntime.dll` is present, and the state of the `GamingServices` and `GamingServicesNet` services. It only reads it.

**Installer.** On the user's machine the installer must make sure Gaming Services is installed (it comes from the Microsoft Store). A title stopping right after `XGameRuntimeInitialize` with the runtime reported MISSING points there. On the development machine it is already installed and running (version 39.119.19001.0 when this was written).

### `xgameruntime.dll` interposer (`crates/decant-xgameruntime`)

A title reaches the game runtime through `InitializeApiImpl*` and `QueryApiImpl(api_id, riid, out)`, which hand back COM style tables. The interposer exports the same seven entry points as the PC runtime (`InitializeApiImpl`, `InitializeApiImplEx`, `InitializeApiImplEx2`, `QueryApiImpl`, `UninitializeApiImpl`, `XErrorReport`, `DllCanUnloadNow`, ordinals pinned by `build.rs`), logs every call, above all which `api_id` tables the title asks for and what the PC runtime answers, and forwards to the PC runtime (`DECANT_XGR_REAL`, default `System32\xgameruntime.dll`). `DECANT_XGR_LOG=<file>` names the log, otherwise it goes to `OutputDebugString`. `DECANT_XGR_TRACE=1` logs every call on a PC runtime table, `DECANT_XGR_PROXY=0` hands the PC objects out unwrapped.

Some tables are served locally, with or without a PC runtime, so that no table the title uses gets our user handle or our queues into the PC runtime (`DECANT_XGR_REAL_USER=1` forwards them when there is a PC runtime): XAsync, XTaskQueue and XThread, XUser with one offline user, XNetworking in an offline state that opens no socket, XDisplay with HDR unavailable, XPackage, XGame, XSystem and its analytics, accessibility, XGameSave (saves in a per user folder, `DECANT_XGS_ROOT` moves it), XPersistentLocalStorage (a host folder with the title's quota, `DECANT_PLS_DIR`) and XStore (the offline answer for an owned retail title, nothing bought and no store contacted). Every other table is forwarded to the PC runtime, or refused with `REGDB_E_CLASSNOTREG` when there is none. Every call on a local table is logged, bounded, with a heartbeat line every 5 s (queues, pending callbacks, pending async operations). The tables and their slot maps are data in `src/tables.rs` of the crate, and the module headers of the crate describe each one.

Without a PC runtime (Windows Sandbox has no Gaming Services), the initialisation is accepted and the other entries return `E_NOTIMPL`.

The crate builds `xgameruntime.dll` with cargo (`build.cmd` does it with a static C runtime, release, `--locked`) and `cargo test -p decant-xgameruntime` runs its test program `xgameruntime_test` with the modes `offline`, `online`, `preload` and `direct` (which compares the interposer with the PC runtime and needs no GDK). `online` and `preload` need the public GDK, set `DECANT_GDK_ROOT` when building.

Measured with the public GDK 2610 and Gaming Services 39.119.19001.0:

* The PC runtime exports the seven entry points above. It does not export `RestartApiImpl` or `RundownApiImpl`, which the `XCurl.dll` of a package names.
* The `XTaskQueue*` functions of the PC GDK ask for the table `{073B7DCB-1FCF-4030-94BE-E3C9EB623428}`, and the PC runtime serves it.
* Placed next to the executable, the interposer is not picked up, because the GDK loader finds the system copy. Loaded first (`LoadLibrary` of its full path, as a launch tool does), every runtime call goes through it.
* The `libHttpClient.GDK.dll` and `XCurl.dll` of a package load `xgameruntime.dll` themselves, by name, rather than importing it.

## Executable selection

The primary executable is the first `ExecutableList` entry without `IsDevOnly="true"`, dev only tools never launch. `gdk_config_reader --exec CONFIG` prints that name and exits 1 when the config has none, `DECANT_EXE_NAME` overrides it. `launch_prep` uses the same rule through the `decant-gdk-config` crate, so `stage` and `launch` never pick a dev only tool that happens to be listed first. `gdk_config_reader --pe-check FILE` is the pre flight `launch` runs before anything executes: a plain PE (MZ → PE signature), x86-64 machine, Windows GUI/CUI subsystem.

## Runtime pinning

The runtime version is pinned by `RUNTIME_TAG` in `crates/decant-linux/src/prep.rs`. The archive is verified against a trust anchor stored in the repository: a literal `RUNTIME_SHA256_PIN` digest, or a minisign public key (`RUNTIME_MINISIGN_PUB`) checking the release's `.minisig` signature. The committed pin is the digest of the pinned release archive, confirmed by verifying its `.minisig` against Ferestre's release signing key and hashing the archive locally, any mismatch makes `runtime` refuse to extract. `RUNTIME_SHA256` from the environment is an explicit user pin, the user then owns that trust decision, and a mismatch is always a hard failure.

## Status

`launch_prep status` reports every component, config reader, runtime archive (with its hash state), installed runtime, shims, Proton prefix, staged title directory, staged config, executable, launch mode, one ok/MISSING line per item. Nothing aborts the report, the exit code is nonzero only after the full report, when something is missing.

## Build and test

Windows: `build.cmd` builds `d3d12_xs.dll`, `d3d12xs_test.exe` and `gdk_config_reader.exe` and every other component with cargo. The config reader tests run with `cargo test -p decant-gdk-config`.

Linux (Rust target x86_64-pc-windows-gnu, Wine):

```sh
make -C src/gdk all tests   # every DLL and test program, needs gcc-mingw-w64-x86-64 as the linker
wine build/gdk/loadtest.exe     "$(winepath -w build/gdk/xmem.dll)"
wine build/gdk/xmem_abitest.exe "$(winepath -w build/gdk/xmem.dll)"
```

Without a build tool chain on Linux, download the workflow artifact `gdk-dlls-x86_64-pc-windows-gnu` of the `wine` job of a Review run: it holds the same DLLs and test programs.

Launch staging (needs a package you may analyse):

```sh
export DECANT_PACKAGE_DIR=/path/to/package       # read-only
launch_prep all       # runtime, shims, Proton prefix, staged install, smoke test
launch_prep status
launch_prep launch    # inside the air-gapped container; refuses until the title executable is staged
```

`launch_prep` writes only under the checkout (`runtime/`, `install/`, `logs/`, `build/`, all ignored by git). It never touches `~/.wine`, Steam or a system Wine, and leaves no temporary copies behind: its scratch directory is a temporary folder removed on exit.

## Launch: container first

`launch` runs package code only inside the air gapped [sandbox](../../sandbox/README.md) by default: it starts the sandbox (`network_mode: none`, read only package mount, isolation verified before anything runs) and executes the staged executable in that container. A host launch, untrusted package code with network access, against the project rules, requires `DECANT_HOST_LAUNCH=1` and prints a warning on every run. The smoke test (`test` step) also runs in the container by default, with `DECANT_HOST_LAUNCH=1` it runs on the host instead, with the same warning.

## DLL overrides

```
xmem,d3d12_xs=n                      shims (app-local)
d3d12,d3d12core,dxgi=n               vkd3d-proton / DXVK
the package's MSVC CRT set=n,b       native first, builtin fallback
xaudio2_9=b                          Wine/FAudio
xgameruntime,gameinput=b             Ferestre builtins
```

(The exact override string is `DLL_OVERRIDES` in `crates/decant-linux/src/prep.rs`. Wine needs the literal module names there.)
