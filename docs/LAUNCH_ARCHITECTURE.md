# Launch architecture

How a Scarlett GDK title is started with Decant, on Windows and on Linux. Commands and options are in [src/gdk/README.md](../src/gdk/README.md). The layers themselves are described in [ARCHITECTURE.md](../ARCHITECTURE.md).

Nothing derived from a package is committed here. The repository records relations and counts, not the file names, import lists or offsets of a package.

## What a title needs

```
title executable (x86-64 PE, GDK, TargetDeviceFamily=Scarlett)
 ├─ xgameruntime.dll .......... GDK runtime, loaded by name       (PC runtime, or decant-xgameruntime in front of it)
 ├─ d3d12_xs.dll .............. Scarlett D3D12.X                    (decant-d3d12x)
 ├─ xmem.dll .................. console allocator                   (decant-xmem)
 ├─ console audio, input ...... AcpHal, xapu, chad, GameInput       (decant-acphal, decant-xapu, decant-chad, decant-gameinput)
 ├─ xg_xs.dll, dstorage ....... layout and storage                  (decant-xg, decant-dstorage)
 ├─ MSVC 14.x CRT ............. the package ships its own copies
 └─ companion DLLs ............ GDK client libraries and middleware
 Data: package archives, loose data folders, MicrosoftGame.config
```

The title imports console modules that Windows does not provide. A missing function or a missing data import (the title imports `IID_ID3D12Device` from `d3d12_xs.dll`, for example) stops it from loading at all, so `audit_xbox_imports` runs first. `decant_stubgen` writes a logging stand in DLL for each module that nothing provides.

## Windows

`decant-launch` stages the package into a writable overlay folder with our DLLs in front, then runs the title under the debugging API. Staging never writes to the package. The shims sit in the application folder of the title, which Windows searches first for any DLL that is not a known system DLL, so no loader trick is needed. The debugger sees every thread from its first instruction and every exception before the title does, and logs every DLL, debug line and exception.

Who answers which calls is in the table of [src/gdk/README.md](../src/gdk/README.md#who-answers-what-on-windows).

The installer must make sure Gaming Services is installed (it comes from the Microsoft Store). A title stopping right after `XGameRuntimeInitialize` with the runtime reported missing points there.

## Linux

`launch_prep` of the `decant-linux` crate prepares a Wine and Proton stack: it downloads the pinned [Ferestre](https://github.com/icex/ferestre) runtime and verifies it against a trust anchor in the repository, builds the shims, creates the prefix, stages the package and launches. Ferestre supplies the GDK API (a Wine builtin `xgameruntime.dll`, patches for GDK packages, a pinned vkd3d-proton). Our DLLs cover what it lacks, the Scarlett only modules.

```
 launch_prep: runtime → stubs → prefix → stage → test → launch (inside sandbox/)
 Ferestre runtime (Proton, Wine fork)               install/title/ (staged, ignored by git)
 ├─ xgameruntime.dll, gameinput.dll  [Ferestre]     ├─ title executable        (user supplied)
 ├─ xaudio2_9.dll                    [Wine, FAudio] ├─ MicrosoftGame.config
 ├─ d3d12, d3d12core                 [vkd3d-proton] ├─ archives, data folders  (symbolic links, read only)
 └─ dxgi                             [DXVK]         ├─ MSVC CRT, companion DLLs (from the package)
                                                    └─ xmem.dll, d3d12_xs.dll  [our DLLs]
```

Everything that only reads files is native Linux code: the shader tools, the texture layout code and the `decant-gdk-config` reader. `cargo test --workspace` runs on Linux directly. Wine is needed only to execute Windows code: the package's DLLs, our DLLs and their tests, and the title. That runs inside [sandbox/](../sandbox/README.md), a container without network.

The Review workflow job `wine` cross builds every GDK DLL for `x86_64-pc-windows-gnu` and runs the test programs under Wine, see [CI.md](CI.md).

### Debugging under Wine

`WINEDEBUG=+loaddll,+seh,err+module` and per channel traces, `winedbg --gdb` for a gdb server on the PE process, Proton's `PROTON_LOG=1`. Our DLLs log through `OutputDebugString` (`+debugstr`). `DECANT_LOG=1` makes `xmem.dll` log through `OutputDebugString`, `DECANT_LOG_PATH` writes a session log. The switches that show what a title does are in [DEBUG_TOOLS.md](DEBUG_TOOLS.md).

In a bare `wineboot` prefix the `d3d12_xs` forwarders fail, because the builtin `d3d12.dll` of Wine needs `libvkd3d-*.dll`. A Proton prefix installs vkd3d-proton, use `launch_prep prefix`.

Wine leaves `PROCESS_MEMORY_COUNTERS_EX.PrivateUsage` at 0. `xmem.dll` falls back to `PagefileUsage`, then `WorkingSetSize`.

## First run of a title

1. Pre flight: the executable must be a plain PE32+ x86-64 file the user may run. If it is encrypted, stop, nothing here decrypts packages. `launch_prep` and `decant-launch` run this check with `decant-gdk-config`.
2. Import diff with `audit_xbox_imports`. Every unresolved name is a stub to write.
3. For a symbol one of our DLLs lacks, add it as a logged `E_NOTIMPL` first, so the first call is visible. For a whole module nothing provides, `decant_stubgen`. Repeat step 2 until nothing is unresolved.
4. Stage and launch inside `sandbox/` (Linux) or Windows Sandbox (Windows). Read the log bottom up: the last unimplemented `d3d12_xs` slot, the last `xgameruntime` call, or the fault frames.
5. Classify the stop: a missing or untrue API (fixable), a console assumption (fixed virtual addresses, GPU address equal to CPU address, hardware queues), or the title refusing to run (out of scope).
6. Switch off what cannot work through the title's own settings: the upscaler, frame generation and ray tracing, unless the matching layer is in place.

## Pitfalls measured on retail titles

Each line is a lead to confirm on the executable at hand, not a rule. The D3D12.X and DirectStorage ones are in [D3D12X_LAYER.md](D3D12X_LAYER.md) and [DSTORAGE_ABI.md](DSTORAGE_ABI.md), the XMem ones in [CONSOLE_LIBS.md](CONSOLE_LIBS.md) section 5.

* **Files** go through `CreateFile2`, not `CreateFileW`, with paths relative to the package root. A clean exit (code 1) right after the GDK services started looked like a runtime refusal, it was the engine not finding its project file. Trace `CreateFile2`. The Xbox game drive `G:` is mapped to the overlay for the length of a run.
* **`xgameruntime.dll` is not imported.** The title loads it by name, then resolves `QueryApiImpl`.
* **User.** The PC GDK runtime has no user for an unregistered title, so sign in is abandoned. The `XUser*` and `XStore*` functions are linked statically into the executable.
* **Frame pacing.** Without `WaitFrameEventX` and the flip events, the game thread waits for a frame start that never comes and the engine's watchdog ends the process after about 90 s. `d3d12_xs` emulates the frame events (device slots 78 to 80).
* **Command lists** are created open on PC. A retail engine expects them closed and calls `Reset` first, which crashed the NVIDIA driver. See [D3D12X_LAYER.md](D3D12X_LAYER.md).
* **Shaders.** The DXIL is stripped and native RDNA2 code compressed with zlib is shipped instead, so there is a black screen without the translator. FXC took minutes on some shaders and tripped the watchdog, DXC compiles in a second.
* **Incomplete package.** Files missing from an extracted image give a misleading fatal error. Check the integrity of the package before looking further.
* **Network.** Once pipelines are created the title calls real servers through WinHTTP. The sandbox has no network, keep it that way.

## Risks

| Risk | Impact | Notes |
|---|---|---|
| No runnable plain PE, console packages are encrypted | Blocks everything | Out of scope, no key or DRM work is done or planned |
| Console assumptions in memory: fixed virtual addresses, large pages, GPU visible allocations | Early crash | `xmem.dll` distinguishes heap and graphics, serves placeholders and physical pages |
| Console built GPU middleware (BVH builder, upscaler kernels) | Feature loss | The DLLs load but must not be driven, use the BVH and FidelityFX layers or switch the feature off |
| Static GDK client behaves as console (device type, runtime checks) | Early exit | Answer as Scarlett where the config says so |
| Integrity checks in a retail build | Silent exit | First suspect our own untrue answers, no bypass work |
| Online services for a console title from a PC client | No online | Offline only, the sandbox has no network anyway |
| Provenance of title data | n/a | Title files, runtimes and generated reports are never committed (`.gitignore`, `check_repository`) |
