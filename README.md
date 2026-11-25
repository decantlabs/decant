# Decant

Decant runs Xbox Series (GDK, Scarlett) titles natively on PC.

The Xbox Series CPU is x86-64, like a PC, so title code runs as it is. What a PC lacks is the console's GPU programs and its system software. Decant supplies both.

> [!NOTE]
> The repository is private for now. Decant is a work in progress. Real titles boot to their first screens in testing, but nothing is claimed as playable yet.

## How it works

Decant has two halves.

**The GPU half.**

* **Shader translator.** Console shaders are precompiled RDNA2 (GFX10) machine code. `decant-shader` translates them to HLSL, DXC compiles that to DXIL, and the result runs on native Direct3D 12 or on vkd3d-proton. See [docs/SHADER_TRANSLATOR.md](docs/SHADER_TRANSLATOR.md).
* **D3D12.X layer.** `d3d12_xs.dll` exposes the console's Direct3D 12 interfaces, in the console's method order, on top of a PC Direct3D 12 device. It handles root signatures, pipelines, texture tiling, frame pacing and presentation. See [docs/D3D12X_LAYER.md](docs/D3D12X_LAYER.md).

**The system half.** Reimplementations of the console APIs a title calls:

* memory (`XMem`), the DirectStorage console ABI, `xgameruntime`, GameInput
* XAudio2, the AcpHal and SHAPE audio stack, XAPU, the XDSP library (`chad`), XMA2
* XG texture layout, BCPack texture decoding, ray tracing acceleration structure builds, FidelityFX forwarding

Around them are a Windows launch tool (`decant-launch`, which runs a staged GDK title under the debugging API), a Linux and Proton launch driver (`launch_prep` of the `decant-linux` crate), and diagnostics.

## Status

Last updated: 25 November 2026. Details and limits are in [STATUS.md](STATUS.md).

| Area | State |
|---|---|
| Shader translation | Console shaders translate to HLSL and DXIL and compile. Execution checks on a real GPU cover a small share of the corpus so far |
| D3D12.X layer | Exists and is tested on a real GPU through console method slots. Pipelines, draws and presentation for real titles are still being completed |
| Console API layers | XMem, DirectStorage, `xgameruntime`, GameInput, XAudio2, AcpHal, XAPU, XDSP, XG, BCPack, BVH and FidelityFX forwarding exist and have tests |
| Launch tools | `decant-launch` on Windows, `launch_prep` for Wine and Proton, import auditing and stand in DLL generation |
| Real titles | Real titles boot to their first screens in testing. Nothing is claimed as playable |

## How to build

You need Windows 11 with the Rust toolchain pinned in `rust-toolchain.toml` and the Visual Studio C++ build tools.

`setup.ps1` checks every tool the project needs (Git, Visual Studio C++ tools and Windows SDK, Rust with the pinned toolchain, LLVM, CMake, Ninja, the Vulkan SDK and a `spirv-val` 2026.x) and offers to install each missing one with winget, asking first. `-Check` only reports, `-Yes` installs without asking.

```bat
powershell -ExecutionPolicy Bypass -File setup.ps1 -Check
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`build.cmd` builds the DLLs and test programs with cargo into `build\` (set `DECANT_BUILD` to use another folder):

```bat
build.cmd
```

Optional environment variables for `build.cmd`:

* `DECANT_GDK_ROOT`: the folder of the public GDK, to build the online test of `xgameruntime`.
* `DECANT_GDEFLATE_ROOT`: the `GDeflate` folder of a microsoft/DirectStorage checkout, to link the GDeflate decoder of the CPU fallback.

PC DirectStorage is loaded at run time, so it is not needed to build.

The tools and tests build with cargo:

```bat
cargo build --release --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

On Linux, `make -C src/gdk` cross builds the DLLs with mingw-w64 and `launch_prep` stages and launches under Wine and Proton. See [src/gdk/README.md](src/gdk/README.md). To build the vkd3d-proton patch, see [third_party/vkd3d-proton/README.md](third_party/vkd3d-proton/README.md).

## Repository layout

The crates are in `crates/`, all named `decant-*`. The full list with one line each is in [crates/README.md](crates/README.md).

| Group | Crates |
|---|---|
| Shader translation | `decant-shader`, `decant-dxc`, `decant-dxil`, `decant-extract-shaders`, `decant-shader-*` |
| Console layers | `decant-d3d12x`, `decant-xmem`, `decant-dstorage`, `decant-dstorage-xcrdapi`, `decant-xgameruntime`, `decant-gameinput`, `decant-xaudio2`, `decant-xma`, `decant-xma2`, `decant-acphal`, `decant-xapu`, `decant-chad`, `decant-xg`, `decant-texture`, `decant-bcpack`, `decant-bvh`, `decant-ffx` |
| Launch | `decant-launch`, `decant-linux`, `decant-gdk-config`, `decant-stub`, `decant-log` |
| Tools and tests | `decant-audit-imports`, `decant-gpu-diag`, `decant-smoketest`, `decant-repo-check`, `decant-testkit`, `decant-spirv-exercise` |

Other folders:

* `docs/`: architecture notes, ABI specifications and the GFX10 tables. Start with [ARCHITECTURE.md](ARCHITECTURE.md).
* `src/gdk/`: the Linux build of the console layers and the launch stack, see its README.
* `sandbox/`: an air gapped container and a Windows Sandbox loop for running untrusted code.
* `third_party/`: code under other licences, with its own README.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Hardware testers can follow [docs/GPU_TESTING.md](docs/GPU_TESTING.md).

## Licence

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 the Decant contributors. Contributors are listed in [CREDITS.md](CREDITS.md).
