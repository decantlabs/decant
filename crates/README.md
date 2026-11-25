# Crates

The workspace is at the repository root. Every component is a Rust crate named `decant-*`. The toolchain is pinned in `rust-toolchain.toml`.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

On Windows, `build.cmd` builds the DLLs and test programs into `build\` (see the [README](../README.md#how-to-build)).

## Shader translation

| Crate | Content |
|---|---|
| `decant-shader` | The translator: console GFX10 (RDNA2) machine code to HLSL in one step, with the real root signature and every stage of the pipeline in hand. See [docs/SHADER_TRANSLATOR.md](../docs/SHADER_TRANSLATOR.md) |
| `decant-dxc` | The one DXC loader: finds `dxcompiler.dll` and `dxil.dll` at run time and compiles HLSL to signed DXIL |
| `decant-dxil` | A DXIL writer without LLVM, and a SPIR-V importer for compute modules |
| `decant-extract-shaders` | `extract_shaders`: finds console shader containers in files by their own headers and isolates the GPU code of each shader |
| `decant-shader-core` | Shared part of the shader research tools: metadata rows, location tables, SPIR-V relocation. Binaries `decant_shader_locate` and `decant_shader_link` |
| `decant-shader-batch` | `decant_shader_batch`, `decant_shader_summary`, `decant_shader_one` and `decant_shader_cache`: batch translation, replay and the translation cache |
| `decant-shader-analysis` | Read only research tools: signatures, resource lists, binding tables and input convention checks |
| `decant-shader-bridge` | `decant_shader_identify`, `decant_shader_adapt`, `decant_shader_serve`, `decant_shader_describe` and `decant_shader_adapt_cache`: the bridge to vkd3d-proton and its offline adaptation cache |
| `decant-shader-harness` | On screen test drivers, the shader gallery, validation by pipeline creation, semantic cases and the translator regression guard |
| `decant-shader-tools` | More research tools: capability count, HLSL probe, instrumentation, SPIR-V reflection and image checks |
| `decant-shader-bench` | `decant_shader_run` (Vulkan through `ash`) and `vkd3d_test` (Direct3D 12): the GPU test benches |
| `decant-shader-ab` | GPU A/B comparison of two shaders on a real D3D12 device with the same generated inputs |
| `decant-spirv-exercise` | A validator exercise: a minimal SPIR-V module and invalid variants checked with a local SPIRV-Tools |

## Console layers

| Crate | Content |
|---|---|
| `decant-d3d12x` | `d3d12_xs.dll`, the D3D12.X layer over PC D3D12: root signature helpers, console ABI proxies in the measured method order, frame events. Status in [docs/D3D12X_LAYER.md](../docs/D3D12X_LAYER.md) |
| `decant-xmem` | `xmem.dll`, the console allocator on Windows virtual memory, with its tests `loadtest` and `xmem_abitest` |
| `decant-dstorage` | `dstorage_xs.dll`, the DirectStorage layer: console ABI read from spec files, over PC DirectStorage or a CPU fallback, with zlib, GDeflate and BCPack. See [docs/DSTORAGE_ABI.md](../docs/DSTORAGE_ABI.md) |
| `decant-dstorage-xcrdapi` | `xcrdapi.dll`, the same layer under the second name a title may import `DStorageGetFactory` from |
| `decant-bcpack` | BCPack decoder for BC1, BC3, BC4, BC5, BC6H and BC7 |
| `decant-texture` | GFX10 texture tiling, surface layout and XG resource layout computation |
| `decant-xg` | `xg_xs.dll`, the console texture and resource layout library |
| `decant-xgameruntime` | `xgameruntime.dll`, the interposer that logs what a title asks the game runtime and forwards to the PC runtime |
| `decant-xaudio2` | `decant_xaudio2.dll`, the XAudio2 layer over `XAudio2_9.dll` |
| `decant-xma` | XMA2 decoding in pure Rust, for the XMA contexts of AcpHal |
| `decant-xma2` | `decant_xma2.dll`, the optional XMA2 decoder plugin of `decant_xaudio2.dll`, built on `decant-xma` |
| `decant-acphal` | `AcpHal.dll`, the audio control processor library: APU heaps, SHAPE contexts, the `IAcpHal` object |
| `decant-xapu` | `xapu.dll`, the XAPU library: Opus decoding and sample rate conversion |
| `decant-chad` | `chad.dll`, the XDSP library: partitioned convolution and FFT streams |
| `decant-gameinput` | `GameInput.dll`, a shim with one always connected gamepad driven by a control file, and `decant-pad`, the tool that writes to that file |
| `decant-ffx` | `decant_ffx.dll`, the FidelityFX shim that forwards to a PC backend you name |
| `decant-bvh` | `decant_bvh.dll`, ray tracing acceleration structure builds on native D3D12 objects |

## Launch

| Crate | Content |
|---|---|
| `decant-launch` | `decant-launch`, the Windows tool that stages a title into an overlay folder with our DLLs first, runs it under the debugging API and logs every DLL, debug line and exception |
| `decant-linux` | The Linux and Proton launch driver (`launch_prep`) and the sandbox runners `run_airgapped` and `test_isolation` |
| `decant-gdk-config` | `gdk_config_reader`: reads `MicrosoftGame.config` and checks the PE headers of the title executable |
| `decant-stub` | Generates stand in DLL projects for console modules a title imports and nothing provides |
| `decant-log` | The session log shared by the DLLs (`DECANT_LOG`, `DECANT_LOG_PATH`) |

## Tools and tests

| Crate | Content |
|---|---|
| `decant-audit-imports` | `audit_xbox_imports`: lists the imports of a GDK build and what this PC or our DLLs can resolve |
| `decant-gpu-diag` | `gpu_diag`: adapters and capabilities, then the native hardware tests, as a record you review before sharing. See [docs/GPU_TESTING.md](../docs/GPU_TESTING.md) |
| `decant-smoketest` | `gdk_smoketest`: loads the staged DLLs and runs light functional checks |
| `decant-repo-check` | `check_repository` and `third_party`: the review check of tracked paths and text, and the setup of the pieces in `third_party/` |
| `decant-testkit` | Helpers of the test programs: loading a DLL and printing checks |

Several crates have their own README with their ABI details.
