# Component evidence

Last updated: 25 November 2026.

Real titles boot to their first screens in testing. Nothing is claimed as playable.

Each row gives what was measured and where the measurement stops. A summary is in the [README](README.md#status).

## Shaders

| Component | Measurement | Boundary |
|---|---|---|
| Decoding | 26,502 of the 26,503 programs of a retail title decode | One program remains |
| Translation | Every program of that corpus translated by the earlier SPIR-V route and accepted by `spirv-val` (Vulkan 1.2): pixel 12,101, mesh 9,282, compute 2,689, vertex 2,236, geometry 195 | Structural validity only. Valid does not mean correct. 22 modules use a fallback, not a proof. The direct HLSL and DXIL route of `decant-shader` is newer and covers a smaller share |
| wave64 on 32 wide subgroups | Two halves emulation on every stage, chosen from the device | The full corpus has not been measured in 32 lane mode |
| Output on a real GPU | Simple reference shaders conform on direct Vulkan and through Direct3D 12 and vkd3d-proton. A few real shaders run on real decoded textures and give the expected result | A handful of real shaders so far |
| Batch bench | Runs translated pixel shaders one process each on synthetic inputs and classifies the output | Synthetic inputs. A few post processing shaders can lose the device and are skipped |
| Load time | 1,344 ms down to 9 ms per shader with the ahead of time translation cache | One machine |

## Console layers

| Component | Measurement | Boundary |
|---|---|---|
| XMem | Synthetic checks pass. A concurrent free of the same block is safe (8 threads, 128 blocks, 300 rounds). All 10 functions a title imports are exported, including the physical page functions on Windows placeholders | `XMemTransferMemory` logs its arguments until its signature is known. The frame unit is 4 KiB where the specification counts 64 KiB |
| XAudio2 layer | 14 synthetic checks, including a 440 Hz sine played through the layer | PC engine only |
| GameInput | `GameInputCreate` and `IUnknown` checked, same API as console | Dynamic Latency Input (console only) is not implemented |
| AcpHal, XAPU, chad, XG | Implemented from specifications in [docs/CONSOLE_LIBS.md](docs/CONSOLE_LIBS.md) and tested through their exports | Specifications are facts to confirm on a title |
| BCPack | BC1, BC3, BC4, BC5, BC6H and BC7 decoding | Checked on synthetic data and calibrated tables |
| BVH builds | One triangle, 2,816 bytes reserved, 1,152 bytes compacted | Bottom level builds only, on native GPU addresses |
| FidelityFX | Context lifecycle and 32 jitter phases, 1280x720 input to 2560x1440 output | Query only, no frame dispatch |
| D3D12.X layer | 52 of 52 real root signature containers serialize again through the layer (0 of 52 on PC D3D12 alone). Console ordered proxies for device, command list and queue, with the order measured on console modules. A command list ran through a proxied queue on an RTX 4080 | Presentation is a hook only. Pipelines, draws and heap objects for real titles are being completed |
| DirectStorage (console) | Reads, queues, statuses, fences, zlib and GDeflate on synthetic files, over PC DirectStorage 1.3 and a CPU fallback. A synthetic caller reads through console vtables | Console interface IDs and method order come from each title's executable |
| Launch tools | `decant-launch` on Windows, `launch_prep` with an air gapped sandbox, an import auditor and a stand in DLL generator | Behaviour on a given title depends on that title |

Method, conventions, tools and open work for shaders: [docs/SHADERS.md](docs/SHADERS.md).
