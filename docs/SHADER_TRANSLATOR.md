# Shader translator

Console GPU programs are precompiled RDNA2 (GFX10) machine code. A PC GPU cannot run them, so Decant translates them. The translator is written in Rust under the project licence (MIT) and was written clean room, from Mesa's MIT licensed AMD sources, the LLVM AMDGPU backend, public AMD and Microsoft documents, and measurements of console shaders.

## How it works

1. **Container.** `decant-shader` reads the console container: code, signatures, declared resources and the binding table.
2. **Decode and lift.** The machine code is decoded, split into basic blocks and lifted to a small intermediate representation. One file per instruction class covers the lifting.
3. **Structure.** Blocks become structured statements: `if`, `else`, and loops with `break` and `continue`.
4. **Resources.** Descriptor loads are traced back to their source and placed against the real root signature of the pipeline.
5. **Stages.** All stages of a pipeline are translated in one call, so the input of a pixel stage is made from the output signature of the stage before it.
6. **HLSL and DXIL.** The result is HLSL, which DXC compiles to DXIL. `decant-dxil` is a DXIL writer without LLVM, used for the SPIR-V importer route and for modules that HLSL cannot carry. DXIL runs on native Direct3D 12 and on vkd3d-proton.

A program the translator cannot carry yet is refused with a stable key such as `op:v_sqrt_f32` or `resource:raw-address`, so a corpus can be counted by cause.

A console compute program that works across a 64 lane wave is printed with `[WaveSize(64)]`. For a host whose waves have at most 32 lanes (`WaveMode::Emulated`, `hlsl/wave_emu.rs`) one console wave becomes 64 threads of the group in group index order, and lane reads and 64 lane ballots go through group shared memory between group barriers. The form is taken only when every thread reaches every such call. Otherwise the program is refused as `stage:wave64-on-32-lane-host`. The thread order is an assumption that was not measured on a console.

## Crates

| Crate | Content |
|---|---|
| `decant-shader` | The translator: tables, decoder, lifting, structuring, resource placement, HLSL printer, cache key and census tool |
| `decant-dxil` | DXIL writer and SPIR-V importer |
| `decant-dxc` | Helper around DXC |
| `decant-shader-core`, `decant-shader-batch`, `decant-shader-analysis`, `decant-shader-bridge`, `decant-shader-harness`, `decant-shader-tools`, `decant-shader-bench`, `decant-shader-ab` | Research, batch, bridge and GPU test tools |
| `decant-extract-shaders` | Finds console shader containers in files you provide and isolates the GPU code of each shader |

The method, the measurements and the tools are in [SHADERS.md](SHADERS.md). The instruction encoding and semantics tables are in [gfx10/](gfx10/encoding.md).

## Licences

The translator is MIT like the rest of the project. The patch to vkd3d-proton in `third_party/vkd3d-proton/` is LGPL 2.1, like vkd3d-proton itself.
