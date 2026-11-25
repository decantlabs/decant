# Shader work: method, results and tools

This page describes how console shaders are studied and what has been verified. The translator itself is described in [SHADER_TRANSLATOR.md](SHADER_TRANSLATOR.md). This repository ships tools only. It does not ship console shader binaries, translated modules or metadata derived from them. You produce those locally from your own titles.

## The chain

1. **Container.** Each console shader is a DXBC container (Shader Model 6.6) that holds no DXIL bytecode. An `XDAT` part holds an `XBCH` block, zlib compressed, which contains an `FSSM` 1.1 container with RDNA2 (GFX10) machine code, the binding tables and the signatures.
2. **Translation.** The machine code is decoded and translated. The main route is `decant-shader`, which produces HLSL for the whole pipeline in one step, then DXC compiles it to DXIL. An earlier route went through SPIR-V and ran on vkd3d-proton.
3. **Validation.** SPIR-V modules are checked with `spirv-val`. DXIL modules are checked with `dxil.dll` and by creating the pipeline on a real driver. This is a structural check with placeholder inputs.
4. **Execution check.** Some modules run on a real GPU and the image is compared with a value computed independently.

Valid does not mean correct. A module can pass every structural check and still compute a wrong image, which is why step 4 exists.

## Results

The figures below were measured on the shader corpus of a retail title (tens of thousands of programs), with the SPIR-V based translator that preceded `decant-shader`.

| Check | Result |
|---|---|
| Instruction decoding | all programs but one decoded |
| Opcode tables against Mesa's ACO GFX10 numbering | 659 identical, 67 the same instruction under its GFX10 name, 0 mismatches |
| Instructions reached in the corpus | tens of millions, none unrecognised |
| Structural validation with `spirv-val` (Vulkan 1.2, `VK_KHR_maintenance8` assumed) | every decoded program, on every stage (pixel, mesh, compute, vertex, geometry) |
| Modules that use a fallback instead of a proof | 22 (a null texture descriptor or a fixed sampler) |
| Modules that read memory by address | about 47 percent, including every mesh shader |
| Execution on a real GPU against an independent result | simple reference shaders conform on direct Vulkan and through Direct3D 12 and vkd3d-proton, plus a few real shaders on real decoded textures |

Mesh modules need SPIRV-Tools 2026.x to validate. Version 2025.1 rejects every one of them.

### Conventions established from the shaders themselves

| Topic | Finding |
|---|---|
| Vertex inputs | Vertex shaders start by calling a routine supplied at run time. The input element with signature register r and component c arrives in vector register `4 + 4r + c`, the vertex index in register 0 and the instance index in register 3 |
| Pixel inputs | Hardware attribute k is the kth non system input register with at least one component read |
| Vertex outputs | The jth non system output register is exported as parameter j |
| Resource bindings | A table after the code maps each Direct3D register to a descriptor table register and an offset in 32 bit words |
| Heaps | Records of kind 5 and 6 in the first table after the code give the scalar registers that hold the resource heap and the sampler heap |

### wave64 on 32 wide GPUs

Console shaders run 64 lanes per wave. NVIDIA and Intel GPUs run 32. The translator emulates wave64 exactly on every stage (compute, mesh, vertex, pixel, geometry, ray query) with two halves, chosen automatically from the device subgroup size. The full corpus has not been measured in 32 lane mode yet.

## Tools

Every tool is a binary of the Rust workspace. Build them with `cargo build --release` in the repository root.

| Tool | Purpose |
|---|---|
| `decant_shader_direct_census` (`decant-shader`) | Count instructions, encoding forms and program shapes over a corpus, and report the first refusal of each program |
| `decant_shader_metadata`, `decant_shader_outputs` | Input and output signatures and resource lists of every shader |
| `fssm_tables`, `fssm_bindings` | Dump, decode and verify the binding tables placed after the code |
| `decant_shader_batch`, `decant_shader_summary` | Translate and validate in batches, then group failures by cause |
| `decant_shader_one` | Replay selected shaders |
| `decant_shader_locate`, `decant_shader_link` | Put module inputs and outputs at Direct3D signature registers, or link a pixel module to a vertex module |
| `decant_shader_identify`, `decant_shader_adapt`, `decant_shader_adapt_cache` | Give each resource of a translated module its Direct3D register, rewrite the resources for a target layout, and keep the result in an offline cache |
| `decant_shader_describe` | Write a module and a JSON description for each shader (stage, signatures, resources, push constants, known limits) |
| `decant_shader_regress` | Regression guard: translate a fixed sample with two builds, compare the modules, run `spirv-val` again and report what changed |
| `decant_shader_run`, `vkd3d_test`, `decant_shader_gallery` | GPU benches: Vulkan direct, Direct3D 12, and a bulk run of many pixel shaders on synthetic inputs |
| `decant_shader_capabilities`, `decant_shader_hlsl_probe`, `spirv_reflect` | Count modules that rely on addressed memory, probe HLSL and DXIL conversion, and read the interface of a SPIR-V module |
| `decant_shader_dxil_cache`, `decant_shader_spv_measure` (`decant-dxil`) | Build DXIL override files from a SPIR-V cache and measure the SPIR-V to DXIL importer |

The crates behind these tools are listed in [crates/README.md](../crates/README.md).

## The vkd3d-proton bridge

`third_party/vkd3d-proton/vkd3d-decant-shader.patch` adds 57 lines to `libs/vkd3d-shader/dxil.c`. It does nothing unless `VKD3D_DECANT_SHADER_DIR` is set. When it is set, the hook writes the container and the interface vkd3d-proton chose for each shader, then runs `VKD3D_DECANT_SHADER_COMMAND`. That command places a translated module in `VKD3D_SHADER_OVERRIDE`. See [third_party/vkd3d-proton/README.md](../third_party/vkd3d-proton/README.md).

Known limits of the bridge:

* Files are keyed by the hash of the container only. The interface vkd3d-proton chose depends on the root signature, so the same shader compiled for two root signatures at the same time can overwrite the other's files.
* The hook runs before the mesh shader check, so mesh shaders start a translation that is never used.
* Every bound buffer size read by a translated shader is replaced by 65,536 bytes, the largest constant buffer view.

## Open work

| Item | What is missing |
|---|---|
| Resource arrays, heaps indexed at run time, several immediate samplers, root constants | Not handled by the rewriting tool |
| Mesh shaders through vkd3d-proton | vkd3d-proton reflects the bytecode before it replaces a mesh shader |
| Memory read by address | An address table kept up to date on the Direct3D side |
| Voxelization style geometry shaders | The real output topology and an interface that matches the vertex shader in front of them |
| Execution checks | Wider coverage of real shaders on real data, and the full corpus in 32 lane mode |

## How to help

Review the method and the statistics, extend the test benches with new synthetic cases, or work on an item above. Follow [CONTRIBUTING.md](../CONTRIBUTING.md). Keep every input synthetic.
