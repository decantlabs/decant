# decant-shader

The direct shader translator: console shader programs, GFX10 (RDNA2) machine code in
FSSM containers, become HLSL in one step at the moment the title creates a pipeline, with the real
root signature and every stage of the pipeline in hand. DXC compiles the HLSL to DXIL, which runs on
native Direct3D 12 and on vkd3d-proton.

## Clean room

The crate is written clean room. Sources:

* Mesa `src/amd` (MIT): the instruction table (`aco_opcodes.py`) and the bit layouts
  (`aco_assembler.cpp`).
* The LLVM AMDGPU backend (Apache 2.0 with the LLVM exception) and public AMD documents.
* Microsoft's HLSL, DXC and DXIL documentation.
* This project's own crates, specifications and measurements on the title's shaders.

## Use

```rust
use decant_shader::{CountingHost, Options, RootSignature, container, translate};

let vs = container::program(&vertex_container_bytes, "vs")?;
let ps = container::program(&pixel_container_bytes, "ps")?;
let translation = translate(&[vs, ps], &root_signature, &Options::default(), &host)?;
for stage in &translation.stages {
    // stage.hlsl, stage.entry, stage.profile, stage.dxc_args go to DXC.
    // stage.resources and stage.needs say what the layer has to bind.
}
```

`translate` takes all stages of a pipeline in one call because it links them: the input struct of
the pixel stage is made from the output signature of the stage before it. A program it cannot carry
yet is refused with a `Refusal` whose `key()` is stable (`op:v_sqrt_f32`, `resource:raw-address`,
`stage:compute-not-written`).

## Layout

| Path | Content |
|---|---|
| `src/api.rs`, `src/refusal.rs` | the public types, frozen after phase 0 |
| `src/tables/` | the instruction table (`gfx10.rs`, generated) and the lookup |
| `src/decode/` | words to `Insn` records, the encoder that inverts it, the listing |
| `src/asm/` | the assembler for synthetic test programs |
| `src/container.rs` | the console container: code, signatures, declared resources, binding table |
| `src/cfg.rs` | reachable code as basic blocks |
| `src/exec.rs` | which scalar register pairs hold lane masks |
| `src/ir/` | the intermediate representation |
| `src/lift/` | one file per instruction class: instruction to IR statements, with its HLSL helpers |
| `src/structure/` | blocks to structured statements: `if`, `else`, loops with `break` and `continue` |
| `src/resource/` | descriptor tracing, dead code removal, placement against the root signature |
| `src/stage/` | entry state, inputs, exports and linking per stage |
| `src/hlsl/` | the printer |
| `src/cache.rs` | the cache key of a translation |
| `src/census.rs`, `src/corpus.rs`, `src/bin/` | the corpus census tool |

## The census

`decant_shader_direct_census <folder of console shader containers> [--top N] [--stage S] [--show ID...]
[--blocked FILE]` counts instructions, encoding forms and program shapes over a corpus, and for each
program the first refusal that stops its translation. Work is steered by that report. Everything it
prints comes from the input corpus.

## The instruction table

`src/tables/gfx10.rs` lists every instruction with a GFX10 opcode number: name, format, number,
timing class and operand slots. It is data taken from Mesa's `aco_opcodes.py` at revision
`f8281cfa33e2`, written by a one off script that imports that Python file and prints one row per
instruction whose `gfx10` number is set. To refresh it, do the same against a newer Mesa and keep
the header lines.

## Tests

`cargo test -p decant-shader` runs everywhere. The tests build their programs with the assembler.
The ones that compile the HLSL need `dxc.exe` (searched in `DECANT_SHADER_DXC`, then in the Windows SDK)
and skip the compile step with a message when it is missing.
