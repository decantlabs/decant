# decant-dxil

A DXIL writer without LLVM. It takes a small scalar SSA IR and returns the bytes of a DXBC container
that holds DXIL (LLVM 3.7 bitcode) for Shader Model 6.6, with the container digest computed by the
crate itself. A PC D3D12 runtime accepts the result without running `dxil.dll`, and `dxil.dll`
validates it.

The crate was written clean room from public
documents only: the LLVM bitcode format, the LLVM 3.7 sources, DXIL.rst, DxilConstants.h, hctdb.py,
the DXC container headers, d3d12.h and RFC 1321. Licence: the project licence. Dependencies: `serde`,
`serde_json` and `thiserror`. On Windows the validator helper and the tests use the `windows` crate.

## Status

| Phase | Content | State |
|---|---|---|
| 1 | bitstream writer, module, metadata, container, digest, empty compute shader | done |
| 2 | buffers (raw, structured, typed), constant buffers, arithmetic, groupshared, barrier, atomics | done |
| 3 | control flow: branches, loops, phis, preheaders, dedicated exits, critical edges, dead phis, reverse postorder layout | done |
| 3 | wave operations, `WaveSize`, descriptor heap handles (`createHandleFromHeap`), heap flags in the binding report | done |
| 4 | vertex and pixel stages, signatures (ISG1, OSG1, PSV0 tables, dx.viewIdState), loadInput, storeOutput, discard, derivatives, textures and samplers (load, store, sample family, gather, dimensions, LOD), heap samplers | done |
| 5 | SPIR-V importer for compute modules, corpus measurement tool | first version, see below |

Compute, vertex and pixel shaders are written. Only one function per shader, no calls. No `f16` values, no `i64`
atomics, ranges of size 1 only. No multisampled textures, no f16 texels, no geometry, hull, domain or mesh stages, no view id.

## SPIR-V importer (phase 5)

`decant_dxil::spirv::import(words)` reads a compute module and returns the IR (`Imported`):
vectors, matrices, structs and arrays are scalarised, local variables become SSA values with phis
(Braun et al.), `OpFunctionCall` is expanded in the SPIR-V before the translation, `OpSwitch`
becomes a chain of compares, GLSL.std.450 is translated to DXIL operations, and the descriptor variables
become bound resources (synthetic mapping: the next register of the class in the space of the
set, `ImportedResource` lists which SPIR-V descriptor each IR resource came from, so a host can
rename them to the real layout). `half` values are carried as `f32` values that hold exactly
representable numbers, packing and unpacking go through the 16 bit conversion operations. What
cannot be translated is an `ImportError` whose `category` is a stable key (`op:OpFoo`,
`memory:physical-storage-buffer`, `type:int`, ...), so a corpus can be counted by cause.

### Memory read by device address

DXIL has no device pointers, so a `PhysicalStorageBuffer` access (`OpConvertUToPtr`, loads, stores and
atomics through such a pointer, access chains and pointer bitcasts on it) becomes a raw buffer
access at a 32 bit byte offset into one window of device memory. The binding contract for the host
(`Imported::device_memory` names the two IR resources, `ImportedResource` lists them with set 14):

| Register | Root parameter | Content |
|---|---|---|
| `u0` of space 14 (`RWByteAddressBuffer`) | a root UAV | the window: its GPU virtual address is the base address |
| `b0` of space 14 (16 bytes) | a root CBV or root constants | dwords 0 and 1: the 64 bit base address, low dword first |

An access at the 64 bit address `A` reads the window at byte `(A, base)` truncated to 32 bits. So
every address the shader forms must lie within 4 GiB above the base address and inside the memory
the root UAV exposes. The shader does not check the range. A host that translates console addresses
with a table (as the translated shaders do) chooses the base as the lowest address it hands out. A root
descriptor has no size, so this relies on the GPU virtual address space being flat. D3D12 leaves
accesses outside the resource a root descriptor was made from undefined, and the test
(`device_addresses_become_raw_buffer_accesses_in_a_window`) only covers one buffer. Whether
neighbouring allocations are reachable has to be measured on each GPU.

### Descriptor arrays

A descriptor variable that is an array of more than one descriptor (`OpTypeArray` or
`OpTypeRuntimeArray`) becomes one resource with a range. A constant or dynamic index becomes the
index of `createHandleFromBinding` (marked non uniform). A runtime array has an unbounded range
(`range_size` is `u32::MAX`) and gets a register space of its own from 1000 up, because D3D12 does not
allow it to overlap anything. `ImportedResource::space` says which space an IR resource uses. A
bounded array advances the next register of its class in the space by its length.

### Placing descriptors (`import_with`)

`import` numbers registers in order of first use. A host that must hit the registers of the console
shader (so that the title's root signature covers them) calls `import_with(words, &Bindings)`. A
`Bindings` maps `(set, binding, array index)` to a `Target`:

| Target | Becomes |
|---|---|
| `Register { Srv, space, register }` | a raw buffer SRV, a texture SRV |
| `Register { Uav, .. }` | a raw buffer UAV, a storage image, a texture that is only fetched (the console shader read a read write texture) |
| `Register { Cbv, .. }` | a constant buffer of 64 KiB (a storage buffer is read as one: its size is not known here) |
| `Register { Sampler, .. }` | a sampler |
| `Heap(i)` | `ResourceDescriptorHeap[i]`, a handle made at each use, `resource_heap` set in the report |
| `SamplerHeap(i)` | `SamplerDescriptorHeap[i]`, `sampler_heap` set in the report |

A binding that is an array is placed per element and the shader may only use constant indices
(`binding:dynamic-index` otherwise, unless the array is read through the heap, see the user data mirror below). An element the shader uses and the map lacks is refused with
`binding:unmapped`, a class that does not fit the use (an image at a `Cbv`) with `binding:class`,
and two different resources at one register with `binding:conflict`. Views of one buffer as `uint`, `ushort` and `uchar` are several
variables at one binding: they share one IR resource per placed register.

### Console semantic names

`Bindings::name(output, location, component, semantic, index)` renames the element of the interface
variable at a `Location` and `Component` (elements packed into one register differ by it): a vertex input `ATTRIB 3` becomes `NORMAL 0` and so on, so that the title's input layout
matches without change. The register of a named element is its location, which is the register
of the console signature (unnamed interpolants sit at the location plus one, as above), pixel
outputs stay `SV_Target`, and built ins keep their system value names. `Bindings::system_register(output, "SV_Position", r)` (also
`SV_ClipDistance`, `SV_CullDistance`) puts those built ins at the register of the console signature,
else `SV_Position` is at register 0, which collides with a named element there. A host that
puts the locations at the console's signature registers (the `locate` step of the translator
output) names each location with the console container's semantic. `tests/spirv_names.rs` compares
the `ISG1` and `OSG1` parts with what DXC writes for the same shader in HLSL (identical bytes),
for a vertex shader with `POSITION`, `NORMAL` and `TEXCOORD` inputs and a `TEXCOORD` output.

### The user data mirror (descriptors chosen at run time)

A console shader reaches its descriptors through user SGPRs: a table is the address held in `sK`, and the
shader may read the descriptor at a byte offset that is not a constant (a bindless index it loaded from
a buffer). Direct3D has no such registers, so the layer appends one root parameter, the **user data
mirror**, and the shader computes the heap index itself. The contract:

| | |
|---|---|
| Root parameter | one root constant buffer view, `b1` of register space 15 (`USER_DATA_REGISTER`, `USER_DATA_SPACE`), shader visibility all, appended after the title's parameters (two dwords of root signature). `b0` of space 15 stays the push block of a module without a push layout, and spaces 14 and 1000 up are taken as above. |
| Size | 32 dwords, 128 bytes (`USER_DATA_DWORDS`): the user SGPR image of the draw or dispatch, dword `K` is `sK`. |
| Packing | the console rule: root parameters in order, a root constant takes one SGPR (its value), a root descriptor two (low dword in `sK`, high in `sK+1`: the console address), a table one. Compute starts at `s4`, graphics stages at `s8`, up to `s31`. Dwords no root parameter fills are 0. |
| A table | dword `K` is the **heap index** of the table's first descriptor (GPU handle minus the start of the heap, divided by the increment), in the resource heap for a resource table, in the sampler heap for a sampler table. The title's heap memory holds PC descriptors at the console's indices (`title_heap`), so descriptor `n` of the table is heap element `mirror[K] + n`. |
| Written by | the layer, whenever the title sets a root argument, from its shadow of the root arguments, it must be current at each draw and dispatch. |

`Target::HeapAt(IndexExpr)` and `Target::SamplerHeapAt(IndexExpr)` place a descriptor at a heap index the shader
computes. `Target::ConstantHeapAt(IndexExpr)` is the same for a buffer the console binding table names with a `b` register:
its slot holds a constant buffer view, so the storage buffer is read as a constant buffer, as at a `Cbv` register. `IndexExpr` has constants, `User(K)` (mirror dword `K`), adds and shifts, and
`BufferWord { buffer, offset }`, a dword read from one of the shader's own buffers (an index kept in a constant
buffer). The console read at table `sK`, byte offset `O` becomes `User(K) + O / 32` (`O / 16` in a sampler
table), a bindless read `table[ shift(index, 5) ]` becomes `User(K) + index`. `Imported::user_data` lists the dwords
read, and the mirror is the constant buffer of the module (the `.json` of `decant_shader_dxil_cache` carries it as
`"user_data":{"space":15,"register":1,"dwords":32,"used":[8,9]}` apart from `registers`, so the layer appends the
parameter only when it is present).

**Push data words** that the probe names (`pushreg <word> s<N>` in the `.src` file, `PushLayout::registers`) read mirror dword
`N`, so a prefix word is the value the console shader would have found in `sN` (a root constant), or the heap index of a table.
`dispatch.<axis>` words map to the workgroup id.

**Root descriptors** (`Target::Address(K)`, a buffer whose console origin is `GetUserData(sK)`): mirror dwords `K` and `K+1` hold the
64 bit GPU address of the buffer in PC memory, and the shader reads and writes it through the device memory window described above
(`u0` and `b0` of space 14, the window and its base address). So a layer that fills the mirror for a root descriptor writes the translated
PC address, not the console address. Memory read by device address (`PhysicalStorageBuffer`) is imported in placed modules the same
way, and the addresses it reads are PC addresses, as the page table of the translator (below) gives them.

**The translator's own buffers**: every module declares bindings 53 (the page table: one 64 bit PC address per 16 KiB console page),
54 (pages touched) and 55 and 56 (plain `uint` buffers whose role the probe does not state). `decant_shader_dxil_cache` places them in space 13:
`t0` for 53, `u0`, `u1`, `u2` for 54, 55 and 56, and lists the ones a shader uses under `"service"` in the `.json`. The layer must append
a root SRV and root UAVs for them (zero filled buffers serve 55 and 56 as far as known).

The `.json` of `decant_shader_dxil_cache` keeps the mirror, window and service buffers in `registers` too: a layer that does not know them
finds registers its root signature lacks and keeps the title's bytecode. A layer that appends them reads `"user_data"`
(`{"space":15,"register":1,"dwords":32,"used":[8,9]}`), `"window"` (`{"space":14,"uav_register":0,"base_register":0}`) and
`"service"` and checks the other registers only.

An array indexed at run time is a heap array when every placed element is in a heap: the shader selects among the
element indices (`createHandleFromHeap`, non uniform). A binding that is a register for constant indices can
carry a second placement for the run time case, `Bindings::bind_dynamic`, which the bridge fills with the
mirror based heap index of the element (`User(K) + O / 32`).

### Push constants of a placed module

`Bindings::push` takes a `PushLayout` made from the `push A B C` line of the translator's `.src` file
(`data_end = A`, `offsets_end = B`, `sizes_end = C`). The block is then not a constant buffer any
more, each word is resolved at import:

| Words | Meaning | Becomes |
|---|---|---|
| `[0, A)` | values of console scalar registers | the workgroup id when listed in `group_id` (`SV_GroupID`), else refused (`push:prefix`) |
| `[A, B)` | byte offset of each buffer in its binding, several to a dword | 0, a Direct3D view starts at its first byte |
| `[B, C)` | byte size of each buffer, one dword per buffer in rank order | `GetDimensions` of the buffer (`PushLayout::buffers[k]` names it) |

The size of a buffer is the size of its view. A root descriptor has no size and reports 0, so a shader
that reads size words needs its buffers in a descriptor table or the heap (the device test
`push_words_are_group_id_zero_offsets_and_buffer_sizes` does it with heap UAVs and runs on the 3090). A
size word of a root descriptor buffer (`Target::Address`) reads 4294967295 (it has no size), one of a buffer placed at a constant buffer register reads 65536 (a constant buffer cannot be
larger, so a bounds check passes) and does not declare the buffer. A size word of a buffer the layout
does not name is refused (`push:size-unmapped`). Corpus note: in the cache the second field
counts offset dwords of 4 buffers each and the third ends the size dwords, so `push 0 5 25` is 20 buffers.

### The DXIL cache (`decant_shader_dxil_cache`)

`decant_shader_dxil_cache <input folder> <output folder> [--measure <stage>] [--stage S] [--jobs N] [--every N]`
(crate `decant-shader-bridge`) builds the override files of the d3d12-xs layer from the SPIR-V cache: for
each shader `<fnv1a64 of the console container, 16 hex digits>.dxil` and a `.json` with the registers it
needs, `{"registers":[{"class":"srv","register":188,"space":0}],"resource_heap":false,"sampler_heap":false}`.
Per module it locates the interface (as `decant_shader_describe` does), identifies the resources
through the FSSM binding table and the `.src` file, places them (`Bindings`), reads the `push A B C`
line, names the vertex and pixel interface with the console semantics and the registers of the
console signature, imports, writes and checks the container with dxil.dll. It prints how many
modules each stage carries and the refusals by cause. Measured on the whole cache of a retail title's
shader corpus (every vertex and pixel module, 2,292 of the 2,689 compute shaders):

| Stage | Modules | Carried | Main refusals |
|---|---|---|---|
| vertex | 2,222 | 2,097 | several immediate samplers 88, signature overlaps 23 |
| pixel | 11,998 | 7,054 | several immediate samplers 3,755, a buffer the expression elides 580, 64 bit texels 416 |
| compute | 2,292 | 1,244 | several immediate samplers 782, two resources at one register 151 |

(Before the user data mirror: vertex 1,052, pixel 2,346, compute 89.) The registers and their spaces come from the console container itself: its `PSV0` part lists every resource
with class, register range, register space and kind (the records the Direct3D debug layer reads at pipeline
creation). `decant_shader_dxil_cache` matches each identified register (`t11`, `b27`) to a record of the same class, the
right kind of resource (buffer, texture, sampler) first, and takes the space from it. A register that two
spaces have is ambiguous (the first record is taken and the module is counted), a register no record has means
`identify` chose wrongly (counted, and a use of it is refused as `binding:identify-off`). Without a `PSV0` the space
`DECANT_SHADER_DXIL_SPACE` (default 1) is used. The summary prints, per stage, how many modules had a register the
container lacks, ambiguous registers, and spaces other than the default, with an example. The tool runs at most 8 jobs.

`--space-hints <file>` reads what the d3d12-xs layer wrote with `DECANT_D3D12X_SPACE_HINTS` (`<hash> <class letter> <register> <old space> <new space>`
per register of a refused override) and puts those registers of the listed hashes in the hinted space, the most frequent one when root
signatures disagree. `--only-hinted` builds only those hashes, so a run writes just the files to replace.

A refusal says what the console shader does that has no fixed place: several immediate samplers are
samplers the console shader builds from constants, which need the static samplers of the title's
root signature (not placed yet, one immediate sampler with one static sampler is). Descriptors chosen at run time use the user
data mirror, root descriptors the window, the push data words the mirror too.

### Vertex and pixel shaders

`import` takes the first entry point of the module, a vertex or fragment shader as well as a compute
shader. Interface variables and the members of interface blocks (`gl_PerVertex`) become signature
elements. The host that wires vertex buffers and render targets needs this mapping
(`Imported::io` lists, per element, the location or the built in):

| SPIR-V | Signature element |
|---|---|
| vertex input, `Location` L | `ATTRIB` with index L in register L |
| vertex output and fragment input, `Location` L | `TEXCOORD` with index L in register L + 1 (`Component` is the first column) |
| fragment output, `Location` L | `SV_Target` L in register L |
| `Position`, `FragCoord` | `SV_Position`, register 0, linear without perspective |
| `ClipDistance`, `CullDistance` | `SV_ClipDistance`, `SV_CullDistance`, four floats to a register |
| `VertexIndex`, `InstanceIndex` | `SV_VertexID`, `SV_InstanceID` (unsigned) |
| `FrontFacing`, `PrimitiveId` | `SV_IsFrontFace`, `SV_PrimitiveID`, constant interpolation |
| `SampleId` | `SV_SampleIndex`, no register |
| `FragDepth`, `SampleMask`, `FragStencilRefEXT` | `SV_Depth`, `SV_Coverage`, `SV_StencilRef`, no register |
| `BaryCoordKHR`, `BaryCoordNoPerspKHR` | `SV_Barycentrics` 0 and 1, no register (the `Barycentrics` feature) |
| `HelperInvocation` | `IsHelperLane` (Shader Model 6.6) |
| `SubgroupLocalInvocationId`, `SubgroupSize` | wave operations |

Interpolation follows `Flat`, `NoPerspective`, `Centroid` and `Sample`, integers are constant. `OpKill` is
a discard. `OpDPdx` and its relatives are derivatives, `OpFwidth` is the sum of their absolute values.
`PointSize` is dropped. A sampler used for depth compare operations becomes a comparison sampler,
a separate resource when the same variable is also used to sample.

Interface masks follow what DXC writes (checked against DXC output, see the signature notes
above): a component used anywhere in an element counts for every row of that element, and an input
that is indexed with a row that is not a constant is marked in the dynamic mask of the `PSV0` element.

Not translated yet: `OpArrayLength`, 64 bit floats, multisampled images, hull, domain, geometry and mesh
stages, interpolation functions (`InterpolateAt...`), view id.

`decant_shader_spv_measure <folder> [--limit N] [--examples K] [--dump <folder>]` imports, writes and
validates (with dxil.dll) every `*.spv` under a folder and prints how many pass and the failures by
cause. The modules are only read. Measured on the compute modules of the translation of a retail title's
shader corpus (2,292 modules of the 2,689 compute shaders):

| Step | Modules |
|---|---|
| translated by the probe | 2,292 |
| imported | 1,410 |
| written as a container | 1,347 |
| accepted by the validator of dxil.dll | 1,347 |

Failures: dynamic descriptor array index 805, a type mismatch of `OpNot` on 64 bit values 63 (fixed in
the next version), 8 and 16 bit integers 53, `OpImageTexelPointer` 20, 64 bit float 4. Every
container that was written validates. `tests/spirv.rs` runs three hand written modules on a device
and compares with a Rust reference (a loop through function variables, a function call, a workgroup
array with a barrier, a switch, and device addresses).

## Use

```rust
use decant_dxil::ir::*;

let mut shader = Shader::compute([64, 1, 1], vec![Block::new(vec![/* instructions */], Terminator::Ret)]);
shader.resources.push(Resource::raw(ResourceClass::Uav, 0, 0)); // u0, space 0
let out = decant_dxil::write(&shader)?;
// out.container: the DXBC bytes, out.report: the bindings the root signature has to provide
```

`write` returns `Error::Invalid`, `Unsupported`, `Irreducible` or `UseBeforeDef` with a message that
names the value, resource or block.

## The IR (`ir.rs`)

* One function per shader. `Shader { stage, resources, groupshared, blocks }`. Block 0 is the
  entry. Every block is `phis`, `insts` and one terminator (`Br`, `CondBr`, `Ret`).
* Types: `Bool` (i1), `I32`, `I64`, `F32` (and `F16` reserved). No vectors, a front end scalarises.
* Values are named by `ValueId`, chosen by the front end and unique in the shader. An `Inst` has a
  list of result ids: none for stores, one for most operations, several for loads of several
  components and for the carry operations. An `Operand` is a value or a constant (`Const`, floats are
  bit patterns).
* Signedness is in the operation (`UDiv`, `SDiv`), not in the type.
* Resources are bound ranges (class, kind, space, register, size), referred to by their index in
  `Shader::resources`. The lowering creates the handles in the entry block
  (`createHandleFromBinding` then `annotateHandle`) and writes `dx.resources`, `PSV0` and the shader
  flags. `BindingReport` lists what the root signature must provide.
* Buffer addressing: raw buffer, `index` is a byte offset and `offset` is `None`. Structured buffer,
  `index` is the element and `offset` is the byte offset inside it. Typed buffer, `index` is the
  element and a store must write all four components.
* Groupshared memory is an array of `I32` or `F32` reached by element index (`GsLoad`, `GsStore`,
  `GsAtomic`, `GsCmpXchg`). There are no other pointers.
* `Barrier` must only be placed in uniform control flow. The IR cannot prove that.
* `FirstBitHigh` is the DXIL operation: the distance from the most significant bit.

### Control flow

`write` first normalises the graph (`cfg.rs`): unreachable blocks go, the entry gets no predecessor,
irreducible graphs are refused, an empty latch is folded into its predecessor, every loop gets one
preheader and dedicated exits, critical edges that are not back edges are split, dead phis are
removed (through chains), and the blocks are laid out in reverse postorder. Every operand except a
phi input is then defined earlier in the stream, so the writer needs no forward reference records.
A phi must have exactly one input for each predecessor of its block.

### The `.sxir` file

`Shader::to_sxir()` writes and `Shader::from_sxir()` reads the IR as UTF 8 JSON, the `serde`
representation of the types of `ir.rs` (pretty printed, trailing newline). Enums are externally
tagged, constants of the IR are `{"I32": 5}` and so on. The format follows the Rust types, so it
changes when `ir.rs` changes. Both sides of the process boundary use this crate to read and write
it. Tests check that the text round trips and that the container written from the round tripped IR
is byte for byte the same.

## Output

`write` returns a DXBC container with the parts `SFI0`, `ISG1`, `OSG1`, `PSV0`, `HASH` and `DXIL`:

* `DXIL`: the program header, then unabbreviated LLVM 3.7 bitcode (module version 1). Attribute
  groups, types, module level constants, metadata (`dx.version`, `dx.valver`, `dx.shaderModel`,
  `dx.resources`, `dx.entryPoints`), the 14 fixed metadata kinds, a value symbol table and one
  function block. DXIL version 1.6, validator version 1.6, shader model 6.6.
* `PSV0`: runtime info version 2 (48 bytes) with the thread group size and the resource table in the
  order constant buffers, samplers, SRVs, UAVs.
* `SFI0` and the shader flags (tag 0 of the entry properties) are computed from the code:
  raw and structured buffers (bit 4), `i64` (bit 20 and 0x8000), typed UAV loads (bit 13 and 0x800).
* `HASH` is 20 zero bytes. The container digest at offset 4 is `ComputeHashRetail` over the bytes
  from offset 20. `container::dxbc_checksum` is a copy of `d3d12_xs::rts0::dxbc_checksum` (our own
  code). `d3d12-xs` is not changed, a test proves both give the same digest for lengths 0 to 200.

## Tests

`cargo test -p decant-dxil` runs on every machine. Parts that need Windows tooling skip with a
message when it is missing.

| Test file | What it checks | Needs |
|---|---|---|
| `src/bitstream.rs`, `src/container.rs`, `src/cfg.rs` | VBR, signed VBR, block lengths, a reader that decodes the stream back, the digest rule, graph passes | nothing |
| `tests/structure.rs` | parts and sizes, block order of the module, golden bytes of the empty shader, `.sxir` round trip, error cases | nothing |
| `tests/digest.rs` | the digest equals `d3d12_xs::rts0::dxbc_checksum` | nothing |
| `tests/validator.rs` | every test shader validates through `IDxcValidator` of `dxil.dll` in place, and the digest dxil.dll writes equals ours | `dxil.dll` |
| `tests/negative.rs` | damaged containers are rejected (proof that the tests can fail) | `dxil.dll` |
| `tests/random.rs` | 300 random structured graphs with nested ifs and loops with phis: layout invariant, 40 of them validated | `dxil.dll` for the validation |
| `tests/device.rs` | pipeline creation, dispatch and read back on a D3D12 device | a device with Shader Model 6.6 |

`dxil.dll` is searched in `DECANT_SHADER_DXIL_DLL`, then `Windows Kits\10\bin\<newest>\x64`. The device
test uses the default adapter and falls back to WARP. `DECANT_SHADER_DEVICE=warp` forces WARP and
`DECANT_SHADER_DEVICE=hardware` forbids it. `DECANT_SHADER_DUMP_DIR=<folder>` keeps the containers written by the
tests, for `dxc -dumpbin`.

The test shaders (`tests/common/shaders.rs`) are the shaders of the specification: empty compute,
thread id store, buffer copy (raw and structured), wave operations (`WaveSize` 4 and 32 run, others validate), a UAV taken from the descriptor heap, arithmetic (16 operations checked against a Rust
reference), loop with a branch (n = 0, 1, 7, 1000), groupshared with a barrier, and atomics on a UAV
and on groupshared memory.

## Findings about the specification

* Unabbreviated records, no function attributes on the entry, module level constants only, an
  internal groupshared global with an `undef` initialiser, no `dx.typeAnnotations`, a zero `HASH`
  part, no `STAT` part and the part order `SFI0 ISG1 OSG1 PSV0 HASH DXIL` all validate and run.
* The 14 fixed metadata kind records are not required: the validator and the drivers accept the
  module with and without them. They are written (`WRITE_METADATA_KINDS` in `lib.rs`).
* The PSV0 resource order constant buffers, SRVs, UAVs validates (see `resource_zoo`).
* The class of the wave bit count operation 135 is `waveAllOp` (the spec names it `waveAllBitCount`).
  The validator names the expected class in its error, which is how this was found. Operation 136
  is `wavePrefixBitCount`.
* A heap resource must not set the raw and structured buffer flag (bit 4), only declared bound ones do.
* `dx.op.calculateLOD` is declared with the overload `f32` (`dx.op.calculateLOD.f32`).
* The metadata usage mask of a signature element (extra tag 3) counts columns of the element, the
  `ISG1` and `OSG1` masks count components of the register.
* `ISG1` and `OSG1` share the name string of system value elements (`SV_Target` 0 and 1) and give every
  arbitrary semantic its own string. The strings are in declaration order while the rows are sorted by
  register, one row of the part for each row of an element. The `PSV0` string table has one string per
  arbitrary semantic and none for system values.
* The output rows of `OSG1` carry the complement of the written components in their read write mask,
  the input rows the components always read.
* The `PSV0` input to output table covers whole vectors (four rows per input register). The
  `dx.viewIdState` table stops at the highest declared scalar (`register * 4 + component`).
* Reading an undefined component of `GetDimensions` is rejected by the validator, so the lowering
  returns 0 for the components a texture kind does not define.

## Graphics

A vertex or pixel shader declares `Shader::inputs` and `Shader::outputs` (`SigElement`): a semantic name
and index, a `Semantic`, a component type, an interpolation mode, a register (`start_row`, `start_col`),
`rows` and `cols`. An element without a register (`SV_Depth`, `SV_Coverage`, `SV_SampleIndex`) uses
`SigElement::UNALLOCATED` as `start_row`. `LoadInput` and `StoreOutput` name an element, a row inside it
(a dynamic row index is allowed) and a column inside it. The crate derives the used masks, `ISG1`,
`OSG1`, the tables of `PSV0`, `dx.viewIdState` and the input to output table from the data flow of the IR,
including control dependence. For the test shaders the three signature parts and `PSV0` are byte for byte
what DXC 1.9 writes with `-validator-version 1.6` for the same HLSL.

Textures: kinds 1D, 2D, 3D, cube and the arrays of 1D, 2D and cube. Operations: `TexLoad`, `TexStore`,
the `Sample` family (`SampleMode`), `Gather`, `GetDimensions`, `CalculateLod`.
