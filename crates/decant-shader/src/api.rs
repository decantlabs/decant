//! The public types of the translator. The runtime (`d3d12-xs`) and the tools build these and
//! read these. Nothing here depends on how a pass works, so the passes can change freely.
//!
//! This file is frozen after phase 0: a change here is a change of the runtime interface and
//! goes through the owner of the crate.

/// A shader stage of a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    Vertex,
    Pixel,
    Geometry,
    Hull,
    Domain,
    Compute,
    Mesh,
    Amplification,
}

impl StageProgram {
    /// A program with nothing but its stage and code, for tools and tests to fill in. Writing
    /// `..StageProgram::new(stage, code)` keeps a struct literal valid when a field is added.
    pub fn new(stage: Stage, code: Vec<u32>) -> StageProgram {
        StageProgram {
            stage,
            code,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: Vec::new(),
            binding_table: Vec::new(),
            threads: [0; 3],
            lds_bytes: 0,
            geometry: None,
            user_data: Vec::new(),
            param_exports: Vec::new(),
            pixel_inputs: Vec::new(),
            name: String::new(),
        }
    }
}

impl Stage {
    /// The profile prefix DXC takes (`vs`, `ps`, ...).
    pub fn profile_prefix(self) -> &'static str {
        match self {
            Stage::Vertex => "vs",
            Stage::Pixel => "ps",
            Stage::Geometry => "gs",
            Stage::Hull => "hs",
            Stage::Domain => "ds",
            Stage::Compute => "cs",
            Stage::Mesh => "ms",
            Stage::Amplification => "as",
        }
    }

    /// The stage code of `shaders_gpu.pack` (`decant_extract_shaders::output::stage_code`).
    pub fn from_pack_code(code: u8) -> Option<Stage> {
        Some(match code {
            0 => Stage::Pixel,
            1 => Stage::Vertex,
            2 => Stage::Geometry,
            3 => Stage::Hull,
            4 => Stage::Domain,
            5 => Stage::Compute,
            13 => Stage::Mesh,
            _ => return None,
        })
    }

    /// The first user data register of the stage, as a machine register. Measured on the
    /// title's containers (24 Nov 2026): compute programs start at `s4`, vertex and pixel
    /// programs at `s8`, mesh and geometry programs at `s16` (every binding record of the
    /// 9,282 mesh and 195 geometry programs names a register 8 below the one the code loads
    /// from). The archive holds no hull, domain or amplification program.
    pub fn first_user_sgpr(self) -> u8 {
        match self {
            Stage::Compute => 4,
            Stage::Mesh | Stage::Geometry => 16,
            _ => 8,
        }
    }

    /// The machine registers that hold user data at entry: 28 for compute, 24 for the
    /// graphics stages.
    pub fn user_registers(self) -> std::ops::Range<u8> {
        let first = self.first_user_sgpr();
        if self == Stage::Compute {
            first..32
        } else {
            first..first + 24
        }
    }

    /// The mirror dword (and the register number the container's tables use) of machine
    /// register `machine`: the mirror keeps compute at dword 4 and every graphics stage at
    /// dword 8, whatever register the stage really starts at.
    pub fn mirror_dword(self, machine: u8) -> u8 {
        match self {
            Stage::Compute => machine,
            _ => machine.wrapping_sub(self.first_user_sgpr()).wrapping_add(8),
        }
    }
}

/// A register class of Direct3D.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegisterClass {
    /// `t` registers.
    Srv,
    /// `u` registers.
    Uav,
    /// `b` registers.
    Cbv,
    /// `s` registers.
    Sampler,
}

impl RegisterClass {
    pub fn letter(self) -> char {
        match self {
            RegisterClass::Srv => 't',
            RegisterClass::Uav => 'u',
            RegisterClass::Cbv => 'b',
            RegisterClass::Sampler => 's',
        }
    }
}

/// One element of an input or output signature of the console container (`ISG1`, `OSG1`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SigElement {
    pub semantic: String,
    pub index: u32,
    /// The Direct3D system value code (0 for none, 1 for `SV_Position`, 64 for `SV_Target`, ...).
    pub system_value: u32,
    /// The component type code of the signature: 1 unsigned, 2 signed, 3 float.
    pub component_type: u32,
    pub register: u32,
    /// The components the element has, bit 0 for x.
    pub mask: u8,
    /// The second mask byte of the signature. For an input: the components the program reads.
    /// For an output it is the complement of the components the program writes.
    pub rw_mask: u8,
}

/// A resource the console container declares (its `PSV0` part), with the place the binding
/// table after the code gives it when it has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeclaredResource {
    pub class: RegisterClass,
    /// First and last register of the range.
    pub lower: u32,
    pub upper: u32,
    pub space: u32,
    /// The `PSV0` resource type code (1 sampler, 2 constant buffer, 3 typed SRV, 4 raw SRV,
    /// 5 structured SRV, 6 to 9 the UAV kinds).
    pub psv_type: u32,
    /// The `PSV0` resource kind code (the DXIL resource kind: 2 a 2D texture, 12 a structured
    /// buffer, 13 a constant buffer, ...). 0 when the container has the short record.
    pub psv_kind: u32,
}

/// One record of the binding table the console compiler places after the code: the descriptor
/// of register `register` of `class` is at dword `word_offset` of the table whose address is in
/// user data register `sgpr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TableBinding {
    pub class: RegisterClass,
    pub register: u32,
    pub sgpr: u8,
    pub word_offset: u32,
}

/// The runtime information of a geometry program in `PSV0`. Measured on the title: all 195
/// geometry programs take triangles (3), write a triangle strip (5) and at most 3 vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryInfo {
    /// The Direct3D input primitive code (3 a triangle).
    pub input_primitive: u32,
    /// The Direct3D output topology code (5 a triangle strip).
    pub output_topology: u32,
    /// The output streams the program writes, bit 0 for stream 0.
    pub stream_mask: u32,
    pub max_vertices: u32,
}

/// What a user data register holds, by the root layout in the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UserDataKind {
    /// The address of a descriptor table.
    Table,
    /// The first of `count` root constants of constant buffer register `register`.
    Constants { register: u32, count: u32 },
    /// The address of a root shader resource view at `t` register `register`, in this register
    /// and the next.
    RootSrv { register: u32 },
    /// The base of the resource descriptor heap, for descriptors chosen at run time.
    ResourceHeap,
    /// The base of the sampler descriptor heap.
    SamplerHeap,
}

/// One user data register of the root layout in the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UserDataSlot {
    pub sgpr: u8,
    pub kind: UserDataKind,
}

/// One record of the parameter export table of an FSSM block: parameter slot `slot` (export
/// target `32 + slot`) carries the output element in signature register `register`. Two 16 bit
/// registers can share one slot, as halves of its dwords.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamExport {
    pub slot: u8,
    /// The record kind: 6 a parameter, 7 a clip distance.
    pub kind: u8,
    pub register: u32,
    /// The fourth byte of the record: the register again for a 16 bit register, else 0.
    pub sixteen_bit_register: u32,
    /// Bit 0: the low halves of the slot are used, bit 1: the high halves.
    pub flags: u8,
}

/// One record of the pixel input table of an FSSM block: hardware attribute `attr` carries the
/// input element in signature register `register`. Measured on all 12,101 pixel programs of the
/// title (24 Nov 2026).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelInput {
    /// The attribute number the interpolation instructions name.
    pub attr: u8,
    /// The record kind: 6 an interpolated input, 7 a clip distance.
    pub kind: u8,
    /// The signature register of the input element.
    pub register: u32,
    /// The second register when two 16 bit elements share the attribute: halves 0 to 3 of the
    /// attribute are `register`, halves 4 to 7 this one.
    pub high_register: Option<u32>,
    /// The dword channels of the attribute that are used, bit 0 for x.
    pub channels: u8,
    /// 0x01 constant over the primitive, 0x02 perspective, 0x08 a 16 bit attribute, 0x10
    /// interpolated with the 16 bit instructions.
    pub flags: u8,
}

/// One program of a pipeline, as the title hands it over.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StageProgram {
    pub stage: Stage,
    /// The instruction words of the FSSM block, with the six word signature at the start and
    /// without the padding at the end.
    pub code: Vec<u32>,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub resources: Vec<DeclaredResource>,
    /// The binding table of the FSSM block, as far as its record kinds are known.
    pub binding_table: Vec<TableBinding>,
    /// The thread group size of a compute or mesh program, from the runtime information of
    /// `PSV0`. Zeros for the other stages.
    pub threads: [u32; 3],
    /// The size of the group shared memory (the local data share) the program uses, in bytes,
    /// from the FSSM block. Zero for a program without such memory.
    pub lds_bytes: u32,
    /// What `PSV0` says about a geometry program. `None` for the other stages.
    pub geometry: Option<GeometryInfo>,
    /// The root layout the program was compiled for (the first table of the FSSM block): what
    /// each user data register holds. Registers are in the numbering of the container, which
    /// is the mirror dword (`Stage::mirror_dword`).
    pub user_data: Vec<UserDataSlot>,
    /// The parameter export table of the FSSM block: which output register each parameter
    /// slot (export target 32 and up) carries. Empty for stages without parameter exports and
    /// for programs that come without a container.
    pub param_exports: Vec<ParamExport>,
    /// The pixel input table of the FSSM block: which input register each hardware attribute
    /// carries. Empty for other stages and for programs that come without a container. The
    /// pixel stage then falls back to ranking the read inputs.
    pub pixel_inputs: Vec<PixelInput>,
    /// A name for messages. It is not part of the cache key.
    pub name: String,
}

/// One range of a descriptor table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RootRange {
    pub class: RegisterClass,
    /// `u32::MAX` for an unbounded range.
    pub count: u32,
    pub base_register: u32,
    pub space: u32,
    /// The offset of the range in the table, in descriptors.
    pub offset: u32,
}

/// What a root parameter is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RootParamKind {
    Table(Vec<RootRange>),
    /// 32 bit constants at a `b` register.
    Constants {
        register: u32,
        space: u32,
        count: u32,
    },
    /// A root descriptor.
    Descriptor {
        class: RegisterClass,
        register: u32,
        space: u32,
    },
}

/// One root parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RootParam {
    pub kind: RootParamKind,
    /// The `D3D12_SHADER_VISIBILITY` code (0 for all stages).
    pub visibility: u32,
}

/// A static sampler of the root signature. Only its place matters to the translator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StaticSampler {
    pub register: u32,
    pub space: u32,
    pub visibility: u32,
}

/// The root signature the title pairs the pipeline with, as the layer parsed it. The layer's own
/// appended parameters (the user data mirror) are not in `params`: [`Options::mirror`] names them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RootSignature {
    pub params: Vec<RootParam>,
    pub static_samplers: Vec<StaticSampler>,
    /// The `D3D12_ROOT_SIGNATURE_FLAGS` the PC root signature will have.
    pub flags: u32,
}

/// What the layer appends to every PC root signature for translated programs, all in one
/// register space.
///
/// * The **mirror**: a constant buffer at `b<register>` (`d3d12-xs`, `user_mirror.rs`), 32
///   dwords, dword `K` is what user data register `K` holds on the console, and a table is the
///   host heap index of its first descriptor.
/// * The **heap map**: a raw buffer at `t<heap_map>` with one dword per console heap page of
///   `1 << HEAP_PAGE_SHIFT` slots, the host index of the first slot of that page. The title's
///   resource heap is larger than a PC heap may be, so the layer places console pages where it
///   has room. A program that takes a descriptor by its console heap index reads
///   `map[index >> HEAP_PAGE_SHIFT] + (index & (page size - 1))`. Table reads do not use it.
/// * The **view shadow**: a raw buffer at `t<view_shadow>` with four dwords per HOST heap slot,
///   indexed by the heap index the program uses for `ResourceDescriptorHeap`. It says what the
///   title's view in that slot was, because the layer does not create every view as the title
///   asked for it:
///   * **Buffer views are raw views.** A console buffer descriptor has a base, a stride, a
///     count and a format, and the instruction decides whether it reads raw dwords or decoded
///     elements. One PC view cannot serve both, and a structured buffer read on PC uses the
///     declared stride or the stride of the view depending on the driver (measured, 24 Nov
///     2026). So the layer creates every structured AND every typed buffer view of the title as
///     a raw view over the same bytes, and the row is: dword 0 the element size in bytes (the
///     stride), dword 1 the element count, dword 2 the DXGI format number of the title's view
///     (0 for a view the title made raw or structured), dword 3 the start offset (next point).
///     The program does the address arithmetic of the machine (`index * stride + offset`) and
///     decodes a formatted element itself.
///   * **The start offset.** A raw view on PC must start on 16 bytes
///     (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`, the device is removed otherwise), a console view
///     starts on 4. So every buffer view the layer makes raw starts on the 16 bytes at or below
///     the title's first byte, and bits 0 to 3 of dword 3 hold the byte offset of the title's
///     first byte in the PC view (0, 4, 8 or 12). This also holds for a view the title made raw
///     itself: its row is stride 0, dword 1 the size in dwords, dword 2 zero, dword 3 the
///     offset (0 when it starts on 16 bytes). The printer adds `row.w & 0xF` to every byte
///     address on a buffer in a heap slot (`hlsl::buffer::add_view_offsets`), whenever there is
///     a view shadow. A layer that never moves a view leaves the bits zero. A buffer bound as a
///     root descriptor has no row: the layer never moves it.
///   * **The rule that follows:** once the layer makes a slot raw, every program that reads
///     that slot reads it raw and decodes, also a program that declares a typed resource for
///     it. No text may declare `Buffer<T>` or `StructuredBuffer<T>` for a heap slot.
///   * A stride of 0 in a row means that the slot holds no raw view of elements: a constant
///     buffer, a view the title made raw, or a view whose element size is no multiple of 4 (the
///     layer keeps that one as the title made it and logs it). A program that expects elements
///     there reads zero.
///   * Counters of unordered access views are dropped, a raw view cannot carry one.
///   * **The row of an image view carries its size**, for programs that take a texture from the
///     whole heap and only ask for its size, so that nothing in the program says what kind of
///     texture it is: dwords 0 and 1 zero, dword 2 = `(width - 1) | (height - 1) << 16`, dword 3
///     = the last slice (bits 0 to 12) | the last mip level << 16 (bits 16 to 19) | the image
///     type << 28 (8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array, 14 2D multisample, 15
///     2D multisample array). The image type is never 0 and bits 28 to 31 of a buffer row are
///     always 0, so the two cannot be confused. A layer that leaves image rows zero makes such
///     a program see a 1 by 1 texture.
///
/// Both buffers belong to the view heap that is bound, which is why they are root parameters
/// and not heap slots: a program is compiled once and runs with any heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MirrorAbi {
    /// The `b` register of the mirror.
    pub register: u32,
    pub space: u32,
    /// The `t` register of the heap map. `None` when the layer has none: a program that takes
    /// a descriptor by its console heap index is then refused with `resource:no-heap-map`.
    pub heap_map: Option<u32>,
    /// The `t` register of the view shadow. `None` when the layer has none: a program that
    /// reads a structured or typed buffer view is then refused with `resource:buffer-structured`.
    pub view_shadow: Option<u32>,
}

impl Default for MirrorAbi {
    fn default() -> Self {
        MirrorAbi {
            register: 1,
            space: 15,
            heap_map: Some(0),
            view_shadow: Some(1),
        }
    }
}

/// Console heap pages have `1 << HEAP_PAGE_SHIFT` descriptor slots, see [`MirrorAbi`].
pub const HEAP_PAGE_SHIFT: u32 = 12;

/// How the lanes of a wave are modelled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WaveMode {
    /// One lane per shader invocation, no operation looks at another lane. A program that needs
    /// another lane is refused.
    #[default]
    Scalar,
    /// One lane per invocation, operations across lanes use the wave intrinsics of HLSL with
    /// `[WaveSize(n)]`. `lanes` is 32 or 64.
    Native { lanes: u8 },
    /// Operations across lanes go through group shared memory, for a 64 lane program on a GPU
    /// with smaller waves. Compute only.
    Emulated,
}

/// What the caller wants.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Options {
    /// The mirror the layer appends to the root signature, when it can. Without it a resource
    /// that no range of the root signature covers is refused.
    pub mirror: Option<MirrorAbi>,
    pub wave: WaveMode,
    /// Shader model minor version of the profile (`6` for `ps_6_6`).
    pub shader_model_minor: u8,
    /// Put the instruction listing into the HLSL as comments.
    pub listing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            mirror: Some(MirrorAbi::default()),
            wave: WaveMode::Scalar,
            shader_model_minor: 6,
            listing: false,
        }
    }
}

/// Where a resource of the translated shader is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// A register that the root signature covers.
    Register {
        class: RegisterClass,
        register: u32,
        space: u32,
    },
    /// `ResourceDescriptorHeap[mirror[dword] + slot]` (the sampler heap for a sampler).
    HeapAtMirror {
        class: RegisterClass,
        dword: u8,
        slot: u32,
    },
    /// A descriptor the program chooses at run time by its console heap index (bindless). The
    /// translated program turns the console index into a host index through the heap map
    /// (see [`MirrorAbi`]) and reads `ResourceDescriptorHeap[host]`. A sampler index is
    /// used as it is, in the sampler heap. The whole heap is reachable through such a
    /// descriptor. It needs the direct heap indexing flag of its heap.
    HeapAtConsoleIndex { class: RegisterClass },
    /// A sampler the console program built from four constants (`words`, the console sampler
    /// descriptor): `SamplerDescriptorHeap[index]`, with the index the [`Host`] gave for the words.
    HeapSampler { words: [u32; 4], index: u32 },
}

/// One resource of a translated stage, for the layer's checks and logs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundResource {
    /// The name in the HLSL text.
    pub name: String,
    pub placement: Placement,
}

/// What the layer has to provide for a translated stage beyond the title's root signature.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Needs {
    /// Mirror dwords the shader reads. Empty when it does not use the mirror.
    pub mirror_dwords: Vec<u8>,
    /// The shader indexes `ResourceDescriptorHeap` (root signature flag `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`).
    pub resource_heap: bool,
    /// The shader indexes `SamplerDescriptorHeap` (flag `SAMPLER_HEAP_DIRECTLY_INDEXED`).
    pub sampler_heap: bool,
    /// The shader reads the heap map of [`MirrorAbi`].
    pub heap_map: bool,
    /// The shader reads the view shadow of [`MirrorAbi`].
    pub view_shadow: bool,
}

/// One translated stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TranslatedStage {
    pub stage: Stage,
    pub hlsl: String,
    /// The entry point name in `hlsl`.
    pub entry: String,
    /// The DXC target profile, for example `ps_6_6`.
    pub profile: String,
    /// Extra DXC arguments the text relies on.
    pub dxc_args: Vec<String>,
    pub resources: Vec<BoundResource>,
    pub needs: Needs,
}

/// The result of [`crate::translate`]: one entry per stage program, in the order given.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Translation {
    pub stages: Vec<TranslatedStage>,
}

/// One stage compiled to a DXIL container.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompiledStage {
    pub stage: Stage,
    pub dxil: Vec<u8>,
    pub resources: Vec<BoundResource>,
    pub needs: Needs,
}

/// What the translator asks the layer while it translates.
pub trait Host {
    /// The index in the sampler heap of the sampler with these four console descriptor words.
    /// The layer keeps one registry for the whole run, so the same words always give the same
    /// index. `None` when the layer cannot provide such samplers: the program is then refused.
    fn sampler_index(&self, words: [u32; 4]) -> Option<u32>;
}

/// A [`Host`] for tests and tools: it numbers the samplers in the order they are asked for.
#[derive(Debug, Default)]
pub struct CountingHost {
    seen: std::sync::Mutex<Vec<[u32; 4]>>,
}

impl CountingHost {
    /// The sampler words asked for so far, by index.
    pub fn samplers(&self) -> Vec<[u32; 4]> {
        self.seen.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

impl Host for CountingHost {
    fn sampler_index(&self, words: [u32; 4]) -> Option<u32> {
        let mut seen = self.seen.lock().ok()?;
        let at = seen.iter().position(|w| *w == words).unwrap_or_else(|| {
            seen.push(words);
            seen.len() - 1
        });
        u32::try_from(at).ok()
    }
}

/// An HLSL compiler. The runtime implements it over `dxcompiler.dll`, the tests over `dxc.exe`.
pub trait Compiler {
    /// Compiles `hlsl` and returns the DXIL container, or the compiler's messages.
    fn compile(
        &self,
        hlsl: &str,
        entry: &str,
        profile: &str,
        args: &[String],
    ) -> Result<Vec<u8>, String>;
}
