//! The console D3D12 method order, as data. The Rust form of the former
//! `src/gdk/d3d12x_abi.h`.
//!
//! The console device, command list, queue and descriptor heap do not use the PC vtable order
//! everywhere. Each table maps a console slot to what the proxy does with it:
//!
//! * `Fwd` relays to the PC object, at the PC slot `pc + (console slot - first)`.
//! * `Custom` is a method of the proxy (IUnknown, object wrapping, signature or convention changes).
//! * `Unknown` logs the slot with the hint and answers `E_NOTIMPL`.
//!
//! Sources, per row: "measured" means call sites in this project's own package DLLs, read with
//! `crates/decant-audit-imports` (`d3d12x_vtprobe`). "Other runtimes" means slot numbers that other
//! console compatibility runtimes measured on retail titles. "assumed" means the PC order, not
//! contradicted. All of them are leads to measure again on the title in hand with `d3d12x_vtprobe`.
//! Changing a row needs no change to the proxy code. Slots are absolute (IUnknown is 0 to 2).
//!
//! Only facts are recorded here (slot numbers and argument shapes). Internal Microsoft method names
//! read in driver strings are not reused. Nothing here comes from the console SDK. The `what` texts
//! are written to the log as they are.

/// What the proxy does with a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleKind {
    /// Relay to the PC object, PC slot `pc + (slot - first)`.
    Fwd,
    /// A method of the proxy.
    Custom,
    /// Log the slot and answer `E_NOTIMPL`.
    Unknown,
}

/// A run of console slots `first..=last` and what happens to them.
#[derive(Clone, Copy, Debug)]
pub struct SlotRule {
    pub first: u16,
    pub last: u16,
    pub kind: RuleKind,
    /// `Fwd` only: the PC slot of `first`.
    pub pc: u16,
    /// Method, source or hint, written to the log.
    pub what: &'static str,
}

const fn fwd(first: u16, last: u16, pc: u16, what: &'static str) -> SlotRule {
    SlotRule { first, last, kind: RuleKind::Fwd, pc, what }
}

const fn custom(first: u16, last: u16, what: &'static str) -> SlotRule {
    SlotRule { first, last, kind: RuleKind::Custom, pc: 0, what }
}

const fn unknown(first: u16, last: u16, what: &'static str) -> SlotRule {
    SlotRule { first, last, kind: RuleKind::Unknown, pc: 0, what }
}

/// Maps the Xbox only extended depth stencil formats 117 to 120 (R16G8_TYPELESS, D16_UNORM_S8_UINT,
/// R16_UNORM_X8_TYPELESS, X16_TYPELESS_G8_UINT) to their 24 bit PC counterparts 44 to 47
/// (R24G8_TYPELESS, D24_UNORM_S8_UINT, R24_UNORM_X8_TYPELESS, X24_TYPELESS_G8_UINT). Every other
/// value is returned unchanged.
pub fn pc_dxgi_format(fmt: u32) -> u32 {
    if (117..=120).contains(&fmt) { fmt - 73 } else { fmt }
}

/// Little endian `u32` at byte `at` of a description.
fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

const DESC_DIMENSION: usize = 0;
const DESC_ALIGNMENT: std::ops::Range<usize> = 8..16;
const DESC_FORMAT: usize = 0x20;
const DESC_LAYOUT: usize = 0x2C;
const DESC_FLAGS: usize = 0x30;
const DIMENSION_BUFFER: u32 = 1;

/// Rewrites a console `D3D12_RESOURCE_DESC` (56 bytes, same layout as the PC one, device slots 27
/// and 29) into one the PC runtime accepts. The `Alignment` is cleared because the driver
/// recomputes it. The `Layout` is forced to ROW_MAJOR (1) for a buffer and UNKNOWN (0) otherwise.
/// Only the low byte of `Flags` is a PC flag, the higher bits are console tile layout markers. The
/// format goes through [`pc_dxgi_format`].
pub fn pc_resource_desc(desc: &mut [u8; 0x38]) {
    let is_buffer = word(desc, DESC_DIMENSION) == DIMENSION_BUFFER;
    let format = pc_dxgi_format(word(desc, DESC_FORMAT));
    let mut flags = word(desc, DESC_FLAGS) & 0xFF;
    // A depth stencil target cannot also be a render target (0x1), allow unordered access (0x4) or
    // allow simultaneous access (0x20) on PC.
    if flags & 0x2 != 0 {
        flags &= !(0x1 | 0x4 | 0x20);
    }
    // The packed 16 bit formats B5G6R5 (85), B5G5R5A1 (86) and B4G4R4A4 (115) have no typed
    // unordered access support on PC, so the flag is dropped and the render target use stays. A
    // retail title was seen creating a large B5G5R5A1 target with that flag and trapping on the
    // refusal.
    if matches!(format, 85 | 86 | 115) {
        flags &= !0x4;
    }
    // B4G4R4A4 cannot be a render target on PC either.
    if format == 115 {
        flags &= !0x1;
    }
    desc[DESC_ALIGNMENT].fill(0);
    desc[DESC_FORMAT..DESC_FORMAT + 4].copy_from_slice(&format.to_le_bytes());
    desc[DESC_LAYOUT..DESC_LAYOUT + 4].copy_from_slice(&u32::from(is_buffer).to_le_bytes());
    desc[DESC_FLAGS..DESC_FLAGS + 4].copy_from_slice(&flags.to_le_bytes());
}

/// The initial state of a texture without the bits its PC flags (after [`pc_resource_desc`])
/// forbid. The console lets a title name a state its own flags allow. A retail title was seen
/// creating a cube depth target with unordered access in the unordered access state: the PC copy
/// loses that flag and the PC refuses the state with E_INVALIDARG. Buffers keep their state.
pub fn pc_initial_state(desc: &[u8; 0x38], state: u32) -> u32 {
    if word(desc, DESC_DIMENSION) == DIMENSION_BUFFER {
        return state;
    }
    let flags = word(desc, DESC_FLAGS);
    // Bits 25 and up are not resource states. Vertex and constant buffer (0x1), index buffer
    // (0x2), stream out (0x100) and indirect argument (0x200) states only exist for buffers.
    let mut allowed = state & 0x01FF_FFFF & !(0x1 | 0x2 | 0x100 | 0x200);
    // Render target state needs the render target flag.
    if flags & 0x1 == 0 {
        allowed &= !0x4;
    }
    // Unordered access state needs the unordered access flag.
    if flags & 0x4 == 0 {
        allowed &= !0x8;
    }
    // Depth write (0x10) and depth read (0x20) need the depth stencil flag.
    if flags & 0x2 == 0 {
        allowed &= !(0x10 | 0x20);
    }
    // Shader resource states (0x40, 0x80) are refused when the resource denies shader resources.
    if flags & 0x8 != 0 {
        allowed &= !(0x40 | 0x80);
    }
    allowed
}

/// The optimized clear value (`D3D12_CLEAR_VALUE`, 24 bytes, format first) for a resource whose
/// PC description is `desc` (after [`pc_resource_desc`]). There is none unless the resource is a
/// render target (0x1) or depth target (0x2), because the PC refuses one otherwise. The format is
/// the PC format of a console one, and for a depth target the depth member of a typeless family.
pub fn pc_clear_value(desc: &[u8; 0x38], clear: &[u8; 24]) -> Option<[u8; 24]> {
    let flags = word(desc, DESC_FLAGS);
    if flags & 0x3 == 0 {
        return None;
    }
    let mut value = *clear;
    let mut format = pc_dxgi_format(word(clear, 0));
    if flags & 0x2 != 0 {
        // Typeless to depth: R32G8X24 (19, 21 to 20), R32 (39, 41 to 40), R24G8 (44, 46 to 45),
        // R16 (53, 56 to 55).
        format = match format {
            19 | 21 => 20,
            39 | 41 => 40,
            44 | 46 => 45,
            53 | 56 => 55,
            other => other,
        };
    }
    value[0..4].copy_from_slice(&format.to_le_bytes());
    Some(value)
}

/// The typeless parent of a DXGI format: formats with the same parent may view each other's
/// resources on PC. A format outside a typed group is its own family.
pub fn format_family(f: u32) -> u32 {
    match f {
        1..=4 => 1,
        5..=8 => 5,
        9..=14 => 9,
        15..=18 => 15,
        19..=22 => 19,
        23..=25 => 23,
        27..=32 => 27,
        33..=38 => 33,
        39..=43 => 39,
        44..=47 => 44,
        48..=52 => 48,
        53..=59 => 53,
        60..=64 => 60,
        70..=72 => 70,
        73..=75 => 73,
        76..=78 => 76,
        79..=81 => 79,
        82..=84 => 82,
        87 | 90 | 91 => 90,
        88 | 92 | 93 => 92,
        94..=96 => 94,
        97..=99 => 97,
        other => other,
    }
}

/// Whether a view of format `view` may be made of a resource of format `resource` on PC. The
/// console lets a view reinterpret the bits (a retail title views an R32_UINT texture as R9G9B9E5); PC
/// D3D12 does not, and as the view calls return nothing it removes the device instead.
/// `view` 0 (UNKNOWN) takes the resource's format.
pub fn view_format_fits(view: u32, resource: u32) -> bool {
    view == 0 || resource == 0 || format_family(view) == format_family(resource)
}

/// First time `flag` moves from false to true: returns true. Used so a layout decision logs
/// itself exactly once.
fn latch_once(flag: &std::sync::atomic::AtomicBool) -> bool {
    !flag.swap(true, std::sync::atomic::Ordering::AcqRel)
}

/// Device slots that only the newer console device layout calls: `GetResourceAllocationInfo2` (108),
/// `CreateCommittedResource3` (127) and `CreatePlacedResourceX` with an initial barrier layout
/// (133). A call to any of them proves the newer layout. A plain membership test, so it can be unit
/// tested without a live device.
pub fn is_newer_layout_slot(slot: u32) -> bool {
    matches!(slot, 108 | 127 | 133)
}

static NEWER_LAYOUT_LATCHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LAYOUT_FORCE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();

/// `DECANT_D3D12X_LAYOUT=early|late|auto`, read once (default: unset, behaves like `auto`): forces
/// the newer ("late") or the measured default ("early") console device layout, or leaves it to be
/// decided at run time from the title's own calls. Named like the project's other `DECANT_...`
/// switches.
fn layout_forced() -> Option<bool> {
    *LAYOUT_FORCE.get_or_init(|| match std::env::var("DECANT_D3D12X_LAYOUT").ok().as_deref() {
        Some("late") => Some(true),
        Some("early") => Some(false),
        _ => None,
    })
}

/// Whether the console device should use the newer layout's slot bindings right now.
pub fn newer_layout() -> bool {
    layout_forced().unwrap_or_else(|| NEWER_LAYOUT_LATCHED.load(std::sync::atomic::Ordering::Acquire))
}

/// `ID3D12Device11` `{5405c344-d457-444e-b4dd-2366e45aee39}`: a title that creates its device as
/// this interface has the newer console layout. A retail title was seen doing so at its first
/// device creation, long before it calls any of the slots of [`is_newer_layout_slot`].
pub const IID_DEVICE11: [u8; 16] = [0x44, 0xc3, 0x05, 0x54, 0x57, 0xd4, 0x4e, 0x44, 0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39];

/// The title created its device as the interface `iid` (its 16 bytes as in memory). Returns `true`
/// the first time this proves the newer layout, like [`note_newer_device_method`].
pub fn note_device_iid(iid: &[u8; 16]) -> bool {
    if *iid != IID_DEVICE11 || layout_forced().is_some() {
        return false;
    }
    latch_once(&NEWER_LAYOUT_LATCHED)
}

/// The title called device slot `slot`. Returns `true` only for the call that first proves the
/// newer layout, so the caller can log the switch once. Returns `false` when the layout is forced,
/// already latched, or `slot` is not one of [`is_newer_layout_slot`].
pub fn note_newer_device_method(slot: u32) -> bool {
    is_newer_layout_slot(slot) && layout_forced().is_none() && latch_once(&NEWER_LAYOUT_LATCHED)
}

/// Adds one more descriptor to a packed `GetResourceAllocationInfo2` allocation. Given the running
/// `total` size and `align` (maximum alignment) and the `(this_size, this_align)` the PC driver
/// reported for the next descriptor, returns `(offset of this one, new total, new maximum
/// alignment)`. The offset is the old total rounded up to `this_align`. Returns `None` when the
/// descriptor reports size `u64::MAX` or alignment 0, the console convention for "cannot answer",
/// which makes the whole allocation unanswerable.
pub fn accumulate_alloc_info(total: u64, align: u64, this_size: u64, this_align: u64) -> Option<(u64, u64, u64)> {
    if this_size == u64::MAX || this_align == 0 {
        return None;
    }
    let offset = total.next_multiple_of(this_align);
    Some((offset, offset + this_size, align.max(this_align)))
}

pub const DEVICE_SLOTS: u16 = 160;
pub const LIST_SLOTS: u16 = 160;
/// Command-list slots whose unknown methods log their arguments (`proxy`'s `unknown_probe`, one
/// probe per slot): the console list's middle range, where the newer layout's write, wait and
/// barrier methods are.
pub const PROBE_LIST_SLOTS: std::ops::RangeInclusive<u32> = 60..=149;
pub const QUEUE_SLOTS: u16 = 48;
pub const HEAP_SLOTS: u16 = 16;
pub const ALLOCATOR_SLOTS: u16 = 12;
pub const RESOURCE_SLOTS: u16 = 16;
pub const FENCE_SLOTS: u16 = 16;
pub const DMA_LIST_SLOTS: u16 = 32;

/// The seven kinds of proxied object, in the order of the tables below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Device = 0,
    List = 1,
    Queue = 2,
    Heap = 3,
    Allocator = 4,
    Resource = 5,
    Fence = 6,
    /// A command list created with the console's DMA list IID (device slots 73 and 12).
    DmaList = 7,
}

impl Kind {
    pub const ALL: [Kind; 8] = [Kind::Device, Kind::List, Kind::Queue, Kind::Heap, Kind::Allocator, Kind::Resource, Kind::Fence, Kind::DmaList];

    pub fn from_index(index: u32) -> Option<Kind> {
        Kind::ALL.get(index as usize).copied()
    }

    /// The name used in the log lines.
    pub fn name(self) -> &'static str {
        ["device", "command list", "queue", "descriptor heap", "command allocator", "resource", "fence", "DMA command list"][self as usize]
    }

    pub fn slots(self) -> u16 {
        [DEVICE_SLOTS, LIST_SLOTS, QUEUE_SLOTS, HEAP_SLOTS, ALLOCATOR_SLOTS, RESOURCE_SLOTS, FENCE_SLOTS, DMA_LIST_SLOTS][self as usize]
    }

    pub fn rules(self) -> &'static [SlotRule] {
        [DEVICE_RULES, LIST_RULES, QUEUE_RULES, HEAP_RULES, ALLOCATOR_RULES, RESOURCE_RULES, FENCE_RULES, DMA_LIST_RULES][self as usize]
    }

    /// The rule that covers `slot`, if any.
    pub fn rule(self, slot: u32) -> Option<&'static SlotRule> {
        self.rules().iter().find(|r| slot >= u32::from(r.first) && slot <= u32::from(r.last))
    }
}

/// ID3D12Device (console): the PC order of the base ID3D12Device from 0 to 43, console methods from
/// 44 on (another retail title: 44 to 160 are all console methods, not PC ones). Nothing from 44 on is relayed by position.
pub const DEVICE_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown (IDXGIDevice answered by a DXGI device object: another retail title asks for it right after creation)"),
    fwd(3, 7, 3, "ID3D12Object, GetNodeCount (assumed)"),
    custom(8, 8, "CreateCommandQueue (PC form), wrapped (another retail title: exists beside 48)"),
    custom(9, 9, "CreateCommandAllocator (PC form), wrapped: the title calls slot 9 of the allocator (memory callbacks), which the PC object does not have"),
    custom(10, 10, "CreateGraphicsPipelineState (PC form): state enums made valid for PC as on 60"),
    custom(11, 11, "CreateComputePipelineState (PC form; assumed, never called by the package): a DXIL override is looked up before the host builds it, DECANT_D3D12X_DXIL_OVERRIDE"),
    custom(12, 12, "CreateCommandList (PC form), wrapped (assumed PC slot; console titles use 73)"),
    fwd(13, 13, 13, "CheckFeatureSupport (another retail title; some console structures are 64 bytes against 60 on PC, relayed as is)"),
    custom(14, 14, "CreateDescriptorHeap, wrapped: the console heap returns its start handles in rax (measured riid position, also seen in another retail title)"),
    custom(15, 15, "GetDescriptorHandleIncrementSize (PC order; other runtimes measured 15, 18, 19 and 15, 17, 22, 23): console heap type 4 asked as type 0"),
    custom(
        16,
        16,
        "CreateRootSignature (PC form): console-only RTS0 flag bits cleared from the blob before the host sees it (round 13 of the port); DECANT_D3D12X_RS_HEAP=1 also adds CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED",
    ),
    custom(17, 24, "CBV/SRV/UAV/RTV/DSV/Sampler/CopyDescriptors: translate title-memory CPU handles to host heap handles (PC 17..24)"),
    custom(25, 26, "GetResourceAllocationInfo, GetCustomHeapProperties: structure returned by value; a console caller may pass (result, this) or (this, result) - whichever argument is one of our own device proxies is this, detected per call (another retail title)"),
    custom(27, 27, "CreateCommittedResource (PC form): the console D3D12_RESOURCE_DESC is sanitized the same way as CreatePlacedResource below"),
    fwd(28, 28, 28, "CreateHeap (PC form; no console-only D3D12_HEAP_FLAGS bits are known to need masking yet)"),
    custom(29, 29, "CreatePlacedResource (PC form), the heap argument unwrapped; the console D3D12_RESOURCE_DESC (56 bytes, same layout as the PC one) has an Alignment the driver recomputes, a Layout only known for buffers, and Flags bits above the low byte with no PC meaning"),
    fwd(30, 30, 30, "CreateReservedResource (PC order, measured)"),
    custom(31, 31, "CreateSharedHandle (PC order): the object may be a proxied resource, unwrapped"),
    fwd(32, 33, 32, "OpenSharedHandle, OpenSharedHandleByName (PC order, measured)"),
    custom(34, 35, "MakeResident, Evict: proxied objects unwrapped (PC order)"),
    custom(36, 36, "CreateFence (PC order), wrapped: the newer layout's fence slot 12 hands out the address of the completed value"),
    fwd(37, 40, 37, "PC order (measured: 38 GetCopyableFootprints, other runtimes agree)"),
    custom(41, 41, "CreateCommandSignature (PC form): console indirect argument type 105 (DispatchX) is PC Dispatch"),
    custom(42, 42, "GetResourceTiling (PC order): the tiled resource may be a proxy, unwrapped"),
    fwd(43, 43, 43, "GetAdapterLuid (PC order; returns a LUID by value, console convention not measured)"),
    custom(44, 44, "placed resource at a virtual address (VA, desc, state, clear, riid, out) - backed by a sanitized committed resource on PC until VA placement exists"),
    unknown(45, 47, "console device method"),
    custom(48, 48, "console queue creation (console desc, riid, out); first 16 bytes = PC desc (other runtimes)"),
    unknown(49, 51, "console device method"),
    custom(52, 52, "SetDebugFlagsX: S_OK, nothing to set on PC"),
    unknown(53, 54, "console device method"),
    custom(55, 55, "derived pipeline (base PSO, patch count, patches of 0x14C bytes, riid, out), measured by another runtime; not supported"),
    custom(56, 56, "SerializePipelineStateX: E_ACCESSDENIED, the console driver form cannot be made"),
    custom(57, 57, "DeserializeGraphicsPipelineStateX(cache entry, riid, out): a pipeline cache miss, out cleared, D3D12_ERROR_DRIVER_VERSION_MISMATCH; the title skips the entry and creates the pipeline from its desc (60) when it needs it"),
    unknown(58, 59, "console device method"),
    custom(60, 60, "graphics pipeline (desc in PC layout, extension count, extensions, riid, out), measured by another runtime"),
    custom(61, 61, "CreateSamplerX (0x6C-byte console desc, anisotropic filter only) -> PC CreateSampler"),
    custom(62, 62, "compute pipeline (desc, extension count, extensions, riid, out) - measured with zero extensions, other runtimes agree"),
    unknown(63, 65, "console device method"),
    custom(66, 66, "SerializePipelineStateX (compute): E_ACCESSDENIED"),
    custom(67, 67, "DeserializeComputePipelineStateX(cache entry, riid, out), the compute twin of 57 (same shape, on a miss the caller creates through 62): a cache miss like 57"),
    unknown(68, 68, "GetGpuHardwareConfigurationX (public SystemInfo sample); structure not public"),
    unknown(69, 69, "console device method"),
    custom(70, 71, "CreateShaderResourceViewAt / CreateUnorderedAccessViewAt (views by GPU VA)"),
    unknown(72, 72, "console device method"),
    custom(73, 73, "command list creation (parameters {type at 0, node mask at 8}, allocator, PSO, riid, out) (other runtimes)"),
    custom(74, 75, "RegisterPagePoolX / UnregisterPagePoolX"),
    custom(76, 76, "console committed resource, CreateCommittedResource shape (measured: riid 7th argument, other runtimes agree): the desc sanitised as for 27"),
    unknown(77, 77, "console device method"),
    custom(78, 80, "SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX (other runtimes, public samples): frame pacing emulated"),
    custom(81, 81, "frame statistics by token (token, type, count*, statistics*): count set to 0, S_OK, no statistics kept (the record structure is not measured)"),
    unknown(82, 82, "performance counters (other runtimes)"),
    unknown(83, 85, "console device method"),
    custom(86, 86, "CreatePipelineState from stream (unpacked to plain graphics/compute desc, or our mesh desc made on first bind through ID3D12Device2::CreatePipelineState)"),
    unknown(87, 87, "release (other runtimes)"),
    unknown(88, 107, "console device method"),
    custom(108, 108, "GetResourceAllocationInfo2 (newer layout; a call here latches the newer layout, D-2026-11-23-2): same struct-return ambiguity as 25/26, console D3D12_RESOURCE_DESC1[] packed with the same accumulation as 25"),
    unknown(109, 126, "console device method"),
    custom(127, 127, "CreateCommittedResource3 with initial D3D12_BARRIER_LAYOUT (newer layout; latches newer layout, D-2026-11-23-2), ALLOW_DISPLAY masked"),
    unknown(128, 132, "console device method"),
    custom(133, 133, "CreatePlacedResourceX with initial D3D12_BARRIER_LAYOUT (newer layout; latches newer layout, D-2026-11-23-2), backed by a sanitized committed resource on PC"),
    unknown(134, DEVICE_SLOTS - 1, "console device method"),
];

/// ID3D12GraphicsCommandList (console). PC order to 26, the query block moved up to 28..31, then the
/// PC order shifted by 4 up to DiscardResource (55), the PC order without shift for 56..59.
pub const LIST_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 8, 3, "PC order"),
    custom(9, 9, "Close (PC order, another runtime measured 9): a list the PC runtime refuses to close is remembered and left out of ExecuteCommandLists, which would remove the device"),
    custom(10, 10, "Reset (older layout): the allocator argument may be a proxy and is unwrapped"),
    fwd(11, 11, 11, "ClearState (PC order)"),
    custom(12, 14, "DrawInstanced, DrawIndexedInstanced, Dispatch (PC order, measured: 14 Dispatch; other runtimes measured 15, 16, 17): the user data mirror is written first when the root arguments changed (DECANT_D3D12X_DXIL_OVERRIDE)"),
    custom(15, 15, "CopyBufferRegion (PC order): both buffers may be proxies, unwrapped"),
    custom(16, 16, "CopyTextureRegion (PC order): the copy locations are copied and their resources unwrapped"),
    custom(17, 17, "CopyResource (PC order): both resources unwrapped"),
    custom(18, 18, "CopyTiles (PC order): the tiled resource and the buffer unwrapped"),
    custom(19, 19, "ResolveSubresource (PC order): both resources unwrapped"),
    custom(20, 20, "IASetPrimitiveTopology: maps Xbox RECT_LIST (0x11) to PC TRIANGLELIST (4)"),
    custom(21, 22, "RSSetViewports, RSSetScissorRects (PC order): relayed, and noted for the draw probe (DECANT_D3D12X_PRESENT_PROBE)"),
    fwd(23, 24, 23, "PC order"),
    custom(25, 25, "SetPipelineState (PC order, measured): relayed, and noted for the render target probe (DECANT_D3D12X_PRESENT_PROBE)"),
    custom(26, 26, "ResourceBarrier (PC order, measured): the barriers are copied and their resources unwrapped"),
    custom(27, 27, "ExecuteBundle (other runtimes), proxied bundle unwrapped"),
    fwd(28, 29, 52, "BeginQuery, EndQuery (PC 52, 53; other runtimes, consistent with the shift of 4)"),
    custom(30, 30, "ResolveQueryData (PC 54): the destination buffer unwrapped"),
    custom(31, 31, "SetPredication (PC 55): the buffer unwrapped"),
    custom(32, 32, "SetDescriptorHeaps (PC 28), proxied heaps unwrapped (other runtimes)"),
    custom(33, 40, "PC + 4 (measured: 33, 37, 39; other runtimes measured 34, 35, 36), PC 29..36: Set{Compute,Graphics}RootSignature, RootDescriptorTable, Root32BitConstant, Root32BitConstants, relayed and recorded in the shadow of the root arguments (user data mirror)"),
    custom(41, 46, "SetCompute/GraphicsRootConstantBufferView, ShaderResourceView, UnorderedAccessView (PC 37..42, measured: 43, 45; another runtime measured 41): an address that is upload memory of slot 148 becomes its GPU address"),
    custom(47, 47, "IASetIndexBuffer (PC 43, measured: 47): the view is remembered for DrawIndexedX, upload memory addresses translated"),
    custom(48, 48, "IASetVertexBuffers (PC 44; another runtime measured 48): upload memory addresses in the views translated"),
    fwd(49, 49, 45, "SOSetTargets (PC 45)"),
    custom(50, 52, "OMSetRenderTargets, ClearDepthStencilView, ClearRenderTargetView (PC 46..48, another runtime measured 50, 51, 52): title-memory CPU handles translated to host heap handles (an untranslated handle crashes the PC runtime in ClearRenderTargetView)"),
    custom(53, 53, "ClearUnorderedAccessViewUint (PC 49): the resource unwrapped"),
    custom(54, 54, "ClearUnorderedAccessViewFloat (PC 50): the resource unwrapped"),
    custom(55, 55, "DiscardResource (PC 51): the resource unwrapped"),
    fwd(56, 58, 56, "SetMarker, BeginEvent, EndEvent: PC order, no shift"),
    custom(59, 59, "ExecuteIndirect (PC order, older layout): the argument and count buffers unwrapped (another runtime measured 59 with the PC arguments); newer layout: not ExecuteIndirect, logged and skipped"),
    unknown(60, 60, "memory copy (other runtimes, older layout); newer layout: not known"),
    custom(61, 61, "ExecuteIndirect, newer layout (PC 59; measured: signature, 1, buffer, offset, NULL, 0); older layout: a fill method (other runtimes), logged"),
    custom(62, 62, "CopyMemoryToMemoryX(destination, source, size, flags), newer layout (older 60): a buffer copy between buffers the proxy knows by address; older layout: logged"),
    custom(63, 65, "FillMemoryWith32BitValueX, 64BitValueX, 128BitValueX(address, size, value or pointer), newer layout (older 61 to 63): copies from a pattern buffer; older layout: logged"),
    unknown(66, 66, "console-only"),
    custom(67, 67, "texture cache flush (other runtimes): nothing to do on PC"),
    fwd(68, 68, 62, "OMSetDepthBounds, two floats (PC 62, ID3D12GraphicsCommandList1), other runtimes"),
    unknown(69, 69, "console-only (other runtimes: end-of-pipe writes, waits, predication)"),
    custom(70, 70, "OMSetDepthBounds, newer layout (PC 62)"),
    custom(71, 72, "Write32/64BitValueBottomOfPipeX, newer layout (values to title memory, done by the CPU after the GPU)"),
    custom(73, 74, "Wait32/64BitValueX, newer layout (memory waits, evaluated before the list runs)"),
    custom(75, 75, "pixel shader depth order(0 or 1 from the bound program), newer layout (older 73; a retail title calls it after SetPipelineState when the value changes): logged, nothing to set on PC"),
    unknown(76, 77, "console-only (other runtimes: end-of-pipe writes, waits, predication)"),
    unknown(78, 78, "console dispatch (other runtimes, older layout)"),
    unknown(79, 79, "console-only"),
    custom(80, 80, "DispatchX(x, y, z, flags), newer layout (PC 14; measured: n, 1, 1, 0x8020); older layout: logged"),
    unknown(81, 81, "console indexed draw (other runtimes)"),
    unknown(82, 82, "console-only"),
    custom(83, 83, "newer layout: DrawIndexedX(index address, format, count, flags), a temporary index view with the bound one restored; older layout: CopyTextureRegion with an 8th argument, always 0 (other runtimes)"),
    unknown(84, 84, "console resource copy (other runtimes); arguments not known"),
    custom(85, 85, "CopyTextureRegionX, newer layout (8 arguments, flags last; PC 16)"),
    custom(86, 86, "CopyResourceX, newer layout (PC 17)"),
    custom(87, 87, "SetPredicationBufferX({ address, operation } or null), newer layout (older 85; PC 55 on the buffer holding the address); older layout: logged"),
    unknown(88, 89, "console-only"),
    custom(90, 91, "Write32BitValueTopOfPipeX, Write64BitValueTopOfPipeX, newer layout (older 88, 89): values to title memory, done by the CPU after the GPU like slots 71 and 72; older layout: logged"),
    custom(92, 92, "Write32BitValueBottomOfShaderX, newer layout (older 90): same CPU write; older layout: logged"),
    unknown(93, 94, "console-only"),
    custom(95, 95, "KickoffX, newer layout (older 93): logged, the commands run when the list is executed; older layout: logged"),
    custom(96, 96, "RSSetDepthBiasX(bias, clamp, slope), newer layout (older 94): ID3D12GraphicsCommandList9::RSSetDepthBias (PC 82) when the PC list has it; older layout: logged"),
    unknown(97, 99, "console-only"),
    custom(100, 100, "DispatchMeshX(x, flags, indirect arguments, payload), newer layout (older 98): PC DispatchMesh(x, 1, 1), an indirect count is logged and left out"),
    unknown(101, 103, "console-only"),
    unknown(104, 109, "PC 60..65 per another runtime, which contradicts the OMSetDepthBounds measured at 68: not relayed"),
    unknown(110, 123, "console-only"),
    unknown(124, 125, "RSSetShadingRate, RSSetShadingRateImage (PC 77, 78) per other runtimes: not relayed"),
    custom(126, 126, "shading rate, newer layout (older 124): ignored, no variable rate on PC here"),
    custom(127, 127, "DispatchMesh(x, y, z), older layout (other runtimes): PC ID3D12GraphicsCommandList6::DispatchMesh; newer layout: not known, logged"),
    unknown(128, 128, "console-only"),
    custom(129, 129, "DispatchMesh(x, y, z), newer layout (older 127): PC ID3D12GraphicsCommandList6::DispatchMesh with the user data mirror prepared as for a draw"),
    unknown(130, 130, "console-only"),
    custom(131, 132, "pipeline syncs at the top and the bottom of the pipe (flags, control, address, size), newer layout (older 129, 130): nothing to do when no memory is named"),
    unknown(133, 138, "console-only"),
    custom(139, 139, "Reset, with one extra argument (newer layout, D-2026-11-23-2: slot 10's older-layout Reset shifted here)"),
    unknown(140, 140, "console-only"),
    unknown(141, 141, "console-only (another runtime: binds a compute scratch buffer)"),
    custom(142, 142, "state call with one u64 argument, result unused, newer layout (purpose not measured): ignored"),
    unknown(143, 145, "console-only"),
    custom(146, 146, "UAV barrier used between BVH build passes, newer layout (one global UAV barrier)"),
    unknown(147, 147, "console-only"),
    custom(148, 148, "AllocateUploadMemory(size, alignment): CPU memory the title writes, whose address the proxy also accepts as a GPU address in root descriptors and index and vertex buffers; freed when the command allocator is reset"),
    unknown(149, LIST_SLOTS - 1, "console-only"),
];

/// ID3D12CommandQueue (console): PC order to GetDesc (18), Signal with flags at 23, PresentX at 30
/// (measured in another retail title). The PC vtable holds private entries after 18 that the runtime calls itself. The proxy
/// never touches the PC vtable, so those numbers keep their console meaning here.
pub const QUEUE_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 7, 3, "ID3D12Object, GetDevice (assumed)"),
    custom(8, 8, "UpdateTileMappings (PC order): the tiled resource unwrapped"),
    custom(9, 9, "CopyTileMappings (PC order): both resources unwrapped"),
    custom(10, 10, "ExecuteCommandLists, unwraps proxied lists (other runtimes)"),
    fwd(11, 13, 11, "SetMarker, BeginEvent, EndEvent (PC order in both layouts)"),
    custom(14, 15, "older layout: Signal, Wait (PC 14, 15), fence proxies unwrapped; newer layout: two console methods inserted here, logged"),
    custom(16, 17, "older layout: GetTimestampFrequency, GetClockCalibration (other runtimes measured 16, 17; the console answers 99,750,000 Hz, the PC frequency is kept); newer layout: Signal, Wait (PC 14, 15), measured as (fence, 1), (fence, 2)"),
    custom(18, 18, "newer layout: GetTimestampFrequency (PC 16), measured (one u64 out); older layout: GetDesc (PC 18), structure returned through a hidden pointer"),
    custom(19, 19, "newer layout: GetClockCalibration (PC 17), inferred from the shift of slot 18; older layout: unknown console method"),
    custom(20, QUEUE_SLOTS - 1, "older numbering (newer layout: slot minus two): 18 GetDesc, 23 Signal(fence, value, flags) (other runtimes), 30 PresentX(plane count, planes, parameters) (other runtimes), 33 ExecuteCommandListsX(count, lists, value, flags); the rest answer S_OK and are logged"),
];

/// ID3D12DescriptorHeap (console). Same methods as the PC. The start handles come back in rax
/// instead of through a hidden result pointer (other runtimes).
pub const HEAP_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 8, 3, "ID3D12Object, GetDevice, GetDesc (assumed; GetDesc returns a structure by value, console convention not measured)"),
    custom(9, 10, "GetCPUDescriptorHandleForHeapStart, GetGPUDescriptorHandleForHeapStart: handle in rax (other runtimes)"),
    unknown(11, HEAP_SLOTS - 1, "not a PC descriptor heap method"),
];

/// ID3D12CommandAllocator (console). PC order to Reset (8), then slot 9: the title calls
/// `slot9(this, fn, ctx)` once per allocator of its ring (measured on a retail title, a
/// "set memory callbacks" method: `fn` takes size and alignment and calls the title's allocator).
pub const ALLOCATOR_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 7, 3, "ID3D12Object, GetDevice (PC order)"),
    custom(8, 8, "Reset (PC order): the upload memory the lists took from this allocator (list slot 148) is freed"),
    custom(9, 9, "memory callbacks (function, context): recorded in the proxy, never called"),
    unknown(10, ALLOCATOR_SLOTS - 1, "console allocator method"),
];

/// ID3D12Resource (console). The PC order, as the C layer's resource table has it (PC 0..14). Only
/// resources placed at a console address (device slots 44 and 133) are proxied; the others are PC
/// objects handed out as they are.
pub const RESOURCE_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 6, 3, "ID3D12Object: GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName (PC order)"),
    fwd(7, 7, 7, "GetDevice (PC order; the PC device comes back as it is)"),
    custom(8, 8, "Map (PC order): a placed resource PC cannot map answers its title address (Map fallback, UMA)"),
    custom(9, 9, "Unmap (PC order): skipped after a Map that answered the title address"),
    fwd(10, 10, 10, "GetDesc (PC order; returns a structure by value, console convention not measured)"),
    custom(11, 11, "GetGPUVirtualAddress (PC order): PC answers 0 for a texture, a placed resource answers its title address"),
    fwd(12, 14, 12, "WriteToSubresource, ReadFromSubresource, GetHeapProperties (PC order)"),
    unknown(15, RESOURCE_SLOTS - 1, "console resource method"),
];

/// ID3D12Fence (console). PC order to GetCreationFlags (11). Slot 12 exists only in the newer
/// layout: the address of the completed value, which a retail title polls with MONITORX and MWAITX
/// (a retail title calls it, then reads the returned u64 in a loop).
pub const FENCE_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 7, 3, "ID3D12Object, GetDevice (PC order)"),
    custom(8, 8, "GetCompletedValue: the cell's answer (rules R1 and R2: MAX only on a removed device)"),
    fwd(9, 9, 9, "SetEventOnCompletion (PC order)"),
    custom(10, 10, "Signal from the CPU (PC order), then the completed value cell is refreshed"),
    fwd(11, 11, 11, "GetCreationFlags (PC order, ID3D12Fence1)"),
    custom(12, 12, "the address of the completed value (newer layout), a cell kept equal to the PC value"),
    unknown(13, FENCE_SLOTS - 1, "console fence method"),
];

/// A console DMA command list, created with IID `30bc77b9-8379-4b25-95de-b8588c42275d` (device
/// slots 73 and 12). Close at 9 and Reset at 10, then the copy methods: 11 to 14 take the arguments
/// of the PC's slots 15 to 18 (CopyBufferRegion, CopyTextureRegion, CopyResource, CopyTiles; a
/// title was seen calling 12 with CopyTextureRegion's arguments). What follows is not known, so
/// each further slot answers S_OK and does nothing, which keeps a title that calls them going.
pub const DMA_LIST_RULES: &[SlotRule] = &[
    custom(0, 2, "IUnknown"),
    fwd(3, 8, 3, "ID3D12Object, GetDevice, GetType (PC order)"),
    custom(9, 9, "Close (PC order), a refused close remembered as for the other lists"),
    custom(10, 10, "Reset (PC order): the allocator may be a proxy and is unwrapped"),
    custom(11, 11, "CopyBufferRegion (PC 15): both buffers unwrapped"),
    custom(12, 12, "CopyTextureRegion (PC 16): the copy locations are copied and their resources unwrapped"),
    custom(13, 13, "CopyResource (PC 17): both resources unwrapped"),
    custom(14, 14, "CopyTiles (PC 18): the tiled resource and the buffer unwrapped"),
    custom(15, DMA_LIST_SLOTS - 1, "unknown DMA list method: answers S_OK and does nothing"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The desc a retail title stops on: a square cube R16_TYPELESS target with depth, unordered access
    /// and console flags (0x80300006), as `pc_resource_desc` leaves it.
    fn cube_depth_desc() -> [u8; 0x38] {
        let mut d = [0u8; 0x38];
        d[0] = 3;
        d[0x10..0x18].copy_from_slice(&768u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&768u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&6u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&53u32.to_le_bytes());
        d[0x24] = 1;
        d[0x30..0x34].copy_from_slice(&0x8030_0006u32.to_le_bytes());
        pc_resource_desc(&mut d);
        d
    }

    #[test]
    fn a_depth_target_loses_the_unordered_access_state_with_its_flag() {
        let d = cube_depth_desc();
        assert_eq!(u32::from_le_bytes([d[0x30], d[0x31], d[0x32], d[0x33]]), 0x2);
        assert_eq!(pc_initial_state(&d, 0x8), 0, "unordered access");
        assert_eq!(pc_initial_state(&d, 0x10), 0x10, "depth write stays");
        assert_eq!(pc_initial_state(&d, 0x4), 0, "render target");
        assert_eq!(pc_initial_state(&d, 0xAC3), 0x8C0, "generic read without the buffer bits");
        let mut buffer = [0u8; 0x38];
        buffer[0] = 1;
        assert_eq!(pc_initial_state(&buffer, 0xAC3), 0xAC3, "a buffer keeps its state");
    }

    #[test]
    fn a_clear_value_takes_the_depth_format_and_is_dropped_without_a_target() {
        let d = cube_depth_desc();
        let mut clear = [0u8; 24];
        clear[0..4].copy_from_slice(&53u32.to_le_bytes());
        clear[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        let c = pc_clear_value(&d, &clear).unwrap();
        assert_eq!(u32::from_le_bytes([c[0], c[1], c[2], c[3]]), 55, "D16_UNORM");
        assert_eq!(c[4..], clear[4..]);
        let mut plain = d;
        plain[0x30..0x34].fill(0);
        assert_eq!(pc_clear_value(&plain, &clear), None);
    }

    #[test]
    fn views_must_stay_in_the_resource_format_family() {
        assert!(view_format_fits(41, 42), "R32_FLOAT view of R32_UINT");
        assert!(view_format_fits(29, 27), "sRGB view of a typeless RGBA8 resource");
        assert!(view_format_fits(0, 42), "UNKNOWN takes the resource format");
        assert!(view_format_fits(91, 87), "BGRA sRGB view of BGRA");
        assert!(!view_format_fits(67, 42), "R9G9B9E5 view of R32_UINT is refused on PC");
        assert!(!view_format_fits(28, 87), "RGBA view of BGRA");
    }

    #[test]
    fn the_fence_table_relays_the_pc_slots_and_answers_slot_12() {
        let k = Kind::Fence;
        assert_eq!((k.slots(), k.name()), (16, "fence"));
        for s in [3u32, 7, 9, 11] {
            let r = k.rule(s).unwrap();
            assert_eq!((r.kind, r.pc + (s as u16 - r.first)), (RuleKind::Fwd, s as u16), "slot {s}");
        }
        // GetCompletedValue (8) is answered by the fence cell rule (rules R1 and R2), not relayed.
        assert_eq!(k.rule(8).unwrap().kind, RuleKind::Custom, "slot 8 is the settled answer");
        assert_eq!(k.rule(10).unwrap().kind, RuleKind::Custom);
        assert_eq!(k.rule(12).unwrap().kind, RuleKind::Custom);
        assert_eq!(k.rule(13).unwrap().kind, RuleKind::Unknown);
        assert_eq!(Kind::from_index(6), Some(Kind::Fence));
        assert_eq!(Kind::Device.rule(36).unwrap().kind, RuleKind::Custom, "CreateFence wraps");
    }

    #[test]
    fn the_allocator_table_relays_the_pc_slots_and_records_slot_9() {
        let k = Kind::Allocator;
        assert_eq!(k.slots(), 12);
        assert_eq!(k.name(), "command allocator");
        assert_eq!(k.rule(0).unwrap().kind, RuleKind::Custom);
        for s in 3..=7u32 {
            let r = k.rule(s).unwrap();
            assert_eq!((r.kind, r.pc + (s as u16 - r.first)), (RuleKind::Fwd, s as u16), "slot {s}");
        }
        assert_eq!(k.rule(8).unwrap().kind, RuleKind::Custom, "Reset frees the upload memory");
        assert_eq!(k.rule(9).unwrap().kind, RuleKind::Custom);
        for s in 10..=11u32 {
            assert_eq!(k.rule(s).unwrap().kind, RuleKind::Unknown, "slot {s}");
        }
        assert!(k.rule(12).is_none());
        assert_eq!(Kind::from_index(4), Some(Kind::Allocator));
        assert_eq!(Kind::from_index(5), Some(Kind::Resource));
        assert_eq!(Kind::from_index(7), Some(Kind::DmaList));
        assert_eq!(Kind::from_index(8), None);
    }

    #[test]
    fn the_resource_table_is_the_pc_order_with_placed_resources_only_proxied() {
        let k = Kind::Resource;
        assert_eq!(k.slots(), 16);
        assert_eq!(k.name(), "resource");
        for s in [0u32, 1, 2, 7] {
            assert_eq!(k.rule(s).unwrap().kind, if s < 3 { RuleKind::Custom } else { RuleKind::Fwd }, "slot {s}");
        }
        for s in 3..=6u32 {
            let r = k.rule(s).unwrap();
            assert_eq!((r.kind, r.pc + (s as u16 - r.first)), (RuleKind::Fwd, s as u16), "slot {s}");
        }
        for s in [10u32, 12, 13, 14] {
            let r = k.rule(s).unwrap();
            assert_eq!((r.kind, r.pc + (s as u16 - r.first)), (RuleKind::Fwd, s as u16), "slot {s}");
        }
        assert_eq!(k.rule(8).unwrap().kind, RuleKind::Custom, "Map is the placed fallback");
        assert_eq!(k.rule(9).unwrap().kind, RuleKind::Custom, "Unmap is skipped after the fallback");
        assert_eq!(k.rule(11).unwrap().kind, RuleKind::Custom, "GetGPUVirtualAddress is the placed fallback");
        assert_eq!(k.rule(15).unwrap().kind, RuleKind::Unknown);
    }

    #[test]
    fn the_resource_arguments_of_the_relays_are_unwrapped_by_custom_slots() {
        // list: CopyBufferRegion, CopyTextureRegion, CopyResource, CopyTiles, ResolveSubresource,
        // ResourceBarrier, ResolveQueryData, SetPredication, ClearUAV Uint/Float, DiscardResource,
        // ExecuteIndirect
        for s in [15u32, 16, 17, 18, 19, 26, 30, 31, 53, 54, 55, 59, 83, 85, 86] {
            assert_eq!(Kind::List.rule(s).unwrap().kind, RuleKind::Custom, "list slot {s}");
        }
        // queue: UpdateTileMappings, CopyTileMappings
        assert_eq!(Kind::Queue.rule(8).unwrap().kind, RuleKind::Custom);
        assert_eq!(Kind::Queue.rule(9).unwrap().kind, RuleKind::Custom);
        // device: CreateSharedHandle, GetResourceTiling
        assert_eq!(Kind::Device.rule(31).unwrap().kind, RuleKind::Custom);
        assert_eq!(Kind::Device.rule(42).unwrap().kind, RuleKind::Custom);
        // list: OMSetRenderTargets, ClearDepthStencilView, ClearRenderTargetView translate CPU handles
        for s in 50..=52 {
            assert_eq!(Kind::List.rule(s).unwrap().kind, RuleKind::Custom, "list slot {s}");
        }
        // the rows that take no resource stay relays, and keep their PC slots
        for (kind, s, pc) in [
            (Kind::List, 11, 11),
            (Kind::List, 23, 23),
            (Kind::List, 28, 52),
            (Kind::List, 49, 45),
            (Kind::List, 56, 56),
            (Kind::Device, 30, 30),
            (Kind::Device, 32, 32),
            (Kind::Device, 43, 43),
            (Kind::Queue, 3, 3),
            (Kind::Queue, 7, 7),
        ] {
            let r = kind.rule(s).unwrap();
            assert_eq!((r.kind, r.pc + (s as u16 - r.first)), (RuleKind::Fwd, pc), "{} slot {s}", kind.name());
        }
    }

    #[test]
    fn device_slot_9_and_list_slot_10_are_proxy_methods() {
        assert_eq!(Kind::Device.rule(9).unwrap().kind, RuleKind::Custom);
        assert_eq!(Kind::Device.rule(10).unwrap().kind, RuleKind::Custom, "graphics PSO with valid state enums");
        assert_eq!(Kind::List.rule(10).unwrap().kind, RuleKind::Custom);
        let r = Kind::List.rule(11).unwrap();
        assert_eq!((r.kind, r.pc), (RuleKind::Fwd, 11));
    }

    #[test]
    fn device_slot_81_frame_statistics_is_a_proxy_method() {
        assert_eq!(Kind::Device.rule(81).unwrap().kind, RuleKind::Custom);
        assert_eq!(Kind::Device.rule(80).unwrap().kind, RuleKind::Custom, "the frame clock next to it");
    }

    #[test]
    fn list_slot_126_shading_rate_is_ignored_and_its_neighbours_stay_unknown() {
        let l = Kind::List;
        assert_eq!(l.rule(126).unwrap().kind, RuleKind::Custom);
        assert!(l.rule(126).unwrap().what.contains("newer layout"));
        for s in [124u32, 125] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "list slot {s}");
        }
    }

    #[test]
    fn mesh_dispatches_are_list_slots_127_older_and_129_and_100_newer() {
        let l = Kind::List;
        for s in [100u32, 127, 129] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Custom, "list slot {s}");
            assert!(l.rule(s).unwrap().what.contains("DispatchMesh"), "list slot {s}");
        }
        for s in [99u32, 101, 128, 130] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "list slot {s}");
        }
    }

    #[test]
    fn list_slots_131_and_132_are_pipeline_syncs_and_their_neighbours_stay_unknown() {
        let l = Kind::List;
        assert_eq!(l.rule(131).unwrap().kind, RuleKind::Custom);
        assert_eq!(l.rule(132).unwrap().kind, RuleKind::Custom);
        for s in [128u32, 130, 133] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "list slot {s}");
        }
    }

    #[test]
    fn list_slot_142_state_call_is_ignored_and_143_stays_unknown() {
        let l = Kind::List;
        assert_eq!(l.rule(142).unwrap().kind, RuleKind::Custom);
        assert_eq!(l.rule(143).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(141).unwrap().kind, RuleKind::Unknown);
    }

    #[test]
    fn every_slot_has_exactly_one_rule_and_runs_are_in_order() {
        for kind in Kind::ALL {
            let mut next = 0u16;
            for rule in kind.rules() {
                assert_eq!(rule.first, next, "{} has a gap or an overlap at slot {next}", kind.name());
                assert!(rule.last >= rule.first);
                next = rule.last + 1;
            }
            assert_eq!(next, kind.slots(), "{} table must end at its slot count", kind.name());
        }
    }

    #[test]
    fn the_measured_console_slots_are_where_the_c_table_put_them() {
        let device = Kind::Device;
        assert_eq!(device.rule(8).unwrap().kind, RuleKind::Custom);
        assert_eq!(device.rule(16).unwrap().kind, RuleKind::Custom, "CreateRootSignature: root-signature-at-device port");
        assert_eq!(device.rule(64).unwrap().kind, RuleKind::Unknown);
        assert_eq!(device.rule(76).unwrap().kind, RuleKind::Custom, "committed resource, sanitised like 27");
        for s in [12u32, 13, 14, 33, 40] {
            assert_eq!(Kind::List.rule(s).unwrap().kind, RuleKind::Custom, "list slot {s} feeds the user data mirror");
        }
        assert_eq!(Kind::List.rule(139).unwrap().kind, RuleKind::Custom);
        assert_eq!(Kind::Queue.rule(30).unwrap().kind, RuleKind::Custom);
        let q13 = Kind::Queue.rule(13).unwrap();
        assert_eq!((q13.kind, q13.first, q13.pc), (RuleKind::Fwd, 11, 11), "11 to 13 stay relayed by position");
        assert_eq!(Kind::Queue.rule(14).unwrap().kind, RuleKind::Custom, "14 and 15 depend on the layout");
        assert_eq!(Kind::Queue.rule(16).unwrap().kind, RuleKind::Custom, "16 and 17 depend on the layout");
        assert_eq!(Kind::Queue.rule(17).unwrap().kind, RuleKind::Custom, "16 and 17 depend on the layout");
        assert_eq!(Kind::Queue.rule(18).unwrap().kind, RuleKind::Custom, "GetTimestampFrequency in the newer layout, GetDesc in the older");
        assert_eq!(Kind::Queue.rule(19).unwrap().kind, RuleKind::Custom, "GetClockCalibration in the newer layout");
        assert_eq!(Kind::Queue.rule(20).unwrap().kind, RuleKind::Custom, "20 and up go through the older numbering");
        assert_eq!(Kind::Heap.rule(10).unwrap().kind, RuleKind::Custom);
        assert!(Kind::Heap.rule(16).is_none());
    }

    #[test]
    fn relayed_rows_stay_inside_the_pc_slot_space() {
        for kind in Kind::ALL {
            for rule in kind.rules().iter().filter(|r| r.kind == RuleKind::Fwd) {
                assert!(rule.pc + (rule.last - rule.first) < 160, "{}: {}", kind.name(), rule.what);
            }
        }
    }

    #[test]
    fn a_console_resource_desc_loses_its_alignment_tile_flags_and_console_layout() {
        let read = |d: &[u8; 0x38], at: usize| u32::from_le_bytes(d[at..at + 4].try_into().unwrap());
        // A buffer: alignment cleared, layout ROW_MAJOR, only the low flag byte survives.
        let mut buffer = [0u8; 0x38];
        buffer[0] = 1;
        buffer[8..16].copy_from_slice(&0x1_0000u64.to_le_bytes());
        buffer[0x2C] = 0x22;
        buffer[0x30..0x34].copy_from_slice(&0xABCD_0008u32.to_le_bytes());
        pc_resource_desc(&mut buffer);
        assert_eq!(buffer[8..16], [0u8; 8]);
        assert_eq!(read(&buffer, 0x2C), 1);
        assert_eq!(read(&buffer, 0x30), 0x08);

        // A texture: layout UNKNOWN, a console depth format mapped, depth drops the incompatible flags.
        let mut texture = [0u8; 0x38];
        texture[0] = 3;
        texture[0x20..0x24].copy_from_slice(&118u32.to_le_bytes());
        texture[0x2C] = 1;
        texture[0x30..0x34].copy_from_slice(&0x0010_0027u32.to_le_bytes());
        pc_resource_desc(&mut texture);
        assert_eq!(read(&texture, 0x2C), 0);
        assert_eq!(read(&texture, 0x20), 45);
        assert_eq!(read(&texture, 0x30), 0x02);
    }

    #[test]
    fn only_the_four_console_depth_formats_are_remapped() {
        assert_eq!((117..=120).map(pc_dxgi_format).collect::<Vec<_>>(), [44, 45, 46, 47]);
        for f in [0, 28, 116, 121, 200] {
            assert_eq!(pc_dxgi_format(f), f);
        }
    }

    #[test]
    fn packed_16_bit_targets_lose_only_the_uav_flag() {
        for (fmt, want) in [(86u32, 0x1u32), (85, 0x1), (115, 0x0), (87, 0x5)] {
            let mut t = [0u8; 0x38];
            t[0] = 3; // D3D12_RESOURCE_DIMENSION_TEXTURE2D
            t[0x20..0x24].copy_from_slice(&fmt.to_le_bytes());
            t[0x30..0x34].copy_from_slice(&0x5u32.to_le_bytes()); // RENDER_TARGET | UNORDERED_ACCESS
            pc_resource_desc(&mut t);
            assert_eq!(u32::from_le_bytes(t[0x30..0x34].try_into().unwrap()), want, "format {fmt}");
        }
    }

    #[test]
    fn newer_layout_slots_are_108_127_and_133_only() {
        for s in [108u32, 127, 133] {
            assert!(super::is_newer_layout_slot(s), "slot {s}");
        }
        for s in [0u32, 25, 41, 61, 107, 134, 159] {
            assert!(!super::is_newer_layout_slot(s), "slot {s}");
        }
    }

    #[test]
    fn probe_range_is_inside_the_list_and_covers_the_newer_write_wait_and_barrier_slots() {
        let r = super::PROBE_LIST_SLOTS;
        assert_eq!(r.clone().count(), 90);
        assert!(*r.end() < u32::from(super::LIST_SLOTS));
        for s in [71u32, 72, 73, 74, 146] {
            assert!(r.contains(&s), "slot {s}");
        }
    }

    #[test]
    fn the_device11_iid_bytes_are_those_of_the_public_guid() {
        // {5405c344-d457-444e-b4dd-2366e45aee39}: data1, data2, data3 little endian, then data4.
        let mut want = Vec::new();
        want.extend_from_slice(&0x5405_c344u32.to_le_bytes());
        want.extend_from_slice(&0xd457u16.to_le_bytes());
        want.extend_from_slice(&0x444eu16.to_le_bytes());
        want.extend_from_slice(&[0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39]);
        assert_eq!(super::IID_DEVICE11.to_vec(), want);
        assert!(!super::note_device_iid(&[0; 16]), "another interface proves nothing");
    }

    #[test]
    fn latch_once_fires_exactly_once() {
        static F: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        assert!(super::latch_once(&F));
        assert!(!super::latch_once(&F));
        assert!(!super::latch_once(&F));
    }

    #[test]
    fn newer_layout_slots_are_custom_and_their_neighbours_keep_their_rules() {
        // queue 20 and up: one handler per slot that maps the newer numbering back by two
        let q = Kind::Queue;
        for s in [20u32, 23, 25, 30, 32, 35, 47] {
            assert_eq!(q.rule(s).unwrap().kind, RuleKind::Custom, "queue slot {s}");
        }
        assert!(q.rule(47).unwrap().what.contains("ExecuteCommandListsX"));

        // list 70 (OMSetDepthBounds) next to 68 (relayed to PC 62) and the unknown 69 and 71..77
        let l = Kind::List;
        let r = l.rule(68).unwrap();
        assert_eq!((r.kind, r.pc), (RuleKind::Fwd, 62));
        assert_eq!(l.rule(70).unwrap().kind, RuleKind::Custom);
        assert!(l.rule(70).unwrap().what.contains("newer layout"));
        assert_eq!(l.rule(71).unwrap().kind, RuleKind::Custom);
        assert_eq!(l.rule(72).unwrap().kind, RuleKind::Custom);
        assert!(l.rule(71).unwrap().what.contains("newer layout"));
        assert_eq!(l.rule(73).unwrap().kind, RuleKind::Custom);
        assert_eq!(l.rule(74).unwrap().kind, RuleKind::Custom);
        for s in [69u32, 76, 77] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "list slot {s}");
        }
        assert_eq!(l.rule(75).unwrap().kind, RuleKind::Custom, "pixel shader depth order");

        // list 85 (CopyTextureRegionX) and 86 (CopyResourceX) next to 83 (custom), 84 and 87
        assert_eq!(l.rule(83).unwrap().kind, RuleKind::Custom);
        assert_eq!(l.rule(84).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(85).unwrap().kind, RuleKind::Custom);
        assert!(l.rule(85).unwrap().what.contains("newer layout"));
        assert_eq!(l.rule(86).unwrap().kind, RuleKind::Custom);
        assert!(l.rule(86).unwrap().what.contains("newer layout"));
        for s in [88u32, 103] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "list slot {s}");
        }
        assert_eq!(l.rule(139).unwrap().kind, RuleKind::Custom, "Reset (newer) unchanged");
        assert_eq!(l.rule(146).unwrap().kind, RuleKind::Custom, "UAV barrier (newer)");
        assert_eq!(l.rule(145).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(147).unwrap().kind, RuleKind::Unknown);
    }

    #[test]
    fn upload_memory_and_the_address_taking_methods_are_custom_with_their_neighbours_relayed() {
        let l = Kind::List;
        for s in 33..=48u32 {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Custom, "slot {s}");
        }
        let r = l.rule(49).unwrap();
        assert_eq!((r.kind, r.pc), (RuleKind::Fwd, 45), "SOSetTargets");
        assert_eq!(l.rule(148).unwrap().kind, RuleKind::Custom, "AllocateUploadMemory");
        assert!(l.rule(148).unwrap().what.contains("AllocateUploadMemory"));
        assert_eq!(l.rule(147).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(149).unwrap().kind, RuleKind::Unknown);
    }

    #[test]
    fn draw_dispatch_and_indirect_slots_of_the_newer_layout_are_custom_and_say_so() {
        let l = Kind::List;
        for (s, word) in [(59u32, "newer layout"), (61, "newer layout"), (80, "newer layout"), (83, "newer layout")] {
            let r = l.rule(s).unwrap();
            assert_eq!(r.kind, RuleKind::Custom, "slot {s}");
            assert!(r.what.contains(word), "slot {s}");
        }
        assert!(l.rule(83).unwrap().what.contains("CopyTextureRegion"), "the older meaning is kept");
        assert!(l.rule(80).unwrap().what.contains("DispatchX"));
        assert_eq!(l.rule(60).unwrap().kind, RuleKind::Unknown);
        for s in 62..=65u32 {
            let r = l.rule(s).unwrap();
            assert_eq!(r.kind, RuleKind::Custom, "slot {s}");
            assert!(r.what.contains("newer layout"), "slot {s}");
        }
        assert!(l.rule(62).unwrap().what.contains("CopyMemoryToMemoryX"));
        assert!(l.rule(65).unwrap().what.contains("128Bit"));
        assert_eq!(l.rule(66).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(67).unwrap().kind, RuleKind::Custom, "texture cache flush unchanged");
        assert_eq!(l.rule(78).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(79).unwrap().kind, RuleKind::Unknown);
        assert_eq!(l.rule(81).unwrap().kind, RuleKind::Unknown);
    }

    #[test]
    fn predication_writes_kickoff_and_depth_bias_slots_are_custom_with_unknown_gaps() {
        let l = Kind::List;
        for s in [87u32, 90, 91, 92, 95, 96] {
            let r = l.rule(s).unwrap();
            assert_eq!(r.kind, RuleKind::Custom, "slot {s}");
            assert!(r.what.contains("newer layout"), "slot {s}");
        }
        for s in [84u32, 88, 89, 93, 94, 97, 103] {
            assert_eq!(l.rule(s).unwrap().kind, RuleKind::Unknown, "slot {s}");
        }
        assert!(l.rule(96).unwrap().what.contains("RSSetDepthBias"));
        assert!(l.rule(87).unwrap().what.contains("SetPredicationBufferX"));
    }

    #[test]
    fn the_dma_list_has_close_reset_and_the_four_copies_and_answers_ok_to_the_rest() {
        let k = Kind::DmaList;
        assert_eq!(k.slots(), 32);
        assert_eq!(k.name(), "DMA command list");
        assert_eq!(k.rule(9).unwrap().kind, RuleKind::Custom, "Close, a refusal remembered");
        for s in 10..=14u32 {
            assert_eq!(k.rule(s).unwrap().kind, RuleKind::Custom, "slot {s}");
        }
        assert!(k.rule(11).unwrap().what.contains("PC 15"));
        assert!(k.rule(14).unwrap().what.contains("PC 18"));
        for s in [15u32, 20, 31] {
            let r = k.rule(s).unwrap();
            assert_eq!(r.kind, RuleKind::Custom, "slot {s}");
            assert!(r.what.contains("S_OK"));
        }
        assert!(k.rule(32).is_none());
        // the graphics list's table is untouched by it
        assert_eq!(Kind::List.slots(), 160);
    }

    #[test]
    fn alloc_info_accumulates_offsets_and_poisons_on_unanswerable() {
        // two 0x1000-aligned, 0x1000-byte allocations pack back to back
        let (o0, t0, a0) = super::accumulate_alloc_info(0, 0, 0x1000, 0x1000).unwrap();
        assert_eq!((o0, t0, a0), (0, 0x1000, 0x1000));
        let (o1, t1, a1) = super::accumulate_alloc_info(t0, a0, 0x1000, 0x1000).unwrap();
        assert_eq!((o1, t1, a1), (0x1000, 0x2000, 0x1000));
        // an unaligned running total still rounds the next offset up to the alignment
        let (o2, t2, _) = super::accumulate_alloc_info(0x10, 0x1000, 0x40, 0x40).unwrap();
        assert_eq!((o2, t2), (0x40, 0x80));
        // a descriptor the PC driver cannot answer poisons the allocation
        assert!(super::accumulate_alloc_info(0, 0, u64::MAX, 0x1000).is_none());
        assert!(super::accumulate_alloc_info(0, 0, 0x1000, 0).is_none());
    }
}
