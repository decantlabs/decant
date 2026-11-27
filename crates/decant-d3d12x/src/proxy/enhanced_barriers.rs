//! Console list slot 146 (`ID3D12GraphicsCommandList7::Barrier`, newer console layout): parses
//! Enhanced Barrier groups (`D3D12_BARRIER_GROUP` with `GLOBAL`, `TEXTURE`, and `BUFFER` barrier
//! arrays) and lowers them to legacy PC `ID3D12GraphicsCommandList::ResourceBarrier` (PC slot 26),
//! updating `detile` and `present` resource state tracking alongside a local per-resource/per-list
//! state tracker for `D3D12_BARRIER_LAYOUT_UNDEFINED` transitions.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::offset_of;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, D3D12_BARRIER_GROUP,
    D3D12_BARRIER_LAYOUT_UNDEFINED, D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON,
    D3D12_BARRIER_SUBRESOURCE_RANGE, D3D12_BARRIER_TYPE_BUFFER, D3D12_BARRIER_TYPE_GLOBAL,
    D3D12_BARRIER_TYPE_TEXTURE, D3D12_BUFFER_BARRIER, D3D12_GLOBAL_BARRIER, D3D12_TEXTURE_BARRIER,
    D3D12_TEXTURE_BARRIER_FLAG_DISCARD,
};

use super::*;

type GetListType = unsafe extern "system" fn(Object) -> u32;

pub(super) const BARRIER_GROUP_BYTES: usize = 16;
pub(super) const GLOBAL_BARRIER_BYTES: usize = 16;
pub(super) const TEXTURE_BARRIER_BYTES: usize = 64;
pub(super) const BUFFER_BARRIER_BYTES: usize = 40;

const BARRIER_TYPE_GLOBAL: u32 = 0;
const BARRIER_TYPE_TEXTURE: u32 = 1;
const BARRIER_TYPE_BUFFER: u32 = 2;

pub(super) const BARRIER_LAYOUT_UNDEFINED: u32 = 0xFFFF_FFFF;
pub(super) const PC_MAX_BARRIER_LAYOUT: u32 = 30;
/// Console-only layout value paired with `Sync = NONE` and `Access = NO_ACCESS` (maps to PC
/// `COMMON` / `PRESENT`).
#[cfg(test)]
const CONSOLE_BARRIER_LAYOUT_PRESENT: u32 = 31;

const BARRIER_ACCESS_COMMON: u32 = 0;
const BARRIER_ACCESS_UNORDERED_ACCESS: u32 = 0x10;
const BARRIER_ACCESS_NO_ACCESS: u32 = 0x8000_0000;
const BARRIER_ACCESS_WRITE_MASK: u32 = 0x8 // RENDER_TARGET
    | 0x10 // UNORDERED_ACCESS
    | 0x20 // DEPTH_STENCIL_WRITE
    | 0x100 // STREAM_OUTPUT
    | 0x400 // COPY_DEST
    | 0x1000 // RESOLVE_DEST
    | 0x8000 // RAYTRACING_ACCELERATION_STRUCTURE_WRITE
    | 0x4_0000 // VIDEO_DECODE_WRITE
    | 0x10_0000 // VIDEO_PROCESS_WRITE
    | 0x40_0000; // VIDEO_ENCODE_WRITE

const TEXTURE_BARRIER_FLAG_DISCARD: u32 = 1;

const LEGACY_BARRIER_TYPE_TRANSITION: u32 = 0;
const LEGACY_BARRIER_TYPE_ALIASING: u32 = 1;
const LEGACY_BARRIER_TYPE_UAV: u32 = 2;
const ALL_SUBRESOURCES: u32 = 0xFFFF_FFFF;

const D3D12_COMMAND_LIST_TYPE_BUNDLE: u32 = 1;
const D3D12_COMMAND_LIST_TYPE_COMPUTE: u32 = 2;
const D3D12_COMMAND_LIST_TYPE_COPY: u32 = 3;

/// Valid legacy `D3D12_RESOURCE_STATES` bits on a `COMPUTE` command list.
const COMPUTE_VALID_STATES: u32 = 0x1 // VERTEX_AND_CONSTANT_BUFFER
    | 0x8 // UNORDERED_ACCESS
    | 0x40 // NON_PIXEL_SHADER_RESOURCE
    | 0x200 // INDIRECT_ARGUMENT / PREDICATION
    | 0x400 // COPY_DEST
    | 0x800 // COPY_SOURCE
    | 0x40_0000; // RAYTRACING_ACCELERATION_STRUCTURE

/// Valid legacy `D3D12_RESOURCE_STATES` bits on a `COPY` command list.
const COPY_VALID_STATES: u32 = 0x400 | 0x800; // COPY_DEST | COPY_SOURCE

// Compile-time ABI assertions verifying that our console Enhanced Barrier offsets and sizes match
// public PC `d3d12.h` structures.
const _: () = {
    assert!(size_of::<D3D12_BARRIER_GROUP>() == BARRIER_GROUP_BYTES);
    assert!(offset_of!(D3D12_BARRIER_GROUP, Type) == 0);
    assert!(offset_of!(D3D12_BARRIER_GROUP, NumBarriers) == 4);
    assert!(offset_of!(D3D12_BARRIER_GROUP, Anonymous) == 8);

    assert!(size_of::<D3D12_GLOBAL_BARRIER>() == GLOBAL_BARRIER_BYTES);
    assert!(offset_of!(D3D12_GLOBAL_BARRIER, SyncBefore) == 0);
    assert!(offset_of!(D3D12_GLOBAL_BARRIER, SyncAfter) == 4);
    assert!(offset_of!(D3D12_GLOBAL_BARRIER, AccessBefore) == 8);
    assert!(offset_of!(D3D12_GLOBAL_BARRIER, AccessAfter) == 12);

    assert!(size_of::<D3D12_TEXTURE_BARRIER>() == TEXTURE_BARRIER_BYTES);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, SyncBefore) == 0);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, SyncAfter) == 4);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, AccessBefore) == 8);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, AccessAfter) == 12);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, LayoutBefore) == 16);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, LayoutAfter) == 20);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, pResource) == 24);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, Subresources) == 32);
    assert!(offset_of!(D3D12_TEXTURE_BARRIER, Flags) == 56);

    assert!(size_of::<D3D12_BARRIER_SUBRESOURCE_RANGE>() == 24);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, IndexOrFirstMipLevel) == 0);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumMipLevels) == 4);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, FirstArraySlice) == 8);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumArraySlices) == 12);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, FirstPlane) == 16);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumPlanes) == 20);

    assert!(size_of::<D3D12_BUFFER_BARRIER>() == BUFFER_BARRIER_BYTES);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, SyncBefore) == 0);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, SyncAfter) == 4);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, AccessBefore) == 8);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, AccessAfter) == 12);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, pResource) == 16);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, Offset) == 24);
    assert!(offset_of!(D3D12_BUFFER_BARRIER, Size) == 32);

    assert!(D3D12_BARRIER_TYPE_GLOBAL.0 as u32 == BARRIER_TYPE_GLOBAL);
    assert!(D3D12_BARRIER_TYPE_TEXTURE.0 as u32 == BARRIER_TYPE_TEXTURE);
    assert!(D3D12_BARRIER_TYPE_BUFFER.0 as u32 == BARRIER_TYPE_BUFFER);
    assert!(D3D12_BARRIER_LAYOUT_UNDEFINED.0 as u32 == BARRIER_LAYOUT_UNDEFINED);
    assert!(D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON.0 as u32 == PC_MAX_BARRIER_LAYOUT);
    assert!(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS.0 as u32 == BARRIER_ACCESS_UNORDERED_ACCESS);
    assert!(D3D12_BARRIER_ACCESS_NO_ACCESS.0 as u32 == BARRIER_ACCESS_NO_ACCESS);
    assert!(D3D12_TEXTURE_BARRIER_FLAG_DISCARD.0 as u32 == TEXTURE_BARRIER_FLAG_DISCARD);
};

#[derive(Default)]
struct FastUsizeHasher(u64);

impl Hasher for FastUsizeHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
        }
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<V> = HashMap<usize, V, BuildHasherDefault<FastUsizeHasher>>;

/// Sanitizes a console `D3D12_BARRIER_LAYOUT` value for PC D3D12.
///
/// Preserves `D3D12_BARRIER_LAYOUT_UNDEFINED` (`0xFFFFFFFF`) and standard PC layouts (`0..=30`).
/// Console-only layouts above `30` (including observed console layout `31` / `0x1f`) map to `0`
/// (`D3D12_BARRIER_LAYOUT_COMMON` / `PRESENT`).
pub(super) fn sanitize_layout(layout: u32) -> u32 {
    if layout == BARRIER_LAYOUT_UNDEFINED || layout <= PC_MAX_BARRIER_LAYOUT {
        layout
    } else {
        0
    }
}

/// Converts a console `D3D12_BARRIER_LAYOUT` into a legacy `D3D12_RESOURCE_STATES` bitmask,
/// returning `None` for `D3D12_BARRIER_LAYOUT_UNDEFINED` (`0xFFFFFFFF`).
pub(super) fn layout_to_legacy_state(layout: u32) -> Option<u32> {
    let sanitized = sanitize_layout(layout);
    if sanitized == BARRIER_LAYOUT_UNDEFINED {
        None
    } else {
        Some(super::barrier_layout_to_state(sanitized))
    }
}

/// Masks a legacy `D3D12_RESOURCE_STATES` value so it is legal on `list_type`.
pub(super) fn clamp_state_for_list_type(state: u32, list_type: u32) -> u32 {
    match list_type {
        D3D12_COMMAND_LIST_TYPE_COMPUTE => state & COMPUTE_VALID_STATES,
        D3D12_COMMAND_LIST_TYPE_COPY => state & COPY_VALID_STATES,
        _ => state,
    }
}

/// Resource metadata recorded at PC resource creation so Enhanced Barrier lowering can expand
/// subresource ranges and choose valid legacy aliasing/UAV barrier forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ResourceMeta {
    pub mip_levels: u32,
    pub array_size: u32,
    pub plane_count: u32,
    /// `true` for placed or reserved PC resources (eligible for `ALIASING(NULL, res)`).
    /// `false` for committed PC resources, where D3D12 forbids non-null aliasing barriers.
    pub aliasable: bool,
    /// `true` when the PC resource carries `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`.
    pub allow_uav: bool,
    /// `true` when metadata came from a known creation descriptor.
    pub known: bool,
}

impl Default for ResourceMeta {
    fn default() -> Self {
        Self {
            mip_levels: 1,
            array_size: 1,
            plane_count: 1,
            aliasable: false,
            allow_uav: false,
            known: false,
        }
    }
}

impl ResourceMeta {
    fn subresource_count(self) -> usize {
        self.mip_levels
            .max(1)
            .saturating_mul(self.array_size.max(1))
            .saturating_mul(self.plane_count.max(1)) as usize
    }
}

fn plane_count_for_format(format: u32) -> u32 {
    match format {
        // Depth + stencil two-plane formats (R32G8X24 / D32_S8X24 / R24G8 / D24_S8 families).
        19..=22 | 44..=47 => 2,
        _ => 1,
    }
}

/// Extracts [`ResourceMeta`] from a sanitized 56-byte `D3D12_RESOURCE_DESC`.
pub(super) fn resource_meta_from_desc(desc: &[u8; 0x38], aliasable: bool) -> ResourceMeta {
    let dim = le_u32(desc, 0);
    let width = u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap_or([0; 8])).max(1);
    let height = le_u32(desc, 0x18).max(1);
    let depth_or_array = u16::from_le_bytes([desc[0x1C], desc[0x1D]]).max(1);
    let raw_mips = u16::from_le_bytes([desc[0x1E], desc[0x1F]]);
    let format = le_u32(desc, 0x20);
    let flags = le_u32(desc, 0x30);
    let allow_uav = (flags & 0x4) != 0;

    if dim == 1 {
        return ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable, allow_uav, known: true };
    }

    let mip_levels = if raw_mips != 0 {
        u32::from(raw_mips)
    } else {
        let max_dim = match dim {
            2 => width,
            3 => width.max(u64::from(height)),
            4 => width.max(u64::from(height)).max(u64::from(depth_or_array)),
            _ => width,
        };
        (64 - max_dim.leading_zeros()).max(1)
    };
    let array_size = if dim == 4 { 1 } else { u32::from(depth_or_array) };
    let plane_count = plane_count_for_format(format);
    ResourceMeta { mip_levels, array_size, plane_count, aliasable, allow_uav, known: true }
}

#[derive(Clone, Debug)]
struct TrackedResource {
    meta: ResourceMeta,
    states: Vec<u32>,
}

fn apply_subresource_state(states: &mut Vec<u32>, count: usize, sub: u32, after: u32) {
    if sub == ALL_SUBRESOURCES || (count <= 1 && sub == 0) {
        states.clear();
        states.push(after);
        return;
    }
    let idx = sub as usize;
    if idx >= count {
        return;
    }
    if states.len() != count {
        let initial = states.first().copied().unwrap_or(0);
        states.resize(count, initial);
    }
    states[idx] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

fn state_for_subresource(states: &[u32], sub: u32) -> u32 {
    if states.len() <= 1 || sub == ALL_SUBRESOURCES {
        states.first().copied().unwrap_or(0)
    } else {
        states.get(sub as usize).copied().unwrap_or_else(|| states.first().copied().unwrap_or(0))
    }
}

/// Per-resource and per-command-list legacy state tracker supporting `UNDEFINED` layout resolution.
#[derive(Default)]
pub(super) struct StateTracker {
    resources: FastMap<TrackedResource>,
    list_notes: FastMap<Vec<(usize, u32, u32)>>,
}

impl StateTracker {
    pub(super) fn note_resource_created(&mut self, res: usize, state: u32, meta: ResourceMeta) {
        if res != 0 {
            self.resources.insert(res, TrackedResource { meta, states: vec![state] });
        }
    }

    pub(super) fn note_list_transitions(&mut self, list: usize, transitions: &[(usize, u32, u32)]) {
        if list == 0 || transitions.is_empty() {
            return;
        }
        self.list_notes.entry(list).or_default().extend(transitions.iter().copied().filter(|&(res, _, _)| res != 0));
    }

    pub(super) fn list_reset(&mut self, list: usize) {
        if let Some(entries) = self.list_notes.get_mut(&list) {
            entries.clear();
        }
    }

    pub(super) fn list_submitted(&mut self, lists: &[usize]) {
        let Self { resources, list_notes } = self;
        for &list in lists {
            let Some(notes) = list_notes.get(&list) else {
                continue;
            };
            for &(res, sub, after) in notes {
                if res == 0 {
                    continue;
                }
                let entry = resources
                    .entry(res)
                    .or_insert_with(|| TrackedResource { meta: ResourceMeta::default(), states: vec![0] });
                let count = entry.meta.subresource_count();
                apply_subresource_state(&mut entry.states, count, sub, after);
            }
        }
    }

    pub(super) fn forget_resource(&mut self, res: usize) {
        self.resources.remove(&res);
    }

    fn effective_state(&self, list: usize, res: usize) -> (ResourceMeta, Vec<u32>) {
        let (meta, mut states) = match self.resources.get(&res) {
            Some(entry) => (entry.meta, entry.states.clone()),
            None => (ResourceMeta::default(), vec![0]),
        };
        if let Some(notes) = self.list_notes.get(&list) {
            let count = meta.subresource_count();
            for &(noted_res, sub, after) in notes {
                if noted_res == res {
                    apply_subresource_state(&mut states, count, sub, after);
                }
            }
        }
        (meta, states)
    }
}

static STATES: Mutex<Option<StateTracker>> = Mutex::new(None);

fn with_states<R>(f: impl FnOnce(&mut StateTracker) -> R) -> R {
    let mut guard = STATES.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(StateTracker::default))
}

/// Records the initial legacy state and metadata of a newly created PC resource.
pub(super) fn note_resource_created(res: usize, state: u32, desc: Option<&[u8; 0x38]>, aliasable: bool) {
    if res == 0 {
        return;
    }
    let meta = match desc {
        Some(d) => resource_meta_from_desc(d, aliasable),
        None => ResourceMeta { aliasable, ..ResourceMeta::default() },
    };
    with_states(|s| s.note_resource_created(res, state, meta));
}

/// Records legacy `ResourceBarrier` transitions on PC command list `list`.
pub(super) fn note_list_transitions(list: usize, transitions: &[(usize, u32, u32)]) {
    if list == 0 || transitions.is_empty() {
        return;
    }
    with_states(|s| s.note_list_transitions(list, transitions));
}

/// Clears pending recorded transitions when PC command list `list` is reset or detached.
pub(super) fn list_reset(list: usize) {
    if list != 0 {
        with_states(|s| s.list_reset(list));
    }
}

/// Commits recorded transitions of submitted PC command lists in execution order.
pub(super) fn list_submitted(lists: &[usize]) {
    if !lists.is_empty() {
        with_states(|s| s.list_submitted(lists));
    }
}

/// Drops tracked state for a released PC resource.
pub(super) fn forget_resource(res: usize) {
    if res != 0 {
        with_states(|s| s.forget_resource(res));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BarrierSubresourceRange {
    pub index_or_first_mip: u32,
    pub num_mips: u32,
    pub first_slice: u32,
    pub num_slices: u32,
    pub first_plane: u32,
    pub num_planes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GlobalBarrier {
    pub sync_before: u32,
    pub sync_after: u32,
    pub access_before: u32,
    pub access_after: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TextureBarrier {
    pub sync_before: u32,
    pub sync_after: u32,
    pub access_before: u32,
    pub access_after: u32,
    pub layout_before: u32,
    pub layout_after: u32,
    pub resource: u64,
    pub subresources: BarrierSubresourceRange,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BufferBarrier {
    pub sync_before: u32,
    pub sync_after: u32,
    pub access_before: u32,
    pub access_after: u32,
    pub resource: u64,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum BarrierGroup {
    Global(Vec<GlobalBarrier>),
    Texture(Vec<TextureBarrier>),
    Buffer(Vec<BufferBarrier>),
    Unknown(u32),
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or([0; 4]))
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or([0; 8]))
}

fn parse_global(raw: &[u8; GLOBAL_BARRIER_BYTES]) -> GlobalBarrier {
    GlobalBarrier {
        sync_before: read_u32(raw, 0),
        sync_after: read_u32(raw, 4),
        access_before: read_u32(raw, 8),
        access_after: read_u32(raw, 12),
    }
}

fn parse_texture(raw: &[u8; TEXTURE_BARRIER_BYTES]) -> TextureBarrier {
    TextureBarrier {
        sync_before: read_u32(raw, 0),
        sync_after: read_u32(raw, 4),
        access_before: read_u32(raw, 8),
        access_after: read_u32(raw, 12),
        layout_before: read_u32(raw, 16),
        layout_after: read_u32(raw, 20),
        resource: read_u64(raw, 24),
        subresources: BarrierSubresourceRange {
            index_or_first_mip: read_u32(raw, 32),
            num_mips: read_u32(raw, 36),
            first_slice: read_u32(raw, 40),
            num_slices: read_u32(raw, 44),
            first_plane: read_u32(raw, 48),
            num_planes: read_u32(raw, 52),
        },
        flags: read_u32(raw, 56),
    }
}

fn parse_buffer(raw: &[u8; BUFFER_BARRIER_BYTES]) -> BufferBarrier {
    BufferBarrier {
        sync_before: read_u32(raw, 0),
        sync_after: read_u32(raw, 4),
        access_before: read_u32(raw, 8),
        access_after: read_u32(raw, 12),
        resource: read_u64(raw, 16),
        offset: read_u64(raw, 24),
        size: read_u64(raw, 32),
    }
}

/// Parses `n` `D3D12_BARRIER_GROUP` structures and their referenced barrier arrays.
///
/// # Safety
/// `groups` is null or points to `n` readable 16-byte `D3D12_BARRIER_GROUP` entries, and each
/// entry's non-null array pointer holds `NumBarriers` readable elements of its type.
unsafe fn parse_groups(n: u32, groups: *const c_void) -> Vec<BarrierGroup> {
    if groups.is_null() || n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        // SAFETY: by the contract above.
        let raw: [u8; BARRIER_GROUP_BYTES] = unsafe { groups.cast::<[u8; BARRIER_GROUP_BYTES]>().add(i).read_unaligned() };
        let ty = read_u32(&raw, 0);
        let count = read_u32(&raw, 4) as usize;
        let ptr = read_u64(&raw, 8) as usize as *const u8;
        match ty {
            BARRIER_TYPE_GLOBAL => {
                let mut items = Vec::with_capacity(if ptr.is_null() { 0 } else { count });
                if !ptr.is_null() {
                    for j in 0..count {
                        // SAFETY: by the contract above.
                        let elem: [u8; GLOBAL_BARRIER_BYTES] = unsafe { ptr.add(j * GLOBAL_BARRIER_BYTES).cast::<[u8; GLOBAL_BARRIER_BYTES]>().read_unaligned() };
                        items.push(parse_global(&elem));
                    }
                }
                out.push(BarrierGroup::Global(items));
            }
            BARRIER_TYPE_TEXTURE => {
                let mut items = Vec::with_capacity(if ptr.is_null() { 0 } else { count });
                if !ptr.is_null() {
                    for j in 0..count {
                        // SAFETY: by the contract above.
                        let elem: [u8; TEXTURE_BARRIER_BYTES] = unsafe { ptr.add(j * TEXTURE_BARRIER_BYTES).cast::<[u8; TEXTURE_BARRIER_BYTES]>().read_unaligned() };
                        items.push(parse_texture(&elem));
                    }
                }
                out.push(BarrierGroup::Texture(items));
            }
            BARRIER_TYPE_BUFFER => {
                let mut items = Vec::with_capacity(if ptr.is_null() { 0 } else { count });
                if !ptr.is_null() {
                    for j in 0..count {
                        // SAFETY: by the contract above.
                        let elem: [u8; BUFFER_BARRIER_BYTES] = unsafe { ptr.add(j * BUFFER_BARRIER_BYTES).cast::<[u8; BUFFER_BARRIER_BYTES]>().read_unaligned() };
                        items.push(parse_buffer(&elem));
                    }
                }
                out.push(BarrierGroup::Buffer(items));
            }
            other => out.push(BarrierGroup::Unknown(other)),
        }
    }
    out
}

fn legacy_transition_barrier(resource: u64, subresource: u32, before: u32, after: u32) -> [u8; BARRIER_BYTES] {
    let mut b = [0u8; BARRIER_BYTES];
    b[0..4].copy_from_slice(&LEGACY_BARRIER_TYPE_TRANSITION.to_le_bytes());
    b[8..16].copy_from_slice(&resource.to_le_bytes());
    b[16..20].copy_from_slice(&subresource.to_le_bytes());
    b[20..24].copy_from_slice(&before.to_le_bytes());
    b[24..28].copy_from_slice(&after.to_le_bytes());
    b
}

fn legacy_aliasing_barrier(resource_after: u64) -> [u8; BARRIER_BYTES] {
    let mut b = [0u8; BARRIER_BYTES];
    b[0..4].copy_from_slice(&LEGACY_BARRIER_TYPE_ALIASING.to_le_bytes());
    b[16..24].copy_from_slice(&resource_after.to_le_bytes());
    b
}

fn legacy_uav_barrier(resource: u64) -> [u8; BARRIER_BYTES] {
    let mut b = [0u8; BARRIER_BYTES];
    b[0..4].copy_from_slice(&LEGACY_BARRIER_TYPE_UAV.to_le_bytes());
    b[8..16].copy_from_slice(&resource.to_le_bytes());
    b
}

#[derive(Debug, PartialEq, Eq)]
enum SubresourceTargets {
    All,
    Indices(Vec<u32>),
}

/// Expands a `D3D12_BARRIER_SUBRESOURCE_RANGE` into either `ALL_SUBRESOURCES` or flat legacy
/// subresource indices using `meta`.
fn expand_subresources(range: BarrierSubresourceRange, meta: ResourceMeta) -> SubresourceTargets {
    if range.num_mips == 0 {
        return if range.index_or_first_mip == ALL_SUBRESOURCES {
            SubresourceTargets::All
        } else {
            SubresourceTargets::Indices(vec![range.index_or_first_mip])
        };
    }

    let num_mips = range.num_mips;
    let num_slices = range.num_slices.max(1);
    let num_planes = range.num_planes.max(1);

    if meta.known {
        if range.index_or_first_mip == 0
            && num_mips >= meta.mip_levels
            && range.first_slice == 0
            && num_slices >= meta.array_size
            && range.first_plane == 0
            && num_planes >= meta.plane_count
        {
            return SubresourceTargets::All;
        }
        let mips = meta.mip_levels.max(1);
        let slices = meta.array_size.max(1);
        let planes = meta.plane_count.max(1);
        let end_mip = range.index_or_first_mip.saturating_add(num_mips).min(mips);
        let end_slice = range.first_slice.saturating_add(num_slices).min(slices);
        let end_plane = range.first_plane.saturating_add(num_planes).min(planes);
        let mut indices = Vec::new();
        for plane in range.first_plane..end_plane {
            for slice in range.first_slice..end_slice {
                for mip in range.index_or_first_mip..end_mip {
                    indices.push(mip + slice * mips + plane * mips * slices);
                }
            }
        }
        return if indices.is_empty() { SubresourceTargets::All } else { SubresourceTargets::Indices(indices) };
    }

    if num_mips == 1 && num_slices == 1 && num_planes == 1 && range.first_slice == 0 && range.first_plane == 0 {
        SubresourceTargets::Indices(vec![range.index_or_first_mip])
    } else {
        SubresourceTargets::All
    }
}

fn buffer_barrier_needs_sync(b: BufferBarrier) -> bool {
    let has_uav = ((b.access_before | b.access_after) & BARRIER_ACCESS_UNORDERED_ACCESS) != 0;
    if has_uav {
        return true;
    }
    if b.sync_before == 0 || b.sync_after == 0 || b.access_before == BARRIER_ACCESS_NO_ACCESS || b.access_after == BARRIER_ACCESS_NO_ACCESS {
        return false;
    }
    b.access_before == BARRIER_ACCESS_COMMON
        || b.access_after == BARRIER_ACCESS_COMMON
        || (b.access_before & BARRIER_ACCESS_WRITE_MASK) != 0
        || (b.access_after & BARRIER_ACCESS_WRITE_MASK) != 0
}

fn global_barrier_needs_sync(g: GlobalBarrier) -> bool {
    g.sync_before != 0
        || g.sync_after != 0
        || (g.access_before != BARRIER_ACCESS_NO_ACCESS && g.access_after != BARRIER_ACCESS_NO_ACCESS)
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(super) struct LoweredBarriers {
    /// Ordered batches to submit via PC slot 26 (`ResourceBarrier`). Separating pre-sync
    /// (aliasing/buffer/global), transitions, and post-UAV prevents duplicate-subresource and
    /// aliasing+transition validation errors in a single `ResourceBarrier` call.
    pub batches: Vec<Vec<[u8; BARRIER_BYTES]>>,
    /// `(pc_res, subresource, state_after)` transitions for `detile` and `present`.
    pub noted: Vec<(usize, u32, u32)>,
    pub unknown_group: Option<u32>,
}

/// Lowers Enhanced Barrier `groups` recorded on PC command list `list_id` (type `list_type`) into
/// legacy PC `ResourceBarrier` batches and updates `tracker`.
pub(super) fn lower_enhanced_barriers(
    list_id: usize,
    list_type: u32,
    groups: &[BarrierGroup],
    tracker: &mut StateTracker,
    is_title_mem: impl Fn(u64) -> bool,
    unwrap_ptr: impl Fn(u64) -> u64,
) -> LoweredBarriers {
    let mut out = LoweredBarriers::default();
    if list_type == D3D12_COMMAND_LIST_TYPE_BUNDLE || groups.is_empty() {
        return out;
    }

    let supports_uav_alias = list_misc::supports_uav_and_aliasing_barriers(list_type);
    let mut pre_sync: Vec<[u8; BARRIER_BYTES]> = Vec::new();
    let mut transition_batches: Vec<Vec<[u8; BARRIER_BYTES]>> = Vec::new();
    let mut cur_transitions: Vec<[u8; BARRIER_BYTES]> = Vec::new();
    let mut cur_seen: Vec<(usize, u32)> = Vec::new();
    let mut post_uav: Vec<[u8; BARRIER_BYTES]> = Vec::new();

    let mut has_global_uav = false;
    let mut has_global_alias = false;
    let mut pre_uav_resources: Vec<u64> = Vec::new();
    let mut pre_alias_resources: Vec<u64> = Vec::new();
    let mut post_uav_resources: Vec<u64> = Vec::new();
    let mut has_post_global_uav = false;

    let push_global_uav = |pre_sync: &mut Vec<[u8; BARRIER_BYTES]>, has_global_uav: &mut bool| {
        if !*has_global_uav {
            pre_sync.push(legacy_uav_barrier(0));
            *has_global_uav = true;
        }
    };
    let push_global_alias = |pre_sync: &mut Vec<[u8; BARRIER_BYTES]>, has_global_alias: &mut bool| {
        if !*has_global_alias {
            pre_sync.push(legacy_aliasing_barrier(0));
            *has_global_alias = true;
        }
    };

    let mut push_transition = |pc_res: usize, sub: u32, before: u32, after: u32| {
        let conflicts = cur_seen
            .iter()
            .any(|&(r, s)| r == pc_res && (s == sub || s == ALL_SUBRESOURCES || sub == ALL_SUBRESOURCES));
        if conflicts && !cur_transitions.is_empty() {
            transition_batches.push(std::mem::take(&mut cur_transitions));
            cur_seen.clear();
        }
        cur_transitions.push(legacy_transition_barrier(pc_res as u64, sub, before, after));
        cur_seen.push((pc_res, sub));
    };

    for group in groups {
        match group {
            BarrierGroup::Global(items) => {
                if supports_uav_alias && items.iter().copied().any(global_barrier_needs_sync) {
                    push_global_uav(&mut pre_sync, &mut has_global_uav);
                    push_global_alias(&mut pre_sync, &mut has_global_alias);
                }
            }
            BarrierGroup::Buffer(items) => {
                if !supports_uav_alias {
                    continue;
                }
                for &b in items {
                    if !buffer_barrier_needs_sync(b) {
                        continue;
                    }
                    let title_mem = b.resource != 0 && is_title_mem(b.resource);
                    let pc_res = if b.resource == 0 { 0 } else { unwrap_ptr(b.resource) };
                    if title_mem || pc_res == 0 {
                        push_global_uav(&mut pre_sync, &mut has_global_uav);
                        push_global_alias(&mut pre_sync, &mut has_global_alias);
                        continue;
                    }
                    let has_uav_access = ((b.access_before | b.access_after) & BARRIER_ACCESS_UNORDERED_ACCESS) != 0;
                    let allow_uav = tracker
                        .resources
                        .get(&(pc_res as usize))
                        .map_or(has_uav_access, |entry| entry.meta.allow_uav);
                    if has_uav_access && allow_uav {
                        if !has_global_uav && !pre_uav_resources.contains(&pc_res) {
                            pre_sync.push(legacy_uav_barrier(pc_res));
                            pre_uav_resources.push(pc_res);
                        }
                    } else {
                        push_global_uav(&mut pre_sync, &mut has_global_uav);
                    }
                }
            }
            BarrierGroup::Texture(items) => {
                for &t in items {
                    if t.resource == 0 {
                        continue;
                    }
                    let pc_res = unwrap_ptr(t.resource) as usize;
                    if pc_res == 0 {
                        continue;
                    }
                    let Some(raw_after) = layout_to_legacy_state(t.layout_after) else {
                        continue;
                    };
                    let after_state = clamp_state_for_list_type(raw_after, list_type);
                    let (meta, states) = tracker.effective_state(list_id, pc_res);
                    let is_undefined = t.layout_before == BARRIER_LAYOUT_UNDEFINED;
                    let is_discard = (t.flags & TEXTURE_BARRIER_FLAG_DISCARD) != 0;

                    if supports_uav_alias && (is_discard || is_undefined) {
                        if meta.aliasable {
                            if !has_global_alias && !pre_alias_resources.contains(&(pc_res as u64)) {
                                pre_sync.push(legacy_aliasing_barrier(pc_res as u64));
                                pre_alias_resources.push(pc_res as u64);
                            }
                        } else {
                            push_global_alias(&mut pre_sync, &mut has_global_alias);
                        }
                    }

                    let targets = expand_subresources(t.subresources, meta);
                    let mut recorded_for_tracker: Vec<(usize, u32, u32)> = Vec::new();
                    if is_undefined {
                        match targets {
                            SubresourceTargets::All => {
                                if states.len() <= 1 {
                                    let before = clamp_state_for_list_type(states.first().copied().unwrap_or(0), list_type);
                                    if before != after_state {
                                        push_transition(pc_res, ALL_SUBRESOURCES, before, after_state);
                                        out.noted.push((pc_res, ALL_SUBRESOURCES, after_state));
                                    }
                                    recorded_for_tracker.push((pc_res, ALL_SUBRESOURCES, after_state));
                                } else {
                                    for (sub, &st) in states.iter().enumerate() {
                                        let before = clamp_state_for_list_type(st, list_type);
                                        if before != after_state {
                                            push_transition(pc_res, sub as u32, before, after_state);
                                            out.noted.push((pc_res, sub as u32, after_state));
                                        }
                                    }
                                    recorded_for_tracker.push((pc_res, ALL_SUBRESOURCES, after_state));
                                }
                            }
                            SubresourceTargets::Indices(subs) => {
                                for sub in subs {
                                    let before = clamp_state_for_list_type(state_for_subresource(&states, sub), list_type);
                                    if before != after_state {
                                        push_transition(pc_res, sub, before, after_state);
                                        out.noted.push((pc_res, sub, after_state));
                                    }
                                    recorded_for_tracker.push((pc_res, sub, after_state));
                                }
                            }
                        }
                    } else {
                        let raw_before = layout_to_legacy_state(t.layout_before).unwrap_or(0);
                        let before_state = clamp_state_for_list_type(raw_before, list_type);
                        match targets {
                            SubresourceTargets::All => {
                                if before_state != after_state {
                                    push_transition(pc_res, ALL_SUBRESOURCES, before_state, after_state);
                                    out.noted.push((pc_res, ALL_SUBRESOURCES, after_state));
                                }
                                recorded_for_tracker.push((pc_res, ALL_SUBRESOURCES, after_state));
                            }
                            SubresourceTargets::Indices(subs) => {
                                for sub in subs {
                                    if before_state != after_state {
                                        push_transition(pc_res, sub, before_state, after_state);
                                        out.noted.push((pc_res, sub, after_state));
                                    }
                                    recorded_for_tracker.push((pc_res, sub, after_state));
                                }
                            }
                        }
                    }
                    tracker.note_list_transitions(list_id, &recorded_for_tracker);

                    let has_uav_access = ((t.access_before | t.access_after) & BARRIER_ACCESS_UNORDERED_ACCESS) != 0;
                    if supports_uav_alias && has_uav_access {
                        let allow_uav = if meta.known { meta.allow_uav } else { true };
                        if allow_uav {
                            if !has_post_global_uav && !post_uav_resources.contains(&(pc_res as u64)) {
                                post_uav.push(legacy_uav_barrier(pc_res as u64));
                                post_uav_resources.push(pc_res as u64);
                            }
                        } else if !has_post_global_uav {
                            post_uav.push(legacy_uav_barrier(0));
                            has_post_global_uav = true;
                        }
                    }
                }
            }
            &BarrierGroup::Unknown(ty) => {
                out.unknown_group.get_or_insert(ty);
                if supports_uav_alias {
                    push_global_uav(&mut pre_sync, &mut has_global_uav);
                    push_global_alias(&mut pre_sync, &mut has_global_alias);
                }
            }
        }
    }

    if !cur_transitions.is_empty() {
        transition_batches.push(cur_transitions);
    }
    if !pre_sync.is_empty() {
        out.batches.push(pre_sync);
    }
    out.batches.extend(transition_batches);
    if !post_uav.is_empty() {
        out.batches.push(post_uav);
    }
    out
}

/// Console list slot 146 (newer layout): `ID3D12GraphicsCommandList7::Barrier(n, groups)`.
///
/// Parses the console Enhanced Barrier groups and records equivalent legacy `ResourceBarrier`
/// batches on the PC command list, keeping `detile` and `present` transition tracking in sync.
pub(super) extern "system" fn list_barrier_x(this: *mut Proxy, n: u32, groups: *const c_void) {
    static N: AtomicI32 = AtomicI32::new(0);
    static UNKNOWN_TY: AtomicI32 = AtomicI32::new(0);
    if !crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 146);
    }
    // SAFETY: `this` is a live command list proxy; `groups` holds `n` readable barrier groups.
    unsafe {
        let pc = (*this).pc;
        crate::capture::flush(this as usize, pc);
        if groups.is_null() || n == 0 {
            return;
        }
        let parsed = parse_groups(n, groups);
        let list_type = method::<GetListType>(pc, 8)(pc);
        let lowered = with_states(|tracker| {
            lower_enhanced_barriers(
                pc as usize,
                list_type,
                &parsed,
                tracker,
                |p| is_title_mem_resource(p as usize as Object),
                |p| pc_pointer(p),
            )
        });
        if let Some(ty) = lowered.unknown_group
            && log_due(&UNKNOWN_TY)
        {
            xs_log!("console list slot 146 (Barrier): unknown D3D12_BARRIER_TYPE {ty}, recorded global UAV and aliasing barrier");
        }
        if !lowered.noted.is_empty() {
            crate::detile::note_transitions(pc as usize, &lowered.noted);
            crate::present::note_list_transitions(pc as usize, &lowered.noted);
        }
        let mut total_barriers = 0usize;
        for batch in &lowered.batches {
            if !batch.is_empty() {
                total_barriers += batch.len();
                method::<Barriers>(pc, 26)(pc, batch.len() as u32, batch.as_ptr().cast());
            }
        }
        if log_due(&N) {
            xs_log!(
                "console list slot 146 (Barrier, groups {n}, list_type {list_type}): lowered to {total_barriers} legacy barriers in {} calls ({} transitions noted)",
                lowered.batches.len(),
                lowered.noted.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(b: &[u8; BARRIER_BYTES], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn u64_at(b: &[u8; BARRIER_BYTES], at: usize) -> u64 {
        u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }

    fn tex_barrier(res: u64, before: u32, after: u32, access_before: u32, access_after: u32, flags: u32) -> TextureBarrier {
        TextureBarrier {
            sync_before: 1,
            sync_after: 1,
            access_before,
            access_after,
            layout_before: before,
            layout_after: after,
            resource: res,
            subresources: BarrierSubresourceRange {
                index_or_first_mip: ALL_SUBRESOURCES,
                num_mips: 0,
                first_slice: 0,
                num_slices: 0,
                first_plane: 0,
                num_planes: 0,
            },
            flags,
        }
    }

    #[test]
    fn sanitize_layout_maps_console_31_and_unknown_to_common_and_keeps_undefined() {
        assert_eq!(sanitize_layout(BARRIER_LAYOUT_UNDEFINED), BARRIER_LAYOUT_UNDEFINED);
        assert_eq!(layout_to_legacy_state(BARRIER_LAYOUT_UNDEFINED), None);

        assert_eq!(sanitize_layout(0), 0);
        assert_eq!(sanitize_layout(2), 2);
        assert_eq!(sanitize_layout(30), 30);
        assert_eq!(sanitize_layout(CONSOLE_BARRIER_LAYOUT_PRESENT), 0);
        assert_eq!(sanitize_layout(32), 0);
        assert_eq!(sanitize_layout(999), 0);

        assert_eq!(layout_to_legacy_state(CONSOLE_BARRIER_LAYOUT_PRESENT), Some(0));
        assert_eq!(layout_to_legacy_state(2), Some(0x4));
        assert_eq!(layout_to_legacy_state(6), Some(0xC0));
        assert_eq!(layout_to_legacy_state(11), Some(0x100_0000));
        assert_eq!(layout_to_legacy_state(19), Some(0xAC3));
        assert_eq!(layout_to_legacy_state(27), Some(0x40));
    }

    #[test]
    fn undefined_texture_barrier_resolves_tracked_state_and_emits_aliasing_before_transition() {
        let mut tracker = StateTracker::default();
        let committed = ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable: false, allow_uav: true, known: true };
        let placed = ResourceMeta { aliasable: true, ..committed };
        tracker.note_resource_created(0xA000, 0x8, committed); // UAV initial state
        tracker.note_resource_created(0xB000, 0xC0, placed); // SRV initial state

        let groups = [BarrierGroup::Texture(vec![
            tex_barrier(0xA001, BARRIER_LAYOUT_UNDEFINED, 2, BARRIER_ACCESS_NO_ACCESS, 0x8, TEXTURE_BARRIER_FLAG_DISCARD),
            tex_barrier(0xB001, BARRIER_LAYOUT_UNDEFINED, 8, BARRIER_ACCESS_NO_ACCESS, 0x400, 0),
        ])];
        let unwrap = |p: u64| p & !1;
        let lowered = lower_enhanced_barriers(0x100, 0, &groups, &mut tracker, |_| false, unwrap);

        // Batch 0: aliasing (global NULL/NULL for committed 0xA000; placed 0xB000 covered once global is present)
        // Batch 1: transitions (0xA000: 0x8 -> 0x4, 0xB000: 0xC0 -> 0x400)
        assert_eq!(lowered.batches.len(), 2);
        assert_eq!(lowered.batches[0].len(), 1);
        assert_eq!(u32_at(&lowered.batches[0][0], 0), LEGACY_BARRIER_TYPE_ALIASING);
        assert_eq!(u64_at(&lowered.batches[0][0], 16), 0);

        assert_eq!(lowered.batches[1].len(), 2);
        assert_eq!(u32_at(&lowered.batches[1][0], 0), LEGACY_BARRIER_TYPE_TRANSITION);
        assert_eq!(u64_at(&lowered.batches[1][0], 8), 0xA000);
        assert_eq!(u32_at(&lowered.batches[1][0], 16), ALL_SUBRESOURCES);
        assert_eq!(u32_at(&lowered.batches[1][0], 20), 0x8);
        assert_eq!(u32_at(&lowered.batches[1][0], 24), 0x4);

        assert_eq!(u64_at(&lowered.batches[1][1], 8), 0xB000);
        assert_eq!(u32_at(&lowered.batches[1][1], 20), 0xC0);
        assert_eq!(u32_at(&lowered.batches[1][1], 24), 0x400);
        assert_eq!(lowered.noted, vec![(0xA000, ALL_SUBRESOURCES, 0x4), (0xB000, ALL_SUBRESOURCES, 0x400)]);

        // A second UNDEFINED barrier on the same command list sees 0xA000 already in 0x4 (RT).
        let groups2 = [BarrierGroup::Texture(vec![tex_barrier(0xA001, BARRIER_LAYOUT_UNDEFINED, 6, 0x8, 0x80, 0)])];
        let lowered2 = lower_enhanced_barriers(0x100, 0, &groups2, &mut tracker, |_| false, unwrap);
        assert_eq!(lowered2.batches.len(), 2);
        assert_eq!(u32_at(&lowered2.batches[1][0], 20), 0x4);
        assert_eq!(u32_at(&lowered2.batches[1][0], 24), 0xC0);

        // Submitting the list commits the new states for subsequent lists.
        tracker.list_submitted(&[0x100]);
        let groups3 = [BarrierGroup::Texture(vec![tex_barrier(0xA001, BARRIER_LAYOUT_UNDEFINED, CONSOLE_BARRIER_LAYOUT_PRESENT, 0x80, BARRIER_ACCESS_NO_ACCESS, 0)])];
        let lowered3 = lower_enhanced_barriers(0x200, 0, &groups3, &mut tracker, |_| false, unwrap);
        assert_eq!(u32_at(&lowered3.batches[1][0], 20), 0xC0);
        assert_eq!(u32_at(&lowered3.batches[1][0], 24), 0);
    }

    #[test]
    fn placed_texture_discard_emits_targeted_aliasing_and_uav_texture_emits_post_uav() {
        let mut tracker = StateTracker::default();
        let placed = ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable: true, allow_uav: true, known: true };
        tracker.note_resource_created(0xC000, 0x8, placed);

        // Same-layout UAV -> UAV barrier with DISCARD flag emits targeted ALIASING(NULL, res)
        // and UAV(res), with no illegal before==after transition.
        let groups = [BarrierGroup::Texture(vec![tex_barrier(0xC000, 3, 3, BARRIER_ACCESS_UNORDERED_ACCESS, BARRIER_ACCESS_UNORDERED_ACCESS, TEXTURE_BARRIER_FLAG_DISCARD)])];
        let lowered = lower_enhanced_barriers(0x10, 0, &groups, &mut tracker, |_| false, |p| p);
        assert_eq!(lowered.batches.len(), 2);
        assert_eq!(u32_at(&lowered.batches[0][0], 0), LEGACY_BARRIER_TYPE_ALIASING);
        assert_eq!(u64_at(&lowered.batches[0][0], 16), 0xC000);
        assert_eq!(u32_at(&lowered.batches[1][0], 0), LEGACY_BARRIER_TYPE_UAV);
        assert_eq!(u64_at(&lowered.batches[1][0], 8), 0xC000);
        assert!(lowered.noted.is_empty());
    }

    #[test]
    fn subresource_range_expands_mips_slices_and_planes_and_handles_divergent_undefined() {
        let mut tracker = StateTracker::default();
        let meta = ResourceMeta { mip_levels: 2, array_size: 2, plane_count: 1, aliasable: false, allow_uav: false, known: true };
        tracker.note_resource_created(0xD000, 0, meta);

        // Transition mip 1 across both array slices (subresources 1 and 3) from COMMON -> COPY_DEST.
        let mut t = tex_barrier(0xD000, 0, 8, 0, 0x400, 0);
        t.subresources = BarrierSubresourceRange {
            index_or_first_mip: 1,
            num_mips: 1,
            first_slice: 0,
            num_slices: 2,
            first_plane: 0,
            num_planes: 1,
        };
        let lowered = lower_enhanced_barriers(0x10, 0, &[BarrierGroup::Texture(vec![t])], &mut tracker, |_| false, |p| p);
        assert_eq!(lowered.batches.len(), 1);
        assert_eq!(lowered.batches[0].len(), 2);
        assert_eq!(u32_at(&lowered.batches[0][0], 16), 1);
        assert_eq!(u32_at(&lowered.batches[0][1], 16), 3);

        // Now subresources 0,2 are COMMON (0) and 1,3 are COPY_DEST (0x400).
        // An ALL_SUBRESOURCES UNDEFINED -> SHADER_RESOURCE (0xC0) emits per-subresource transitions
        // with each subresource's exact before state.
        let all_undef = tex_barrier(0xD000, BARRIER_LAYOUT_UNDEFINED, 6, 0x400, 0x80, 0);
        let lowered2 = lower_enhanced_barriers(0x10, 0, &[BarrierGroup::Texture(vec![all_undef])], &mut tracker, |_| false, |p| p);
        let trans = &lowered2.batches[1];
        assert_eq!(trans.len(), 4);
        assert_eq!((u32_at(&trans[0], 16), u32_at(&trans[0], 20), u32_at(&trans[0], 24)), (0, 0, 0xC0));
        assert_eq!((u32_at(&trans[1], 16), u32_at(&trans[1], 20), u32_at(&trans[1], 24)), (1, 0x400, 0xC0));
        assert_eq!((u32_at(&trans[2], 16), u32_at(&trans[2], 20), u32_at(&trans[2], 24)), (2, 0, 0xC0));
        assert_eq!((u32_at(&trans[3], 16), u32_at(&trans[3], 20), u32_at(&trans[3], 24)), (3, 0x400, 0xC0));
    }

    #[test]
    fn buffer_and_global_barriers_respect_title_mem_and_command_list_type() {
        let mut tracker = StateTracker::default();
        let uav_buf = ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable: false, allow_uav: true, known: true };
        let non_uav_buf = ResourceMeta { allow_uav: false, ..uav_buf };
        tracker.note_resource_created(0xE000, 0, uav_buf);
        tracker.note_resource_created(0xF000, 0, non_uav_buf);

        // Regular UAV buffer -> targeted UAV(0xE000) on DIRECT/COMPUTE.
        let regular_uav = BarrierGroup::Buffer(vec![BufferBarrier {
            sync_before: 0x80,
            sync_after: 0x80,
            access_before: BARRIER_ACCESS_UNORDERED_ACCESS,
            access_after: BARRIER_ACCESS_UNORDERED_ACCESS,
            resource: 0xE000,
            offset: 0,
            size: u64::MAX,
        }]);
        let direct = lower_enhanced_barriers(1, 0, std::slice::from_ref(&regular_uav), &mut tracker, |_| false, |p| p);
        assert_eq!(direct.batches.len(), 1);
        assert_eq!(direct.batches[0], vec![legacy_uav_barrier(0xE000)]);

        // title_mem-backed buffer -> global [UAV(NULL), ALIASING(NULL, NULL)].
        let title_mem = lower_enhanced_barriers(1, 2, std::slice::from_ref(&regular_uav), &mut tracker, |p| p == 0xE000, |p| p);
        assert_eq!(title_mem.batches.len(), 1);
        assert_eq!(title_mem.batches[0], vec![legacy_uav_barrier(0), legacy_aliasing_barrier(0)]);

        // Non-UAV buffer with COPY_DEST -> VERTEX_BUFFER falls back to UAV(NULL) (valid without ALLOW_UAV).
        let copy_to_vb = BarrierGroup::Buffer(vec![BufferBarrier {
            sync_before: 0x200,
            sync_after: 0x8,
            access_before: 0x400,
            access_after: 0x1,
            resource: 0xF000,
            offset: 0,
            size: u64::MAX,
        }]);
        let non_uav = lower_enhanced_barriers(1, 0, &[copy_to_vb], &mut tracker, |_| false, |p| p);
        assert_eq!(non_uav.batches, vec![vec![legacy_uav_barrier(0)]]);

        // Pure read-to-read buffer barrier needs no legacy sync.
        let read_to_read = BarrierGroup::Buffer(vec![BufferBarrier {
            sync_before: 0x80,
            sync_after: 0x10,
            access_before: 0x80, // SRV
            access_after: 0x80,  // SRV
            resource: 0xE000,
            offset: 0,
            size: u64::MAX,
        }]);
        assert!(lower_enhanced_barriers(1, 0, &[read_to_read], &mut tracker, |_| false, |p| p).batches.is_empty());

        // Global barrier emits [UAV(NULL), ALIASING(NULL, NULL)] on DIRECT/COMPUTE, skipped on COPY/BUNDLE.
        let global = BarrierGroup::Global(vec![GlobalBarrier { sync_before: 1, sync_after: 1, access_before: 0, access_after: 0 }]);
        assert_eq!(
            lower_enhanced_barriers(1, 0, std::slice::from_ref(&global), &mut tracker, |_| false, |p| p).batches,
            vec![vec![legacy_uav_barrier(0), legacy_aliasing_barrier(0)]]
        );
        assert_eq!(
            lower_enhanced_barriers(1, 2, std::slice::from_ref(&global), &mut tracker, |_| false, |p| p).batches,
            vec![vec![legacy_uav_barrier(0), legacy_aliasing_barrier(0)]]
        );
        assert!(lower_enhanced_barriers(1, 3, &[global.clone(), regular_uav.clone()], &mut tracker, |_| false, |p| p).batches.is_empty());
        assert!(lower_enhanced_barriers(1, 1, &[global, regular_uav], &mut tracker, |_| false, |p| p).batches.is_empty());
    }

    #[test]
    fn parse_groups_reads_raw_d3d12_binary_layouts_accurately() {
        let mut tex_raw = [0u8; TEXTURE_BARRIER_BYTES];
        tex_raw[0..4].copy_from_slice(&0x40u32.to_le_bytes()); // SyncBefore = RT
        tex_raw[4..8].copy_from_slice(&0x10u32.to_le_bytes()); // SyncAfter = PS
        tex_raw[8..12].copy_from_slice(&0x8u32.to_le_bytes()); // AccessBefore = RT
        tex_raw[12..16].copy_from_slice(&0x80u32.to_le_bytes()); // AccessAfter = SRV
        tex_raw[16..20].copy_from_slice(&2u32.to_le_bytes()); // LayoutBefore = RT
        tex_raw[20..24].copy_from_slice(&CONSOLE_BARRIER_LAYOUT_PRESENT.to_le_bytes()); // LayoutAfter = 31
        tex_raw[24..32].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        tex_raw[32..36].copy_from_slice(&ALL_SUBRESOURCES.to_le_bytes());
        tex_raw[56..60].copy_from_slice(&TEXTURE_BARRIER_FLAG_DISCARD.to_le_bytes());

        let mut buf_raw = [0u8; BUFFER_BARRIER_BYTES];
        buf_raw[0..4].copy_from_slice(&0x80u32.to_le_bytes());
        buf_raw[4..8].copy_from_slice(&0x800u32.to_le_bytes());
        buf_raw[8..12].copy_from_slice(&0x10u32.to_le_bytes());
        buf_raw[12..16].copy_from_slice(&0x200u32.to_le_bytes());
        buf_raw[16..24].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
        buf_raw[24..32].copy_from_slice(&0u64.to_le_bytes());
        buf_raw[32..40].copy_from_slice(&u64::MAX.to_le_bytes());

        let mut glob_raw = [0u8; GLOBAL_BARRIER_BYTES];
        glob_raw[0..4].copy_from_slice(&1u32.to_le_bytes());
        glob_raw[4..8].copy_from_slice(&1u32.to_le_bytes());

        let make_group = |ty: u32, count: u32, ptr: *const u8| {
            let mut g = [0u8; BARRIER_GROUP_BYTES];
            g[0..4].copy_from_slice(&ty.to_le_bytes());
            g[4..8].copy_from_slice(&count.to_le_bytes());
            g[8..16].copy_from_slice(&(ptr as usize as u64).to_le_bytes());
            g
        };
        let groups = [
            make_group(BARRIER_TYPE_GLOBAL, 1, glob_raw.as_ptr()),
            make_group(BARRIER_TYPE_TEXTURE, 1, tex_raw.as_ptr()),
            make_group(BARRIER_TYPE_BUFFER, 1, buf_raw.as_ptr()),
        ];

        // SAFETY: `groups` and all three element arrays are live stack storage.
        let parsed = unsafe { parse_groups(groups.len() as u32, groups.as_ptr().cast()) };
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0], BarrierGroup::Global(vec![GlobalBarrier { sync_before: 1, sync_after: 1, access_before: 0, access_after: 0 }]));
        assert_eq!(
            parsed[1],
            BarrierGroup::Texture(vec![TextureBarrier {
                sync_before: 0x40,
                sync_after: 0x10,
                access_before: 0x8,
                access_after: 0x80,
                layout_before: 2,
                layout_after: CONSOLE_BARRIER_LAYOUT_PRESENT,
                resource: 0x1234_5678,
                subresources: BarrierSubresourceRange {
                    index_or_first_mip: ALL_SUBRESOURCES,
                    num_mips: 0,
                    first_slice: 0,
                    num_slices: 0,
                    first_plane: 0,
                    num_planes: 0,
                },
                flags: TEXTURE_BARRIER_FLAG_DISCARD,
            }])
        );
        assert_eq!(
            parsed[2],
            BarrierGroup::Buffer(vec![BufferBarrier {
                sync_before: 0x80,
                sync_after: 0x800,
                access_before: 0x10,
                access_after: 0x200,
                resource: 0xDEAD_BEEF,
                offset: 0,
                size: u64::MAX,
            }])
        );
    }

    #[test]
    fn duplicate_texture_transitions_in_one_call_split_into_separate_batches() {
        let mut tracker = StateTracker::default();
        let meta = ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable: false, allow_uav: false, known: true };
        tracker.note_resource_created(0x1000, 0, meta);

        let groups = [
            BarrierGroup::Texture(vec![tex_barrier(0x1000, 0, 2, 0, 0x8, 0)]),
            BarrierGroup::Texture(vec![tex_barrier(0x1000, 2, 6, 0x8, 0x80, 0)]),
        ];
        let lowered = lower_enhanced_barriers(1, 0, &groups, &mut tracker, |_| false, |p| p);
        assert_eq!(lowered.batches.len(), 2);
        assert_eq!((u32_at(&lowered.batches[0][0], 20), u32_at(&lowered.batches[0][0], 24)), (0, 0x4));
        assert_eq!((u32_at(&lowered.batches[1][0], 20), u32_at(&lowered.batches[1][0], 24)), (0x4, 0xC0));
    }

    #[test]
    fn resource_meta_extracts_mip_chain_array_size_planes_and_uav_flag() {
        // 2D depth+stencil texture with MipLevels=0 (auto full chain: 16x8 -> 5 mips), 4 slices, 2 planes, no UAV.
        let mut d2d = [0u8; 0x38];
        d2d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d2d[0x10..0x18].copy_from_slice(&16u64.to_le_bytes());
        d2d[0x18..0x1C].copy_from_slice(&8u32.to_le_bytes());
        d2d[0x1C..0x1E].copy_from_slice(&4u16.to_le_bytes());
        d2d[0x1E..0x20].copy_from_slice(&0u16.to_le_bytes());
        d2d[0x20..0x24].copy_from_slice(&45u32.to_le_bytes()); // DXGI_FORMAT_D24_UNORM_S8_UINT
        d2d[0x30..0x34].copy_from_slice(&0x2u32.to_le_bytes()); // ALLOW_DEPTH_STENCIL
        assert_eq!(
            resource_meta_from_desc(&d2d, true),
            ResourceMeta { mip_levels: 5, array_size: 4, plane_count: 2, aliasable: true, allow_uav: false, known: true }
        );

        // 3D texture with Depth=8 keeps array_size=1 and computes full mip chain from max(4, 4, 8) = 4 mips.
        let mut d3d = [0u8; 0x38];
        d3d[0..4].copy_from_slice(&4u32.to_le_bytes());
        d3d[0x10..0x18].copy_from_slice(&4u64.to_le_bytes());
        d3d[0x18..0x1C].copy_from_slice(&4u32.to_le_bytes());
        d3d[0x1C..0x1E].copy_from_slice(&8u16.to_le_bytes());
        d3d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes()); // R8G8B8A8_UNORM
        d3d[0x30..0x34].copy_from_slice(&0x4u32.to_le_bytes()); // ALLOW_UNORDERED_ACCESS
        assert_eq!(
            resource_meta_from_desc(&d3d, false),
            ResourceMeta { mip_levels: 4, array_size: 1, plane_count: 1, aliasable: false, allow_uav: true, known: true }
        );
    }

    #[cfg(windows)]
    #[test]
    fn lowered_enhanced_barriers_close_cleanly_on_real_d3d12_lists() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        use windows::core::Interface;

        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 2,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let buf_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 65536,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let default_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };

        for (name, device) in crate::view_gpu_tests::devices() {
            let mut tex: Option<ID3D12Resource> = None;
            let mut buf: Option<ID3D12Resource> = None;
            // SAFETY: valid D3D12 device and resource descriptions.
            unsafe {
                device
                    .CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut tex)
                    .unwrap_or_else(|e| panic!("{name}: committed texture failed: {e}"));
                device
                    .CreateReservedResource(&buf_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buf)
                    .unwrap_or_else(|e| panic!("{name}: reserved buffer failed: {e}"));
            }
            let tex = tex.expect("texture");
            let buf = buf.expect("buffer");

            for (list_type_enum, list_type_raw) in [
                (D3D12_COMMAND_LIST_TYPE_DIRECT, 0u32),
                (D3D12_COMMAND_LIST_TYPE_COMPUTE, 2u32),
                (D3D12_COMMAND_LIST_TYPE_COPY, 3u32),
            ] {
                // SAFETY: standard D3D12 allocator and command list creation on a live device.
                let (alloc, list): (ID3D12CommandAllocator, ID3D12GraphicsCommandList) = unsafe {
                    let alloc = device.CreateCommandAllocator(list_type_enum).expect("allocator");
                    let list = device.CreateCommandList(0, list_type_enum, &alloc, None).expect("command list");
                    (alloc, list)
                };
                let mut tracker = StateTracker::default();
                let tex_meta = ResourceMeta { mip_levels: 2, array_size: 1, plane_count: 1, aliasable: false, allow_uav: true, known: true };
                let buf_meta = ResourceMeta { mip_levels: 1, array_size: 1, plane_count: 1, aliasable: true, allow_uav: true, known: true };
                tracker.note_resource_created(tex.as_raw() as usize, 0x8, tex_meta);
                tracker.note_resource_created(buf.as_raw() as usize, 0, buf_meta);

                let groups = match list_type_raw {
                    0 => vec![
                        BarrierGroup::Global(vec![GlobalBarrier { sync_before: 1, sync_after: 1, access_before: 0, access_after: 0 }]),
                        BarrierGroup::Texture(vec![
                            tex_barrier(tex.as_raw() as usize as u64, BARRIER_LAYOUT_UNDEFINED, 2, BARRIER_ACCESS_NO_ACCESS, 0x8, TEXTURE_BARRIER_FLAG_DISCARD),
                            tex_barrier(tex.as_raw() as usize as u64, 2, CONSOLE_BARRIER_LAYOUT_PRESENT, 0x8, BARRIER_ACCESS_NO_ACCESS, 0),
                        ]),
                        BarrierGroup::Buffer(vec![BufferBarrier {
                            sync_before: 0x80,
                            sync_after: 0x800,
                            access_before: BARRIER_ACCESS_UNORDERED_ACCESS,
                            access_after: 0x200,
                            resource: buf.as_raw() as usize as u64,
                            offset: 0,
                            size: u64::MAX,
                        }]),
                    ],
                    2 => vec![
                        BarrierGroup::Texture(vec![tex_barrier(tex.as_raw() as usize as u64, 3, 6, BARRIER_ACCESS_UNORDERED_ACCESS, 0x80, 0)]),
                        BarrierGroup::Buffer(vec![BufferBarrier {
                            sync_before: 0x80,
                            sync_after: 0x80,
                            access_before: BARRIER_ACCESS_UNORDERED_ACCESS,
                            access_after: BARRIER_ACCESS_UNORDERED_ACCESS,
                            resource: buf.as_raw() as usize as u64,
                            offset: 0,
                            size: u64::MAX,
                        }]),
                    ],
                    _ => vec![
                        BarrierGroup::Global(vec![GlobalBarrier { sync_before: 1, sync_after: 1, access_before: 0, access_after: 0 }]),
                        BarrierGroup::Texture(vec![tex_barrier(tex.as_raw() as usize as u64, 0, 8, 0, 0x400, 0)]),
                    ],
                };
                let lowered = lower_enhanced_barriers(list.as_raw() as usize, list_type_raw, &groups, &mut tracker, |p| p == buf.as_raw() as usize as u64, |p| p);
                // SAFETY: `list` is a live open D3D12 command list; each batch holds valid 32-byte barriers.
                unsafe {
                    for batch in &lowered.batches {
                        method::<Barriers>(list.as_raw(), 26)(list.as_raw(), batch.len() as u32, batch.as_ptr().cast());
                    }
                    list.Close().unwrap_or_else(|e| panic!("{name}: Close failed for {list_type_enum:?}: {e}"));
                }
                drop((list, alloc));
            }
        }
    }
}
