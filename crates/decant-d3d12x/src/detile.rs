//! Console textures the PC GPU can read. A title places its textures in its own memory
//! (`CreatePlacedResourceX`, device slots 44 and 133) in the console tiling and fills them with the
//! CPU or with DirectStorage. The proxy backs every such texture with a committed PC texture that
//! nobody writes to. This module writes it. A compute shader (`shaders/detile.hlsl`) reads the tiled
//! bytes straight from the window of `title_mem` (the very pages the title wrote) and stores one mip
//! level at a time, row major, in a staging buffer. `CopyTextureRegion` then moves the rows into the
//! PC texture. The staging buffer exists because block compressed textures have no UAV, and the copy
//! converts the format for free. The layout (format, mips, slices, swizzle mode) comes from the
//! resource description the title passed, evaluated through `decant_texture::xg` the way `xg_xs.dll`
//! does (see `detile_plan`).
//!
//! When it runs: a texture is dirty from its creation and whenever the title writes its memory.
//! `flush` fills every dirty texture on the queue of an `ExecuteCommandLists` call before the title's
//! own lists run there (direct queues only). The memory needs a GPU address by then
//! (`title_mem::translate_fresh`), otherwise the texture stays dirty and is tried again at the next
//! few submissions. A texture enters and leaves the copy in the state the title's submitted work
//! left it in. The transitions of each list are noted while it is recorded and applied when it is
//! submitted (`note_transitions`, `list_reset`, `submitted`).
//!
//! Things that mark a texture dirty are its creation, a write reported by the DirectStorage layer
//! (`note_write`, exported by `d3d12_xs.dll` as `DecantD3D12XNoteWrite`, usable by other code for any
//! range) and every CPU write through an ordinary pointer. For the latter `title_mem` copies the
//! pages the title wrote since its last submission (found by page protection in `xmem.dll`) and
//! names their tiles (`note_tiles`), and each texture owning one of those pages is filled again.
//! This is what a video player needs that rewrites its planes every frame (three R8 textures a
//! frame in one retail title). It also covers a texture whose memory the title maps only after
//! creating the texture.
//!
//! Not covered are GPU writes into the memory. A copy into a buffer placed at the same address lands
//! in that buffer and not in the section pages. A texture that the title copies into on the GPU is
//! dropped from the table (`note_gpu_write`), so the detile never overwrites what the GPU wrote.
//! Textures that alias the same memory read the same bytes. Render targets, depth textures and
//! unordered access textures are never detiled.
//!
//! The switch `DECANT_D3D12X_DETILE=0` turns the module off (default on).

#![allow(unsafe_code)]
// `d3d12xs_test` includes this file and uses only part of it.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::ManuallyDrop;

#[derive(Default)]
struct FastIntHasher(u64);

impl Hasher for FastIntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.wrapping_mul(0x517c_c1b7_2722_0a95).wrapping_add(u64::from(b));
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastIntHasher>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<FastIntHasher>>;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{HSTRING, Interface, PCSTR, PCWSTR};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};

use crate::detile_plan::{self, MipJob, Plan};
use crate::log::xs_log;

/// `D3D12_RESOURCE_STATE_COPY_DEST`.
const COPY_DEST: u32 = 0x400;
/// The most staging bytes one list may use.
const LIST_STAGING_LIMIT: u64 = 256 << 20;
/// How many staging bytes a batch collects before it is submitted, unless a single texture needs
/// more. Every list keeps its staging buffer for later batches, and a limit of 256 MiB left 2.8 GiB
/// held by the lists.
const BATCH_STAGING_TARGET: u64 = 32 << 20;
/// Default cap on retained staging bytes across all detiler frames (MiB, 0 disables trimming).
const DEFAULT_STAGING_CAP_MIB: u64 = 256;

fn staging_cap_from(over_ride: Option<&str>) -> u64 {
    over_ride.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(DEFAULT_STAGING_CAP_MIB) << 20
}

fn staging_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| staging_cap_from(std::env::var("DECANT_D3D12X_DETILE_STAGING_CAP_MIB").ok().as_deref()))
}
/// Dispatches (parameter blocks) that fit in one list.
const DISPATCH_SLOTS: usize = 1024;
/// Size of one parameter block. A multiple of 256 because a root constant buffer address needs it.
const PARAM_BYTES: usize = 512;
/// Index of the first equation term in a parameter block.
const FIRST_TERM_WORD: usize = 20;
/// How many submissions a texture whose memory has no GPU address yet is retried at.
const FILL_RETRIES: u32 = 8;

/// Whether the module is on (`DECANT_D3D12X_DETILE` is anything but `0`). Read once.
pub fn enabled() -> bool {
    static SWITCH: OnceLock<bool> = OnceLock::new();
    *SWITCH.get_or_init(|| {
        let on = std::env::var_os("DECANT_D3D12X_DETILE").is_none_or(|v| v != "0");
        if !on {
            xs_log!("detile: DECANT_D3D12X_DETILE=0, placed textures stay unfilled");
        }
        on
    })
}

fn round_up(value: u64, multiple: u64) -> u64 {
    value.div_ceil(multiple) * multiple
}

// ==== system DLL entry points and shader ====

/// An export of a system DLL, looked up at run time because the DLL is not linked.
///
/// # Safety
/// `F` must be the function pointer type of the export `name` of `dll`.
unsafe fn load_export<F: Copy>(dll: PCWSTR, name: &std::ffi::CStr) -> Option<F> {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: both names are zero terminated, and the caller vouches for `F`.
    unsafe {
        let module = LoadLibraryW(dll.0);
        if module.is_null() {
            return None;
        }
        let address = GetProcAddress(module, name.as_ptr().cast())?;
        Some(std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, F>(&address))
    }
}

type SerializeRootSignature = unsafe extern "system" fn(*const D3D12_ROOT_SIGNATURE_DESC, D3D_ROOT_SIGNATURE_VERSION, *mut Option<ID3DBlob>, *mut Option<ID3DBlob>) -> windows::core::HRESULT;
type D3dCompile = unsafe extern "system" fn(*const c_void, usize, PCSTR, *const c_void, *mut c_void, PCSTR, PCSTR, u32, u32, *mut Option<ID3DBlob>, *mut Option<ID3DBlob>) -> windows::core::HRESULT;

/// The bytes a blob holds.
///
/// # Safety
/// The blob must be a live, initialized one.
unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob owns `GetBufferSize` bytes at `GetBufferPointer` for as long as it lives.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }
}

/// Compiles `shaders/detile.hlsl` as a shader model 5.1 compute shader.
fn compile_detile_shader() -> Option<Vec<u8>> {
    let source = include_str!("../shaders/detile.hlsl");
    // SAFETY: the export is `D3DCompile` of the compiler DLL.
    let compile = unsafe { load_export::<D3dCompile>(windows::core::w!("d3dcompiler_47.dll"), c"D3DCompile") }?;
    let (mut code, mut messages): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source buffer and both output slots are valid for the call.
    let hr = unsafe {
        compile(source.as_ptr().cast(), source.len(), PCSTR::null(), std::ptr::null(), std::ptr::null_mut(), PCSTR(c"main".as_ptr().cast()), PCSTR(c"cs_5_1".as_ptr().cast()), 0, 0, &mut code, &mut messages)
    };
    match code {
        Some(code) if hr.is_ok() => {
            // SAFETY: a blob the compiler just returned.
            Some(unsafe { blob_bytes(&code) }.to_vec())
        }
        _ => {
            // SAFETY: the same, for the message blob.
            let text = messages.map(|m| String::from_utf8_lossy(unsafe { blob_bytes(&m) }).into_owned());
            xs_log!("detile: the shader does not compile (hr {:#x}): {}", hr.0, text.unwrap_or_default());
            None
        }
    }
}

// ==== resources and barriers ====

fn buffer_resource(width: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

/// A copy of the interface pointer that does not own a reference, as D3D12 structures hold it.
///
/// # Safety
/// The result must not outlive `resource` and is never dropped (it is a `ManuallyDrop`).
unsafe fn borrowed(resource: &ID3D12Resource) -> ManuallyDrop<Option<ID3D12Resource>> {
    // SAFETY: both types are one non null interface pointer wide.
    unsafe { std::mem::transmute_copy::<ID3D12Resource, ManuallyDrop<Option<ID3D12Resource>>>(resource) }
}

fn state_change(resource: &ID3D12Resource, subresource: u32, from: u32, to: u32) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the barrier is used within the call it is built for.
                pResource: unsafe { borrowed(resource) },
                Subresource: subresource,
                StateBefore: D3D12_RESOURCE_STATES(from as i32),
                StateAfter: D3D12_RESOURCE_STATES(to as i32),
            }),
        },
    }
}

fn uav_state() -> u32 {
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32
}

fn copy_source_state() -> u32 {
    D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32
}

/// Which way the barriers of a texture go around the copy.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    /// From the states the title left to `COPY_DEST`.
    Enter,
    /// From `COPY_DEST` back to the states the title left.
    Leave,
}

/// Appends the barriers for one texture around the copy into `out`. `states` holds one state for all
/// subresources or one per subresource. Subresources that are in `COPY_DEST` already need nothing.
fn append_copy_barriers(out: &mut Vec<D3D12_RESOURCE_BARRIER>, texture: &ID3D12Resource, states: &[u32], leg: Leg) {
    let ends = |state: u32| if leg == Leg::Enter { (state, COPY_DEST) } else { (COPY_DEST, state) };
    if states.len() <= 1 {
        let (from, to) = ends(states.first().copied().unwrap_or(0));
        if from != to {
            out.push(state_change(texture, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to));
        }
        return;
    }
    for (index, &state) in states.iter().enumerate() {
        let (from, to) = ends(state);
        if from != to {
            out.push(state_change(texture, index as u32, from, to));
        }
    }
}

/// The barriers for one texture around the copy. `states` holds one state for all subresources or
/// one per subresource. Subresources that are in `COPY_DEST` already need nothing.
fn copy_barriers(texture: &ID3D12Resource, states: &[u32], leg: Leg) -> Vec<D3D12_RESOURCE_BARRIER> {
    let mut barriers = Vec::new();
    append_copy_barriers(&mut barriers, texture, states, leg);
    barriers
}

// ==== the work of one flush ====

/// The most cached footprint layouts kept by a detiler before clearing on the next miss.
const FOOTPRINT_CACHE_LIMIT: usize = 4096;

/// Identifies a PC texture geometry and the console plan layout it is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FootprintKey {
    width: u64,
    height: u32,
    slices: u16,
    mips: u16,
    format: i32,
    bytes_per_element: u32,
}

/// Copyable footprints and staging byte requirement for one texture geometry.
type CachedFootprint = (Arc<[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]>, u64);
type FootprintCache = HashMap<FootprintKey, Result<CachedFootprint, String>>;

/// Reusable buffers used while preparing and grouping a flush.
#[derive(Default)]
struct DetileScratch {
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    row_bytes: Vec<u64>,
    costs: Vec<(u64, usize)>,
    batches: Vec<Range<usize>>,
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
    bases: Vec<u64>,
}

/// One texture to detile.
pub struct Job<'a> {
    pub texture: ID3D12Resource,
    /// GPU address of the first byte of the texture in the window.
    pub src_gpu: u64,
    pub plan: &'a Plan,
    /// The states of the texture, one for all subresources or one each.
    pub states: &'a [u32],
}

/// A job together with the copy footprints of its PC texture.
struct Prepared<'a> {
    job: &'a Job<'a>,
    footprints: Arc<[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]>,
    staging_bytes: u64,
}

/// Checks that the PC texture has the layout of the console plan and returns the copy footprints of
/// all its subresources, reusing cached footprints when the geometry was already validated.
fn prepare_cached<'a>(
    device: &ID3D12Device,
    cache: &mut FootprintCache,
    scratch: &mut DetileScratch,
    job: &'a Job<'a>,
) -> Result<Prepared<'a>, String> {
    // SAFETY: a live resource.
    let desc = unsafe { job.texture.GetDesc() };
    let (mips, slices) = (job.plan.mips.len() as u32, job.plan.slices);
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || u32::from(desc.MipLevels) != mips || u32::from(desc.DepthOrArraySize) != slices {
        return Err(format!("the PC texture ({} mips, {} slices) does not match the console layout ({mips}, {slices})", desc.MipLevels, desc.DepthOrArraySize));
    }
    let key = FootprintKey {
        width: desc.Width,
        height: desc.Height,
        slices: desc.DepthOrArraySize,
        mips: desc.MipLevels,
        format: desc.Format.0,
        bytes_per_element: job.plan.bytes_per_element,
    };
    if let Some(cached) = cache.get(&key) {
        let (footprints, staging_bytes) = cached.as_ref().map_err(Clone::clone)?;
        return Ok(Prepared { job, footprints: Arc::clone(footprints), staging_bytes: *staging_bytes });
    }
    let count = (mips * slices) as usize;
    scratch.footprints.clear();
    scratch.footprints.resize(count, D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default());
    scratch.rows.clear();
    scratch.rows.resize(count, 0);
    scratch.row_bytes.clear();
    scratch.row_bytes.resize(count, 0);
    let mut total = 0u64;
    // SAFETY: every output array holds `count` entries.
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            count as u32,
            0,
            Some(scratch.footprints.as_mut_ptr()),
            Some(scratch.rows.as_mut_ptr()),
            Some(scratch.row_bytes.as_mut_ptr()),
            Some(&mut total),
        )
    };
    let element_bytes = u64::from(job.plan.bytes_per_element);
    for (mip, level) in job.plan.mips.iter().enumerate() {
        let console_row = u64::from(level.width_elements) * element_bytes;
        if scratch.rows[mip] != level.height_elements || scratch.row_bytes[mip] != console_row {
            let err = format!(
                "mip {mip}: the PC rows ({} of {} bytes) differ from the console elements ({} of {console_row})",
                scratch.rows[mip], scratch.row_bytes[mip], level.height_elements
            );
            if cache.len() >= FOOTPRINT_CACHE_LIMIT {
                cache.clear();
            }
            cache.insert(key, Err(err.clone()));
            return Err(err);
        }
    }
    let staging_bytes = round_up(total, 512);
    if staging_bytes > LIST_STAGING_LIMIT {
        let err = format!("{staging_bytes} bytes of staging are too many");
        if cache.len() >= FOOTPRINT_CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, Err(err.clone()));
        return Err(err);
    }
    let footprints: Arc<[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]> = Arc::from(scratch.footprints.as_slice());
    if cache.len() >= FOOTPRINT_CACHE_LIMIT {
        cache.clear();
    }
    cache.insert(key, Ok((Arc::clone(&footprints), staging_bytes)));
    Ok(Prepared { job, footprints, staging_bytes })
}

/// Splits jobs in order into batches in `batches`. A batch is closed when adding the next job would
/// pass the staging target or the dispatch slots, but a batch always holds at least one job. `costs`
/// holds `(staging bytes, dispatches)` per job.
fn group_batches_into(costs: &[(u64, usize)], batches: &mut Vec<Range<usize>>) {
    batches.clear();
    let (mut start, mut bytes, mut dispatches) = (0, 0u64, 0usize);
    for (index, &(job_bytes, job_dispatches)) in costs.iter().enumerate() {
        if index > start && (bytes + job_bytes > BATCH_STAGING_TARGET || dispatches + job_dispatches > DISPATCH_SLOTS) {
            batches.push(start..index);
            (start, bytes, dispatches) = (index, 0, 0);
        }
        bytes += job_bytes;
        dispatches += job_dispatches;
    }
    if start < costs.len() {
        batches.push(start..costs.len());
    }
}

/// Splits jobs in order into batches. A batch is closed when adding the next job would pass the
/// staging target or the dispatch slots, but a batch always holds at least one job. `costs` holds
/// `(staging bytes, dispatches)` per job.
fn group_batches(costs: &[(u64, usize)]) -> Vec<Range<usize>> {
    let mut batches = Vec::new();
    group_batches_into(costs, &mut batches);
    batches
}

/// The parameter block of the dispatch that detiles `level` of `plan`. The word order is the one
/// `shaders/detile.hlsl` reads.
fn param_block(plan: &Plan, level: &MipJob, row_pitch: u32, slice_stride: u32, staging_offset: u64) -> [u32; PARAM_BYTES / 4] {
    let head = [
        level.width_elements,
        level.height_elements,
        plan.bytes_per_element,
        row_pitch,
        u32::from(plan.linear),
        u32::from(level.in_tail),
        level.tail_x,
        level.tail_y,
        plan.block_width,
        plan.block_height,
        level.blocks_x,
        plan.block_bits,
        level.level_offset,
        plan.slice_size,
        level.pitch_bytes,
        slice_stride,
        staging_offset as u32,
        plan.slices,
        plan.terms.len() as u32,
    ];
    let mut words = [0u32; PARAM_BYTES / 4];
    words[..head.len()].copy_from_slice(&head);
    for (n, &(bit, x, y, z)) in plan.terms.iter().enumerate() {
        let at = FIRST_TERM_WORD + n * 4;
        words[at..at + 4].copy_from_slice(&[bit, x, y, z]);
    }
    words
}

/// How many thread groups cover a row of `level`. Elements narrower than a dword are processed in
/// packs so that the shader writes whole dwords.
fn groups_across(plan: &Plan, level: &MipJob) -> u32 {
    let per_thread = if plan.bytes_per_element < 4 { 4 / plan.bytes_per_element } else { 1 };
    level.width_elements.div_ceil(per_thread).div_ceil(16)
}

/// Staging bytes and PC objects of one reusable command list.
struct Frame {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// The queue the last list ran on (a key of `Detiler::queues`) and the value of that queue's
    /// fence that says the list is finished.
    queue: usize,
    done_at: u64,
    /// The textures the last list writes. A command list holds no reference of its own.
    held: Vec<ID3D12Resource>,
    staging: Option<(ID3D12Resource, u64)>,
    params: ID3D12Resource,
    params_ptr: *mut u8,
}

/// A fence of one queue and the last value signalled on it.
struct QueueFence {
    fence: ID3D12Fence,
    last: u64,
}

/// The compute pipeline and the lists that record its work.
pub struct Detiler {
    device: ID3D12Device,
    signature: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    /// One fence per queue. A single shared fence went backwards when one queue signalled after
    /// another, and a busy frame looked finished.
    queues: HashMap<usize, QueueFence>,
    frames: Vec<Frame>,
    /// Cached copyable footprints by PC resource geometry and element byte width.
    footprint_cache: FootprintCache,
    /// Scratch buffers reused across flushes and batches.
    scratch: DetileScratch,
}

// SAFETY: D3D12 objects are free threaded, and the mapped pointer is only used under the table lock.
unsafe impl Send for Detiler {}

impl Detiler {
    /// Builds the pipeline on `device`. `None` (logged) when the shader compiler or the pipeline
    /// cannot be had.
    pub fn new(device: &ID3D12Device) -> Option<Detiler> {
        let view_parameter = |kind: D3D12_ROOT_PARAMETER_TYPE| D3D12_ROOT_PARAMETER {
            ParameterType: kind,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        // b0 holds the parameter block, t0 the tiled source and u0 the staging buffer.
        let parameters = [view_parameter(D3D12_ROOT_PARAMETER_TYPE_CBV), view_parameter(D3D12_ROOT_PARAMETER_TYPE_SRV), view_parameter(D3D12_ROOT_PARAMETER_TYPE_UAV)];
        let signature_desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 3, pParameters: parameters.as_ptr(), ..Default::default() };
        // SAFETY: the export is `D3D12SerializeRootSignature`.
        let Some(serialize) = (unsafe { load_export::<SerializeRootSignature>(windows::core::w!("d3d12.dll"), c"D3D12SerializeRootSignature") }) else {
            xs_log!("detile: D3D12SerializeRootSignature is not available");
            return None;
        };
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: the description and its parameters live through the call and `blob` is a valid output.
        let serialized = unsafe { serialize(&signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, std::ptr::null_mut()) };
        let blob = blob.filter(|_| serialized.is_ok())?;
        let shader = compile_detile_shader()?;
        // SAFETY: plain calls on objects created here, the blob owns its bytes.
        unsafe {
            let signature: ID3D12RootSignature = device.CreateRootSignature(0, blob_bytes(&blob)).ok()?;
            let pipeline_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(signature.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr().cast(), BytecodeLength: shader.len() },
                ..Default::default()
            };
            let created: windows::core::Result<ID3D12PipelineState> = device.CreateComputePipelineState(&pipeline_desc);
            drop(ManuallyDrop::into_inner(pipeline_desc.pRootSignature));
            match created {
                Ok(pipeline) => Some(Detiler {
                    device: device.clone(),
                    signature,
                    pipeline,
                    queues: HashMap::new(),
                    frames: Vec::new(),
                    footprint_cache: HashMap::new(),
                    scratch: DetileScratch::default(),
                }),
                Err(e) => {
                    xs_log!("detile: the pipeline cannot be created ({e})");
                    None
                }
            }
        }
    }

    /// A fresh frame: allocator, closed list and a mapped upload buffer for the parameter blocks.
    fn new_frame(&self) -> Option<Frame> {
        let params_bytes = (DISPATCH_SLOTS * PARAM_BYTES) as u64;
        // SAFETY: plain D3D12 calls on the device this detiler owns.
        unsafe {
            let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
            list.Close().ok()?;
            let upload = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
            let mut created: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &buffer_resource(params_bytes, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut created).ok()?;
            let params = created?;
            // Names let the DRED breadcrumbs and allocation lists point at these objects.
            let number = self.frames.len();
            let _ = list.SetName(&HSTRING::from(format!("decant detile list {number}")));
            let _ = params.SetName(&HSTRING::from(format!("decant detile parameters {number}")));
            let mut mapped = std::ptr::null_mut();
            params.Map(0, None, Some(&mut mapped)).ok()?;
            crate::mem_owners::add(crate::mem_owners::Kind::DetileParams, crate::mem_owners::Segment::Unknown, 1, params_bytes);
            Some(Frame { allocator, list, queue: 0, done_at: 0, held: Vec::new(), staging: None, params, params_ptr: mapped.cast() })
        }
    }

    /// The index of a frame the GPU has finished with, preferring one whose staging buffer already
    /// holds `needed` bytes so staging buffers do not grow unnecessarily across frames. A new frame
    /// is made when all are busy.
    fn idle_frame(&mut self, needed: u64) -> Option<usize> {
        let queues = &self.queues;
        // SAFETY: the fences are live.
        let completed = |queue: usize| queues.get(&queue).map_or(0, |q| unsafe { q.fence.GetCompletedValue() });
        if let Some(index) = best_finished(
            self.frames.iter().map(|f| (f.queue, f.done_at, f.staging.as_ref().map_or(0, |(_, s)| *s))),
            needed,
            completed,
        ) {
            self.frames[index].held.clear();
            return Some(index);
        }
        let frame = self.new_frame()?;
        self.frames.push(frame);
        Some(self.frames.len() - 1)
    }

    /// Makes sure the frame's staging buffer holds `bytes`, replacing a smaller one.
    fn ensure_staging(&self, frame: &mut Frame, bytes: u64) -> bool {
        if frame.staging.as_ref().is_some_and(|(_, size)| *size >= bytes) {
            return true;
        }
        let size = bytes.next_power_of_two().max(8 << 20);
        let default_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut created: Option<ID3D12Resource> = None;
        // SAFETY: a plain call, and the buffer it replaces was last used by a list the GPU finished.
        let result = unsafe { self.device.CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer_resource(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut created) };
        let (Ok(()), Some(buffer)) = (result, created) else {
            xs_log!("detile: {size} bytes of staging cannot be created");
            return false;
        };
        // SAFETY: a live resource.
        let _ = unsafe { buffer.SetName(&HSTRING::from(format!("decant detile staging {} MiB", size >> 20))) };
        if let Some((_, old)) = &frame.staging {
            crate::mem_owners::remove(crate::mem_owners::Kind::DetileStaging, crate::mem_owners::Segment::Local, 1, *old);
        }
        crate::mem_owners::add(crate::mem_owners::Kind::DetileStaging, crate::mem_owners::Segment::Local, 1, size);
        frame.staging = Some((buffer, size));
        true
    }

    /// Records one batch on a free frame, executes it on `queue` and signals the queue's fence.
    /// Whether the batch was submitted.
    fn run_batch(&mut self, queue: &ID3D12CommandQueue, batch: &[Prepared<'_>]) -> bool {
        let total: u64 = batch.iter().map(|p| p.staging_bytes).sum();
        let Some(index) = self.idle_frame(total) else { return false };
        // The frame leaves the vector while it is recorded, because recording also needs `self`.
        let mut frame = self.frames.swap_remove(index);
        let submitted = self.ensure_staging(&mut frame, total) && self.record(&mut frame, queue, batch);
        self.frames.push(frame);
        submitted
    }

    fn record(&mut self, frame: &mut Frame, queue: &ID3D12CommandQueue, batch: &[Prepared<'_>]) -> bool {
        let Some((staging, _)) = frame.staging.clone() else { return false };
        let key = queue.as_raw() as usize;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.queues.entry(key) {
            // SAFETY: a plain call on the device this detiler owns.
            let Ok(fence) = (unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }) else { return false };
            slot.insert(QueueFence { fence, last: 0 });
        }
        // SAFETY: the allocator and the list are idle, because the frame's fence value on its queue
        // has passed. The staging buffer and the parameter buffer stay with the frame until then and
        // the textures in `held`. The parameter buffer is mapped for its whole life.
        unsafe {
            if frame.allocator.Reset().is_err() || frame.list.Reset(&frame.allocator, None).is_err() {
                return false;
            }
            let list = &frame.list;
            list.SetComputeRootSignature(&self.signature);
            list.SetPipelineState(&self.pipeline);
            self.scratch.barriers.clear();
            self.scratch.barriers.push(state_change(&staging, 0, 0, uav_state()));
            for p in batch {
                append_copy_barriers(&mut self.scratch.barriers, &p.job.texture, p.job.states, Leg::Enter);
            }
            list.ResourceBarrier(&self.scratch.barriers);
            if !dispatch_all(
                list,
                frame.params.GetGPUVirtualAddress(),
                frame.params_ptr,
                staging.GetGPUVirtualAddress(),
                batch,
                &mut self.scratch.bases,
            ) {
                return false;
            }
            list.ResourceBarrier(&[state_change(&staging, 0, uav_state(), copy_source_state())]);
            copy_rows(list, &staging, batch, &self.scratch.bases);
            self.scratch.barriers.clear();
            self.scratch.barriers.push(state_change(&staging, 0, copy_source_state(), 0));
            for p in batch {
                append_copy_barriers(&mut self.scratch.barriers, &p.job.texture, p.job.states, Leg::Leave);
            }
            list.ResourceBarrier(&self.scratch.barriers);
            if list.Close().is_err() {
                return false;
            }
            let Ok(command_list) = list.cast::<ID3D12CommandList>() else { return false };
            frame.held.clear();
            frame.held.extend(batch.iter().map(|p| p.job.texture.clone()));
            queue.ExecuteCommandLists(&[Some(command_list)]);
            crate::diagnose::removed_after(|| format!("a detile list of {} textures", batch.len()));
            frame.queue = key;
            let Some(entry) = self.queues.get_mut(&key) else { return false };
            entry.last += 1;
            if queue.Signal(&entry.fence, entry.last).is_err() {
                // Nothing tells when the list is finished, so the frame is never used again.
                frame.done_at = u64::MAX;
                return false;
            }
            frame.done_at = entry.last;
        }
        true
    }

    /// Detiles `jobs` on `queue`, in as many lists as the staging and dispatch limits need. The
    /// result says per job whether it was submitted. A job whose texture does not match its layout
    /// is skipped and logged. With `wait` the call returns when the GPU is done.
    pub fn run(&mut self, queue: &ID3D12CommandQueue, jobs: &[Job<'_>], wait: bool) -> Vec<bool> {
        let mut submitted = Vec::new();
        self.run_into(queue, jobs, wait, &mut submitted);
        submitted
    }

    /// As [`Detiler::run`], writing the per job outcomes into `submitted` to reuse its allocation.
    pub fn run_into(&mut self, queue: &ID3D12CommandQueue, jobs: &[Job<'_>], wait: bool, submitted: &mut Vec<bool>) {
        let _trace = decant_trace::span("detile", "run_into");
        submitted.clear();
        submitted.resize(jobs.len(), false);
        let mut prepared = Vec::with_capacity(jobs.len());
        let mut origin = Vec::with_capacity(jobs.len());
        for (index, job) in jobs.iter().enumerate() {
            match prepare_cached(&self.device, &mut self.footprint_cache, &mut self.scratch, job) {
                Ok(p) => {
                    prepared.push(p);
                    origin.push(index);
                }
                Err(why) => {
                    static SKIPS_LOGGED: AtomicU32 = AtomicU32::new(0);
                    if SKIPS_LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
                        xs_log!("detile: a texture is skipped: {why}");
                    }
                }
            }
        }
        self.scratch.costs.clear();
        self.scratch.costs.extend(prepared.iter().map(|p| (p.staging_bytes, p.job.plan.mips.len())));
        let mut batches = std::mem::take(&mut self.scratch.batches);
        group_batches_into(&self.scratch.costs, &mut batches);
        for range in &batches {
            if self.run_batch(queue, &prepared[range.clone()]) {
                for &index in &origin[range.clone()] {
                    submitted[index] = true;
                }
            }
        }
        self.scratch.batches = batches;
        if wait {
            self.wait_idle();
        }
        self.trim_staging(staging_cap_bytes());
    }

    /// Drops staging buffers from idle frames when total retained staging exceeds `cap` (`0`
    /// disables trimming).
    fn trim_staging(&mut self, cap: u64) {
        let queues = &self.queues;
        // SAFETY: the fences are live.
        let completed = |queue: usize| queues.get(&queue).map_or(0, |q| unsafe { q.fence.GetCompletedValue() });
        let indices = select_staging_to_trim(
            self.frames.iter().map(|f| (f.queue, f.done_at, f.staging.as_ref().map_or(0, |(_, s)| *s))),
            completed,
            cap,
        );
        for index in indices {
            if let Some((_, size)) = self.frames[index].staging.take() {
                crate::mem_owners::remove(crate::mem_owners::Kind::DetileStaging, crate::mem_owners::Segment::Local, 1, size);
            }
        }
    }

    /// Frames made so far, queues they ran on and the staging bytes they hold.
    fn footprint(&self) -> (usize, usize, u64) {
        (self.frames.len(), self.queues.len(), self.frames.iter().filter_map(|f| f.staging.as_ref().map(|(_, s)| *s)).sum())
    }

    /// Waits until everything submitted so far is done on the GPU, 5 seconds at most per queue.
    pub fn wait_idle(&self) {
        let _trace = decant_trace::span("detile", "wait_idle");
        for q in self.queues.values() {
            // SAFETY: a live fence, and the event is closed before leaving.
            unsafe {
                if q.fence.GetCompletedValue() >= q.last {
                    continue;
                }
                if let Ok(event) = CreateEventW(None, false, false, PCWSTR::null()) {
                    if q.fence.SetEventOnCompletion(q.last, event).is_ok() {
                        WaitForSingleObject(event, 5000);
                    }
                    let _ = CloseHandle(event);
                }
            }
        }
    }
}

/// Records one dispatch per mip level of every job in the batch and writes the staging offset each
/// job starts at into `bases`. Returns `false` when the batch needs more dispatches than the list
/// has slots.
///
/// # Safety
/// `params_ptr` is the mapped start of an upload buffer of `DISPATCH_SLOTS * PARAM_BYTES` bytes at
/// GPU address `params_va`, and `list` is open.
unsafe fn dispatch_all(
    list: &ID3D12GraphicsCommandList,
    params_va: u64,
    params_ptr: *mut u8,
    staging_va: u64,
    batch: &[Prepared<'_>],
    bases: &mut Vec<u64>,
) -> bool {
    let (mut base, mut slot) = (0u64, 0usize);
    bases.clear();
    bases.reserve(batch.len());
    // SAFETY: `list` is open.
    unsafe { list.SetComputeRootUnorderedAccessView(2, staging_va) };
    for p in batch {
        bases.push(base);
        let plan = p.job.plan;
        let mip_count = plan.mips.len();
        // SAFETY: `list` is open.
        unsafe { list.SetComputeRootShaderResourceView(1, p.job.src_gpu) };
        for (mip, level) in plan.mips.iter().enumerate() {
            if slot >= DISPATCH_SLOTS {
                return false;
            }
            let footprint = &p.footprints[mip];
            let slice_stride = if plan.slices > 1 { p.footprints[mip + mip_count].Offset - footprint.Offset } else { 0 };
            let words = param_block(plan, level, footprint.Footprint.RowPitch, slice_stride as u32, base + footprint.Offset);
            // SAFETY: the slot is below `DISPATCH_SLOTS`, so the block lies in the mapped buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), params_ptr.add(slot * PARAM_BYTES), PARAM_BYTES);
                list.SetComputeRootConstantBufferView(0, params_va + (slot * PARAM_BYTES) as u64);
                list.Dispatch(groups_across(plan, level), level.height_elements.div_ceil(8), plan.slices);
            }
            slot += 1;
        }
        base += p.staging_bytes;
    }
    true
}

/// Records the copies from the staging buffer into every subresource of every job.
///
/// # Safety
/// `list` is open and the textures and the staging buffer are live.
unsafe fn copy_rows(list: &ID3D12GraphicsCommandList, staging: &ID3D12Resource, batch: &[Prepared<'_>], bases: &[u64]) {
    for (p, &base) in batch.iter().zip(bases) {
        for (subresource, footprint) in p.footprints.iter().enumerate() {
            let mut placed = *footprint;
            placed.Offset += base;
            // SAFETY: the locations only borrow the resources for the duration of the call.
            unsafe {
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: borrowed(&p.job.texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: subresource as u32 },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: borrowed(staging),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: placed },
                };
                list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
        }
    }
}

/// The index of the first frame, given as `(queue, done at)`, whose queue fence (`completed`) has
/// reached the value. A frame with value 0 was never used and `u64::MAX` marks a failed signal.
fn first_finished(frames: impl Iterator<Item = (usize, u64)>, completed: impl Fn(usize) -> u64) -> Option<usize> {
    frames.enumerate().find(|&(_, (queue, done_at))| done_at == 0 || (done_at != u64::MAX && completed(queue) >= done_at)).map(|(index, _)| index)
}

/// Chooses an idle frame, given as `(queue, done at, staging capacity)`, whose queue fence
/// (`completed`) has reached `done at`. Prefers the smallest idle frame that already fits `needed`
/// bytes so a batch avoids reallocating a small buffer, else the largest idle frame.
fn best_finished(frames: impl Iterator<Item = (usize, u64, u64)>, needed: u64, completed: impl Fn(usize) -> u64) -> Option<usize> {
    let mut best_fit: Option<(usize, u64)> = None;
    let mut best_any: Option<(usize, u64)> = None;
    for (index, (queue, done_at, cap)) in frames.enumerate() {
        let is_idle = done_at == 0 || (done_at != u64::MAX && completed(queue) >= done_at);
        if !is_idle {
            continue;
        }
        if cap >= needed {
            if best_fit.is_none_or(|(_, best_cap)| cap < best_cap) {
                best_fit = Some((index, cap));
            }
        } else if best_any.is_none_or(|(_, best_cap)| cap > best_cap) {
            best_any = Some((index, cap));
        }
    }
    best_fit.or(best_any).map(|(index, _)| index)
}

/// Chooses idle frame indices whose staging buffers should be dropped so retained staging drops to
/// `cap` or below (largest idle buffers first, `cap` 0 disables trimming).
fn select_staging_to_trim(frames: impl Iterator<Item = (usize, u64, u64)>, completed: impl Fn(usize) -> u64, cap: u64) -> Vec<usize> {
    if cap == 0 {
        return Vec::new();
    }
    let mut total = 0u64;
    let mut idle: Vec<(u64, usize)> = Vec::new();
    for (index, (queue, done_at, size)) in frames.enumerate() {
        total += size;
        let is_idle = done_at == 0 || (done_at != u64::MAX && completed(queue) >= done_at);
        if size > 0 && is_idle {
            idle.push((size, index));
        }
    }
    if total <= cap {
        return Vec::new();
    }
    // Largest staging buffers first so retained memory drops with minimal churn.
    idle.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut drop_indices = Vec::new();
    for (size, index) in idle {
        if total <= cap {
            break;
        }
        total = total.saturating_sub(size);
        drop_indices.push(index);
    }
    drop_indices
}

// ==== the registry of placed textures ====

/// Fast path flag indicating whether at least one registered texture may want a fill.
static HAS_PENDING_FILLS: AtomicBool = AtomicBool::new(false);
/// Fast path counter to skip locks when no placed textures are registered.
static TEXTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
static INDEX_SETTLED_UNMAP_GEN: AtomicU64 = AtomicU64::new(0);
static NOTED_COUNT: AtomicUsize = AtomicUsize::new(0);
static FLUSHES_COUNT: AtomicU64 = AtomicU64::new(0);
static DETILED_COUNT: AtomicU64 = AtomicU64::new(0);
static DEFERRED_COUNT: AtomicU64 = AtomicU64::new(0);
/// Cached initialization state of `DETILER`: 0 = uninitialized, 1 = ready, 2 = unavailable.
static DETILER_READY_STATE: AtomicU8 = AtomicU8::new(0);

const COARSE_TILE_WORDS: usize = 1024;
const COARSE_TILE_BUCKETS: usize = COARSE_TILE_WORDS * 64;
static COARSE_TILE_BITS: [AtomicU64; COARSE_TILE_WORDS] = [const { AtomicU64::new(0) }; COARSE_TILE_WORDS];

const COARSE_ADDR_SHIFT: u32 = 16;
const COARSE_ADDR_WORDS: usize = 1024;
const COARSE_ADDR_BUCKETS: usize = COARSE_ADDR_WORDS * 64;
const COARSE_ADDR_MAX_PAGES: u64 = 4096;
static ADDR_COARSE_BITS: [AtomicU64; COARSE_ADDR_WORDS] = [const { AtomicU64::new(0) }; COARSE_ADDR_WORDS];
static HUGE_ADDR_COUNT: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn coarse_tile_bucket(tile: u32) -> usize {
    ((tile >> 4) as usize) & (COARSE_TILE_BUCKETS - 1)
}

#[inline]
fn coarse_tile_bit(tile: u32) -> (usize, u64) {
    let bucket = coarse_tile_bucket(tile);
    (bucket >> 6, 1u64 << (bucket & 63))
}

#[inline]
fn coarse_addr_bucket(page: u64) -> usize {
    (page as usize) & (COARSE_ADDR_BUCKETS - 1)
}

#[inline]
pub fn may_overlap_tile(tile: u32) -> bool {
    if TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let (word, mask) = coarse_tile_bit(tile);
    (COARSE_TILE_BITS[word].load(Ordering::Relaxed) & mask) != 0
}

/// Filters `tiles` in place to those that may overlap a tracked placed texture.
/// Returns `false` if no placed textures exist or no tiles can overlap (so the caller
/// can skip sorting, deduplicating, and calling `note_tiles` completely).
pub fn filter_overlapping_tiles(tiles: &mut Vec<u32>) -> bool {
    if tiles.is_empty() || !enabled() || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let settled = INDEX_SETTLED_UNMAP_GEN.load(Ordering::Acquire);
    if settled == 0 || settled != crate::title_mem::unmap_generation() {
        return true;
    }
    tiles.retain(|&t| {
        let (word, mask) = coarse_tile_bit(t);
        (COARSE_TILE_BITS[word].load(Ordering::Relaxed) & mask) != 0
    });
    !tiles.is_empty()
}

#[inline]
fn may_overlap_addr(va: u64, end: u64) -> bool {
    if HUGE_ADDR_COUNT.load(Ordering::Acquire) > 0 {
        return true;
    }
    let first_page = va >> COARSE_ADDR_SHIFT;
    let last_page = (end - 1) >> COARSE_ADDR_SHIFT;
    if last_page - first_page >= 64 {
        return true;
    }
    for page in first_page..=last_page {
        let bucket = coarse_addr_bucket(page);
        if (ADDR_COARSE_BITS[bucket >> 6].load(Ordering::Relaxed) & (1u64 << (bucket & 63))) != 0 {
            return true;
        }
    }
    false
}

const FLAG_ALIVE: u64 = 1 << 0;
const FLAG_DIRTY: u64 = 1 << 1;
const FLAG_QUEUED: u64 = 1 << 2;
const ATTEMPTS_SHIFT: u32 = 3;
const ATTEMPTS_MASK: u64 = 0x1F << ATTEMPTS_SHIFT;
const GEN_SHIFT: u32 = 8;
const GEN_STEP: u64 = 1 << GEN_SHIFT;

#[inline]
fn attempts_of(state: u64) -> u32 {
    ((state & ATTEMPTS_MASK) >> ATTEMPTS_SHIFT) as u32
}

#[inline]
fn wants_fill_state(state: u64) -> bool {
    (state & FLAG_ALIVE) != 0 && (state & FLAG_DIRTY) != 0 && attempts_of(state) < FILL_RETRIES
}

/// A placed texture the layer fills, with lock-free atomic fill state so `note_write`,
/// `note_tiles`, and `flush` never take a global registry write lock on the hot path.
struct PlacedEntry {
    res: usize,
    va: u64,
    plan: Arc<Plan>,
    /// PC allocation size, for the informational owner statistics.
    bytes: u64,
    /// The states of the title's submitted work, one for all subresources or one each.
    states: Mutex<Arc<Vec<u32>>>,
    /// Packed atomic state: `FLAG_ALIVE`, `FLAG_DIRTY`, `FLAG_QUEUED`, `attempts`, and `write_gen`.
    fill_state: AtomicU64,
}

impl std::fmt::Debug for PlacedEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlacedEntry").field("res", &self.res).field("va", &self.va).finish()
    }
}

impl PlacedEntry {
    fn new(res: usize, va: u64, plan: Arc<Plan>, bytes: u64, state: u32, dirty: bool) -> Self {
        let initial = FLAG_ALIVE | if dirty { FLAG_DIRTY | GEN_STEP } else { 0 };
        Self { res, va, plan, bytes, states: Mutex::new(Arc::new(vec![state])), fill_state: AtomicU64::new(initial) }
    }

    #[inline]
    fn is_dirty(&self) -> bool {
        (self.fill_state.load(Ordering::Acquire) & FLAG_DIRTY) != 0
    }

    #[inline]
    fn attempts(&self) -> u32 {
        attempts_of(self.fill_state.load(Ordering::Acquire))
    }

    #[inline]
    fn wants_fill(&self) -> bool {
        wants_fill_state(self.fill_state.load(Ordering::Acquire))
    }

    #[inline]
    fn mark_dead(&self) {
        self.fill_state.fetch_and(!(FLAG_ALIVE | FLAG_DIRTY), Ordering::AcqRel);
    }

    /// Marks the texture dirty with a fresh retry budget and advances its write generation.
    /// Returns `true` when `FLAG_QUEUED` transitioned `0 -> 1` so the caller should push it.
    #[inline]
    fn invalidate_mark(&self) -> bool {
        let mut should_enqueue = false;
        let _ = self.fill_state.try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
            if (old & FLAG_ALIVE) == 0 {
                should_enqueue = false;
                return None;
            }
            should_enqueue = (old & FLAG_QUEUED) == 0;
            let cleared = old & !ATTEMPTS_MASK;
            Some((cleared | FLAG_DIRTY | FLAG_QUEUED).wrapping_add(GEN_STEP))
        });
        should_enqueue
    }

    /// Sets `FLAG_QUEUED` if the entry is still alive, dirty, below retry limit, and not queued.
    #[inline]
    fn try_set_queued(&self) -> bool {
        self.fill_state
            .try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                if !wants_fill_state(old) || (old & FLAG_QUEUED) != 0 {
                    None
                } else {
                    Some(old | FLAG_QUEUED)
                }
            })
            .is_ok()
    }

    /// Clears `FLAG_QUEUED` upon dequeue in `flush` and returns the current `write_gen` if the
    /// entry still wants a fill.
    #[inline]
    fn dequeue_for_flush(&self) -> Option<u64> {
        let prev = self.fill_state.fetch_and(!FLAG_QUEUED, Ordering::AcqRel);
        let now = prev & !FLAG_QUEUED;
        wants_fill_state(now).then_some(now >> GEN_SHIFT)
    }

    /// Increments retry attempts after an unreachable span if no newer invalidation arrived.
    /// Returns `true` when the caller should re-enqueue for the next flush.
    #[inline]
    fn record_unreachable(&self, snap_gen: u64) -> bool {
        let mut need_enqueue = false;
        let _ = self.fill_state.try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
            if (old & FLAG_ALIVE) == 0 || (old & FLAG_DIRTY) == 0 || (old >> GEN_SHIFT) != snap_gen {
                need_enqueue = false;
                return None;
            }
            let att = attempts_of(old);
            if att >= FILL_RETRIES {
                need_enqueue = false;
                return None;
            }
            let next_att = att + 1;
            let updated = (old & !ATTEMPTS_MASK) | (u64::from(next_att) << ATTEMPTS_SHIFT);
            if next_att < FILL_RETRIES && (updated & FLAG_QUEUED) == 0 {
                need_enqueue = true;
                Some(updated | FLAG_QUEUED)
            } else {
                need_enqueue = false;
                Some(updated)
            }
        });
        need_enqueue
    }

    /// Records the outcome of a GPU fill for `snap_gen`. A concurrent invalidation advances
    /// `write_gen`, so this never clears a newer write's dirty flag.
    #[inline]
    fn settle_gen(&self, snap_gen: u64, filled: bool) {
        let _ = self.fill_state.try_update(Ordering::AcqRel, Ordering::Acquire, |old| {
            if (old & FLAG_ALIVE) == 0 || (old >> GEN_SHIFT) != snap_gen {
                return None;
            }
            if filled {
                Some(old & !FLAG_DIRTY)
            } else {
                Some((old & !ATTEMPTS_MASK) | (u64::from(FILL_RETRIES) << ATTEMPTS_SHIFT))
            }
        });
    }
}

const TEXTURE_SHARDS: usize = 16;

#[inline]
fn texture_shard(res: usize) -> usize {
    (res >> 4) & (TEXTURE_SHARDS - 1)
}

type TextureShardMap = FastMap<usize, Arc<PlacedEntry>>;
static TEXTURES: [RwLock<Option<TextureShardMap>>; TEXTURE_SHARDS] = [const { RwLock::new(None) }; TEXTURE_SHARDS];

#[inline]
fn with_texture_shard_read<R>(res: usize, f: impl FnOnce(&TextureShardMap) -> R) -> Option<R> {
    let guard = TEXTURES[texture_shard(res)].read().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map(f)
}

#[inline]
fn with_texture_shard_write<R>(res: usize, f: impl FnOnce(&mut TextureShardMap) -> R) -> R {
    let mut guard = TEXTURES[texture_shard(res)].write().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(FastMap::default))
}

#[inline]
fn is_placed(res: usize) -> bool {
    with_texture_shard_read(res, |m| m.contains_key(&res)) == Some(true)
}

fn lookup_placed_entry(res: usize) -> Option<Arc<PlacedEntry>> {
    with_texture_shard_read(res, |m| m.get(&res).cloned()).flatten()
}

/// Queue of entries waiting for a detile flush. Each entry appears at most once because
/// `FLAG_QUEUED` gates insertion.
static PENDING_FILLS: Mutex<Vec<Arc<PlacedEntry>>> = Mutex::new(Vec::new());

fn enqueue_pending_single(entry: Arc<PlacedEntry>) {
    let mut q = decant_trace::timed_lock("detile::PENDING", || PENDING_FILLS.lock().unwrap_or_else(|e| e.into_inner()));
    q.push(entry);
    HAS_PENDING_FILLS.store(true, Ordering::Release);
}

fn enqueue_pending_batch(entries: impl Iterator<Item = Arc<PlacedEntry>>) {
    let mut iter = entries.peekable();
    if iter.peek().is_none() {
        return;
    }
    let mut q = decant_trace::timed_lock("detile::PENDING", || PENDING_FILLS.lock().unwrap_or_else(|e| e.into_inner()));
    q.extend(iter);
    HAS_PENDING_FILLS.store(true, Ordering::Release);
}

fn drain_pending_entries(out: &mut Vec<Arc<PlacedEntry>>) {
    out.clear();
    let mut q = decant_trace::timed_lock("detile::PENDING", || PENDING_FILLS.lock().unwrap_or_else(|e| e.into_inner()));
    std::mem::swap(&mut *q, out);
    HAS_PENDING_FILLS.store(false, Ordering::Release);
}

fn invalidate_entries(entries: &[Arc<PlacedEntry>]) {
    if entries.is_empty() {
        return;
    }
    let mut to_enqueue: Vec<Arc<PlacedEntry>> = Vec::new();
    for entry in entries {
        if entry.invalidate_mark() {
            to_enqueue.push(Arc::clone(entry));
        }
    }
    if !to_enqueue.is_empty() {
        enqueue_pending_batch(to_enqueue.into_iter());
    }
}

// ==== size-tiered address index for O(log N) `note_write` ====

const SMALL_ADDR_MAX: u64 = 256 << 10;
const MEDIUM_ADDR_MAX: u64 = 4 << 20;

struct AddrEntry {
    res: usize,
    va: u64,
    end: u64,
    entry: Arc<PlacedEntry>,
}

struct AddrMeta {
    va: u64,
    size: u64,
    huge: bool,
}

/// Separate address index so `note_write` only takes a read lock here and never contends with
/// `flush`, `note_tiles`, `submitted`, or `note_transitions`.
struct AddressIndex {
    small: BTreeMap<u64, Vec<AddrEntry>>,
    small_largest: u64,
    medium: BTreeMap<u64, Vec<AddrEntry>>,
    medium_largest: u64,
    large: Vec<AddrEntry>,
    by_res: FastMap<usize, AddrMeta>,
    coarse_counts: Vec<u32>,
    sync_globals: bool,
}

impl Default for AddressIndex {
    fn default() -> Self {
        Self::new(false)
    }
}

impl AddressIndex {
    fn new(sync_globals: bool) -> Self {
        Self {
            small: BTreeMap::new(),
            small_largest: 0,
            medium: BTreeMap::new(),
            medium_largest: 0,
            large: Vec::new(),
            by_res: FastMap::default(),
            coarse_counts: Vec::new(),
            sync_globals,
        }
    }

    fn update_coarse_bits(&mut self, va: u64, size: u64, add: bool) -> bool {
        if size == 0 {
            return false;
        }
        let end = va.saturating_add(size);
        let first_page = va >> COARSE_ADDR_SHIFT;
        let last_page = (end - 1) >> COARSE_ADDR_SHIFT;
        let huge = last_page - first_page >= COARSE_ADDR_MAX_PAGES;
        if !self.sync_globals {
            return huge;
        }
        if huge {
            if add {
                HUGE_ADDR_COUNT.fetch_add(1, Ordering::AcqRel);
            } else {
                HUGE_ADDR_COUNT.fetch_sub(1, Ordering::AcqRel);
            }
            return true;
        }
        if self.coarse_counts.is_empty() {
            if !add {
                return false;
            }
            self.coarse_counts.resize(COARSE_ADDR_BUCKETS, 0);
        }
        for page in first_page..=last_page {
            let bucket = coarse_addr_bucket(page);
            if add {
                let c = &mut self.coarse_counts[bucket];
                if *c == 0 {
                    ADDR_COARSE_BITS[bucket >> 6].fetch_or(1u64 << (bucket & 63), Ordering::Release);
                }
                *c += 1;
            } else {
                let c = &mut self.coarse_counts[bucket];
                if *c > 0 {
                    *c -= 1;
                    if *c == 0 {
                        ADDR_COARSE_BITS[bucket >> 6].fetch_and(!(1u64 << (bucket & 63)), Ordering::Release);
                    }
                }
            }
        }
        false
    }

    fn insert(&mut self, res: usize, va: u64, size: u64, entry: Arc<PlacedEntry>) {
        self.remove(res);
        let end = va.saturating_add(size);
        let huge = self.update_coarse_bits(va, size, true);
        self.by_res.insert(res, AddrMeta { va, size, huge });
        let item = AddrEntry { res, va, end, entry };
        if size <= SMALL_ADDR_MAX {
            self.small.entry(va).or_default().push(item);
            self.small_largest = self.small_largest.max(size);
        } else if size <= MEDIUM_ADDR_MAX {
            self.medium.entry(va).or_default().push(item);
            self.medium_largest = self.medium_largest.max(size);
        } else {
            self.large.push(item);
        }
    }

    fn remove(&mut self, res: usize) -> bool {
        let Some(meta) = self.by_res.remove(&res) else { return false };
        if self.sync_globals {
            if meta.huge {
                HUGE_ADDR_COUNT.fetch_sub(1, Ordering::AcqRel);
            } else {
                self.update_coarse_bits(meta.va, meta.size, false);
            }
        }
        if meta.size <= SMALL_ADDR_MAX {
            if let Some(list) = self.small.get_mut(&meta.va) {
                list.retain(|e| e.res != res);
                if list.is_empty() {
                    self.small.remove(&meta.va);
                }
            }
        } else if meta.size <= MEDIUM_ADDR_MAX {
            if let Some(list) = self.medium.get_mut(&meta.va) {
                list.retain(|e| e.res != res);
                if list.is_empty() {
                    self.medium.remove(&meta.va);
                }
            }
        } else {
            self.large.retain(|e| e.res != res);
        }
        true
    }

    fn collect_overlapping(&self, start: u64, end: u64, out: &mut Vec<Arc<PlacedEntry>>) {
        out.clear();
        if !self.small.is_empty() {
            for (_, list) in self.small.range(start.saturating_sub(self.small_largest)..end) {
                for e in list {
                    if start < e.end && e.va < end {
                        out.push(Arc::clone(&e.entry));
                    }
                }
            }
        }
        if !self.medium.is_empty() {
            for (_, list) in self.medium.range(start.saturating_sub(self.medium_largest)..end) {
                for e in list {
                    if start < e.end && e.va < end {
                        out.push(Arc::clone(&e.entry));
                    }
                }
            }
        }
        for e in &self.large {
            if start < e.end && e.va < end {
                out.push(Arc::clone(&e.entry));
            }
        }
    }

    #[cfg(test)]
    fn overlapping(&self, start: u64, end: u64) -> Vec<usize> {
        let mut entries = Vec::new();
        self.collect_overlapping(start, end, &mut entries);
        entries.into_iter().map(|e| e.res).collect()
    }
}

static ADDR_INDEX: RwLock<Option<AddressIndex>> = RwLock::new(None);

fn with_addr_write<R>(f: impl FnOnce(&mut AddressIndex) -> R) -> R {
    let mut guard = decant_trace::timed_lock("detile::ADDR_WRITE", || ADDR_INDEX.write().unwrap_or_else(|e| e.into_inner()));
    f(guard.get_or_insert_with(|| AddressIndex::new(true)))
}

fn with_addr_read<R>(f: impl FnOnce(&AddressIndex) -> R) -> Option<R> {
    let guard = decant_trace::timed_lock("detile::ADDR_READ", || ADDR_INDEX.read().unwrap_or_else(|e| e.into_inner()));
    guard.as_ref().map(f)
}

// ==== reverse tile index for `note_tiles` ====

enum TileRefs {
    One(usize),
    Two([usize; 2]),
    Many(Vec<usize>),
}

impl TileRefs {
    #[inline]
    fn push(&mut self, res: usize) {
        match self {
            TileRefs::One(a) => *self = TileRefs::Two([*a, res]),
            TileRefs::Two([a, b]) => *self = TileRefs::Many(vec![*a, *b, res]),
            TileRefs::Many(v) => v.push(res),
        }
    }

    #[inline]
    fn retain_not(&mut self, res: usize) -> bool {
        match self {
            TileRefs::One(a) => *a == res,
            TileRefs::Two([a, b]) => {
                if *a == res {
                    *self = TileRefs::One(*b);
                } else if *b == res {
                    *self = TileRefs::One(*a);
                }
                false
            }
            TileRefs::Many(v) => {
                v.retain(|&r| r != res);
                v.is_empty()
            }
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        match self {
            TileRefs::One(a) => std::slice::from_ref(a),
            TileRefs::Two(pair) => pair.as_slice(),
            TileRefs::Many(v) => v.as_slice(),
        }
    }
}

struct TileEntry {
    va: u64,
    size: u64,
    entry: Arc<PlacedEntry>,
    tiles: Option<Vec<u32>>,
}

struct PrecomputedFullIndex {
    tile_to_textures: FastMap<u32, TileRefs>,
    unmapped_textures: FastSet<usize>,
    coarse_counts: Vec<u32>,
    coarse_bits: [u64; COARSE_TILE_WORDS],
}

#[derive(Debug)]
enum IndexPlan {
    Full {
        unmap_gen: u64,
        items: Vec<(usize, u64, u64, Arc<PlacedEntry>)>,
        spans: Vec<(u64, u64)>,
        cap_hint: usize,
    },
    Incremental {
        unmap_gen: u64,
        items: Vec<(usize, u64, u64, Arc<PlacedEntry>)>,
        spans: Vec<(u64, u64)>,
    },
}

/// Reverse index from mirror tiles to placed textures, protected by its own `RwLock` so fast-path
/// `note_tiles` lookups only take a read lock and never contend with `flush`, `note_write`, or `submitted`.
struct TileIndex {
    entries: FastMap<usize, TileEntry>,
    tile_to_textures: FastMap<u32, TileRefs>,
    unmapped_textures: FastSet<usize>,
    pending_index: FastSet<usize>,
    layout_gen: u64,
    last_map_gen: u64,
    last_tile_gen: u64,
    coarse_counts: Vec<u32>,
    sync_globals: bool,
}

impl Default for TileIndex {
    fn default() -> Self {
        Self::new(false)
    }
}

impl TileIndex {
    fn new(sync_globals: bool) -> Self {
        Self {
            entries: FastMap::default(),
            tile_to_textures: FastMap::default(),
            unmapped_textures: FastSet::default(),
            pending_index: FastSet::default(),
            layout_gen: 0,
            last_map_gen: 0,
            last_tile_gen: 0,
            coarse_counts: Vec::new(),
            sync_globals,
        }
    }

    #[inline]
    fn add_coarse_tile(&mut self, tile: u32) {
        if self.coarse_counts.is_empty() {
            self.coarse_counts.resize(COARSE_TILE_BUCKETS, 0);
        }
        let bucket = coarse_tile_bucket(tile);
        let c = &mut self.coarse_counts[bucket];
        if *c == 0 && self.sync_globals {
            COARSE_TILE_BITS[bucket >> 6].fetch_or(1u64 << (bucket & 63), Ordering::Relaxed);
        }
        *c += 1;
    }

    #[inline]
    fn remove_coarse_tile(&mut self, tile: u32) {
        if self.coarse_counts.is_empty() {
            return;
        }
        let bucket = coarse_tile_bucket(tile);
        let c = &mut self.coarse_counts[bucket];
        if *c > 0 {
            *c -= 1;
            if *c == 0 && self.sync_globals {
                COARSE_TILE_BITS[bucket >> 6].fetch_and(!(1u64 << (bucket & 63)), Ordering::Relaxed);
            }
        }
    }

    fn update_settled_gen(&self) {
        if !self.sync_globals {
            return;
        }
        let settled = if !self.entries.is_empty() && self.pending_index.is_empty() && self.unmapped_textures.is_empty() {
            self.layout_gen
        } else {
            0
        };
        INDEX_SETTLED_UNMAP_GEN.store(settled, Ordering::Release);
    }

    fn insert(&mut self, res: usize, va: u64, size: u64, entry: Arc<PlacedEntry>) {
        self.remove(res);
        self.entries.insert(res, TileEntry { va, size, entry, tiles: None });
        self.pending_index.insert(res);
        if self.sync_globals {
            INDEX_SETTLED_UNMAP_GEN.store(0, Ordering::Release);
        }
    }

    fn remove(&mut self, res: usize) -> bool {
        let Some(te) = self.entries.remove(&res) else { return false };
        self.pending_index.remove(&res);
        self.unmapped_textures.remove(&res);
        if let Some(tiles) = &te.tiles {
            for &tile in tiles {
                if let Some(refs) = self.tile_to_textures.get_mut(&tile)
                    && refs.retain_not(res)
                {
                    self.tile_to_textures.remove(&tile);
                    self.remove_coarse_tile(tile);
                }
            }
        }
        self.update_settled_gen();
        true
    }

    fn remove_many(&mut self, resources: &[usize]) {
        let mut any = false;
        for &res in resources {
            if self.remove(res) {
                any = true;
            }
        }
        if any {
            self.update_settled_gen();
        }
    }

    #[inline]
    fn needs_update(&self, current_unmap: u64, current_map: u64, current_tile: u64) -> bool {
        if self.layout_gen != current_unmap || !self.pending_index.is_empty() {
            return true;
        }
        (current_tile != self.last_tile_gen || current_map != self.last_map_gen) && !self.unmapped_textures.is_empty()
    }

    fn plan_index_update(&self, current_unmap: u64, current_map: u64, current_tile: u64) -> Option<IndexPlan> {
        if self.layout_gen != current_unmap {
            let items: Vec<(usize, u64, u64, Arc<PlacedEntry>)> = self.entries.iter().map(|(&res, t)| (res, t.va, t.size, Arc::clone(&t.entry))).collect();
            let spans: Vec<(u64, u64)> = items.iter().map(|&(_, va, len, _)| (va, len)).collect();
            return Some(IndexPlan::Full {
                unmap_gen: current_unmap,
                items,
                spans,
                cap_hint: self.tile_to_textures.len(),
            });
        }
        let check_unmapped = (current_tile != self.last_tile_gen || current_map != self.last_map_gen) && !self.unmapped_textures.is_empty();
        if !self.pending_index.is_empty() || check_unmapped {
            let mut items = Vec::with_capacity(self.pending_index.len() + if check_unmapped { self.unmapped_textures.len() } else { 0 });
            for &res in &self.pending_index {
                if let Some(t) = self.entries.get(&res) {
                    items.push((res, t.va, t.size, Arc::clone(&t.entry)));
                }
            }
            if check_unmapped {
                for &res in &self.unmapped_textures {
                    if !self.pending_index.contains(&res)
                        && let Some(t) = self.entries.get(&res)
                    {
                        items.push((res, t.va, t.size, Arc::clone(&t.entry)));
                    }
                }
            }
            let spans: Vec<(u64, u64)> = items.iter().map(|&(_, va, len, _)| (va, len)).collect();
            return Some(IndexPlan::Incremental { unmap_gen: current_unmap, items, spans });
        }
        None
    }

    fn build_full_index_maps(
        items: &[(usize, u64, u64, Arc<PlacedEntry>)],
        tiles_per_item: &[Option<Vec<u32>>],
        capacity_hint: usize,
    ) -> PrecomputedFullIndex {
        let mut tile_to_textures: FastMap<u32, TileRefs> = FastMap::with_capacity_and_hasher(capacity_hint.max(items.len() * 2), Default::default());
        let mut unmapped_textures: FastSet<usize> = FastSet::default();
        let mut coarse_counts = vec![0u32; COARSE_TILE_BUCKETS];
        let mut coarse_bits = [0u64; COARSE_TILE_WORDS];
        for ((res, _, _, _), maybe_tiles) in items.iter().zip(tiles_per_item) {
            match maybe_tiles {
                Some(tiles) => {
                    for &tile in tiles {
                        match tile_to_textures.entry(tile) {
                            std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().push(*res),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(TileRefs::One(*res));
                                let bucket = coarse_tile_bucket(tile);
                                if coarse_counts[bucket] == 0 {
                                    coarse_bits[bucket >> 6] |= 1u64 << (bucket & 63);
                                }
                                coarse_counts[bucket] += 1;
                            }
                        }
                    }
                }
                None => {
                    unmapped_textures.insert(*res);
                }
            }
        }
        PrecomputedFullIndex { tile_to_textures, unmapped_textures, coarse_counts, coarse_bits }
    }

    fn apply_precomputed_full_index(
        &mut self,
        items: &[(usize, u64, u64, Arc<PlacedEntry>)],
        tiles_per_item: Vec<Option<Vec<u32>>>,
        mut precomputed: PrecomputedFullIndex,
        (layout_gen, map_gen, tile_gen): (u64, u64, u64),
        layout_current: bool,
    ) -> (FastMap<u32, TileRefs>, FastSet<usize>) {
        std::mem::swap(&mut self.tile_to_textures, &mut precomputed.tile_to_textures);
        std::mem::swap(&mut self.unmapped_textures, &mut precomputed.unmapped_textures);
        std::mem::swap(&mut self.coarse_counts, &mut precomputed.coarse_counts);
        if self.sync_globals {
            for (slot, &val) in COARSE_TILE_BITS.iter().zip(&precomputed.coarse_bits) {
                slot.store(val, Ordering::Relaxed);
            }
        }
        let mut live_matched = 0usize;
        for ((res, _, _, entry), maybe_tiles) in items.iter().zip(tiles_per_item) {
            let Some(te) = self.entries.get_mut(res).filter(|t| Arc::ptr_eq(&t.entry, entry)) else {
                continue;
            };
            live_matched += 1;
            self.pending_index.remove(res);
            te.tiles = maybe_tiles;
        }
        if layout_current && live_matched == self.entries.len() {
            self.layout_gen = layout_gen;
            self.last_map_gen = map_gen;
            self.last_tile_gen = tile_gen;
        } else {
            self.layout_gen = 0;
        }
        self.update_settled_gen();
        (precomputed.tile_to_textures, precomputed.unmapped_textures)
    }

    #[cfg(test)]
    fn apply_full_index(
        &mut self,
        items: &[(usize, u64, u64, Arc<PlacedEntry>)],
        tiles_per_item: &[Option<Vec<u32>>],
        layout_gen: u64,
        map_gen: u64,
        tile_gen: u64,
        layout_current: bool,
    ) {
        let precomputed = Self::build_full_index_maps(items, tiles_per_item, self.tile_to_textures.len());
        let _ = self.apply_precomputed_full_index(items, tiles_per_item.to_vec(), precomputed, (layout_gen, map_gen, tile_gen), layout_current);
    }

    fn apply_incremental_index(
        &mut self,
        items: &[(usize, u64, u64, Arc<PlacedEntry>)],
        tiles_per_item: &[Option<Vec<u32>>],
        map_gen: u64,
        map_current: bool,
        tile_gen: u64,
        tile_current: bool,
    ) {
        for ((res, _, _, entry), maybe_tiles) in items.iter().zip(tiles_per_item) {
            let Some(te) = self.entries.get_mut(res).filter(|t| Arc::ptr_eq(&t.entry, entry)) else {
                continue;
            };
            self.pending_index.remove(res);
            self.unmapped_textures.remove(res);
            let old_tiles = te.tiles.take();
            if let Some(old_tiles) = old_tiles {
                for tile in old_tiles {
                    if let Some(refs) = self.tile_to_textures.get_mut(&tile)
                        && refs.retain_not(*res)
                    {
                        self.tile_to_textures.remove(&tile);
                        self.remove_coarse_tile(tile);
                    }
                }
            }
            match maybe_tiles {
                Some(tiles) => {
                    for &tile in tiles {
                        match self.tile_to_textures.entry(tile) {
                            std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().push(*res),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                e.insert(TileRefs::One(*res));
                                self.add_coarse_tile(tile);
                            }
                        }
                    }
                    if let Some(te) = self.entries.get_mut(res) {
                        te.tiles = Some(tiles.clone());
                    }
                }
                None => {
                    self.unmapped_textures.insert(*res);
                }
            }
        }
        if map_current {
            self.last_map_gen = map_gen;
        }
        if tile_current {
            self.last_tile_gen = tile_gen;
        }
        self.update_settled_gen();
    }

    /// Collects unique entries whose mirror tiles appear in `tiles` under a shared read lock.
    fn collect_hits(&self, tiles: &[u32], report: bool, hit_res: &mut Vec<usize>, out: &mut Vec<Arc<PlacedEntry>>) {
        hit_res.clear();
        out.clear();
        if !self.tile_to_textures.is_empty() {
            for &tile in tiles {
                if let Some(refs) = self.tile_to_textures.get(&tile) {
                    hit_res.extend_from_slice(refs.as_slice());
                }
            }
            if hit_res.len() > 1 {
                hit_res.sort_unstable();
                hit_res.dedup();
            }
            for &res in hit_res.iter() {
                if let Some(te) = self.entries.get(&res) {
                    out.push(Arc::clone(&te.entry));
                }
            }
        }
        if report {
            let waiting = self.entries.values().filter(|t| t.entry.is_dirty() && t.entry.attempts() == 0).count();
            let filled = self.entries.values().filter(|t| !t.entry.is_dirty()).count() as u32;
            let unmapped = self.unmapped_textures.len() as u32;
            let first_unmapped = self.unmapped_textures.iter().next().and_then(|res| self.entries.get(res)).map_or(0, |t| t.va);
            let marked = out.len() as u32;
            let untouched = (self.entries.len() as u32).saturating_sub(marked + unmapped);
            xs_log!(
                "detile: {} textures, {} pages written: {marked} marked, {waiting} wait for a fill, {untouched} not written ({filled} of them filled), {unmapped} with pages that are not mapped (first at {first_unmapped:#x})",
                self.entries.len(),
                tiles.len()
            );
        }
    }
}

static TILE_INDEX: RwLock<Option<TileIndex>> = RwLock::new(None);
static INDEX_REBUILD_LOCK: Mutex<()> = Mutex::new(());

fn with_tile_write<R>(f: impl FnOnce(&mut TileIndex) -> R) -> R {
    let mut guard = decant_trace::timed_lock("detile::TILE_WRITE", || TILE_INDEX.write().unwrap_or_else(|e| e.into_inner()));
    f(guard.get_or_insert_with(|| TileIndex::new(true)))
}

fn with_tile_read<R>(f: impl FnOnce(&TileIndex) -> R) -> Option<R> {
    let guard = decant_trace::timed_lock("detile::TILE_READ", || TILE_INDEX.read().unwrap_or_else(|e| e.into_inner()));
    guard.as_ref().map(f)
}

/// Experiments for the removal of the device with block compressed textures, set by environment.
#[derive(Clone, Copy)]
struct Tuning {
    /// `DECANT_D3D12X_DETILE_DEFER=1`: a texture whose span got tile mappings that the next sync
    /// applies waits for that sync.
    defer_fresh_spans: bool,
    /// `DECANT_D3D12X_DETILE_WAIT=1`: every flush waits until the GPU is done with its lists.
    wait_each_flush: bool,
}

fn tuning() -> Tuning {
    static TUNING: OnceLock<Tuning> = OnceLock::new();
    *TUNING.get_or_init(|| {
        let is_set = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
        let t = Tuning { defer_fresh_spans: is_set("DECANT_D3D12X_DETILE_DEFER"), wait_each_flush: is_set("DECANT_D3D12X_DETILE_WAIT") };
        if t.defer_fresh_spans || t.wait_each_flush {
            xs_log!("detile: deferral of fresh spans {}, waiting for each flush {}", t.defer_fresh_spans, t.wait_each_flush);
        }
        t
    })
}

struct DetilerState {
    detiler: Option<Detiler>,
    unavailable: bool,
}

static DETILER: Mutex<DetilerState> = Mutex::new(DetilerState { detiler: None, unavailable: false });

fn with_detiler<R>(f: impl FnOnce(&mut DetilerState) -> R) -> R {
    let mut guard = decant_trace::timed_lock("detile::DETILER", || DETILER.lock().unwrap_or_else(|e| e.into_inner()));
    f(&mut guard)
}

const NOTED_SHARDS: usize = 16;

#[inline]
fn noted_shard(list: usize) -> usize {
    (list >> 4) & (NOTED_SHARDS - 1)
}

#[derive(Default)]
struct NotedShard {
    lists: FastMap<usize, Vec<(usize, u32, u32)>>,
}

static NOTED: [Mutex<Option<NotedShard>>; NOTED_SHARDS] = [const { Mutex::new(None) }; NOTED_SHARDS];

fn with_noted_shard<R>(list: usize, f: impl FnOnce(&mut NotedShard) -> R) -> R {
    let mut guard = NOTED[noted_shard(list)].lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(NotedShard::default))
}

/// One line about the detile state for the device removal report. It never blocks, a busy shard
/// is reported as such.
pub fn summary() -> String {
    let mut textures_len = 0usize;
    let mut dirty = 0usize;
    let mut retried = 0usize;
    for shard in &TEXTURES {
        let Ok(guard) = shard.try_read() else { return "detile: the table is busy".into() };
        if let Some(map) = guard.as_ref() {
            textures_len += map.len();
            for entry in map.values() {
                if entry.is_dirty() {
                    dirty += 1;
                }
                if entry.wants_fill() {
                    retried += 1;
                }
            }
        }
    }
    let flushes = FLUSHES_COUNT.load(Ordering::Relaxed);
    let detiled = DETILED_COUNT.load(Ordering::Relaxed);
    let deferred = DEFERRED_COUNT.load(Ordering::Relaxed);
    let (frames, queues, staging) = DETILER
        .try_lock()
        .ok()
        .and_then(|d| d.detiler.as_ref().map(Detiler::footprint))
        .unwrap_or((0, 0, 0));
    format!(
        "detile: {textures_len} textures ({dirty} dirty, {retried} of them still tried), {flushes} flushes, {detiled} detiled, {deferred} put off, {frames} lists on {queues} queues with {} MiB of staging",
        staging >> 20
    )
}

/// Whether the title selected the version 4 swizzle equations. `xg_xs.dll` keeps the flag.
pub fn hardware_v4() -> bool {
    // SAFETY: looks up a module that may or may not be loaded, and an export of it by name.
    unsafe {
        let module = GetModuleHandleW(windows_sys::w!("xg_xs.dll"));
        if module.is_null() {
            return false;
        }
        match GetProcAddress(module, c"DecantXgHardwareV4".as_ptr().cast()) {
            Some(export) => std::mem::transmute::<unsafe extern "system" fn() -> isize, unsafe extern "system" fn() -> u32>(export)() != 0,
            None => false,
        }
    }
}

fn le_u32(raw: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(raw[at..at + 4].try_into().unwrap_or_default())
}

static LOGGED_KINDS_COUNT: AtomicUsize = AtomicUsize::new(0);
static LOGGED_KINDS: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
static LOGGED_REFUSED_COUNT: AtomicUsize = AtomicUsize::new(0);
static LOGGED_REFUSED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// The title placed a texture at `va` (device slots 44 and 133). `res` is the PC resource, `raw` the
/// description the title passed, `state` the initial state and `bytes` the PC allocation size. A
/// texture whose layout cannot be detiled is logged with its reason.
pub fn register(res: usize, va: u64, raw: &[u8; 0x38], state: u32, bytes: u64) {
    if !enabled() || va == 0 || res == 0 {
        return;
    }
    let v4 = hardware_v4();
    match detile_plan::plan_shared(raw, v4) {
        Ok(plan) => {
            let format = le_u32(raw, 0x20);
            if LOGGED_KINDS_COUNT.load(Ordering::Relaxed) < 64 {
                let mut guard = LOGGED_KINDS.lock().unwrap_or_else(|e| e.into_inner());
                let set = guard.get_or_insert_with(HashSet::new);
                if set.len() < 64 && set.insert((format, plan.mode)) {
                    LOGGED_KINDS_COUNT.store(set.len(), Ordering::Relaxed);
                    let width = u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap_or_default());
                    xs_log!(
                        "detile: a texture to detile, ok format {format} mode {} {width}x{} mips {} slices {} v4 {v4} at {va:#x}, {} bytes",
                        plan.mode,
                        le_u32(raw, 0x18),
                        plan.mips.len(),
                        plan.slices,
                        plan.size
                    );
                }
            }
            let size = plan.size;
            let entry = Arc::new(PlacedEntry::new(res, va, plan, bytes, state, true));
            let old = with_texture_shard_write(res, |shard| {
                let prev = shard.insert(res, Arc::clone(&entry));
                if prev.is_none() {
                    TEXTURE_COUNT.fetch_add(1, Ordering::AcqRel);
                }
                prev
            });
            if let Some(prev) = old {
                prev.mark_dead();
                crate::mem_owners::remove(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, 1, prev.bytes);
            }
            crate::mem_owners::add(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, 1, bytes);
            with_addr_write(|addr| addr.insert(res, va, size, Arc::clone(&entry)));
            with_tile_write(|ti| ti.insert(res, va, size, Arc::clone(&entry)));
            if entry.try_set_queued() {
                enqueue_pending_single(entry);
            }
        }
        Err(why) => {
            if LOGGED_REFUSED_COUNT.load(Ordering::Relaxed) >= 64 {
                return;
            }
            let reason = why.split(':').next().unwrap_or("");
            let mut guard = LOGGED_REFUSED.lock().unwrap_or_else(|e| e.into_inner());
            let set = guard.get_or_insert_with(HashSet::new);
            if set.len() < 64 && !set.contains(reason) {
                set.insert(reason.to_string());
                LOGGED_REFUSED_COUNT.store(set.len(), Ordering::Relaxed);
                xs_log!("detile: a placed texture at {va:#x} is not detiled: {why}");
            }
        }
    }
}

/// What the registry knows about PC resource `res`, for the descriptor diagnostics: its title
/// address and size, whether its pages are mapped, whether it waits for a fill and how many of the
/// first 64 KiB of its bytes are not zero. `None` for a resource the registry does not hold.
pub fn describe(res: usize) -> Option<String> {
    let entry = lookup_placed_entry(res)?;
    let (va, size, mode, dirty, attempts) = (entry.va, entry.plan.size, entry.plan.mode, entry.is_dirty(), entry.attempts());
    let mapped = crate::title_mem::tiles_of(va, size).is_some();
    let probe = size.min(64 << 10) as usize;
    let content = if crate::title_mem::writable(va as usize, probe) {
        // SAFETY: `writable` says that the `probe` bytes at `va` are committed title memory.
        let bytes = unsafe { std::slice::from_raw_parts(va as *const u8, probe) };
        format!("{} of the first {probe} bytes not zero", bytes.iter().filter(|&&b| b != 0).count())
    } else {
        "bytes not readable".into()
    };
    let fill = if dirty { format!("waits for a fill ({attempts} tries)") } else { "filled".into() };
    Some(format!("detile va {va:#x} {size} bytes mode {mode}, pages {}, {fill}, {content}", if mapped { "mapped" } else { "NOT mapped" }))
}

fn remove_resource_everywhere(res: usize) -> bool {
    if !is_placed(res) {
        return false;
    }
    let removed = with_texture_shard_write(res, |shard| {
        let entry = shard.remove(&res)?;
        entry.mark_dead();
        let remaining = TEXTURE_COUNT.fetch_sub(1, Ordering::AcqRel) - 1;
        if remaining == 0 {
            INDEX_SETTLED_UNMAP_GEN.store(0, Ordering::Release);
        }
        Some(entry)
    });
    let Some(entry) = removed else { return false };
    crate::mem_owners::remove(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, 1, entry.bytes);
    with_addr_write(|addr| addr.remove(res));
    with_tile_write(|ti| ti.remove(res));
    true
}

/// The resource was released.
pub fn forget(res: usize) {
    if TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let _ = remove_resource_everywhere(res);
}

/// Removes a batch of released resources under single `ADDR_INDEX` and `TILE_INDEX` write locks.
pub fn forget_many(resources: &[usize]) {
    if resources.is_empty() || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut removed_ids: Vec<usize> = Vec::new();
    let mut removed_bytes = 0u64;
    let mut removed_count = 0usize;
    for &res in resources {
        if !is_placed(res) {
            continue;
        }
        let entry = with_texture_shard_write(res, |shard| {
            let entry = shard.remove(&res)?;
            entry.mark_dead();
            Some(entry)
        });
        if let Some(entry) = entry {
            removed_ids.push(res);
            removed_bytes += entry.bytes;
            removed_count += 1;
        }
    }
    if removed_ids.is_empty() {
        return;
    }
    let prev = TEXTURE_COUNT.fetch_sub(removed_count, Ordering::AcqRel);
    if prev <= removed_count {
        INDEX_SETTLED_UNMAP_GEN.store(0, Ordering::Release);
    }
    crate::mem_owners::remove(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, removed_count as u64, removed_bytes);
    with_addr_write(|addr| {
        for &res in &removed_ids {
            addr.remove(res);
        }
    });
    with_tile_write(|ti| ti.remove_many(&removed_ids));
}

/// Existing view and Unmap callers report possible writes. The placed memory is tracked by
/// `note_tiles` and explicit writes arrive through `note_write`, so nothing is left to do here.
pub fn mark_resource(_res: usize) {}

std::thread_local! {
    static ADDR_HIT_SCRATCH: std::cell::RefCell<Vec<Arc<PlacedEntry>>> = const { std::cell::RefCell::new(Vec::new()) };
    static TILE_RES_SCRATCH: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    static TILE_HIT_SCRATCH: std::cell::RefCell<Vec<Arc<PlacedEntry>>> = const { std::cell::RefCell::new(Vec::new()) };
    static SUBMITTED_SCRATCH: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The title or a layer wrote `len` bytes at title address `va` (a DirectStorage completion or a
/// copy).
pub fn note_write(va: u64, len: u64) {
    if !enabled() || len == 0 || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let end = va.saturating_add(len);
    if !may_overlap_addr(va, end) {
        return;
    }
    ADDR_HIT_SCRATCH.with(|cell| {
        let mut hits = cell.borrow_mut();
        hits.clear();
        let _ = with_addr_read(|addr| addr.collect_overlapping(va, end, &mut hits));
        if !hits.is_empty() {
            invalidate_entries(&hits);
            hits.clear();
        }
    });
}

/// With the present probe on, true at most once every five seconds.
fn probe_line_due() -> bool {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    static PROBE: OnceLock<bool> = OnceLock::new();
    if !*PROBE.get_or_init(|| std::env::var_os("DECANT_D3D12X_PRESENT_PROBE").is_some_and(|v| v != "0")) {
        return false;
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let due = last.is_none_or(|t| t.elapsed().as_secs() >= 5);
    if due {
        *last = Some(Instant::now());
    }
    due
}

type DropPayload = (FastMap<u32, TileRefs>, FastSet<usize>);

fn background_drop_sender() -> Option<&'static std::sync::mpsc::Sender<DropPayload>> {
    static DROP_TX: OnceLock<Option<std::sync::mpsc::Sender<DropPayload>>> = OnceLock::new();
    DROP_TX
        .get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel::<DropPayload>();
            match std::thread::Builder::new().name("detile-drop".into()).spawn(move || {
                while let Ok(payload) = rx.recv() {
                    drop(payload);
                }
            }) {
                Ok(_) => Some(tx),
                Err(_) => None,
            }
        })
        .as_ref()
}

fn enqueue_background_drop(old_map: FastMap<u32, TileRefs>, old_unmapped: FastSet<usize>) {
    if old_map.len() > 2048
        && let Some(tx) = background_drop_sender()
    {
        if let Err(std::sync::mpsc::SendError((m, u))) = tx.send((old_map, old_unmapped)) {
            drop(m);
            drop(u);
        }
        return;
    }
    drop(old_map);
    drop(old_unmapped);
}

fn tiles_of_batch_parallel(spans: &[(u64, u64)], tiles_out: &mut Vec<Option<Vec<u32>>>) {
    crate::title_mem::tiles_of_batch(spans, tiles_out);
}

/// The title wrote the pages with mirror tiles `tiles` since its last submission (`title_mem::sync`).
/// Every texture that owns one of them becomes dirty, and so does one whose memory had no GPU
/// address at its first tries.
pub fn note_tiles(tiles: &[u32]) {
    static ANNOUNCED: AtomicU32 = AtomicU32::new(0);
    if !enabled() || tiles.is_empty() || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let report = probe_line_due();
    let current_unmap = crate::title_mem::unmap_generation();
    let current_map = crate::title_mem::mapping_generation();
    let current_tile = crate::title_mem::tile_generation();

    let marked = TILE_RES_SCRATCH.with(|res_cell| {
        TILE_HIT_SCRATCH.with(|hit_cell| {
            let mut hit_res = res_cell.borrow_mut();
            let mut hits = hit_cell.borrow_mut();
            hit_res.clear();
            hits.clear();

            let fast = with_tile_read(|ti| {
                if ti.entries.is_empty() {
                    return Ok(());
                }
                if ti.needs_update(current_unmap, current_map, current_tile) {
                    return Err(());
                }
                ti.collect_hits(tiles, report, &mut hit_res, &mut hits);
                Ok(())
            });
            match fast {
                None => return 0usize,
                Some(Ok(())) => {}
                Some(Err(())) => {
                    let _rebuild_guard = decant_trace::timed_lock("detile::TILE_REBUILD", || INDEX_REBUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner()));
                    let build_unmap = crate::title_mem::unmap_generation();
                    let build_map = crate::title_mem::mapping_generation();
                    let build_tile = crate::title_mem::tile_generation();
                    let plan_or_done = with_tile_read(|ti| {
                        if ti.entries.is_empty() {
                            return None;
                        }
                        if let Some(plan) = ti.plan_index_update(build_unmap, build_map, build_tile) {
                            return Some(plan);
                        }
                        ti.collect_hits(tiles, report, &mut hit_res, &mut hits);
                        None
                    })
                    .flatten();
                    if let Some(plan) = plan_or_done {
                        let mut tiles_out = Vec::new();
                        match plan {
                            IndexPlan::Full { unmap_gen: _, items, spans, cap_hint } => {
                                tiles_of_batch_parallel(&spans, &mut tiles_out);
                                let precomputed = TileIndex::build_full_index_maps(&items, &tiles_out, cap_hint);
                                let (old_map, old_unmapped) = with_tile_write(|ti| {
                                    let old = ti.apply_precomputed_full_index(
                                        &items,
                                        tiles_out,
                                        precomputed,
                                        (build_unmap, build_map, build_tile),
                                        crate::title_mem::unmap_generation() == build_unmap,
                                    );
                                    ti.collect_hits(tiles, report, &mut hit_res, &mut hits);
                                    old
                                });
                                enqueue_background_drop(old_map, old_unmapped);
                            }
                            IndexPlan::Incremental { unmap_gen: snap_unmap, items, spans } => {
                                tiles_of_batch_parallel(&spans, &mut tiles_out);
                                with_tile_write(|ti| {
                                    if crate::title_mem::unmap_generation() != snap_unmap {
                                        ti.layout_gen = 0;
                                    } else {
                                        ti.apply_incremental_index(
                                            &items,
                                            &tiles_out,
                                            build_map,
                                            crate::title_mem::mapping_generation() == build_map,
                                            build_tile,
                                            crate::title_mem::tile_generation() == build_tile,
                                        );
                                    }
                                    ti.collect_hits(tiles, report, &mut hit_res, &mut hits);
                                });
                            }
                        }
                    }
                }
            }

            let count = hits.len();
            if count > 0 {
                invalidate_entries(&hits);
                hits.clear();
            }
            hit_res.clear();
            count
        })
    });
    if marked > 0 && ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("detile: {marked} textures whose memory the title wrote ({} pages written) are filled again", tiles.len());
    }
}

/// The title's GPU work writes into the texture (a copy). The CPU view is stale from then on, so
/// the texture is not detiled any more.
pub fn note_gpu_write(res: usize) {
    if TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    if remove_resource_everywhere(res) {
        static ANNOUNCED: AtomicU32 = AtomicU32::new(0);
        if ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 4 {
            xs_log!("detile: a texture the title copies into on the GPU is no longer detiled");
        }
    }
}

/// The title recorded `transitions` (resource, subresource, state after) on PC list `list`.
pub fn note_transitions(list: usize, transitions: &[(usize, u32, u32)]) {
    if !enabled() || transitions.is_empty() || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut filtered: Vec<(usize, u32, u32)> = Vec::new();
    let mut last = (0usize, false);
    for &(res, sub, after) in transitions {
        let placed = if res == last.0 && res != 0 {
            last.1
        } else {
            let p = is_placed(res);
            last = (res, p);
            p
        };
        if placed {
            filtered.push((res, sub, after));
        }
    }
    if filtered.is_empty() {
        return;
    }
    with_noted_shard(list, |n| {
        let entry = n.lists.entry(list);
        let was_absent = matches!(entry, std::collections::hash_map::Entry::Vacant(_));
        entry.or_default().extend(filtered);
        if was_absent {
            NOTED_COUNT.fetch_add(1, Ordering::Release);
        }
    });
}

/// PC list `list` starts a new recording, so the transitions noted for it are dropped.
pub fn list_reset(list: usize) {
    if NOTED_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    with_noted_shard(list, |n| {
        if let Some(entries) = n.lists.get_mut(&list) {
            entries.clear();
        }
    });
}

/// Updates `states` for a transition of subresource `sub` (or of all, `0xFFFFFFFF`) to `after` in a
/// texture with `count` subresources. All equal states collapse to a single entry.
pub fn apply_transition(states: &mut Vec<u32>, count: usize, sub: u32, after: u32) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (count <= 1 && sub == 0) {
        states.clear();
        states.push(after);
        return;
    }
    let index = sub as usize;
    if index >= count {
        return;
    }
    if states.len() != count {
        let initial = states.first().copied().unwrap_or(0);
        states.resize(count, initial);
    }
    states[index] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

/// The PC lists `lists` were submitted in this order, so the textures now are in the states that
/// the transitions of those lists left them in.
pub fn submitted(lists: &[usize]) {
    if lists.is_empty() || NOTED_COUNT.load(Ordering::Acquire) == 0 || TEXTURE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    SUBMITTED_SCRATCH.with(|cell| {
        let mut changes = cell.borrow_mut();
        changes.clear();
        for &list in lists {
            with_noted_shard(list, |n| {
                if let Some(entries) = n.lists.get(&list) {
                    changes.extend_from_slice(entries);
                }
            });
        }
        if changes.is_empty() {
            return;
        }
        let mut last: Option<(usize, Arc<PlacedEntry>)> = None;
        for &(res, sub, after) in changes.iter() {
            let entry = match &last {
                Some((prev_res, prev_entry)) if *prev_res == res => Some(Arc::clone(prev_entry)),
                _ => {
                    let found = lookup_placed_entry(res);
                    if let Some(ref e) = found {
                        last = Some((res, Arc::clone(e)));
                    }
                    found
                }
            };
            if let Some(entry) = entry {
                let count = (entry.plan.mips.len() as u32 * entry.plan.slices) as usize;
                let mut states = entry.states.lock().unwrap_or_else(|e| e.into_inner());
                apply_transition(Arc::make_mut(&mut *states), count, sub, after);
            }
        }
        changes.clear();
    });
}

fn needs_fresh_sync(defer_fresh_spans: bool, translated: &[Option<(crate::title_mem::Span, bool)>]) -> bool {
    !defer_fresh_spans && translated.iter().any(|m| matches!(m, Some((_, true))))
}

struct OwnedJob {
    texture: ID3D12Resource,
    src_gpu: u64,
    plan: Arc<Plan>,
    states: Arc<Vec<u32>>,
}

#[derive(Default)]
struct FlushScratch {
    pending_entries: Vec<Arc<PlacedEntry>>,
    pending_items: Vec<(Arc<PlacedEntry>, u64)>,
    pending_spans: Vec<(u64, u64)>,
    translated: Vec<Option<(crate::title_mem::Span, bool)>>,
    ready_entries: Vec<(Arc<PlacedEntry>, u64, u64)>,
    requeue: Vec<Arc<PlacedEntry>>,
    owned_jobs: Vec<OwnedJob>,
    settled_entries: Vec<(Arc<PlacedEntry>, u64)>,
    outcome: Vec<bool>,
}

impl FlushScratch {
    fn clear_all(&mut self) {
        self.pending_entries.clear();
        self.pending_items.clear();
        self.pending_spans.clear();
        self.translated.clear();
        self.ready_entries.clear();
        self.requeue.clear();
        self.owned_jobs.clear();
        self.settled_entries.clear();
        self.outcome.clear();
    }
}

std::thread_local! {
    static FLUSH_SCRATCH: std::cell::RefCell<FlushScratch> = std::cell::RefCell::new(FlushScratch::default());
}

struct FlushScratchGuard(FlushScratch);

impl FlushScratchGuard {
    fn take() -> Self {
        let mut s = FLUSH_SCRATCH.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        s.clear_all();
        Self(s)
    }
}

impl Drop for FlushScratchGuard {
    fn drop(&mut self) {
        self.0.clear_all();
        let returned = std::mem::take(&mut self.0);
        let _ = FLUSH_SCRATCH.try_with(|cell| {
            *cell.borrow_mut() = returned;
        });
    }
}

fn collect_pending_for_flush(s: &mut FlushScratch) {
    drain_pending_entries(&mut s.pending_entries);
    s.pending_items.clear();
    s.pending_spans.clear();
    for entry in s.pending_entries.drain(..) {
        if let Some(snap_gen) = entry.dequeue_for_flush() {
            s.pending_spans.push((entry.va, entry.plan.size));
            s.pending_items.push((entry, snap_gen));
        }
    }
}

/// Detiles every dirty texture on the PC direct queue `queue`, ahead of the title's lists that run
/// on it.
///
/// # Safety
/// `queue` is a live PC command queue.
pub unsafe fn flush(queue: *mut c_void) {
    if !enabled() || TEXTURE_COUNT.load(Ordering::Acquire) == 0 || !HAS_PENDING_FILLS.load(Ordering::Acquire) {
        return;
    }
    let _trace = decant_trace::span("detile", "flush");
    let queue_raw = queue;
    // SAFETY: by the contract of this function.
    let Some(queue) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return };
    // SAFETY: a live queue.
    if unsafe { queue.GetDesc() }.Type != D3D12_COMMAND_LIST_TYPE_DIRECT {
        return;
    }
    let started = Instant::now();
    let tuning = tuning();
    if !crate::title_mem::enabled() {
        if DETILER_READY_STATE.load(Ordering::Acquire) != 2 {
            with_detiler(|d| {
                if !d.unavailable {
                    d.unavailable = true;
                    DETILER_READY_STATE.store(2, Ordering::Release);
                    xs_log!("detile: title memory is off (DECANT_D3D12X_TITLE_MEM=1), placed textures stay unfilled");
                }
            });
        }
        return;
    }
    let Some(device) = crate::title_mem::device() else { return };
    let detiler_state = DETILER_READY_STATE.load(Ordering::Acquire);
    if detiler_state == 2 {
        return;
    }
    if detiler_state == 0 {
        let ready = with_detiler(|d| {
            if d.unavailable {
                DETILER_READY_STATE.store(2, Ordering::Release);
                return false;
            }
            if d.detiler.is_none() {
                d.detiler = Detiler::new(&device);
                if d.detiler.is_none() {
                    d.unavailable = true;
                    DETILER_READY_STATE.store(2, Ordering::Release);
                    xs_log!("detile: not available, placed textures stay unfilled");
                    return false;
                }
            }
            DETILER_READY_STATE.store(1, Ordering::Release);
            true
        });
        if !ready {
            return;
        }
    }

    let mut scratch_guard = FlushScratchGuard::take();
    let s = &mut scratch_guard.0;

    collect_pending_for_flush(s);
    if s.pending_items.is_empty() {
        return;
    }

    // Translate outside all registry/index locks so title_mem::STATE never nests under detile locks.
    crate::title_mem::translate_fresh_batch(&s.pending_spans, &mut s.translated);
    if needs_fresh_sync(tuning.defer_fresh_spans, &s.translated) {
        enqueue_pending_batch(s.pending_items.drain(..).filter_map(|(entry, _)| entry.try_set_queued().then_some(entry)));
        s.pending_spans.clear();
        // SAFETY: `queue_raw` is a live direct queue and no detile locks are held.
        unsafe { crate::title_mem::sync(queue_raw) };
        collect_pending_for_flush(s);
        if s.pending_items.is_empty() {
            return;
        }
        crate::title_mem::translate_fresh_batch(&s.pending_spans, &mut s.translated);
    }

    let mut unreachable = 0u32;
    let mut deferred_here = 0u64;
    for ((entry, snap_gen), mapped) in s.pending_items.drain(..).zip(&s.translated) {
        if !entry.wants_fill() {
            continue;
        }
        match mapped {
            // Fresh mappings go out with the next sync, before the next submission.
            Some((_, true)) if tuning.defer_fresh_spans => {
                deferred_here += 1;
                if entry.try_set_queued() {
                    s.requeue.push(entry);
                }
            }
            Some((span, _)) => {
                s.ready_entries.push((entry, snap_gen, span.gpu));
            }
            None => {
                unreachable += 1;
                if entry.record_unreachable(snap_gen) {
                    s.requeue.push(entry);
                }
            }
        }
    }
    if deferred_here > 0 {
        DEFERRED_COUNT.fetch_add(deferred_here, Ordering::Relaxed);
    }
    if !s.requeue.is_empty() {
        enqueue_pending_batch(s.requeue.drain(..));
    }
    if unreachable > 0 && FLUSHES_COUNT.load(Ordering::Relaxed) < 4 {
        xs_log!("detile: {unreachable} textures have no GPU address for their memory yet");
    }
    if s.ready_entries.is_empty() {
        return;
    }

    s.ready_entries.sort_unstable_by_key(|(entry, _, _)| entry.res);
    for (entry, snap_gen, gpu) in s.ready_entries.drain(..) {
        let raw = entry.res as *mut c_void;
        let cloned = with_texture_shard_read(entry.res, |shard| {
            let live = shard.get(&entry.res)?;
            if !Arc::ptr_eq(live, &entry) {
                return None;
            }
            // SAFETY: the shard holds `entry`, so `forget` / `forget_many` cannot return and
            // release the resource while this read guard is held.
            unsafe { ID3D12Resource::from_raw_borrowed(&raw) }.cloned()
        })
        .flatten();
        if let Some(texture) = cloned {
            let states = entry.states.lock().unwrap_or_else(|e| e.into_inner()).clone();
            s.owned_jobs.push(OwnedJob { texture, src_gpu: gpu, plan: Arc::clone(&entry.plan), states });
            s.settled_entries.push((entry, snap_gen));
        }
    }
    let ready_count = s.settled_entries.len();
    if ready_count == 0 {
        return;
    }

    // Execute GPU detile work under DETILER without holding any registry/index locks so
    // note_tiles, note_write, register, and submitted never contend with command list recording.
    let jobs: Vec<Job<'_>> = s
        .owned_jobs
        .iter()
        .map(|j| Job { texture: j.texture.clone(), src_gpu: j.src_gpu, plan: &j.plan, states: &j.states })
        .collect();
    let footprint = with_detiler(|d| {
        let detiler = d.detiler.as_mut()?;
        detiler.run_into(queue, &jobs, tuning.wait_each_flush, &mut s.outcome);
        Some(detiler.footprint())
    });
    drop(jobs);
    s.owned_jobs.clear();
    let Some((frames, queues, staging)) = footprint else {
        return;
    };

    let mut filled_bytes = 0u64;
    let mut detiled_here = 0u64;
    for ((entry, snap_gen), &ok) in s.settled_entries.drain(..).zip(&s.outcome) {
        entry.settle_gen(snap_gen, ok);
        if ok {
            filled_bytes += entry.plan.size;
            detiled_here += 1;
        }
    }
    if detiled_here > 0 {
        DETILED_COUNT.fetch_add(detiled_here, Ordering::Relaxed);
    }
    let flush_num = FLUSHES_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if flush_num <= 10 || flush_num.is_multiple_of(100) {
        let detiled_total = DETILED_COUNT.load(Ordering::Relaxed);
        let deferred_total = DEFERRED_COUNT.load(Ordering::Relaxed);
        xs_log!(
            "detile: flush {flush_num}: {ready_count} textures ({} MiB of tiled memory) in {} ms, {detiled_total} in all, {deferred_total} put off, {frames} lists on {queues} queues with {} MiB of staging",
            filled_bytes >> 20,
            started.elapsed().as_millis(),
            staging >> 20
        );
    }
    let streaming_units = (ready_count as u64).max(filled_bytes >> 20);
    crate::title_mem::note_streaming_work(streaming_units);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_plan(size: u64) -> Plan {
        Plan { bytes_per_element: 4, linear: true, mode: 0, block_bits: 0, block_width: 1, block_height: 1, slice_size: 0, slices: 1, mips: Vec::new(), terms: Vec::new(), size }
    }

    fn placed_entry(res: usize, va: u64, size: u64, dirty: bool) -> Arc<PlacedEntry> {
        Arc::new(PlacedEntry::new(res, va, Arc::new(flat_plan(size)), size, 0, dirty))
    }

    fn test_insert_global(res: usize, va: u64, size: u64, dirty: bool) -> Arc<PlacedEntry> {
        let entry = placed_entry(res, va, size, dirty);
        with_texture_shard_write(res, |shard| {
            if shard.insert(res, Arc::clone(&entry)).is_none() {
                TEXTURE_COUNT.fetch_add(1, Ordering::AcqRel);
            }
        });
        with_addr_write(|addr| addr.insert(res, va, size, Arc::clone(&entry)));
        with_tile_write(|ti| ti.insert(res, va, size, Arc::clone(&entry)));
        if dirty && entry.try_set_queued() {
            enqueue_pending_single(Arc::clone(&entry));
        }
        entry
    }

    fn clear_global_pending_for(res: usize) {
        let mut drained = Vec::new();
        drain_pending_entries(&mut drained);
        let mut keep = Vec::new();
        for e in drained {
            if e.res == res {
                let _ = e.dequeue_for_flush();
            } else {
                keep.push(e);
            }
        }
        if !keep.is_empty() {
            enqueue_pending_batch(keep.into_iter());
        }
    }

    #[test]
    fn overlap_search_includes_a_big_texture_that_starts_far_below() {
        let mut idx = AddressIndex::new(false);
        idx.insert(1, 0x1000, 0x10_0000, placed_entry(1, 0x1000, 0x10_0000, false));
        idx.insert(2, 0x8_0000, 0x100, placed_entry(2, 0x8_0000, 0x100, false));
        idx.insert(3, 0x20_0000, 0x100, placed_entry(3, 0x20_0000, 0x100, false));
        let mut hit = idx.overlapping(0x8_0080, 0x8_0081);
        hit.sort_unstable();
        assert_eq!(hit, [1, 2]);
        assert!(idx.overlapping(0x20_0100, 0x20_0200).is_empty(), "a write right behind a texture misses it");
    }

    #[test]
    fn inserting_again_moves_the_texture_and_removing_forgets_the_address() {
        let mut idx = AddressIndex::new(false);
        idx.insert(1, 0x1000, 0x1000, placed_entry(1, 0x1000, 0x1000, false));
        idx.insert(2, 0x8_0000, 0x100, placed_entry(2, 0x8_0000, 0x100, false));
        idx.insert(1, 0x30_0000, 0x10, placed_entry(1, 0x30_0000, 0x10, false));
        assert_eq!(idx.overlapping(0x1000, 0x1100), Vec::<usize>::new());
        assert_eq!(idx.overlapping(0x30_0000, 0x30_0001), [1]);
        assert!(idx.remove(2));
        assert!(!idx.remove(2));
        assert!(!idx.small.contains_key(&0x8_0000));
    }

    #[test]
    fn batches_close_at_the_staging_target_or_the_dispatch_slots() {
        let target = BATCH_STAGING_TARGET;
        assert_eq!(group_batches(&[]), Vec::<Range<usize>>::new());
        assert_eq!(group_batches(&[(target, 1), (1, 1)]), [0..1, 1..2], "the target is passed by the second job");
        assert_eq!(group_batches(&[(target / 2, 1), (target / 2, 1), (1, 1)]), [0..2, 2..3]);
        assert_eq!(group_batches(&[(target * 4, 1), (target * 4, 1)]), [0..1, 1..2], "a big job is alone but never dropped");
        let slots = DISPATCH_SLOTS;
        assert_eq!(group_batches(&[(1, slots - 1), (1, 1), (1, 1)]), [0..2, 2..3]);
    }

    #[test]
    fn a_frame_is_free_when_the_fence_of_its_own_queue_has_passed() {
        // Frame 0 waits for value 5 of queue 1, frame 1 for value 1 of queue 2, which finished first.
        let completed = |queue: usize| if queue == 1 { 4 } else { 1 };
        assert_eq!(first_finished([(1, 5), (2, 1)].into_iter(), completed), Some(1));
        assert_eq!(first_finished([(1, 5)].into_iter(), completed), None);
        assert_eq!(first_finished([(1, 0)].into_iter(), completed), Some(0), "never used");
        assert_eq!(first_finished([(2, u64::MAX)].into_iter(), |_| u64::MAX - 1), None, "a failed signal retires the frame");
    }

    #[test]
    fn parameter_block_puts_the_geometry_first_and_the_terms_at_word_20() {
        let mut plan = flat_plan(0);
        plan.linear = false;
        plan.slices = 2;
        plan.block_bits = 16;
        plan.terms = vec![(0, 1, 0, 0), (1, 0, 2, 0)];
        let level = MipJob { width_elements: 64, height_elements: 32, in_tail: false, tail_x: 0, tail_y: 0, blocks_x: 4, level_offset: 0x100, pitch_bytes: 256 };
        let words = param_block(&plan, &level, 512, 0x4000, 0x1_0000_0040);
        assert_eq!(words[..4], [64, 32, 4, 512]);
        assert_eq!(words[4], 0, "tiled");
        assert_eq!(words[12..19], [0x100, 0, 256, 0x4000, 0x40, 2, 2], "the staging offset is cut to 32 bits");
        assert_eq!(words[FIRST_TERM_WORD..FIRST_TERM_WORD + 8], [0, 1, 0, 0, 1, 0, 2, 0]);
        assert!(words[FIRST_TERM_WORD + 8..].iter().all(|&w| w == 0));
    }

    #[test]
    fn narrow_elements_are_packed_four_bytes_per_thread() {
        let level = |width| MipJob { width_elements: width, height_elements: 1, in_tail: false, tail_x: 0, tail_y: 0, blocks_x: 1, level_offset: 0, pitch_bytes: 0 };
        let mut plan = flat_plan(0);
        assert_eq!(groups_across(&plan, &level(64)), 4);
        plan.bytes_per_element = 1;
        assert_eq!(groups_across(&plan, &level(64)), 1, "64 bytes are 16 dwords");
        assert_eq!(groups_across(&plan, &level(65)), 2);
    }

    #[test]
    fn transitions_apply_per_subresource_and_collapse_when_equal() {
        let mut states = vec![0x80];
        apply_transition(&mut states, 3, 1, 0x400);
        assert_eq!(states, [0x80, 0x400, 0x80]);
        apply_transition(&mut states, 3, 7, 0x1);
        assert_eq!(states, [0x80, 0x400, 0x80], "a subresource the texture lacks changes nothing");
        apply_transition(&mut states, 3, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, 0xC0);
        assert_eq!(states, [0xC0]);
        apply_transition(&mut states, 2, 0, 0x400);
        apply_transition(&mut states, 2, 1, 0x400);
        assert_eq!(states, [0x400]);
    }

    #[test]
    fn a_write_makes_a_filled_texture_due_once_and_a_gpu_write_ends_tracking() {
        let mut raw = [0u8; 0x38];
        raw[0..4].copy_from_slice(&3u32.to_le_bytes());
        raw[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        raw[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        raw[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        raw[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        raw[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        raw[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        let res = 0x7101;
        register(res, 0x10_0000, &raw, 0, 64 * 64 * 4);
        // One pass of a flush that always succeeds: how many textures it fills.
        let fill_pass = || {
            let Some(entry) = lookup_placed_entry(res) else { return 0 };
            let Some(snap_gen) = entry.dequeue_for_flush() else { return 0 };
            clear_global_pending_for(res);
            entry.settle_gen(snap_gen, true);
            1
        };
        assert_eq!(fill_pass(), 1, "the first fill");
        assert_eq!(fill_pass(), 0, "nothing changed since");
        note_write(0x20_0000, 16);
        assert_eq!(fill_pass(), 0, "a write elsewhere");
        note_write(0x10_0000 + 100, 4);
        assert_eq!(fill_pass(), 1, "a write inside");
        assert_eq!(fill_pass(), 0);
        mark_resource(res);
        assert_eq!(fill_pass(), 0, "resource marks alone do not matter");
        note_gpu_write(res);
        assert!(!is_placed(res));
    }

    #[test]
    fn a_failed_fill_stops_the_retries_until_the_next_write() {
        let entry = placed_entry(1, 0x1000, 0x100, true);
        assert!(entry.wants_fill());
        let snap = entry.dequeue_for_flush().expect("dirty");
        entry.settle_gen(snap, false);
        assert!(!entry.wants_fill());
        assert!(entry.invalidate_mark());
        assert!(entry.wants_fill());
    }

    #[test]
    fn best_finished_prefers_the_smallest_idle_frame_that_already_fits_needed_staging() {
        let completed = |queue: usize| if queue == 1 { 5 } else { 0 };
        let frames = [(1, 2, 8 << 20), (1, 3, 64 << 20), (1, 4, 32 << 20), (2, 1, 16 << 20)];
        assert_eq!(best_finished(frames.into_iter(), 20 << 20, completed), Some(2), "32 MiB fits 20 MiB with less slack than 64 MiB");
        assert_eq!(best_finished(frames.into_iter(), 8 << 20, completed), Some(0), "exact 8 MiB match");
        assert_eq!(best_finished(frames.into_iter(), 128 << 20, completed), Some(1), "none fits so the largest idle frame is chosen");
        assert_eq!(best_finished([(2, 1, 16 << 20), (1, u64::MAX, 32 << 20)].into_iter(), 8 << 20, completed), None);
    }

    #[test]
    fn group_batches_into_reuses_the_batch_buffer_and_respects_both_limits() {
        let mut batches = Vec::with_capacity(16);
        batches.push(99..100);
        let target = BATCH_STAGING_TARGET;
        let slots = DISPATCH_SLOTS;
        group_batches_into(&[(target / 4, 10), (target / 4, slots - 5), (target / 2, 5), (target, 1)], &mut batches);
        assert_eq!(batches, [0..1, 1..3, 3..4]);
        let cap = batches.capacity();
        group_batches_into(&[], &mut batches);
        assert!(batches.is_empty() && batches.capacity() == cap);
    }

    #[test]
    fn pending_set_tracks_dirty_textures_through_inserts_writes_and_removals() {
        let dirty_one = placed_entry(10, 0x1000, 0x1000, true);
        let clean_two = placed_entry(20, 0x4000, 0x1000, false);
        assert!(dirty_one.wants_fill() && !clean_two.wants_fill());
        assert!(clean_two.invalidate_mark());
        assert!(clean_two.wants_fill());
        assert!(!clean_two.invalidate_mark(), "already queued");
        dirty_one.mark_dead();
        clean_two.mark_dead();
        assert!(!dirty_one.wants_fill() && !clean_two.wants_fill());
    }

    #[test]
    fn reverse_tile_index_updates_incrementally_and_rebuilds_on_layout_change() {
        let mut ti = TileIndex::new(false);
        let e1 = placed_entry(1, 0x1000, 0x1000, false);
        let e2 = placed_entry(2, 0x4000, 0x1000, false);
        let e3 = placed_entry(3, 0x8000, 0x1000, false);
        ti.insert(1, 0x1000, 0x1000, Arc::clone(&e1));
        ti.insert(2, 0x4000, 0x1000, Arc::clone(&e2));
        assert!(ti.pending_index.contains(&1) && ti.pending_index.contains(&2));

        // Initial full build at unmap_gen 7, map_gen 10, tile_gen 3.
        let items = vec![(1, 0x1000, 0x1000, Arc::clone(&e1)), (2, 0x4000, 0x1000, Arc::clone(&e2))];
        ti.apply_full_index(&items, &[Some(vec![10, 11]), None], 7, 10, 3, true);
        assert!(ti.pending_index.is_empty());
        assert_eq!(ti.layout_gen, 7);
        assert_eq!(ti.last_map_gen, 10);
        assert_eq!(ti.last_tile_gen, 3);
        assert!(ti.unmapped_textures.contains(&2) && !ti.unmapped_textures.contains(&1));

        // Inserting a new texture only adds it to pending_index without invalidating layout_gen.
        ti.insert(3, 0x8000, 0x1000, Arc::clone(&e3));
        assert_eq!(ti.layout_gen, 7);
        assert!(ti.pending_index.contains(&3));

        // Incremental update resolves texture 3 and previously unmapped texture 2 at map_gen 11, tile_gen 4.
        ti.apply_incremental_index(
            &[(3, 0x8000, 0x1000, Arc::clone(&e3)), (2, 0x4000, 0x1000, Arc::clone(&e2))],
            &[Some(vec![12]), Some(vec![11, 13])],
            11,
            true,
            4,
            true,
        );
        assert!(ti.pending_index.is_empty());
        assert!(ti.unmapped_textures.is_empty());
        assert_eq!(ti.last_map_gen, 11);
        assert_eq!(ti.last_tile_gen, 4);

        let mut hit_res = Vec::new();
        let mut hits = Vec::new();
        ti.collect_hits(&[99], false, &mut hit_res, &mut hits);
        assert!(hits.is_empty());
        ti.collect_hits(&[10, 11], false, &mut hit_res, &mut hits);
        assert_eq!(hits.len(), 2);
        for h in &hits {
            let _ = h.invalidate_mark();
        }
        assert!(e1.wants_fill() && e2.wants_fill() && !e3.wants_fill());

        // Removing texture 2 cleans tile_to_textures incrementally without invalidating layout_gen.
        assert!(ti.remove(2));
        assert_eq!(ti.layout_gen, 7);
        assert!(!ti.tile_to_textures.contains_key(&13));
        ti.collect_hits(&[11, 13], false, &mut hit_res, &mut hits);
        assert_eq!(hits.len(), 1);

        // Structural layout rebuild replaces all mappings at unmap_gen 8.
        ti.apply_full_index(
            &[(1, 0x1000, 0x1000, Arc::clone(&e1)), (3, 0x8000, 0x1000, Arc::clone(&e3))],
            &[Some(vec![20]), Some(vec![21])],
            8,
            12,
            5,
            true,
        );
        assert_eq!(ti.layout_gen, 8);
        ti.collect_hits(&[10], false, &mut hit_res, &mut hits);
        assert!(hits.is_empty());
        ti.collect_hits(&[20, 21], false, &mut hit_res, &mut hits);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn staging_cap_trims_largest_idle_frames_down_to_budget() {
        assert_eq!(staging_cap_from(None), DEFAULT_STAGING_CAP_MIB << 20);
        assert_eq!(staging_cap_from(Some("0")), 0);
        assert_eq!(staging_cap_from(Some(" 128 ")), 128 << 20);

        let frames = [(1, 10, 256 << 20), (1, 4, 128 << 20), (1, 3, 512 << 20), (1, 5, 128 << 20)];
        let trimmed = select_staging_to_trim(frames.into_iter(), |_| 5, 256 << 20);
        assert_eq!(trimmed, vec![2, 1, 3], "busy frame 0 stays untouched, largest idle frames go first");

        assert!(select_staging_to_trim(frames.into_iter(), |_| 5, 0).is_empty(), "cap 0 disables trimming");
    }

    #[test]
    fn fresh_span_sync_predicate_respects_defer_flag() {
        let span = crate::title_mem::Span { resource: 0, gpu: 0x1000, offset: 0, left: 0x1000 };
        let mapped = [Some((span, false)), None];
        let fresh = [Some((span, false)), Some((span, true))];
        assert!(!needs_fresh_sync(false, &mapped));
        assert!(needs_fresh_sync(false, &fresh));
        assert!(!needs_fresh_sync(true, &fresh));
    }

    #[test]
    fn detiler_lock_does_not_block_registry_access() {
        let _detiler_guard = DETILER.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            let _ = with_tile_read(|ti| ti.entries.len());
            let _ = with_addr_read(|addr| addr.by_res.len());
        })
        .join()
        .unwrap();
        assert!(!summary().is_empty());
    }

    #[test]
    fn fast_tile_index_handles_empty_and_deduplicates_multi_tile_hits() {
        let mut ti = TileIndex::new(false);
        let mut hit_res = Vec::new();
        let mut hits = Vec::new();
        ti.collect_hits(&[1, 2, 3], false, &mut hit_res, &mut hits);
        assert!(hits.is_empty());
        let e1 = placed_entry(1, 0x1000, 0x2000, false);
        ti.insert(1, 0x1000, 0x2000, Arc::clone(&e1));
        ti.apply_incremental_index(&[(1, 0x1000, 0x2000, Arc::clone(&e1))], &[Some(vec![40, 41, 42])], 1, true, 1, true);
        ti.collect_hits(&[40, 41, 41, 42], false, &mut hit_res, &mut hits);
        assert_eq!(hits.len(), 1);
        assert!(e1.invalidate_mark());
        assert!(e1.wants_fill());
    }

    #[test]
    fn arc_plan_and_states_snapshot_share_without_deep_copy_and_cow_on_transition() {
        let p = placed_entry(1, 0x1000, 0x2000, false);
        let plan_snap = Arc::clone(&p.plan);
        let states_snap = p.states.lock().unwrap().clone();
        assert!(Arc::ptr_eq(&p.plan, &plan_snap));
        {
            let mut guard = p.states.lock().unwrap();
            apply_transition(Arc::make_mut(&mut *guard), 1, 0, 7);
            assert!(!Arc::ptr_eq(&*guard, &states_snap));
            assert_eq!(**guard, [7]);
        }
        assert_eq!(*states_snap, [0]);
    }

    #[test]
    fn registry_rwlock_allows_concurrent_reads_and_fast_misses() {
        let local: RwLock<Option<TileIndex>> = RwLock::new(Some(TileIndex::new(false)));
        let r1 = local.read().unwrap_or_else(|e| e.into_inner());
        let r2 = std::thread::scope(|s| s.spawn(|| local.read().unwrap_or_else(|e| e.into_inner()).is_some()).join().unwrap());
        assert!(r1.is_some() && r2);
        drop(r1);
        note_write(0xDEAD_0000, 64);
        note_transitions(0xDEAD, &[(0xBEEF, 0, 1)]);
        list_reset(0xDEAD);
        submitted(&[0xDEAD]);
        forget(0xBEEF);
        note_gpu_write(0xBEEF);
    }

    #[test]
    fn precomputed_full_index_swaps_maps_and_preserves_concurrent_pending_index() {
        let mut ti = TileIndex::new(false);
        let e1 = placed_entry(1, 0x1000, 0x1000, false);
        let e2 = placed_entry(2, 0x2000, 0x1000, false);
        let e3 = placed_entry(3, 0x3000, 0x1000, false);
        ti.insert(1, 0x1000, 0x1000, Arc::clone(&e1));
        ti.insert(2, 0x2000, 0x1000, Arc::clone(&e2));
        let items = vec![(1, 0x1000, 0x1000, Arc::clone(&e1)), (2, 0x2000, 0x1000, Arc::clone(&e2))];
        let tiles_out = vec![Some(vec![10, 11]), None];
        let precomputed = TileIndex::build_full_index_maps(&items, &tiles_out, 0);
        // Simulate a new texture registered concurrently while tiles_of_batch ran.
        ti.insert(3, 0x3000, 0x1000, Arc::clone(&e3));
        let (old_map, old_unmapped) = ti.apply_precomputed_full_index(&items, tiles_out, precomputed, (42, 7, 7), true);
        assert!(old_map.is_empty() && old_unmapped.is_empty());
        assert_eq!(ti.tile_to_textures.get(&10).map(TileRefs::as_slice), Some(&[1][..]));
        assert!(ti.unmapped_textures.contains(&2));
        assert!(!ti.pending_index.contains(&1) && !ti.pending_index.contains(&2));
        assert!(ti.pending_index.contains(&3), "concurrently inserted texture stays queued");
        assert_eq!(ti.layout_gen, 0, "live_matched != entries.len() keeps layout_gen 0");
    }

    #[test]
    fn mark_tiles_from_index_keeps_already_dirty_zero_attempt_and_requeues_settled() {
        let mut ti = TileIndex::new(false);
        let p1 = placed_entry(1, 0x1000, 0x1000, true);
        let p2 = placed_entry(2, 0x2000, 0x1000, false);
        ti.insert(1, 0x1000, 0x1000, Arc::clone(&p1));
        ti.insert(2, 0x2000, 0x1000, Arc::clone(&p2));
        ti.tile_to_textures.insert(5, TileRefs::Two([1, 2]));
        let mut hit_res = Vec::new();
        let mut hits = Vec::new();
        ti.collect_hits(&[5], false, &mut hit_res, &mut hits);
        assert_eq!(hits.len(), 2);
        for h in &hits {
            let _ = h.invalidate_mark();
        }
        assert!(p1.wants_fill());
        assert!(p2.wants_fill());
        assert!(p2.is_dirty());
    }

    #[test]
    fn tiles_of_batch_parallel_matches_serial_above_threshold() {
        let spans: Vec<(u64, u64)> = (0..1500u64).map(|i| (0x1000_0000 + i * 0x10000, 0x10000)).collect();
        let mut serial = Vec::new();
        crate::title_mem::tiles_of_batch(&spans, &mut serial);
        let mut parallel = Vec::new();
        tiles_of_batch_parallel(&spans, &mut parallel);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn map_uses_incremental_unmapped_resolution_while_unmap_triggers_full_rebuild() {
        let mut ti = TileIndex::new(false);
        let e1 = placed_entry(1, 0x1000, 0x1000, false);
        let e2 = placed_entry(2, 0x4000, 0x1000, false);
        ti.insert(1, 0x1000, 0x1000, Arc::clone(&e1));
        ti.insert(2, 0x4000, 0x1000, Arc::clone(&e2));
        let items = vec![(1, 0x1000, 0x1000, Arc::clone(&e1)), (2, 0x4000, 0x1000, Arc::clone(&e2))];
        ti.apply_full_index(&items, &[Some(vec![10]), None], 1, 1, 1, true);
        assert!(ti.unmapped_textures.contains(&2));
        assert!(ti.plan_index_update(1, 1, 1).is_none());

        // MAP bumps mapping_generation without bumping unmap_generation: only unmapped texture 2 is checked.
        match ti.plan_index_update(1, 2, 1) {
            Some(IndexPlan::Incremental { unmap_gen, items, .. }) => {
                assert_eq!(unmap_gen, 1);
                assert_eq!(items.len(), 1);
                assert_eq!((items[0].0, items[0].1, items[0].2), (2, 0x4000, 0x1000));
            }
            other => panic!("expected incremental unmapped plan on MAP, got {other:?}"),
        }

        // UNMAP/RELEASE bumps unmap_generation: full rebuild covers all placed textures.
        match ti.plan_index_update(2, 3, 1) {
            Some(IndexPlan::Full { unmap_gen, items, .. }) => {
                assert_eq!(unmap_gen, 2);
                assert_eq!(items.len(), 2);
            }
            other => panic!("expected full rebuild plan on UNMAP, got {other:?}"),
        }
    }

    #[test]
    fn background_drop_channel_reuses_single_worker_thread() {
        let tx1 = background_drop_sender().expect("drop worker sender");
        let tx2 = background_drop_sender().expect("drop worker sender");
        assert!(std::ptr::eq(tx1, tx2));

        let mut big_map = FastMap::default();
        for tile in 0..2050u32 {
            big_map.insert(tile, TileRefs::One(tile as usize));
        }
        enqueue_background_drop(big_map, FastSet::default());
        enqueue_background_drop(FastMap::default(), FastSet::default());
    }

    #[test]
    fn forget_many_removes_listed_resources_and_preserves_remaining() {
        let mut ti = TileIndex::new(false);
        ti.insert(1, 0x1000, 0x1000, placed_entry(1, 0x1000, 0x1000, false));
        ti.insert(2, 0x2000, 0x1000, placed_entry(2, 0x2000, 0x1000, false));
        ti.insert(3, 0x3000, 0x1000, placed_entry(3, 0x3000, 0x1000, false));
        ti.remove_many(&[1, 3]);
        assert!(!ti.entries.contains_key(&1));
        assert!(ti.entries.contains_key(&2));
        assert!(!ti.entries.contains_key(&3));

        // Also exercise the public forget_many helper through the global split registry.
        test_insert_global(101, 0x1000, 0x1000, false);
        test_insert_global(102, 0x2000, 0x1000, false);
        test_insert_global(103, 0x3000, 0x1000, false);
        forget_many(&[101, 103]);
        assert!(!is_placed(101) && is_placed(102) && !is_placed(103));
        forget_many(&[102]);
    }

    #[test]
    fn noted_transitions_are_isolated_from_registry_and_reset_clears_list() {
        with_noted_shard(0xABCD, |n| n.lists.clear());
        NOTED_COUNT.store(0, Ordering::Release);
        list_reset(0xABCD);
        submitted(&[0xABCD]);
        with_noted_shard(0xABCD, |n| {
            n.lists.insert(0xABCD, vec![(0x1111, 0, 1)]);
            NOTED_COUNT.fetch_add(1, Ordering::Release);
        });
        list_reset(0xABCD);
        assert!(with_noted_shard(0xABCD, |n| n.lists.get(&0xABCD).is_some_and(Vec::is_empty)));
    }

    #[test]
    fn placed_set_tracks_insert_remove_forget_many_and_skips_registry_for_non_placed() {
        test_insert_global(0x2010, 0x1000, 0x1000, false);
        test_insert_global(0x2020, 0x2000, 0x1000, false);
        test_insert_global(0x2030, 0x3000, 0x1000, false);
        assert!(is_placed(0x2010) && is_placed(0x2020) && is_placed(0x2030));
        assert!(!is_placed(0x9990));

        // Hold ADDR_INDEX and TILE_INDEX exclusive write locks: non-placed callers must return without blocking.
        let addr_guard = ADDR_INDEX.write().unwrap_or_else(|e| e.into_inner());
        let tile_guard = TILE_INDEX.write().unwrap_or_else(|e| e.into_inner());
        forget(0x9990);
        note_gpu_write(0x9990);
        note_transitions(0x7777, &[(0x9990, 0, 1), (0x99A0, 0, 2)]);
        forget_many(&[0x9990, 0x99A0]);
        drop(tile_guard);
        drop(addr_guard);

        assert!(remove_resource_everywhere(0x2010));
        assert!(!is_placed(0x2010));
        assert!(is_placed(0x2020) && is_placed(0x2030));

        forget_many(&[0x2020, 0x2030]);
        assert!(!is_placed(0x2020) && !is_placed(0x2030));
    }

    #[test]
    fn coarse_tile_filter_has_zero_false_negatives_and_rejects_unrelated_tiles() {
        let unmap_gen = crate::title_mem::unmap_generation();
        let map_gen = crate::title_mem::mapping_generation();
        let tile_gen = crate::title_mem::tile_generation();

        let mut empty_tiles = vec![0x10, 0x20];
        assert!(!filter_overlapping_tiles(&mut empty_tiles));

        let e1 = test_insert_global(0x3010, 0x1000, 0x1000, false);
        let e2 = test_insert_global(0x3020, 0x2000, 0x1000, false);
        let mut pending_tiles = vec![0x10, 0x400];
        assert!(filter_overlapping_tiles(&mut pending_tiles));
        assert_eq!(pending_tiles, vec![0x10, 0x400]);

        with_tile_write(|ti| {
            ti.apply_full_index(
                &[(0x3010, 0x1000, 0x1000, e1), (0x3020, 0x2000, 0x1000, e2)],
                &[Some(vec![0x10]), Some(vec![0x200])],
                unmap_gen,
                map_gen,
                tile_gen,
                true,
            );
        });
        assert!(may_overlap_tile(0x10));
        assert!(may_overlap_tile(0x200));
        assert!(!may_overlap_tile(0x400));

        let mut unrelated = vec![0x400, 0x800];
        assert!(!filter_overlapping_tiles(&mut unrelated));
        assert!(unrelated.is_empty());

        let mut mixed = vec![0x400, 0x10, 0x800, 0x200];
        assert!(filter_overlapping_tiles(&mut mixed));
        assert_eq!(mixed, vec![0x10, 0x200]);

        forget_many(&[0x3010]);
        assert!(!may_overlap_tile(0x10));
        assert!(may_overlap_tile(0x200));

        forget_many(&[0x3020]);
        assert!(!may_overlap_tile(0x200));
    }

    #[test]
    fn split_locks_allow_concurrent_note_write_submitted_and_tile_reads_without_blocking() {
        let e1 = test_insert_global(0x4010, 0x1_0000_0000, 0x10000, false);
        let e2 = test_insert_global(0x4020, 0x2_0000_0000, 0x80_0000, false); // > 4 MiB (large tier)

        // Hold TILE_INDEX write lock on thread 0: note_write, note_transitions, submitted, and describe must NOT block.
        let tile_write_guard = TILE_INDEX.write().unwrap_or_else(|e| e.into_inner());
        std::thread::scope(|s| {
            s.spawn(|| {
                note_write(0x3_0000_0000, 0x1000); // coarse miss
                note_write(0x1_0000_1000, 0x1000); // hits small tier
                note_write(0x2_0040_0000, 0x1000); // hits large tier
                note_transitions(0x4000, &[(0x4010, 0, 0x80)]);
                submitted(&[0x4000]);
                list_reset(0x4000);
            })
            .join()
            .unwrap();
        });
        drop(tile_write_guard);

        assert!(e1.wants_fill());
        assert!(e2.wants_fill());
        assert_eq!(e1.states.lock().unwrap().as_slice(), &[0x80]);

        // Hold ADDR_INDEX write lock: note_tiles (fast path), submitted, and describe must NOT block.
        let addr_write_guard = ADDR_INDEX.write().unwrap_or_else(|e| e.into_inner());
        std::thread::scope(|s| {
            s.spawn(|| {
                note_transitions(0x4001, &[(0x4020, 0, 0x400)]);
                submitted(&[0x4001]);
                list_reset(0x4001);
            })
            .join()
            .unwrap();
        });
        drop(addr_write_guard);
        assert_eq!(e2.states.lock().unwrap().as_slice(), &[0x400]);

        clear_global_pending_for(0x4010);
        clear_global_pending_for(0x4020);
        forget_many(&[0x4010, 0x4020]);
        assert!(!may_overlap_addr(0x1_0000_1000, 0x1_0000_2000));
        assert!(!may_overlap_addr(0x2_0040_0000, 0x2_0040_1000));
    }

    #[test]
    fn concurrent_invalidation_during_flush_preserves_dirty_state() {
        let entry = test_insert_global(0x5010, 0x4_0000_0000, 0x10000, true);
        let snap_gen = entry.dequeue_for_flush().expect("initial dirty");
        clear_global_pending_for(0x5010);

        // Simulate a concurrent note_write arriving while flush is detiling `snap_gen`.
        note_write(0x4_0000_0000, 4096);
        // Flush finishes the old `snap_gen` and settles it: newer write must remain dirty and queued.
        entry.settle_gen(snap_gen, true);
        assert!(entry.wants_fill(), "newer write must not be cleared by older snap_gen settle");
        assert!(HAS_PENDING_FILLS.load(Ordering::Acquire));

        let next_gen = entry.dequeue_for_flush().expect("requeued newer write");
        assert!(next_gen > snap_gen);
        clear_global_pending_for(0x5010);
        entry.settle_gen(next_gen, true);
        assert!(!entry.wants_fill());

        forget_many(&[0x5010]);
    }

    #[test]
    fn concurrent_workers_stress_note_write_transitions_and_pending_queue() {
        let base_res = 0x6000usize;
        let count = 32usize;
        for i in 0..count {
            test_insert_global(base_res + i * 16, 0x5_0000_0000 + (i as u64) * 0x2_0000, 0x10000, false);
        }
        std::thread::scope(|s| {
            for t in 0..8usize {
                s.spawn(move || {
                    for iter in 0..200usize {
                        let idx = (t * 7 + iter) % count;
                        let res = base_res + idx * 16;
                        let va = 0x5_0000_0000 + (idx as u64) * 0x2_0000;
                        note_write(va, 4096);
                        note_write(0x9_0000_0000 + (iter as u64) * 4096, 4096);
                        let list = 0x8000 + t * 16;
                        note_transitions(list, &[(res, 0, (iter as u32) & 0x7)]);
                        submitted(&[list]);
                        list_reset(list);
                        if iter % 10 == 0 {
                            let mut drained = Vec::new();
                            drain_pending_entries(&mut drained);
                            for e in drained {
                                if let Some(g) = e.dequeue_for_flush() {
                                    e.settle_gen(g, true);
                                }
                            }
                        }
                    }
                });
            }
        });
        let ids: Vec<usize> = (0..count).map(|i| base_res + i * 16).collect();
        let mut drained = Vec::new();
        drain_pending_entries(&mut drained);
        forget_many(&ids);
    }
}
