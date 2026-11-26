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
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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

/// Scratch buffers owned by [`Registry`] and reused across flushes, transitions, writes and tile
/// notifications to avoid per call heap allocations.
#[derive(Default)]
struct RegistryScratch {
    ready: Vec<(usize, u64)>,
    exhausted: Vec<usize>,
    pending_items: Vec<(usize, u64, u64)>,
    pending_spans: Vec<(u64, u64)>,
    translated: Vec<Option<(crate::title_mem::Span, bool)>>,
    resources: Vec<Option<ID3D12Resource>>,
    job_res: Vec<usize>,
    outcome: Vec<bool>,
    overlapping: Vec<usize>,
    changes: Vec<(usize, u32, u32)>,
}

/// A placed texture the layer fills.
struct Placed {
    va: u64,
    plan: Arc<Plan>,
    /// PC allocation size, for the informational owner statistics.
    bytes: u64,
    /// The states of the title's submitted work, one for all subresources or one each.
    states: Vec<u32>,
    tiles: Option<Vec<u32>>,
    dirty: bool,
    attempts: u32,
}

impl Placed {
    fn wants_fill(&self) -> bool {
        self.dirty && self.attempts < FILL_RETRIES
    }

    /// Records the outcome of a fill. A failed fill stops the retries until the next write.
    fn settle(&mut self, filled: bool) {
        if filled {
            self.dirty = false;
        } else {
            self.attempts = FILL_RETRIES;
        }
    }

    /// The title wrote the memory, so the texture is due again with a fresh set of retries.
    fn invalidate(&mut self) {
        self.dirty = true;
        self.attempts = 0;
        HAS_PENDING_FILLS.store(true, Ordering::Release);
    }
}

#[derive(Default)]
struct Registry {
    textures: BTreeMap<usize, Placed>,
    /// Resources that currently want a fill (`Placed::wants_fill`), so a flush visits only dirty
    /// textures instead of scanning the whole table.
    pending: HashSet<usize>,
    /// Transitions noted per PC command list as `(resource, subresource, state after)`.
    noted: HashMap<usize, Vec<(usize, u32, u32)>>,
    detiler: Option<Detiler>,
    unavailable: bool,
    /// Kinds of textures and refusal reasons logged so far, to log each once.
    logged: HashSet<(u32, u32)>,
    logged_refused: HashSet<String>,
    flushes: u64,
    detiled: u64,
    /// Textures put off because their span got pages whose mappings are not applied yet.
    deferred: u64,
    /// Resources by title address, and the largest span of any of them. A write finds the textures
    /// it overlaps without scanning the whole table. Scanning tens of thousands of textures under the lock for
    /// every DirectStorage request once stalled the loading of a large world.
    by_address: BTreeMap<u64, Vec<usize>>,
    largest: u64,
    /// Cached reverse index from mirror tile index to placed texture resources, maintained
    /// incrementally on insert and remove and rebuilt only when `title_mem::mapping_generation()`
    /// changes structurally.
    tile_to_textures: HashMap<u32, Vec<usize>>,
    unmapped_textures: HashSet<usize>,
    pending_index: HashSet<usize>,
    layout_gen: u64,
    last_tile_gen: u64,
    scratch: RegistryScratch,
}

impl Registry {
    fn insert(&mut self, res: usize, texture: Placed) {
        self.remove(res);
        self.by_address.entry(texture.va).or_default().push(res);
        self.largest = self.largest.max(texture.plan.size);
        self.pending_index.insert(res);
        if texture.wants_fill() {
            self.pending.insert(res);
            HAS_PENDING_FILLS.store(true, Ordering::Release);
        }
        crate::mem_owners::add(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, 1, texture.bytes);
        self.textures.insert(res, texture);
    }

    /// Drops `res`. Whether it was held.
    fn remove(&mut self, res: usize) -> bool {
        let Some(texture) = self.textures.remove(&res) else { return false };
        self.pending.remove(&res);
        self.pending_index.remove(&res);
        self.unmapped_textures.remove(&res);
        if let Some(tiles) = &texture.tiles {
            for &tile in tiles {
                if let Some(list) = self.tile_to_textures.get_mut(&tile) {
                    list.retain(|&r| r != res);
                    if list.is_empty() {
                        self.tile_to_textures.remove(&tile);
                    }
                }
            }
        }
        if self.pending.is_empty() {
            HAS_PENDING_FILLS.store(false, Ordering::Release);
        }
        crate::mem_owners::remove(crate::mem_owners::Kind::DetileTexture, crate::mem_owners::Segment::Local, 1, texture.bytes);
        if let Some(at_address) = self.by_address.get_mut(&texture.va) {
            at_address.retain(|&r| r != res);
            if at_address.is_empty() {
                self.by_address.remove(&texture.va);
            }
        }
        true
    }

    /// Marks `res` dirty and adds it to `pending`.
    fn invalidate_res(&mut self, res: usize) -> bool {
        if let Some(texture) = self.textures.get_mut(&res) {
            texture.invalidate();
            self.pending.insert(res);
            true
        } else {
            false
        }
    }

    /// Fills `out` with the resources whose memory overlaps `[start, end)`.
    fn overlapping_into(&self, start: u64, end: u64, out: &mut Vec<usize>) {
        out.clear();
        for (_, resources) in self.by_address.range(start.saturating_sub(self.largest)..end) {
            for &res in resources {
                if self.textures.get(&res).is_some_and(|t| start < t.va.saturating_add(t.plan.size)) {
                    out.push(res);
                }
            }
        }
    }

    /// The resources whose memory overlaps `[start, end)`.
    fn overlapping(&self, start: u64, end: u64) -> Vec<usize> {
        let mut out = Vec::new();
        self.overlapping_into(start, end, &mut out);
        out
    }

    /// Rebuilds the entire `tile_to_textures` index after a structural layout change.
    fn apply_full_index(
        &mut self,
        items: &[(usize, u64, u64)],
        tiles_per_item: &[Option<Vec<u32>>],
        layout_gen: u64,
        tile_gen: u64,
        layout_current: bool,
    ) {
        self.tile_to_textures.clear();
        self.unmapped_textures.clear();
        let mut live_matched = 0usize;
        for (&(res, va, _), maybe_tiles) in items.iter().zip(tiles_per_item) {
            let Some(texture) = self.textures.get_mut(&res).filter(|t| t.va == va) else {
                continue;
            };
            live_matched += 1;
            self.pending_index.remove(&res);
            match maybe_tiles {
                Some(tiles) => {
                    for &tile in tiles {
                        self.tile_to_textures.entry(tile).or_default().push(res);
                    }
                    texture.tiles = Some(tiles.clone());
                }
                None => {
                    texture.tiles = None;
                    self.unmapped_textures.insert(res);
                }
            }
        }
        if layout_current && live_matched == self.textures.len() {
            self.layout_gen = layout_gen;
            self.last_tile_gen = tile_gen;
        } else {
            self.layout_gen = 0;
        }
    }

    /// Incrementally updates `tile_to_textures` for newly inserted or previously unmapped textures.
    fn apply_incremental_index(
        &mut self,
        items: &[(usize, u64, u64)],
        tiles_per_item: &[Option<Vec<u32>>],
        tile_gen: u64,
        tile_current: bool,
    ) {
        for (&(res, va, _), maybe_tiles) in items.iter().zip(tiles_per_item) {
            let Some(texture) = self.textures.get_mut(&res).filter(|t| t.va == va) else {
                continue;
            };
            self.pending_index.remove(&res);
            self.unmapped_textures.remove(&res);
            if let Some(old_tiles) = texture.tiles.take() {
                for tile in old_tiles {
                    if let Some(list) = self.tile_to_textures.get_mut(&tile) {
                        list.retain(|&r| r != res);
                        if list.is_empty() {
                            self.tile_to_textures.remove(&tile);
                        }
                    }
                }
            }
            match maybe_tiles {
                Some(tiles) => {
                    for &tile in tiles {
                        self.tile_to_textures.entry(tile).or_default().push(res);
                    }
                    texture.tiles = Some(tiles.clone());
                }
                None => {
                    self.unmapped_textures.insert(res);
                }
            }
        }
        if tile_current {
            self.last_tile_gen = tile_gen;
        }
    }

    /// Marks textures whose mirror tiles appear in `tiles` using the cached reverse index.
    fn mark_tiles_from_index(&mut self, tiles: &[u32], report: bool) -> u32 {
        let mut hit = std::mem::take(&mut self.scratch.overlapping);
        hit.clear();
        for &tile in tiles {
            if let Some(list) = self.tile_to_textures.get(&tile) {
                hit.extend_from_slice(list);
            }
        }
        hit.sort_unstable();
        hit.dedup();
        let mut marked = 0u32;
        for &res in &hit {
            if let Some(texture) = self.textures.get_mut(&res) {
                texture.invalidate();
                self.pending.insert(res);
                marked += 1;
            }
        }
        self.scratch.overlapping = hit;
        if report {
            let waiting = self.textures.values().filter(|t| t.dirty && t.attempts == 0).count();
            let filled = self.textures.values().filter(|t| !t.dirty).count() as u32;
            let unmapped = self.unmapped_textures.len() as u32;
            let first_unmapped = self.unmapped_textures.iter().next().and_then(|res| self.textures.get(res)).map_or(0, |t| t.va);
            let untouched = (self.textures.len() as u32).saturating_sub(marked + unmapped);
            xs_log!(
                "detile: {} textures, {} pages written: {marked} marked, {waiting} wait for a fill, {untouched} not written ({filled} of them filled), {unmapped} with pages that are not mapped (first at {first_unmapped:#x})",
                self.textures.len(),
                tiles.len()
            );
        }
        marked
    }
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

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn with_registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    let mut guard = decant_trace::timed_lock("detile::REGISTRY", || REGISTRY.lock().unwrap_or_else(|e| e.into_inner()));
    f(guard.get_or_insert_with(Registry::default))
}

/// One line about the detile state for the device removal report. It never blocks, a busy registry
/// is reported as such.
pub fn summary() -> String {
    let Ok(guard) = REGISTRY.try_lock() else { return "detile: the table is busy".into() };
    let Some(r) = guard.as_ref() else { return "detile: no table".into() };
    let (frames, queues, staging) = r.detiler.as_ref().map_or((0, 0, 0), Detiler::footprint);
    let dirty = r.textures.values().filter(|t| t.dirty).count();
    let retried = r.textures.values().filter(|t| t.wants_fill()).count();
    format!(
        "detile: {} textures ({dirty} dirty, {retried} of them still tried), {} flushes, {} detiled, {} put off, {frames} lists on {queues} queues with {} MiB of staging",
        r.textures.len(),
        r.flushes,
        r.detiled,
        r.deferred,
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

/// The title placed a texture at `va` (device slots 44 and 133). `res` is the PC resource, `raw` the
/// description the title passed, `state` the initial state and `bytes` the PC allocation size. A
/// texture whose layout cannot be detiled is logged with its reason.
pub fn register(res: usize, va: u64, raw: &[u8; 0x38], state: u32, bytes: u64) {
    if !enabled() || va == 0 || res == 0 {
        return;
    }
    let v4 = hardware_v4();
    match detile_plan::plan_shared(raw, v4) {
        Ok(plan) => with_registry(|r| {
            let format = le_u32(raw, 0x20);
            if r.logged.len() < 64 && r.logged.insert((format, plan.mode)) {
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
            r.insert(res, Placed { va, plan, bytes, states: vec![state], tiles: None, dirty: true, attempts: 0 });
        }),
        Err(why) => with_registry(|r| {
            let reason = why.split(':').next().unwrap_or("");
            if r.logged_refused.len() < 64 && !r.logged_refused.contains(reason) {
                r.logged_refused.insert(reason.to_string());
                xs_log!("detile: a placed texture at {va:#x} is not detiled: {why}");
            }
        }),
    }
}

/// What the registry knows about PC resource `res`, for the descriptor diagnostics: its title
/// address and size, whether its pages are mapped, whether it waits for a fill and how many of the
/// first 64 KiB of its bytes are not zero. `None` for a resource the registry does not hold.
pub fn describe(res: usize) -> Option<String> {
    let (va, size, mode, dirty, attempts) = with_registry(|r| r.textures.get(&res).map(|t| (t.va, t.plan.size, t.plan.mode, t.dirty, t.attempts)))?;
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

/// The resource was released.
pub fn forget(res: usize) {
    with_registry(|r| {
        r.remove(res);
    });
}

/// Existing view and Unmap callers report possible writes. The placed memory is tracked by
/// `note_tiles` and explicit writes arrive through `note_write`, so nothing is left to do here.
pub fn mark_resource(_res: usize) {}

/// The title or a layer wrote `len` bytes at title address `va` (a DirectStorage completion or a
/// copy).
pub fn note_write(va: u64, len: u64) {
    if !enabled() || len == 0 {
        return;
    }
    let end = va.saturating_add(len);
    with_registry(|r| {
        let mut hit = std::mem::take(&mut r.scratch.overlapping);
        r.overlapping_into(va, end, &mut hit);
        for &res in &hit {
            r.invalidate_res(res);
        }
        r.scratch.overlapping = hit;
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

/// The title wrote the pages with mirror tiles `tiles` since its last submission (`title_mem::sync`).
/// Every texture that owns one of them becomes dirty, and so does one whose memory had no GPU
/// address at its first tries.
pub fn note_tiles(tiles: &[u32]) {
    static ANNOUNCED: AtomicU32 = AtomicU32::new(0);
    if !enabled() || tiles.is_empty() {
        return;
    }
    let report = probe_line_due();
    let current_layout = crate::title_mem::mapping_generation();
    let current_tile = crate::title_mem::tile_generation();
    let fast = with_registry(|r| {
        if r.textures.is_empty() {
            return Ok(0);
        }
        if r.layout_gen != current_layout {
            let items: Vec<(usize, u64, u64)> = r.textures.iter().map(|(&res, t)| (res, t.va, t.plan.size)).collect();
            let spans: Vec<(u64, u64)> = items.iter().map(|&(_, va, len)| (va, len)).collect();
            return Err((true, current_layout, items, spans));
        }
        let check_unmapped = current_tile != r.last_tile_gen && !r.unmapped_textures.is_empty();
        if !r.pending_index.is_empty() || check_unmapped {
            let mut items = Vec::with_capacity(r.pending_index.len() + if check_unmapped { r.unmapped_textures.len() } else { 0 });
            for &res in &r.pending_index {
                if let Some(t) = r.textures.get(&res) {
                    items.push((res, t.va, t.plan.size));
                }
            }
            if check_unmapped {
                for &res in &r.unmapped_textures {
                    if !r.pending_index.contains(&res)
                        && let Some(t) = r.textures.get(&res)
                    {
                        items.push((res, t.va, t.plan.size));
                    }
                }
            }
            let spans: Vec<(u64, u64)> = items.iter().map(|&(_, va, len)| (va, len)).collect();
            return Err((false, current_layout, items, spans));
        }
        Ok(r.mark_tiles_from_index(tiles, report))
    });
    let marked = match fast {
        Ok(marked) => marked,
        Err((full_rebuild, snap_layout, items, spans)) => {
            let build_layout = crate::title_mem::mapping_generation();
            let build_tile = crate::title_mem::tile_generation();
            let mut tiles_out = Vec::new();
            crate::title_mem::tiles_of_batch(&spans, &mut tiles_out);
            with_registry(|r| {
                if full_rebuild {
                    r.apply_full_index(&items, &tiles_out, build_layout, build_tile, crate::title_mem::mapping_generation() == build_layout);
                } else if crate::title_mem::mapping_generation() != snap_layout {
                    r.layout_gen = 0;
                } else {
                    r.apply_incremental_index(&items, &tiles_out, build_tile, crate::title_mem::tile_generation() == build_tile);
                }
                r.mark_tiles_from_index(tiles, report)
            })
        }
    };
    if marked > 0 && ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("detile: {marked} textures whose memory the title wrote ({} pages written) are filled again", tiles.len());
    }
}

/// The title's GPU work writes into the texture (a copy). The CPU view is stale from then on, so
/// the texture is not detiled any more.
pub fn note_gpu_write(res: usize) {
    with_registry(|r| {
        if r.remove(res) {
            static ANNOUNCED: AtomicU32 = AtomicU32::new(0);
            if ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 4 {
                xs_log!("detile: a texture the title copies into on the GPU is no longer detiled");
            }
        }
    });
}

/// The title recorded `transitions` (resource, subresource, state after) on PC list `list`.
pub fn note_transitions(list: usize, transitions: &[(usize, u32, u32)]) {
    if !enabled() || transitions.is_empty() {
        return;
    }
    with_registry(|r| {
        let mut iter = transitions.iter().filter(|(res, _, _)| r.textures.contains_key(res)).copied().peekable();
        if iter.peek().is_some() {
            r.noted.entry(list).or_default().extend(iter);
        }
    });
}

/// PC list `list` starts a new recording, so the transitions noted for it are dropped.
pub fn list_reset(list: usize) {
    with_registry(|r| {
        if let Some(entries) = r.noted.get_mut(&list) {
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
    if lists.is_empty() {
        return;
    }
    with_registry(|r| {
        if r.noted.is_empty() {
            return;
        }
        let mut changes = std::mem::take(&mut r.scratch.changes);
        changes.clear();
        for list in lists {
            if let Some(entries) = r.noted.get(list) {
                changes.extend_from_slice(entries);
            }
        }
        for &(res, sub, after) in &changes {
            if let Some(texture) = r.textures.get_mut(&res) {
                let count = (texture.plan.mips.len() as u32 * texture.plan.slices) as usize;
                apply_transition(&mut texture.states, count, sub, after);
            }
        }
        r.scratch.changes = changes;
    });
}

/// Detiles every dirty texture on the PC direct queue `queue`, ahead of the title's lists that run
/// on it.
///
/// # Safety
/// `queue` is a live PC command queue.
fn needs_fresh_sync(defer_fresh_spans: bool, translated: &[Option<(crate::title_mem::Span, bool)>]) -> bool {
    !defer_fresh_spans && translated.iter().any(|m| matches!(m, Some((_, true))))
}

fn collect_pending_items(r: &mut Registry, pending_items: &mut Vec<(usize, u64, u64)>, pending_spans: &mut Vec<(u64, u64)>, exhausted: &mut Vec<usize>) {
    pending_items.clear();
    pending_spans.clear();
    exhausted.clear();
    for &res in &r.pending {
        let Some(texture) = r.textures.get(&res) else {
            exhausted.push(res);
            continue;
        };
        if !texture.wants_fill() {
            exhausted.push(res);
            continue;
        }
        pending_items.push((res, texture.va, texture.plan.size));
        pending_spans.push((texture.va, texture.plan.size));
    }
    for &res in exhausted.iter() {
        r.pending.remove(&res);
    }
    if r.pending.is_empty() {
        HAS_PENDING_FILLS.store(false, Ordering::Release);
    }
}

pub unsafe fn flush(queue: *mut c_void) {
    if !enabled() || !HAS_PENDING_FILLS.load(Ordering::Acquire) {
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
    let collected = with_registry(|r| {
        if r.unavailable || r.pending.is_empty() {
            if r.pending.is_empty() {
                HAS_PENDING_FILLS.store(false, Ordering::Release);
            }
            return None;
        }
        if !crate::title_mem::enabled() {
            r.unavailable = true;
            xs_log!("detile: title memory is off (DECANT_D3D12X_TITLE_MEM=1), placed textures stay unfilled");
            return None;
        }
        let device = crate::title_mem::device()?;
        if r.detiler.is_none() {
            r.detiler = Detiler::new(&device);
            if r.detiler.is_none() {
                r.unavailable = true;
                xs_log!("detile: not available, placed textures stay unfilled");
                return None;
            }
        }
        let mut pending_items = std::mem::take(&mut r.scratch.pending_items);
        let mut pending_spans = std::mem::take(&mut r.scratch.pending_spans);
        let mut exhausted = std::mem::take(&mut r.scratch.exhausted);
        collect_pending_items(r, &mut pending_items, &mut pending_spans, &mut exhausted);
        if pending_items.is_empty() {
            r.scratch.pending_items = pending_items;
            r.scratch.pending_spans = pending_spans;
            r.scratch.exhausted = exhausted;
            return None;
        }
        let translated = std::mem::take(&mut r.scratch.translated);
        Some((pending_items, pending_spans, exhausted, translated))
    });
    let Some((mut pending_items, mut pending_spans, mut exhausted, mut translated)) = collected else {
        return;
    };

    // Translate outside REGISTRY so title_mem::STATE never nests under REGISTRY.
    crate::title_mem::translate_fresh_batch(&pending_spans, &mut translated);
    if needs_fresh_sync(tuning.defer_fresh_spans, &translated) {
        // SAFETY: `queue_raw` is a live direct queue and REGISTRY is not held.
        unsafe { crate::title_mem::sync(queue_raw) };
        with_registry(|r| collect_pending_items(r, &mut pending_items, &mut pending_spans, &mut exhausted));
        crate::title_mem::translate_fresh_batch(&pending_spans, &mut translated);
    }

    with_registry(|r| {
        let mut ready = std::mem::take(&mut r.scratch.ready);
        ready.clear();
        let mut resources = std::mem::take(&mut r.scratch.resources);
        resources.clear();
        let mut job_res = std::mem::take(&mut r.scratch.job_res);
        job_res.clear();
        let mut outcome = std::mem::take(&mut r.scratch.outcome);
        outcome.clear();

        let mut unreachable = 0u32;
        exhausted.clear();
        for (&(res, _, _), mapped) in pending_items.iter().zip(&translated) {
            if !r.textures.get(&res).is_some_and(Placed::wants_fill) {
                continue;
            }
            match mapped {
                // Fresh mappings go out with the next sync, before the next submission.
                Some((_, true)) if tuning.defer_fresh_spans => r.deferred += 1,
                Some((span, _)) => ready.push((res, span.gpu)),
                None => {
                    unreachable += 1;
                    if let Some(texture) = r.textures.get_mut(&res) {
                        texture.attempts += 1;
                        if !texture.wants_fill() {
                            exhausted.push(res);
                        }
                    }
                }
            }
        }
        r.scratch.pending_items = pending_items;
        r.scratch.pending_spans = pending_spans;
        r.scratch.translated = translated;
        for &res in &exhausted {
            r.pending.remove(&res);
        }
        r.scratch.exhausted = exhausted;
        if r.pending.is_empty() {
            HAS_PENDING_FILLS.store(false, Ordering::Release);
        }
        if unreachable > 0 && r.flushes < 4 {
            xs_log!("detile: {unreachable} textures have no GPU address for their memory yet");
        }
        if ready.is_empty() {
            r.scratch.ready = ready;
            r.scratch.resources = resources;
            r.scratch.job_res = job_res;
            r.scratch.outcome = outcome;
            return;
        }
        ready.sort_unstable_by_key(|&(res, _)| res);
        resources.extend(ready.iter().map(|&(res, _)| {
            let raw = res as *mut c_void;
            // SAFETY: the registry holds the entry, so the title still owns the resource (it is forgotten before it is released).
            unsafe { ID3D12Resource::from_raw_borrowed(&raw) }.cloned()
        }));
        let mut jobs: Vec<Job<'_>> = Vec::with_capacity(ready.len());
        for (&(res, gpu), resource) in ready.iter().zip(&resources) {
            if let (Some(texture), Some(placed)) = (resource, r.textures.get(&res)) {
                jobs.push(Job { texture: texture.clone(), src_gpu: gpu, plan: &placed.plan, states: &placed.states });
                job_res.push(res);
            }
        }
        let Some(detiler) = r.detiler.as_mut() else {
            drop(jobs);
            r.scratch.ready = ready;
            r.scratch.resources = resources;
            r.scratch.job_res = job_res;
            r.scratch.outcome = outcome;
            return;
        };
        detiler.run_into(queue, &jobs, tuning.wait_each_flush, &mut outcome);
        let (frames, queues, staging) = detiler.footprint();
        drop(jobs);
        let mut filled_bytes = 0u64;
        for (&res, &ok) in job_res.iter().zip(&outcome) {
            if let Some(texture) = r.textures.get_mut(&res) {
                texture.settle(ok);
                if !texture.wants_fill() {
                    r.pending.remove(&res);
                }
                if ok {
                    filled_bytes += texture.plan.size;
                    r.detiled += 1;
                }
            } else {
                r.pending.remove(&res);
            }
        }
        if r.pending.is_empty() {
            HAS_PENDING_FILLS.store(false, Ordering::Release);
        }
        r.flushes += 1;
        if r.flushes <= 10 || r.flushes.is_multiple_of(100) {
            xs_log!(
                "detile: flush {}: {} textures ({} MiB of tiled memory) in {} ms, {} in all, {} put off, {frames} lists on {queues} queues with {} MiB of staging",
                r.flushes,
                ready.len(),
                filled_bytes >> 20,
                started.elapsed().as_millis(),
                r.detiled,
                r.deferred,
                staging >> 20
            );
        }
        resources.clear();
        r.scratch.ready = ready;
        r.scratch.resources = resources;
        r.scratch.job_res = job_res;
        r.scratch.outcome = outcome;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_plan(size: u64) -> Plan {
        Plan { bytes_per_element: 4, linear: true, mode: 0, block_bits: 0, block_width: 1, block_height: 1, slice_size: 0, slices: 1, mips: Vec::new(), terms: Vec::new(), size }
    }

    fn placed(va: u64, size: u64) -> Placed {
        Placed { va, plan: Arc::new(flat_plan(size)), bytes: size, states: vec![0], tiles: None, dirty: false, attempts: 0 }
    }

    #[test]
    fn overlap_search_includes_a_big_texture_that_starts_far_below() {
        let mut r = Registry::default();
        r.insert(1, placed(0x1000, 0x10_0000));
        r.insert(2, placed(0x8_0000, 0x100));
        r.insert(3, placed(0x20_0000, 0x100));
        let mut hit = r.overlapping(0x8_0080, 0x8_0081);
        hit.sort_unstable();
        assert_eq!(hit, [1, 2]);
        assert!(r.overlapping(0x20_0100, 0x20_0200).is_empty(), "a write right behind a texture misses it");
    }

    #[test]
    fn inserting_again_moves_the_texture_and_removing_forgets_the_address() {
        let mut r = Registry::default();
        r.insert(1, placed(0x1000, 0x1000));
        r.insert(2, placed(0x8_0000, 0x100));
        r.insert(1, placed(0x30_0000, 0x10));
        assert_eq!(r.overlapping(0x1000, 0x1100), Vec::<usize>::new());
        assert_eq!(r.overlapping(0x30_0000, 0x30_0001), [1]);
        assert!(r.remove(2));
        assert!(!r.remove(2));
        assert!(!r.by_address.contains_key(&0x8_0000));
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
            with_registry(|r| match r.textures.get_mut(&res) {
                Some(t) if t.wants_fill() => {
                    t.settle(true);
                    r.pending.remove(&res);
                    1
                }
                _ => 0,
            })
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
        assert!(with_registry(|r| !r.textures.contains_key(&res)));
    }

    #[test]
    fn a_failed_fill_stops_the_retries_until_the_next_write() {
        let mut texture = placed(0x1000, 0x100);
        texture.dirty = true;
        assert!(texture.wants_fill());
        texture.settle(false);
        assert!(!texture.wants_fill());
        texture.invalidate();
        assert!(texture.wants_fill());
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
        let mut r = Registry::default();
        let mut dirty_one = placed(0x1000, 0x1000);
        dirty_one.dirty = true;
        r.insert(10, dirty_one);
        r.insert(20, placed(0x4000, 0x1000));
        assert!(r.pending.contains(&10) && !r.pending.contains(&20));
        assert!(r.invalidate_res(20));
        assert!(r.pending.contains(&20));
        assert!(r.remove(10));
        assert!(!r.pending.contains(&10) && r.pending.contains(&20));
        assert!(r.remove(20));
        assert!(r.pending.is_empty());
    }

    #[test]
    fn reverse_tile_index_updates_incrementally_and_rebuilds_on_layout_change() {
        let mut r = Registry::default();
        r.insert(1, placed(0x1000, 0x1000));
        r.insert(2, placed(0x4000, 0x1000));
        assert!(r.pending_index.contains(&1) && r.pending_index.contains(&2));

        // Initial full build at layout_gen 7, tile_gen 3.
        let items = vec![(1, 0x1000, 0x1000), (2, 0x4000, 0x1000)];
        r.apply_full_index(&items, &[Some(vec![10, 11]), None], 7, 3, true);
        assert!(r.pending_index.is_empty());
        assert_eq!(r.layout_gen, 7);
        assert_eq!(r.last_tile_gen, 3);
        assert!(r.unmapped_textures.contains(&2) && !r.unmapped_textures.contains(&1));

        // Inserting a new texture only adds it to pending_index without invalidating layout_gen.
        r.insert(3, placed(0x8000, 0x1000));
        assert_eq!(r.layout_gen, 7);
        assert!(r.pending_index.contains(&3));

        // Incremental update resolves texture 3 and previously unmapped texture 2 at tile_gen 4.
        r.apply_incremental_index(&[(3, 0x8000, 0x1000), (2, 0x4000, 0x1000)], &[Some(vec![12]), Some(vec![11, 13])], 4, true);
        assert!(r.pending_index.is_empty());
        assert!(r.unmapped_textures.is_empty());
        assert_eq!(r.last_tile_gen, 4);

        // Writing tile 99 marks nothing, and writing tiles 10 and 11 marks textures 1 and 2.
        assert_eq!(r.mark_tiles_from_index(&[99], false), 0);
        assert_eq!(r.mark_tiles_from_index(&[10, 11], false), 2);
        assert!(r.pending.contains(&1) && r.pending.contains(&2) && !r.pending.contains(&3));

        // Removing texture 2 cleans tile_to_textures incrementally without invalidating layout_gen.
        assert!(r.remove(2));
        assert_eq!(r.layout_gen, 7);
        assert!(!r.tile_to_textures.contains_key(&13));
        assert_eq!(r.mark_tiles_from_index(&[11, 13], false), 1);

        // Structural layout rebuild replaces all mappings at layout_gen 8.
        r.apply_full_index(&[(1, 0x1000, 0x1000), (3, 0x8000, 0x1000)], &[Some(vec![20]), Some(vec![21])], 8, 5, true);
        assert_eq!(r.layout_gen, 8);
        assert_eq!(r.mark_tiles_from_index(&[10], false), 0);
        assert_eq!(r.mark_tiles_from_index(&[20, 21], false), 2);
    }

    #[test]
    fn staging_cap_trims_largest_idle_frames_down_to_budget() {
        assert_eq!(staging_cap_from(None), DEFAULT_STAGING_CAP_MIB << 20);
        assert_eq!(staging_cap_from(Some("0")), 0);
        assert_eq!(staging_cap_from(Some(" 128 ")), 128 << 20);

        // Frame 0: busy (done_at 10 > completed 5), 256 MiB
        // Frame 1: idle, 128 MiB
        // Frame 2: idle, 512 MiB
        // Frame 3: idle, 128 MiB
        // Total = 1024 MiB, cap = 256 MiB -> drop Frame 2 (512 MiB) and Frame 1 (128 MiB) -> 384 -> drop Frame 3 (128 MiB) -> 256 MiB.
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
}
