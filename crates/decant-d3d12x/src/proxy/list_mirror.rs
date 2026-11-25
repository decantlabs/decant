//! The list methods that feed the user data mirror (`user_mirror`): the console slots that set root
//! signatures, tables and constants record into the shadow of the list and are relayed as before, and
//! the draws and dispatches write the 128 byte image into upload memory and set it as the root
//! constant buffer view of the mirror parameter when the root arguments changed.
//!
//! Console slots 33 to 40 are the PC slots 29 to 36 (`SetComputeRootSignature`,
//! `SetGraphicsRootSignature`, the two `SetRootDescriptorTable`, the two `SetRoot32BitConstant`, the
//! two `SetRoot32BitConstants`), slots 41 to 46 (root views) are in `list_gaps`. Slots 12 to 14 are
//! `DrawInstanced`, `DrawIndexedInstanced` and `Dispatch`. All of it costs one atomic load per call
//! until a root signature with the mirror exists, which only `DECANT_D3D12X_DXIL_OVERRIDE` makes.

use super::*;
use crate::slot_tags::{State, Sync, Took};
use crate::user_mirror::{self, Bind};
use crate::window_service;

type SetRootSig = unsafe extern "system" fn(Object, Object);
type SetTable = unsafe extern "system" fn(Object, u32, u64);
type SetConstant = unsafe extern "system" fn(Object, u32, u32, u32);
type SetConstants = unsafe extern "system" fn(Object, u32, u32, *const u32, u32);
type RootVa = unsafe extern "system" fn(Object, u32, u64);
type DrawInstanced = unsafe extern "system" fn(Object, u32, u32, u32, u32);
type DrawIndexedInstanced = unsafe extern "system" fn(Object, u32, u32, u32, i32, u32);
type DispatchPc = unsafe extern "system" fn(Object, u32, u32, u32);

/// The bytes of a root constant buffer view of the mirror: one 256 byte aligned block.
const MIRROR_ALIGN: usize = 256;

/// Records the PC heaps `heaps` as the bound ones of the list (console slot 32).
pub(super) fn note_heaps(this: *mut Proxy, heaps: &[Object]) {
    if !user_mirror::active() {
        return;
    }
    let refs: Vec<_> = heaps.iter().filter_map(|&h| user_mirror::heap_ref(h as usize)).collect();
    user_mirror::with_shadow(this as usize, |s| s.set_heaps(&refs));
}

/// Records a root view address (console slots 41 to 46). The mirror carries the PC GPU address
/// the address translates to (the importer's contract: the shader reaches a root descriptor's
/// buffer through the device memory window by that address), the title address when none.
pub(super) fn note_address(this: *mut Proxy, bind: Bind, index: u32, va: u64) {
    if user_mirror::active() {
        let pc_va = if va == 0 { 0 } else { super::pc_gpu_address(va, 4).unwrap_or(va) };
        if pc_va != 0 && window_service::active() {
            crate::service_mem::outside_window(pc_va);
        }
        user_mirror::with_shadow(this as usize, |s| s.set_address(bind, index, pc_va));
    }
}

/// Sets the window and service parameters of the root signature of `bind` on `this` before a draw or
/// dispatch, once per root signature set and window change. Without buffers (or without the device
/// memory) the parameters are set to 0 so that no root parameter stays unset.
fn prepare_service(this: *mut Proxy, bind: Bind) {
    static N: AtomicI32 = AtomicI32::new(0);
    // SAFETY: `this` is a live list proxy.
    let pc = unsafe { (*this).pc };
    // SAFETY: `pc` is the live PC list.
    let snap = unsafe { crate::service_mem::current(pc.cast()) };
    let epoch = snap.map_or(0, |s| s.epoch);
    let Some(sig) = window_service::pending(this as usize, bind, epoch) else { return };
    let (cbv, srv, uav) = if bind == Bind::Compute { (37, 39, 41) } else { (38, 40, 42) };
    let base = snap.and_then(|s| s.window).map_or(0, |w| w.base);
    let set = |slot: usize, index: Option<u32>, va: u64| {
        if let Some(index) = index {
            // SAFETY: the PC root signature of `bind` has a root descriptor of this kind at `index`.
            unsafe { method::<RootVa>(pc, slot)(pc, index, va) };
        }
    };
    set(uav, sig.window_uav, base);
    if sig.window_base.is_some() {
        let block = super::list_gaps::allocate_upload(this, 16, MIRROR_ALIGN).map(|cpu| (cpu, crate::upload_mem::translate(cpu as u64)));
        match block {
            Some((cpu, gpu)) if gpu != cpu as u64 => {
                let mut bytes = [0u8; 16];
                bytes[..8].copy_from_slice(&base.to_le_bytes());
                // SAFETY: `cpu` is a mapped block of at least 16 bytes that this list owns.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), cpu as *mut u8, 16) };
                set(cbv, sig.window_base, gpu);
            }
            _ => {
                // No block to read the base from: the parameter must still be set, so point it at the table.
                set(cbv, sig.window_base, snap.map_or(0, |s| s.table_gpu));
                if log_due(&N) {
                    xs_log!("window: no upload memory for the base address of the window, the root CBV is not the base");
                }
            }
        }
    }
    set(srv, sig.table, snap.map_or(0, |s| s.table_gpu));
    for (i, index) in sig.bits.iter().enumerate() {
        set(uav, *index, snap.map_or(0, |s| s.bits_gpu[i]));
    }
    window_service::mark_set(this as usize, bind, epoch);
}

/// Writes the mirror of `bind` before a draw or dispatch when the root signature has one and the root
/// arguments changed since it was last written, and sets the window and service parameters.
pub(super) fn prepare(this: *mut Proxy, bind: Bind) {
    static N: AtomicI32 = AtomicI32::new(0);
    if window_service::active() {
        prepare_service(this, bind);
    }
    if user_mirror::active() {
        sync_tables(this, bind);
        if crate::kind_check::on() {
            check_kinds(this, bind);
        }
    }
    super::heap_pages::flush();
    if !user_mirror::active() {
        return;
    }
    set_root_views(this, bind);
    let Some((index, image)) = user_mirror::prepare(this as usize, bind) else { return };
    let Some(cpu) = super::list_gaps::allocate_upload(this, user_mirror::BYTES, MIRROR_ALIGN) else {
        user_mirror::prepare_failed(this as usize, bind);
        return;
    };
    let gpu = crate::upload_mem::translate(cpu as u64);
    if gpu == cpu as u64 {
        user_mirror::prepare_failed(this as usize, bind);
        if log_due(&N) {
            xs_log!("console list: the user data mirror memory {cpu:#x} has no GPU address, the mirror is not set");
        }
        return;
    }
    // SAFETY: `cpu` is a mapped block of at least `BYTES` bytes that this list owns, `this` is a live
    // list proxy and the PC root signature of `bind` has the mirror at `index`.
    unsafe {
        std::ptr::copy_nonoverlapping(image.as_ptr().cast::<u8>(), cpu as *mut u8, user_mirror::BYTES);
        let pc = (*this).pc;
        let slot = if bind == Bind::Compute { 37 } else { 38 };
        method::<RootVa>(pc, slot)(pc, index, gpu);
    }
}

/// Outcome of resolving one internal metadata root SRV (`decant_heap_map` or `decant_view_shadow`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RootViewBind {
    /// Bound to the heap's metadata buffer GPU VA.
    Buffer(u64),
    /// Metadata buffer missing; bound to the zeroed fallback buffer GPU VA.
    Fallback(u64),
    /// Metadata buffer missing and no fallback GPU VA was available.
    Missing,
}

/// Resolves and binds the `decant_heap_map` and `decant_view_shadow` root SRVs for `views`.
/// When a metadata buffer is missing (or 0), requests a fallback GPU VA via `fallback` at most
/// once and binds it when non-zero so the root parameter is never left unbound or stale.
fn bind_root_views(
    views: user_mirror::RootViews,
    heap_map_va: Option<u64>,
    view_shadow_va: Option<u64>,
    mut fallback: impl FnMut() -> u64,
    mut bind: impl FnMut(u32, u64),
) -> (Option<RootViewBind>, Option<RootViewBind>) {
    let mut cached_fallback: Option<u64> = None;
    let mut resolve = |index: Option<u32>, va: Option<u64>| -> Option<RootViewBind> {
        let index = index?;
        match va.filter(|&a| a != 0) {
            Some(address) => {
                bind(index, address);
                Some(RootViewBind::Buffer(address))
            }
            None => {
                let fb = *cached_fallback.get_or_insert_with(&mut fallback);
                if fb != 0 {
                    bind(index, fb);
                    Some(RootViewBind::Fallback(fb))
                } else {
                    Some(RootViewBind::Missing)
                }
            }
        }
    };
    let map_state = resolve(views.heap_map, heap_map_va);
    let shadow_state = resolve(views.view_shadow, view_shadow_va);
    (map_state, shadow_state)
}

/// Sets the root views of `bind` (`user_mirror::RootViews`): the heap map `t0` to the page map of the
/// bound view heap (`title_heap::page_map_address`) and the view shadow `t1` to its shadow buffer
/// (`view_shadow`), when the root signature has them and the root signature or the heaps changed
/// since they were set. A directly translated shader reads the host index of a console heap index
/// in the first and the record stride and count of a heap slot in the second. When either metadata
/// buffer is missing, the device-owned zeroed fallback buffer is bound instead of leaving the root
/// SRV unbound or stale.
fn set_root_views(this: *mut Proxy, bind: Bind) {
    static NO_MAP: AtomicI32 = AtomicI32::new(0);
    static NO_SHADOW: AtomicI32 = AtomicI32::new(0);
    let Some((views, heap_start)) = user_mirror::prepare_root_views(this as usize, bind) else { return };
    let slot = if bind == Bind::Compute { 39 } else { 40 };
    let map_va = views.heap_map.and_then(|_| crate::title_heap::page_map_address(heap_start));
    let shadow_va = views.view_shadow.and_then(|_| crate::view_shadow::buffer_gpu(heap_start));
    let (map_state, shadow_state) = bind_root_views(
        views,
        map_va,
        shadow_va,
        || {
            // SAFETY: `this` is a live list proxy, its PC list is live.
            unsafe { crate::fallback::address((*this).pc) }
        },
        |index, address| {
            // SAFETY: `this` is a live list proxy, the PC root signature of `bind` has a root shader
            // resource view at `index`, and `address` is the GPU address of a live buffer.
            unsafe {
                let pc = (*this).pc;
                method::<RootVa>(pc, slot)(pc, index, address);
            }
        },
    );
    match map_state {
        Some(RootViewBind::Fallback(fb)) if log_due(&NO_MAP) => {
            xs_log!("console list: the view heap at GPU handle {heap_start:#x} has no page map, binding zeroed fallback buffer at {fb:#x}");
        }
        Some(RootViewBind::Missing) if log_due(&NO_MAP) => {
            xs_log!("console list: the view heap at GPU handle {heap_start:#x} has no page map and no zeroed fallback buffer, the heap map is not set");
        }
        _ => {}
    }
    match shadow_state {
        Some(RootViewBind::Fallback(fb)) if log_due(&NO_SHADOW) => {
            xs_log!("console list: the view heap at GPU handle {heap_start:#x} has no view shadow, binding zeroed fallback buffer at {fb:#x}");
        }
        Some(RootViewBind::Missing) if log_due(&NO_SHADOW) => {
            xs_log!("console list: the view heap at GPU handle {heap_start:#x} has no view shadow and no zeroed fallback buffer, the view shadow is not set");
        }
        _ => {}
    }
}

/// Size and format of the resource of a view for the descriptor diagnostics, as they were when the
/// view was made (the resource itself may be gone), "-" for none.
fn describe_resource(v: &crate::view_diag::View) -> String {
    if v.res == 0 {
        return "-".into();
    }
    let (w, h, f, flags) = v.shape;
    let detile = crate::detile::describe(v.res).map(|d| format!(", {d}")).unwrap_or_default();
    format!("{:#x} {w}x{h} format {f} flags {flags:#x}{detile}", v.res)
}

/// Logs the user data mirror of the `n`th draw of its render target and pipeline pair when the
/// descriptor diagnostics want it: every table with the views of its first slots and the first
/// bytes of the same slots in the title's heap memory.
fn describe_draw(this: *mut Proxy, n: u64) {
    describe_bind(this, n, Bind::Graphics);
}

/// The same for the `n`th dispatch of a compute pipeline, which names the views a compute pass
/// reads and writes (eight slots of each table, a pass that fills textures has its outputs there).
pub(super) fn describe_dispatch(this: *mut Proxy, n: u64) {
    describe_bind(this, n, Bind::Compute);
}

fn describe_bind(this: *mut Proxy, n: u64, bind: Bind) {
    if n == 0 || !crate::view_diag::wanted(n) {
        return;
    }
    let list = this as usize;
    let (rt, pso) = crate::rt_diag::list_state(list);
    let (what, slots) = match bind {
        Bind::Graphics => ("draw", 4u32),
        Bind::Compute => ("dispatch", 8u32),
    };
    let object = user_mirror::with_shadow(list, |s| s.signature(bind));
    let Some(sig) = user_mirror::signature(object) else {
        xs_log!("view probe: {what} {n} of target {rt:#x} pipeline {pso:#x}: root signature {object:#x} has no mirror");
        return;
    };
    let (image, heaps) = user_mirror::with_shadow(list, |s| (s.image(bind, &sig), s.heaps()));
    if let Some(layout) = crate::view_diag::signature_once(object) {
        xs_log!("view probe: root signature {object:#x} of pipeline {pso:#x}: {layout}");
    }
    let mut out = format!("view probe: {what} {n} of target {rt:#x} pipeline {pso:#x}, mirror {:x?}", &image[8..]);
    let packing = match bind {
        Bind::Graphics => &sig.graphics,
        Bind::Compute => &sig.compute,
    };
    for (i, placed) in packing.placed.iter().enumerate() {
        let Some(p) = placed else { continue };
        let user_mirror::Slot::Table { sampler } = p.slot else { continue };
        let index = image[p.sgpr as usize];
        out += &format!("; param {i} s{} table index {index}", p.sgpr);
        if sampler {
            continue;
        }
        let Some(heap) = heaps[0].and_then(|h| crate::view_diag::heap(h.gpu_start)) else {
            out += " (no heap)";
            continue;
        };
        // The mirror holds the host index, the slots below are the title's.
        let Some(index) = heaps[0].and_then(|h| crate::title_heap::console_index(h.gpu_start, index)) else {
            out += " (host slot holds no console page)";
            continue;
        };
        if index != image[p.sgpr as usize] {
            out += &format!(" (console slot {index})");
        }
        if index >= heap.host_count {
            out += &format!(" past the {} host slots", heap.host_count);
        }
        for k in 0..slots {
            let slot = index + k;
            let host = heap.host_cpu + slot as usize * heap.increment as usize;
            let view = if slot < heap.host_count { crate::view_diag::view(host) } else { None };
            let made = match view {
                Some(v) if v.res == 0 && v.va != 0 && (crate::title_mem::writable(v.va as usize, 32) || crate::title_heap::translate_cpu(v.va as usize) != Some(v.va as usize)) => {
                    // SAFETY: title memory that `xmem.dll` maps or the memory of a title heap, readable.
                    let words: Vec<u32> = (0..8).map(|w| unsafe { std::ptr::read_unaligned((v.va as usize + w * 4) as *const u32) }).collect();
                    {
                        crate::view_diag::watch(v.va);
                        format!("{} va {:#x} at PC {:x?} holds {words:x?}", v.kind, v.va, super::pc_gpu_address(v.va, 256))
                    }
                }
                Some(v) => format!("{} fmt {} dim {} va {:#x} res {}", v.kind, v.format, v.dim, v.va, describe_resource(&v)),
                None => "never written".into(),
            };
            let title = if heap.title_base != 0 && slot < heap.title_count {
                let at = heap.title_base + slot as usize * heap.increment as usize;
                // SAFETY: the title's heap memory is committed and readable for `title_count` slots
                // (checked when the heap was registered) and lives as long as the heap.
                let words: Vec<u64> = (0..heap.increment as usize / 8).map(|w| unsafe { std::ptr::read_unaligned((at + w * 8) as *const u64) }).collect();
                format!("{words:x?}")
            } else {
                "-".into()
            };
            out += &format!(" [{k}: {made}; title {title}]");
        }
    }
    xs_log!("{out}");
}

/// The descriptor kind check (`kind_check`, `DECANT_D3D12X_KIND_CHECK`) of the pipeline of `bind` on the list,
/// after the table sync: every table offset its shaders read is compared with the host slot, and each
/// mismatch is logged (rate limited, see `kind_check::count`).
fn check_kinds(this: *mut Proxy, bind: Bind) {
    use crate::kind_check as kc;
    let list = this as usize;
    let Some((pso, decl)) = kc::pipeline_of(list, bind == Bind::Graphics) else { return };
    if decl.stages.iter().all(|s| s.views.is_empty()) {
        return;
    }
    let object = user_mirror::with_shadow(list, |s| s.signature(bind));
    let Some(sig) = user_mirror::signature(object) else { return };
    let (image, heaps) = user_mirror::with_shadow(list, |s| (s.image(bind, &sig), s.heaps()));
    let Some(heap_ref) = heaps[0] else { return };
    let Some(heap) = crate::view_diag::heap(heap_ref.gpu_start) else { return };
    let packing = match bind {
        Bind::Graphics => &sig.graphics,
        Bind::Compute => &sig.compute,
    };
    // The root parameter that is the resource table in the mirror dword: only such a dword is a table.
    let param_of = |dword: u8| packing.placed.iter().position(|p| p.as_ref().is_some_and(|p| p.sgpr as usize == usize::from(dword) && matches!(p.slot, user_mirror::Slot::Table { sampler: false })));
    // The console index of a table (the host index when it moved), and the title's words of a slot.
    let base_of = |dword: u8| -> Option<(u32, u32, bool)> {
        param_of(dword)?;
        let host = *image.get(usize::from(dword))?;
        match crate::title_heap::console_index(heap_ref.gpu_start, host) {
            Some(console) => Some((host, console, true)),
            None => Some((host, host, false)),
        }
    };
    let words_of = |index: u32| -> Option<Vec<u64>> {
        if heap.title_base == 0 || index >= heap.title_count {
            return None;
        }
        let at = heap.title_base + index as usize * heap.increment as usize;
        // SAFETY: the title's heap memory is committed and readable for `title_count` slots (checked when
        // the heap was registered) and lives as long as the heap.
        Some((0..heap.increment as usize / 8).map(|w| unsafe { std::ptr::read_unaligned((at + w * 8) as *const u64) }).collect())
    };
    let slot_of = |dword: u8, slot: u32| -> Option<kc::Slot> {
        let (_, console, in_title) = base_of(dword)?;
        let index = console.checked_add(slot)?;
        if index >= heap.host_count {
            return None;
        }
        let host = heap.host_cpu + index as usize * heap.increment as usize;
        let held = match crate::view_diag::view(host) {
            None => None,
            Some(v) => Some(kc::Held::from_view(v.kind, v.res, v.va, v.dim)?),
        };
        let title_zero = if in_title { words_of(index).map(|w| w.iter().all(|&x| x == 0)) } else { None };
        Some(kc::Slot { held, title_zero })
    };
    for m in kc::mismatches(&decl, slot_of) {
        let param = param_of(m.want.dword).map_or(u32::MAX, |p| p as u32);
        let (due, total) = kc::count(pso, param, m.want.slot);
        if let kc::Due::Line(n) = due {
            let (host, console, in_title) = base_of(m.want.dword).unwrap_or((0, 0, false));
            let words = if in_title { words_of(console.saturating_add(m.want.slot)) } else { None };
            let source = match m.stage.source {
                kc::Source::Direct => "direct",
                kc::Source::Override => "override",
            };
            kc::log(&format!(
                "kind check: list {list:#x} pipeline {pso:#x} {} {:016x} ({source}) param {param} table base host index {host} console index {console} offset {}: wants {}, found {}; title words {}; mismatch {n} of this slot",
                m.stage.stage,
                m.stage.hash,
                m.want.slot,
                m.want.describe(),
                m.found,
                words.map_or("not known".to_string(), |w| format!("{w:x?}")),
            ));
        }
        if let Some(line) = total {
            kc::log(&line);
        }
    }
}

/// The zero slot null view (`zero_null`, `DECANT_D3D12X_ZERO_SLOT_NULL`): before the sync of a draw, each
/// table offset the bound pipeline's shaders declare whose title words are all zero gets a null view of
/// the declared kind in the host slot the sync leaves alone (the slot has no tag). Written once for each
/// slot until it is tagged or copied over (`zero_null::due`, `zero_null::mark`). The sync, which follows, copies the slot
/// into a table it moves for this draw, so the moved table has it too. Slots with a tag or with title
/// words are not touched.
fn null_zero_slots(this: *mut Proxy, bind: Bind) {
    use crate::{kind_check as kc, zero_null as zn};
    let list = this as usize;
    let Some((pso, decl)) = kc::pipeline_of(list, bind == Bind::Graphics) else { return };
    if decl.stages.iter().all(|s| s.views.is_empty()) {
        return;
    }
    let (object, heaps) = user_mirror::with_shadow(list, |s| (s.signature(bind), s.heaps()));
    let Some(sig) = user_mirror::signature(object) else { return };
    let Some(heap_ref) = heaps[0] else { return };
    let Some(v) = crate::title_heap::visible_for(heap_ref.gpu_start) else { return };
    let inc = v.increment as usize;
    if v.heap_type != 0 || v.title_base == 0 || inc < 8 {
        return;
    }
    let packing = match bind {
        Bind::Graphics => &sig.graphics,
        Bind::Compute => &sig.compute,
    };
    // The root parameter that is the resource table in the mirror dword: only such a dword is a table.
    let param_of = |dword: u8| packing.placed.iter().position(|p| p.as_ref().is_some_and(|p| p.sgpr as usize == usize::from(dword) && matches!(p.slot, user_mirror::Slot::Table { sampler: false })));
    let tables: Vec<Option<u64>> = user_mirror::with_shadow(list, |s| (0..packing.placed.len()).map(|p| s.table_handle(bind, p)).collect());
    let limit = u64::from(v.title_count.min(v.host_count));
    for stage in &decl.stages {
        for want in &stage.views {
            if zn::null_of(want).is_none() {
                continue;
            }
            let Some(param) = param_of(want.dword) else { continue };
            let Some(handle) = tables.get(param).copied().flatten().filter(|&h| h >= v.gpu_start) else { continue };
            // where the sync reads the title's slots and writes the host slots: the console handle of the table
            let Some(console) = crate::title_heap::console_handle(handle) else { continue };
            if console < v.gpu_start {
                continue;
            }
            let first = (console - v.gpu_start) / inc as u64;
            let index = first + u64::from(want.slot);
            if index >= limit {
                continue;
            }
            let off = index as usize * inc;
            let mut words = [0u64; 4];
            let n = (inc / 8).min(words.len());
            for (w, word) in words.iter_mut().enumerate().take(n) {
                // SAFETY: the title's heap memory is committed and readable for `title_count` slots (checked
                // when the heap was registered), `index` is below it, and a slot has `inc` bytes.
                *word = unsafe { std::ptr::read_unaligned((v.title_base + off + w * 8) as *const u64) };
            }
            let Some(null) = zn::decide(want, &words[..n]) else { continue };
            let dst = v.host_cpu + off;
            if !zn::due(dst, null) {
                continue;
            }
            let before = if crate::view_diag::enabled() { crate::view_diag::view(dst).map_or("no record".to_string(), |h| format!("{} dim {}", h.kind, h.dim)) } else { "not recorded".to_string() };
            if !write_null(this, &v, dst, null) {
                continue;
            }
            // marked after the write: a list on another thread that checked in between writes the same view too
            zn::mark(dst, null);
            let (total, line, total_line) = zn::count();
            if line {
                xs_log!(
                    "zero slot null: list {list:#x} pipeline {pso:#x} {} {:016x} param {param} table base console index {first} (handle {handle:#x}) offset {}: title words zero, wrote {} into host slot {dst:#x}, which held {before}; write {total}",
                    stage.stage,
                    stage.hash,
                    want.slot,
                    null.name()
                );
            }
            if total_line {
                xs_log!("zero slot null: {total} null views written so far");
            }
        }
    }
}

/// Writes the null view `null` into host slot `dst` of heap `v` (and the records that follow a view). `false`
/// when none could be made (the zeroed fallback buffer for a constant buffer view does not exist).
fn write_null(this: *mut Proxy, v: &crate::title_heap::Visible, dst: usize, null: crate::zero_null::Null) -> bool {
    use crate::zero_null::{self as zn, Null};
    let device = v.device as Object;
    let mut dim = 0;
    let kind = match null {
        Null::Cbv => {
            // SAFETY: `this` is a live list proxy, its PC list is live.
            let location = unsafe { crate::fallback::address((*this).pc) };
            if location == 0 {
                return false;
            }
            let mut desc = [0u8; 16];
            desc[..8].copy_from_slice(&location.to_le_bytes());
            desc[8..12].copy_from_slice(&65536u32.to_le_bytes());
            // SAFETY: `device` is the live PC device of the heap, `dst` a host slot of a view heap.
            unsafe { method::<CreateCbv>(device, 17)(device, desc.as_ptr().cast(), dst) };
            "cbv"
        }
        Null::Srv(d) => {
            let Some(desc) = zn::srv_desc(d) else { return false };
            dim = u32::from_le_bytes([desc[4], desc[5], desc[6], desc[7]]);
            // SAFETY: as above, `desc` is a shader resource view description of 0x28 bytes, of no resource.
            unsafe { method::<CreateSrv>(device, 18)(device, std::ptr::null_mut(), desc.as_ptr().cast(), dst) };
            "srv at"
        }
        Null::Uav(d) => {
            let Some(desc) = zn::uav_desc(d) else { return false };
            dim = u32::from_le_bytes([desc[4], desc[5], desc[6], desc[7]]);
            // SAFETY: as above, `desc` is an unordered access view description of 0x28 bytes, of no resource.
            unsafe { method::<CreateUav>(device, 19)(device, std::ptr::null_mut(), std::ptr::null_mut(), desc.as_ptr().cast(), dst) };
            "uav at"
        }
    };
    crate::title_heap::note_cbv(dst, None);
    crate::view_shadow::view_made(dst, [0; 4]);
    // a null view for the kind check too: va 0 is null for a constant buffer view
    crate::view_diag::view_made(dst, crate::view_diag::View { res: 0, shape: (0, 0, 0, 0), format: 0, dim, va: 0, kind });
    // the slot of a paged heap is in its backing heap: the flush that follows brings it to the host pages
    crate::title_heap::written(dst, 1);
    true
}

/// Copies the host views of the tagged title slots of the tables set since the last draw or dispatch
/// into the host slots the shaders read (`title_heap::table_copies`), for a paged heap into its
/// backing heap, from where the flush after this brings them to the host pages. A slot of an earlier
/// draw's table in the window, whose tag's source slot holds a newer view by now, keeps the view it
/// took (`slot_tags`).
fn sync_tables(this: *mut Proxy, bind: Bind) {
    let hosts = user_mirror::with_shadow(this as usize, |s| s.take_unsynced(bind));
    restore_moved(this, bind, &hosts);
    if crate::zero_null::on() {
        null_zero_slots(this, bind);
    }
    for host in hosts {
        let Some(handle) = crate::title_heap::console_handle(host) else { continue };
        let Some(v) = crate::title_heap::visible_for(handle) else { continue };
        // SAFETY: the window of a table stays inside the title memory of its heap (committed and
        // readable for `title_count` slots, checked when the heap was registered).
        let copies = crate::title_heap::table_copies(&v, handle, |a| unsafe { std::ptr::read_unaligned(a as *const [u64; 3]) });
        // constant buffer copies that would change what an earlier draw of this list reads
        let mut own: Vec<(usize, u64, u32)> = Vec::new();
        for c in copies {
            let (dst, src) = (c.dst, c.src());
            let state = crate::slot_tags::state(dst, c.word);
            let now = if v.heap_type == 0 { crate::title_heap::unaligned_cbv(src) } else { None };
            note_old_tag(this, &v, handle, host, &c, state);
            let buffer = match crate::slot_tags::sync(state, now, |t| crate::cbv_copies::is_live(t.list, t.recording)) {
                Sync::Keep => continue,
                Sync::View => None,
                Sync::Buffer(range) => Some(range),
                // the title may have given the memory back since
                Sync::Again((va, size)) if !readable(va, size) => continue,
                Sync::Again(range) => {
                    crate::slot_tags::count(&crate::slot_tags::COUNTS.again_copied);
                    Some(range)
                }
            };
            let conflict = buffer.is_some_and(|(va, size)| crate::cbv_copies::conflicts(this as usize, dst, va, size));
            // the copy of a table that moves is not in the table's own slot
            let recording = if buffer.is_some() && !conflict { crate::cbv_copies::recording(this as usize) } else { 0 };
            crate::slot_tags::taken(dst, Took { word: c.word, list: this as usize, present: crate::title_heap::presents(), buffer, recording });
            if let Some((va, size)) = buffer {
                if conflict {
                    own.push((dst, va, size));
                    continue;
                }
                copied_cbv(this, v.device as Object, dst, va, size);
                crate::title_heap::written(dst, 1);
                continue;
            }
            if dst == src {
                continue;
            }
            // SAFETY: `v.device` is the live PC device of the heap, both slots are host slots of the
            // heap type.
            unsafe { method::<CopyDescriptorsSimplePc>(v.device as Object, 24)(v.device as Object, 1, dst, src, v.heap_type) };
            if v.heap_type == 0 {
                super::heap_pages::shadow_copied(dst, src, 1);
            }
            crate::title_heap::written(dst, 1);
            crate::view_diag::copied(dst, src, 1, v.increment);
            crate::zero_null::forget(dst, 1, v.increment);
        }
        if !own.is_empty() {
            move_table(this, bind, &v, host, handle, &own);
        }
    }
}

/// Whether the `size` bytes of a constant buffer at title address `va` that a slot took before can be read
/// now: title memory `xmem.dll` maps, or the memory of a title heap (the title's constants of a draw lie
/// beside the table of the draw).
fn readable(va: u64, size: u32) -> bool {
    let (va, size) = (va as usize, size as usize);
    let in_heap = |a: usize| crate::title_heap::translate_cpu(a) != Some(a);
    crate::title_mem::writable(va, size) || (in_heap(va) && in_heap(va + size.max(1) - 1))
}

/// Counts and logs (rate limited) a tag of the window of the table at `handle` (host handle `host`)
/// whose source slot holds a newer view (`slot_tags`).
fn note_old_tag(this: *mut Proxy, v: &crate::title_heap::Visible, handle: u64, host: u64, c: &crate::title_heap::TableSlot, state: State) {
    static KEPT: AtomicI32 = AtomicI32::new(0);
    static ELSEWHERE: AtomicI32 = AtomicI32::new(0);
    static LOST: AtomicI32 = AtomicI32::new(0);
    use crate::slot_tags::{COUNTS, count};
    let list = this as usize;
    let first = v.host_cpu + (handle - v.gpu_start) as usize;
    let at = (c.dst - first) / v.increment as usize;
    let src = c.src();
    match state {
        State::Current => {}
        State::Held(t) if t.list != list || t.present != crate::title_heap::presents() => {
            count(&COUNTS.elsewhere);
            if log_due(&ELSEWHERE) {
                xs_log!("console list {list:#x}: the table at {host:#x} reaches slot {at}, which list {:#x} took at present {} and whose source slot {src:#x} holds a newer view now", t.list, t.present);
            }
        }
        State::Held(t) => {
            count(&COUNTS.kept);
            if log_due(&KEPT) {
                let what = |b: Option<(u64, u32)>| b.map_or("a buffer on 256 bytes or another view".to_owned(), |(va, size)| format!("a copy of {va:#x} ({size} bytes)"));
                xs_log!(
                    "console list {list:#x}: an earlier draw's view is kept: the table at {host:#x} reaches slot {at} of its table, the slot holds {}, source slot {src:#x} holds {} now",
                    what(t.buffer),
                    what(crate::title_heap::unaligned_cbv(src))
                );
            }
        }
        State::Lost => {
            count(&COUNTS.lost);
            if log_due(&LOST) {
                xs_log!("console list {list:#x}: slot {at} of the table at {host:#x} never took the view its tag names, source slot {src:#x} holds a newer one now");
            }
        }
    }
}

/// The table the title bound at host handle `host` (console handle `console`, of heap `v`) holds
/// constant buffer copies `own` whose slots an earlier draw of this list reads with other bytes: on
/// the console each draw reads what its table held when it was recorded, on the PC every draw of
/// the list reads the slot as the GPU finds it. So the table moves for this draw to host slots of
/// its own (`title_heap::move_table`) with the copies there, and back before the next draw. Without
/// room for it, the copies go into the shared slots as before.
fn move_table(this: *mut Proxy, bind: Bind, v: &crate::title_heap::Visible, host: u64, console: u64, own: &[(usize, u64, u32)]) {
    static MOVED: AtomicI32 = AtomicI32::new(0);
    static WRAPPED: AtomicI32 = AtomicI32::new(0);
    let inc = v.increment as usize;
    let first = v.host_cpu + ((console - v.gpu_start) / inc as u64) as usize * inc;
    let Some(m) = crate::title_heap::move_table(console, crate::title_heap::TABLE_WINDOW, &mut super::heap_pages::copy_views) else {
        static FULL: AtomicI32 = AtomicI32::new(0);
        let full = crate::title_heap::ring_full();
        if full != 0 && full.is_power_of_two() && log_due(&FULL) {
            xs_log!("console list {:#x}: the ring of moved tables holds only tables written less than {} presents ago, {full} tables not moved (copied into the shared slots)", this as usize, crate::title_heap::KEEP_PRESENTS);
        }
        for &(dst, va, size) in own {
            copied_cbv(this, v.device as Object, dst, va, size);
            crate::title_heap::written(dst, 1);
        }
        return;
    };
    for &(dst, va, size) in own {
        copied_cbv(this, v.device as Object, m.cpu + (dst - first), va, size);
    }
    let params = user_mirror::with_shadow(this as usize, |s| s.retarget(bind, host, m.gpu));
    set_pc_tables(this, bind, &params, m.gpu);
    crate::cbv_copies::moved(this as usize, bind as usize, host, m.gpu);
    if log_due(&MOVED) {
        xs_log!("console list {:#x}: the table at {host:#x} moves to {:#x} for one draw, {} constant buffer copies differ from an earlier draw's, the first in slot {}", this as usize, m.gpu, own.len(), own.first().map_or(0, |o| (o.0 - first) / inc));
    }
    if m.wrapped && log_due(&WRAPPED) {
        xs_log!("console list {:#x}: the ring of moved tables came round, the oldest moved tables are written again", this as usize);
    }
}

/// Points the tables moved for the last draw of `bind` back at the title's slots. `set` are the tables
/// set since then.
fn restore_moved(this: *mut Proxy, bind: Bind, set: &[u64]) {
    static AGAIN: AtomicI32 = AtomicI32::new(0);
    for (host, moved) in crate::cbv_copies::take_moved(this as usize, bind as usize) {
        let params = user_mirror::with_shadow(this as usize, |s| s.retarget(bind, moved, host));
        if !params.is_empty() && !set.contains(&host) {
            crate::slot_tags::count(&crate::slot_tags::COUNTS.again);
            if log_due(&AGAIN) {
                xs_log!("console list {:#x}: the table at {host:#x}, moved for the last draw, is read again without being set, from the title's slots", this as usize);
            }
        }
        set_pc_tables(this, bind, &params, host);
    }
}

/// Sets root parameters `params` of `bind` of the PC list to the table at `handle`.
fn set_pc_tables(this: *mut Proxy, bind: Bind, params: &[u32], handle: u64) {
    let slot = if bind == Bind::Compute { 31 } else { 32 };
    for &index in params {
        // SAFETY: `this` is a live list proxy; the PC slot takes (index, handle).
        unsafe {
            let pc = (*this).pc;
            method::<SetTable>(pc, slot)(pc, index, handle);
        }
    }
}

/// Makes a constant buffer view in host slot `dst` of a copy of the `size` title bytes at `va`, which
/// does not start on 256 bytes as a PC view must. The copy is taken now, when the draw is recorded,
/// in upload memory of the list.
fn copied_cbv(this: *mut Proxy, device: Object, dst: usize, va: u64, size: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    let bytes = (size as usize).next_multiple_of(256).max(256);
    let Some(cpu) = super::list_gaps::allocate_upload(this, bytes, MIRROR_ALIGN) else { return };
    let gpu = crate::upload_mem::translate(cpu as u64);
    if gpu == cpu as u64 {
        return;
    }
    // SAFETY: `va` is the title's constant buffer of `size` bytes (readable title memory, the title
    // made a view of it), `cpu` a mapped block of `bytes` bytes this list owns.
    unsafe {
        std::ptr::copy_nonoverlapping(va as *const u8, cpu as *mut u8, size as usize);
        std::ptr::write_bytes((cpu + size as usize) as *mut u8, 0, bytes - size as usize);
    }
    let mut desc = [0u8; 16];
    desc[..8].copy_from_slice(&gpu.to_le_bytes());
    desc[8..12].copy_from_slice(&(bytes as u32).to_le_bytes());
    // SAFETY: `device` is the live PC device, `dst` a host slot of a view heap.
    unsafe { method::<CreateCbv>(device, 17)(device, desc.as_ptr().cast(), dst) };
    crate::view_shadow::view_made(dst, [0; 4]);
    crate::view_diag::view_made(dst, crate::view_diag::View { res: 0, shape: (0, 0, 0, 0), format: 0, dim: 0, va, kind: "cbv copy" });
    // taken again at the list's submission where the title changed the bytes since (`cbv_copies`)
    let reused = crate::cbv_copies::recorded(this as usize, crate::cbv_copies::CbvCopy { dst, va, size, cpu });
    if log_due(&N) {
        xs_log!("console list: constant buffer view of {va:#x} ({size} bytes, not on 256 bytes) made over a copy of {bytes} bytes at {gpu:#x}, host slot {dst:#x}");
    }
    static REUSED: AtomicI32 = AtomicI32::new(0);
    if reused && log_due(&REUSED) {
        xs_log!("console list {:#x}: host slot {dst:#x} already held a constant buffer copy of another range in this list, the GPU reads only the last ({va:#x}, {size} bytes)", this as usize);
    }
}

/// Console slots 33 and 34: `Set{Compute,Graphics}RootSignature(signature)` (PC 29, 30). Root signature
/// objects are PC objects.
macro_rules! root_signature_method {
    ($name:ident, $pc:literal, $bind:expr) => {
        pub(super) extern "system" fn $name(this: *mut Proxy, signature: Object) {
            if user_mirror::active() {
                user_mirror::with_shadow(this as usize, |s| s.set_signature($bind, signature as usize));
            }
            if window_service::active() {
                window_service::note_signature(this as usize, $bind, signature as usize);
            }
            // SAFETY: `this` is a live list proxy; the PC slot takes the root signature.
            unsafe {
                let pc = (*this).pc;
                method::<SetRootSig>(pc, $pc)(pc, signature);
            }
        }
    };
}

root_signature_method!(list_set_compute_root_signature, 29, Bind::Compute);
root_signature_method!(list_set_graphics_root_signature, 30, Bind::Graphics);

/// Console slots 35 and 36: `Set{Compute,Graphics}RootDescriptorTable(index, GPU handle)` (PC 31, 32).
/// A table of a paged heap gets its host handle (`heap_pages`), which the shadow keeps too, so the
/// mirror holds the host index.
macro_rules! root_table_method {
    ($name:ident, $pc:literal, $bind:expr) => {
        pub(super) extern "system" fn $name(this: *mut Proxy, index: u32, handle: u64) {
            let handle = super::heap_pages::table_handle(this as usize, $bind, index, handle);
            if user_mirror::active() {
                user_mirror::with_shadow(this as usize, |s| s.set_table($bind, index, handle));
            }
            // SAFETY: `this` is a live list proxy; the PC slot takes (index, handle).
            unsafe {
                let pc = (*this).pc;
                method::<SetTable>(pc, $pc)(pc, index, handle);
            }
        }
    };
}

root_table_method!(list_set_compute_root_table, 31, Bind::Compute);
root_table_method!(list_set_graphics_root_table, 32, Bind::Graphics);

/// Console slots 37 and 38: `Set{Compute,Graphics}Root32BitConstant(index, value, offset)` (PC 33, 34).
macro_rules! root_constant_method {
    ($name:ident, $pc:literal, $bind:expr) => {
        pub(super) extern "system" fn $name(this: *mut Proxy, index: u32, value: u32, offset: u32) {
            if user_mirror::active() {
                user_mirror::with_shadow(this as usize, |s| s.set_constants($bind, index, offset, &[value]));
            }
            // SAFETY: `this` is a live list proxy; the PC slot takes (index, value, offset).
            unsafe {
                let pc = (*this).pc;
                method::<SetConstant>(pc, $pc)(pc, index, value, offset);
            }
        }
    };
}

root_constant_method!(list_set_compute_root_constant, 33, Bind::Compute);
root_constant_method!(list_set_graphics_root_constant, 34, Bind::Graphics);

/// The largest `count` of `SetRoot32BitConstants` recorded (a root signature has 64 dwords).
const MAX_CONSTANTS: u32 = 64;

/// Console slots 39 and 40: `Set{Compute,Graphics}Root32BitConstants(index, count, data, offset)`
/// (PC 35, 36).
macro_rules! root_constants_method {
    ($name:ident, $pc:literal, $bind:expr) => {
        pub(super) extern "system" fn $name(this: *mut Proxy, index: u32, count: u32, data: *const u32, offset: u32) {
            if user_mirror::active() && !data.is_null() && count <= MAX_CONSTANTS {
                // SAFETY: the caller passes `count` readable dwords.
                let values: Vec<u32> = (0..count as usize).map(|i| unsafe { data.add(i).read_unaligned() }).collect();
                user_mirror::with_shadow(this as usize, |s| s.set_constants($bind, index, offset, &values));
            }
            // SAFETY: `this` is a live list proxy; the PC slot takes (index, count, data, offset).
            unsafe {
                let pc = (*this).pc;
                method::<SetConstants>(pc, $pc)(pc, index, count, data, offset);
            }
        }
    };
}

root_constants_method!(list_set_compute_root_constants, 35, Bind::Compute);
root_constants_method!(list_set_graphics_root_constants, 36, Bind::Graphics);

/// Console slot 12: `DrawInstanced` (PC 12).
pub(super) extern "system" fn list_draw_instanced(this: *mut Proxy, vertices: u32, instances: u32, start_vertex: u32, start_instance: u32) {
    if crate::pso_diag::skip(this as usize) {
        return;
    }
    prepare(this, Bind::Graphics);
    let n = crate::rt_diag::draw(this as usize, "DrawInstanced", vertices);
    describe_draw(this, n);
    crate::draw_diag::describe(this as usize, n, "DrawInstanced", None, start_vertex, instances);
    crate::flicker_probe::draw(this as usize, crate::flicker_probe::Draw::Plain { vertices, first: start_vertex });
    crate::uav_barrier_merge::mark_pending(this as usize);
    if vertices > 0
        && vertices.is_multiple_of(3)
        && let Some(vb) = crate::rect_list::rect_vb0(this as usize)
        && super::list_gaps::draw_rects(this, vb, vertices, instances, start_vertex, start_instance)
    {
        return;
    }
    // SAFETY: `this` is a live list proxy.
    unsafe {
        let pc = (*this).pc;
        method::<DrawInstanced>(pc, 12)(pc, vertices, instances, start_vertex, start_instance);
    }
}

/// Console slot 13: `DrawIndexedInstanced` (PC 13).
pub(super) extern "system" fn list_draw_indexed_instanced(this: *mut Proxy, indices: u32, instances: u32, start_index: u32, base_vertex: i32, start_instance: u32) {
    if crate::pso_diag::skip(this as usize) {
        return;
    }
    prepare(this, Bind::Graphics);
    let n = crate::rt_diag::draw(this as usize, "DrawIndexedInstanced", indices);
    describe_draw(this, n);
    crate::draw_diag::describe(this as usize, n, "DrawIndexedInstanced", Some((indices, start_index, base_vertex)), 0, instances);
    crate::flicker_probe::draw(this as usize, crate::flicker_probe::Draw::Indexed { count: indices, first: start_index, base: base_vertex });
    crate::uav_barrier_merge::mark_pending(this as usize);
    // SAFETY: `this` is a live list proxy.
    unsafe {
        let pc = (*this).pc;
        method::<DrawIndexedInstanced>(pc, 13)(pc, indices, instances, start_index, base_vertex, start_instance);
    }
}

/// Console slot 14: `Dispatch` (PC 14).
pub(super) extern "system" fn list_dispatch(this: *mut Proxy, x: u32, y: u32, z: u32) {
    if crate::pso_diag::skip(this as usize) {
        return;
    }
    prepare(this, Bind::Compute);
    crate::rt_diag::count(&crate::rt_diag::DISPATCHES);
    describe_dispatch(this, crate::draw_diag::dispatch(this as usize, x, y, z));
    crate::uav_barrier_merge::mark_pending(this as usize);
    // SAFETY: `this` is a live list proxy.
    unsafe {
        let pc = (*this).pc;
        method::<DispatchPc>(pc, 14)(pc, x, y, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_real_metadata_buffers_without_querying_fallback() {
        let views = user_mirror::RootViews { heap_map: Some(4), view_shadow: Some(5) };
        let mut fallback_calls = 0usize;
        let mut bound = Vec::new();
        let state = bind_root_views(
            views,
            Some(0x1000_0000),
            Some(0x2000_0000),
            || {
                fallback_calls += 1;
                0xDEAD_0000
            },
            |index, va| bound.push((index, va)),
        );
        assert_eq!(state, (Some(RootViewBind::Buffer(0x1000_0000)), Some(RootViewBind::Buffer(0x2000_0000))));
        assert_eq!(bound, vec![(4, 0x1000_0000), (5, 0x2000_0000)]);
        assert_eq!(fallback_calls, 0);
    }

    #[test]
    fn binds_fallback_when_either_or_both_metadata_buffers_are_missing() {
        let views = user_mirror::RootViews { heap_map: Some(2), view_shadow: Some(3) };

        // Missing heap map only.
        let mut bound = Vec::new();
        let state = bind_root_views(views, None, Some(0x2000_0000), || 0xFB00_0000, |i, va| bound.push((i, va)));
        assert_eq!(state, (Some(RootViewBind::Fallback(0xFB00_0000)), Some(RootViewBind::Buffer(0x2000_0000))));
        assert_eq!(bound, vec![(2, 0xFB00_0000), (3, 0x2000_0000)]);

        // Missing view shadow only.
        bound.clear();
        let state = bind_root_views(views, Some(0x1000_0000), None, || 0xFB00_0000, |i, va| bound.push((i, va)));
        assert_eq!(state, (Some(RootViewBind::Buffer(0x1000_0000)), Some(RootViewBind::Fallback(0xFB00_0000))));
        assert_eq!(bound, vec![(2, 0x1000_0000), (3, 0xFB00_0000)]);

        // Both missing: fallback is queried once and bound to both slots.
        bound.clear();
        let mut fallback_calls = 0usize;
        let state = bind_root_views(
            views,
            None,
            Some(0),
            || {
                fallback_calls += 1;
                0xFB00_0000
            },
            |i, va| bound.push((i, va)),
        );
        assert_eq!(state, (Some(RootViewBind::Fallback(0xFB00_0000)), Some(RootViewBind::Fallback(0xFB00_0000))));
        assert_eq!(bound, vec![(2, 0xFB00_0000), (3, 0xFB00_0000)]);
        assert_eq!(fallback_calls, 1);
    }

    #[test]
    fn reports_missing_without_binding_when_fallback_is_unavailable() {
        let views = user_mirror::RootViews { heap_map: Some(2), view_shadow: Some(3) };
        let mut bound = Vec::new();
        let state = bind_root_views(views, None, None, || 0, |i, va| bound.push((i, va)));
        assert_eq!(state, (Some(RootViewBind::Missing), Some(RootViewBind::Missing)));
        assert!(bound.is_empty());
    }

    #[test]
    fn skips_unconfigured_root_view_slots() {
        let views = user_mirror::RootViews { heap_map: None, view_shadow: Some(7) };
        let mut fallback_calls = 0usize;
        let mut bound = Vec::new();
        let state = bind_root_views(
            views,
            None,
            None,
            || {
                fallback_calls += 1;
                0xFB00_0000
            },
            |i, va| bound.push((i, va)),
        );
        assert_eq!(state, (None, Some(RootViewBind::Fallback(0xFB00_0000))));
        assert_eq!(bound, vec![(7, 0xFB00_0000)]);
        assert_eq!(fallback_calls, 1);
    }
}
