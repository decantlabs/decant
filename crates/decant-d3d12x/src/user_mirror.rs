//! The user data mirror: the 32 dword user SGPR image of a draw or dispatch, which a translated
//! shader (`decant-dxil`, "The user data mirror") reads from the root constant buffer view `b1` of
//! register space 15.
//!
//! The console packs the root arguments of a draw into user SGPRs: root parameters in order, a root
//! constant one SGPR each, a root descriptor two (the low and the high dword of the console address),
//! a table one. Compute starts at `s4`, graphics at `s8`, both end at `s31`. The layer keeps a shadow
//! of the root arguments per command list ([`Shadow`]), and before a draw or dispatch writes the
//! image into upload memory when the shadow changed since the last write. A table is the heap
//! index of its first descriptor: the GPU handle minus the start of the bound heap, divided by the
//! increment, in the resource heap for a resource table and in the sampler heap for a sampler table.
//!
//! Everything here is pure except the global tables, so it is tested on any host.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::rts0::{RegisterClass, RootLayout, RootParamKind};

/// Dwords of the mirror.
pub const DWORDS: usize = 32;
/// Bytes of the mirror.
pub const BYTES: usize = DWORDS * 4;
/// The first user SGPR of a compute shader.
pub const COMPUTE_FIRST: u32 = 4;
/// The first user SGPR of the graphics stages.
pub const GRAPHICS_FIRST: u32 = 8;

/// Which pipeline the root arguments belong to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Compute = 0,
    Graphics = 1,
}

impl Bind {
    /// The first user SGPR of this kind of pipeline.
    pub fn first_sgpr(self) -> u32 {
        match self {
            Bind::Compute => COMPUTE_FIRST,
            Bind::Graphics => GRAPHICS_FIRST,
        }
    }
}

/// What a root parameter is, as far as the packing goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// A descriptor table, one SGPR. `sampler` for a table of samplers.
    Table { sampler: bool },
    /// Root constants, one SGPR each.
    Constants { count: u32 },
    /// A root descriptor, two SGPRs.
    Descriptor,
}

impl Slot {
    /// The SGPRs the parameter takes.
    pub fn dwords(self) -> u32 {
        match self {
            Slot::Table { .. } => 1,
            Slot::Constants { count } => count,
            Slot::Descriptor => 2,
        }
    }
}

/// Where a root parameter lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    /// The SGPR (the mirror dword) of its first dword.
    pub sgpr: u32,
    pub slot: Slot,
}

/// The console packing of one root signature for one kind of pipeline: `placed[i]` for root parameter
/// `i`, `None` for the mirror itself and for a parameter that does not fit in the 32 SGPRs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packing {
    pub placed: Vec<Option<Placed>>,
}

/// The slot a root parameter takes.
fn slot_of(kind: &RootParamKind) -> Slot {
    match kind {
        RootParamKind::Table(ranges) => Slot::Table { sampler: ranges.first().is_some_and(|r| r.class == RegisterClass::Sampler) },
        RootParamKind::Constants { count, .. } => Slot::Constants { count: *count },
        RootParamKind::Descriptor { .. } => Slot::Descriptor,
    }
}

/// Packs the parameters of `layout` in order, starting at the first SGPR of `bind`. The mirror
/// parameter (see [`RootLayout::mirror_index`]) is left out. Once a parameter does not fit before
/// `s32` it and every later one are `None`, as the console runs out of user data there too.
pub fn pack(layout: &RootLayout, bind: Bind) -> Packing {
    let mirror = layout.mirror_index();
    let mut next = bind.first_sgpr();
    let mut full = false;
    let mut placed = Vec::with_capacity(layout.params.len());
    for (i, param) in layout.params.iter().enumerate() {
        if Some(i) == mirror || full {
            placed.push(None);
            continue;
        }
        let slot = slot_of(&param.kind);
        if next + slot.dwords() > DWORDS as u32 {
            full = true;
            placed.push(None);
            continue;
        }
        placed.push(Some(Placed { sgpr: next, slot }));
        next += slot.dwords();
    }
    Packing { placed }
}

/// The heap index of the descriptor at GPU handle `handle` in a heap that starts at `start` with
/// descriptors `increment` bytes apart. `None` below the start or for a zero increment or an index
/// that does not fit.
pub fn heap_index(handle: u64, start: u64, increment: u32) -> Option<u32> {
    if increment == 0 || handle < start {
        return None;
    }
    u32::try_from((handle - start) / u64::from(increment)).ok()
}

/// The root shader resource views the layer appends next to the mirror, by root parameter index:
/// the heap map `t0` and the view shadow `t1` of the mirror's space. Both are buffers of the bound
/// view heap, set again after each root signature or heap change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RootViews {
    pub heap_map: Option<u32>,
    pub view_shadow: Option<u32>,
}

/// A root signature the layer made with the mirror parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MirrorSig {
    /// The root parameter index of the mirror.
    pub index: u32,
    /// The root parameter indices of the heap map and of the view shadow, when the root signature
    /// has them.
    pub root_views: RootViews,
    pub compute: Packing,
    pub graphics: Packing,
}

impl MirrorSig {
    /// `None` when `layout` has no mirror parameter.
    pub fn from_layout(layout: &RootLayout) -> Option<MirrorSig> {
        let index = u32::try_from(layout.mirror_index()?).ok()?;
        let at = |i: Option<usize>| i.and_then(|i| u32::try_from(i).ok());
        let root_views = RootViews { heap_map: at(layout.heap_map_index()), view_shadow: at(layout.view_shadow_index()) };
        Some(MirrorSig { index, root_views, compute: pack(layout, Bind::Compute), graphics: pack(layout, Bind::Graphics) })
    }

    fn packing(&self, bind: Bind) -> &Packing {
        match bind {
            Bind::Compute => &self.compute,
            Bind::Graphics => &self.graphics,
        }
    }
}

/// A descriptor heap the title can bind: what a GPU handle needs to become an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapRef {
    pub sampler: bool,
    pub gpu_start: u64,
    pub increment: u32,
}

/// The value of one root argument the title set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    Unset,
    /// The GPU descriptor handle of a table.
    Table(u64),
    /// Root constants, by offset from the start of the parameter.
    Constants(Vec<u32>),
    /// The console address of a root CBV, SRV or UAV.
    Address(u64),
}

/// What is on the PC list: the last image written for a pipeline kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Written {
    image: Option<[u32; DWORDS]>,
}

/// The root arguments of one command list since its last reset.
#[derive(Clone, Debug, Default)]
pub struct Shadow {
    sig: [usize; 2],
    args: [Vec<Arg>; 2],
    dirty: [bool; 2],
    written: [Written; 2],
    /// The bound resource heap and sampler heap.
    heaps: [Option<HeapRef>; 2],
    /// Table handles set since the last draw or dispatch of each kind, whose slots are not looked at
    /// for tags yet (`title_heap::sync_table`).
    unsynced: [Vec<u64>; 2],
    /// Whether the root views (`RootViews`) of each pipeline kind have to be set (again): after a
    /// root signature was set, which drops the root arguments, and after other heaps were bound.
    views_stale: [bool; 2],
}

impl Shadow {
    /// `object` is the root signature of `bind` from now on. The arguments are gone, and the mirror
    /// must be set again whatever it held.
    pub fn set_signature(&mut self, bind: Bind, object: usize) {
        let b = bind as usize;
        self.sig[b] = object;
        self.args[b].clear();
        self.dirty[b] = true;
        self.written[b] = Written::default();
        self.views_stale[b] = true;
    }

    /// The GPU start of the bound view heap when the root views of `bind` have to be set (again),
    /// and marks them as set. Without a view heap the question stays open for the next draw.
    pub fn take_root_views(&mut self, bind: Bind) -> Option<u64> {
        let b = bind as usize;
        if !self.views_stale[b] {
            return None;
        }
        let heap = self.heaps[0]?;
        self.views_stale[b] = false;
        Some(heap.gpu_start)
    }

    /// The root signature object set for `bind`, 0 for none.
    pub fn signature(&self, bind: Bind) -> usize {
        self.sig[bind as usize]
    }

    fn arg_mut(&mut self, bind: Bind, index: u32) -> Option<&mut Arg> {
        const MAX_PARAMS: usize = 64;
        let b = bind as usize;
        let index = index as usize;
        if index >= MAX_PARAMS {
            return None;
        }
        if self.args[b].len() <= index {
            self.args[b].resize(index + 1, Arg::Unset);
        }
        self.dirty[b] = true;
        self.args[b].get_mut(index)
    }

    pub fn set_table(&mut self, bind: Bind, index: u32, handle: u64) {
        if let Some(a) = self.arg_mut(bind, index) {
            *a = Arg::Table(handle);
        }
        let pending = &mut self.unsynced[bind as usize];
        if !pending.contains(&handle) && pending.len() < 64 {
            pending.push(handle);
        }
    }

    /// Points every table parameter of `bind` that is set to `from` at `to` instead, without making
    /// it a table to sync: the parameters changed, which the PC list must be told too.
    pub fn retarget(&mut self, bind: Bind, from: u64, to: u64) -> Vec<u32> {
        let b = bind as usize;
        let mut changed = Vec::new();
        for (i, a) in self.args[b].iter_mut().enumerate() {
            if *a == Arg::Table(from) {
                *a = Arg::Table(to);
                changed.push(i as u32);
            }
        }
        if !changed.is_empty() {
            self.dirty[b] = true;
        }
        changed
    }

    /// The table handles set since the last call for `bind`.
    pub fn take_unsynced(&mut self, bind: Bind) -> Vec<u64> {
        std::mem::take(&mut self.unsynced[bind as usize])
    }

    pub fn set_address(&mut self, bind: Bind, index: u32, va: u64) {
        if let Some(a) = self.arg_mut(bind, index) {
            *a = Arg::Address(va);
        }
    }

    /// `values` go to the dwords from `offset` of the root constants at `index`.
    pub fn set_constants(&mut self, bind: Bind, index: u32, offset: u32, values: &[u32]) {
        let Some(end) = (offset as usize).checked_add(values.len()).filter(|&e| e <= 64) else { return };
        if let Some(a) = self.arg_mut(bind, index) {
            let mut words = match std::mem::replace(a, Arg::Unset) {
                Arg::Constants(w) => w,
                _ => Vec::new(),
            };
            if words.len() < end {
                words.resize(end, 0);
            }
            words[offset as usize..end].copy_from_slice(values);
            *a = Arg::Constants(words);
        }
    }

    /// The heaps set by `SetDescriptorHeaps`. A descriptor table is read against them, so both
    /// pipeline kinds are dirty.
    pub fn set_heaps(&mut self, heaps: &[HeapRef]) {
        self.heaps = [heaps.iter().find(|h| !h.sampler).copied(), heaps.iter().find(|h| h.sampler).copied()];
        self.dirty = [true, true];
        self.views_stale = [true, true];
    }

    /// The GPU handle of the table the title set at root parameter `param` of `bind`, `None` when it
    /// set none since the last root signature.
    pub fn table_handle(&self, bind: Bind, param: usize) -> Option<u64> {
        match self.args[bind as usize].get(param)? {
            Arg::Table(h) => Some(*h),
            _ => None,
        }
    }

    /// The heap index of the table at `handle`: in the heap of the table's kind, else in the other one.
    fn table_index(&self, handle: u64, sampler: bool) -> u32 {
        let kind = usize::from(sampler);
        for heap in [self.heaps[kind], self.heaps[1 - kind]].into_iter().flatten() {
            if let Some(i) = heap_index(handle, heap.gpu_start, heap.increment) {
                return i;
            }
        }
        0
    }

    /// The image of the SGPRs for `bind` under `sig`.
    pub fn image(&self, bind: Bind, sig: &MirrorSig) -> [u32; DWORDS] {
        let mut image = [0u32; DWORDS];
        for (i, placed) in sig.packing(bind).placed.iter().enumerate() {
            let Some(p) = placed else { continue };
            let arg = self.args[bind as usize].get(i).unwrap_or(&Arg::Unset);
            let at = p.sgpr as usize;
            match (p.slot, arg) {
                (Slot::Table { sampler }, Arg::Table(h)) => image[at] = self.table_index(*h, sampler),
                (Slot::Constants { count }, Arg::Constants(words)) => {
                    for (k, w) in words.iter().take(count as usize).enumerate() {
                        image[at + k] = *w;
                    }
                }
                (Slot::Descriptor, Arg::Address(va)) => {
                    image[at] = *va as u32;
                    image[at + 1] = (*va >> 32) as u32;
                }
                _ => {}
            }
        }
        image
    }

    /// The image to write before a draw or dispatch of `bind`, when the arguments changed since the
    /// last write and the image differs from the one the PC list holds. Marks it as written.
    pub fn take_image(&mut self, bind: Bind, sig: &MirrorSig) -> Option<[u32; DWORDS]> {
        let b = bind as usize;
        if !self.dirty[b] {
            return None;
        }
        self.dirty[b] = false;
        let image = self.image(bind, sig);
        if self.written[b].image == Some(image) {
            return None;
        }
        self.written[b].image = Some(image);
        Some(image)
    }

    /// The image was not written after all: the next draw tries again.
    /// The bound view heap and sampler heap.
    pub fn heaps(&self) -> [Option<HeapRef>; 2] {
        self.heaps
    }

    pub fn untake(&mut self, bind: Bind) {
        let b = bind as usize;
        self.written[b] = Written::default();
        self.dirty[b] = true;
    }
}

// ---- global tables ----

/// Whether any root signature with the mirror exists, so that the common case takes no lock.
static ANY: AtomicBool = AtomicBool::new(false);
static SIGS: Mutex<Option<HashMap<usize, Arc<MirrorSig>>>> = Mutex::new(None);
static HEAPS: Mutex<Option<HashMap<usize, HeapRef>>> = Mutex::new(None);
static SHADOWS: Mutex<Option<HashMap<usize, Shadow>>> = Mutex::new(None);

/// Whether the layer makes the mirror: `DECANT_D3D12X_DXIL_OVERRIDE` or `DECANT_D3D12X_DIRECT` is set and
/// `DECANT_D3D12X_USER_MIRROR` is not `0`.
pub fn wanted() -> bool {
    (crate::dxil_override::configured_dir().is_some() || crate::compile_service::direct_enabled()) && std::env::var_os("DECANT_D3D12X_USER_MIRROR").is_none_or(|v| v != "0")
}

/// Whether the shadow has to be kept (a root signature with the mirror was made).
pub fn active() -> bool {
    ANY.load(Ordering::Acquire)
}

/// Remembers the root signature object `object` with the mirror, from the layout of the blob the
/// host got. Returns whether the layout has the mirror.
pub fn remember_signature(object: usize, layout: &RootLayout) -> bool {
    let Some(sig) = MirrorSig::from_layout(layout) else { return false };
    SIGS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(object, Arc::new(sig));
    ANY.store(true, Ordering::Release);
    true
}

/// The mirror of root signature object `object`, when it has one.
pub fn signature(object: usize) -> Option<Arc<MirrorSig>> {
    SIGS.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&object).cloned()
}

/// A shader visible heap at PC heap object `heap` was created.
pub fn heap_created(heap: usize, sampler: bool, gpu_start: u64, increment: u32) {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(heap, HeapRef { sampler, gpu_start, increment });
}

/// What is known of the PC heap object `heap`.
pub fn heap_ref(heap: usize) -> Option<HeapRef> {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&heap).copied()
}

/// Runs `f` on the shadow of `list` (made empty at first use).
pub fn with_shadow<R>(list: usize, f: impl FnOnce(&mut Shadow) -> R) -> R {
    let mut g = SHADOWS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new).entry(list).or_default())
}

/// `list` was reset: its root arguments are gone.
pub fn forget(list: usize) {
    if let Some(m) = SHADOWS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&list);
    }
}

/// The root parameter index of the mirror and its image to write before a draw or dispatch of `bind`
/// on `list`, when its root signature has the mirror and the arguments changed.
pub fn prepare(list: usize, bind: Bind) -> Option<(u32, [u32; DWORDS])> {
    if !active() {
        return None;
    }
    let object = with_shadow(list, |s| s.signature(bind));
    let sig = signature(object)?;
    let image = with_shadow(list, |s| s.take_image(bind, &sig))?;
    Some((sig.index, image))
}

/// The root views of the root signature of `bind` and the GPU start of the bound view heap, when
/// they have to be set on `list` before a draw or dispatch: its root signature has one of them, and
/// the root signature or the heaps changed since they were set.
pub fn prepare_root_views(list: usize, bind: Bind) -> Option<(RootViews, u64)> {
    if !active() {
        return None;
    }
    let (object, stale) = with_shadow(list, |s| (s.signature(bind), s.views_stale[bind as usize]));
    if !stale {
        return None;
    }
    let views = signature(object)?.root_views;
    if views == RootViews::default() {
        return None;
    }
    let gpu_start = with_shadow(list, |s| s.take_root_views(bind))?;
    Some((views, gpu_start))
}

/// The image of [`prepare`] could not be written: try again at the next draw.
pub fn prepare_failed(list: usize, bind: Bind) {
    with_shadow(list, |s| s.untake(bind));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rts0::{RootParam, RootRange};

    fn param(kind: RootParamKind, visibility: u32) -> RootParam {
        RootParam { kind, visibility }
    }

    fn table(class: RegisterClass) -> RootParam {
        param(RootParamKind::Table(vec![RootRange { class, count: 4, base: 0, space: 0, flags: 0, offset: 0 }]), 0)
    }

    fn consts(count: u32) -> RootParam {
        param(RootParamKind::Constants { register: 0, space: 0, count }, 0)
    }

    fn view() -> RootParam {
        param(RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 0 }, 0)
    }

    fn mirror() -> RootParam {
        param(RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, 0)
    }

    fn layout(params: Vec<RootParam>) -> RootLayout {
        RootLayout { version: 2, flags: 0, params, static_samplers: Vec::new() }
    }

    fn sgprs(p: &Packing) -> Vec<Option<u32>> {
        p.placed.iter().map(|x| x.map(|p| p.sgpr)).collect()
    }

    #[test]
    fn the_root_views_are_due_after_a_root_signature_or_other_heaps() {
        let heap_map = param(RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 15 }, 0);
        let shadow = param(RootParamKind::Descriptor { class: RegisterClass::Srv, register: 1, space: 15 }, 0);
        let sig = MirrorSig::from_layout(&layout(vec![table(RegisterClass::Srv), mirror(), heap_map, shadow])).expect("a mirror");
        assert_eq!((sig.index, sig.root_views), (1, RootViews { heap_map: Some(2), view_shadow: Some(3) }));
        assert_eq!(MirrorSig::from_layout(&layout(vec![table(RegisterClass::Srv), mirror()])).expect("a mirror").root_views, RootViews::default());
        let views = HeapRef { sampler: false, gpu_start: 0x4000, increment: 32 };
        let samplers = HeapRef { sampler: true, gpu_start: 0x9000, increment: 32 };
        let mut s = Shadow::default();
        assert_eq!(s.take_root_views(Bind::Graphics), None, "nothing is set yet");
        s.set_signature(Bind::Graphics, 7);
        assert_eq!(s.take_root_views(Bind::Graphics), None, "no view heap yet: asked again later");
        s.set_heaps(&[samplers, views]);
        assert_eq!(s.take_root_views(Bind::Graphics), Some(0x4000));
        assert_eq!(s.take_root_views(Bind::Graphics), None, "set once");
        assert_eq!(s.take_root_views(Bind::Compute), Some(0x4000), "each pipeline kind has its own root arguments");
        // Another heap, then another root signature: due again each time.
        s.set_heaps(&[HeapRef { gpu_start: 0x8000, ..views }]);
        assert_eq!(s.take_root_views(Bind::Graphics), Some(0x8000));
        s.set_signature(Bind::Graphics, 8);
        assert_eq!(s.take_root_views(Bind::Graphics), Some(0x8000));
        assert_eq!(s.take_root_views(Bind::Compute), Some(0x8000), "the heap change reached compute too");
        assert_eq!(s.take_root_views(Bind::Compute), None);
        // The global form asks the root signature of the list.
        let list = 0x51D0_0001;
        assert!(remember_signature(0x51D0_0100, &layout(vec![table(RegisterClass::Srv), mirror(), heap_map_param()])));
        assert!(remember_signature(0x51D0_0200, &layout(vec![table(RegisterClass::Srv), mirror()])));
        with_shadow(list, |s| {
            s.set_heaps(&[views]);
            s.set_signature(Bind::Graphics, 0x51D0_0200);
        });
        assert_eq!(prepare_root_views(list, Bind::Graphics), None, "a root signature without the parameters");
        with_shadow(list, |s| s.set_signature(Bind::Graphics, 0x51D0_0100));
        let views = RootViews { heap_map: Some(2), view_shadow: None };
        assert_eq!(prepare_root_views(list, Bind::Graphics), Some((views, 0x4000)));
        assert_eq!(prepare_root_views(list, Bind::Graphics), None);
        forget(list);
    }

    fn heap_map_param() -> RootParam {
        param(RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 15 }, 0)
    }

    #[test]
    fn parameters_are_packed_in_order_from_s4_for_compute_and_s8_for_graphics() {
        // table, constants x3, root descriptor, sampler table, mirror
        let l = layout(vec![table(RegisterClass::Srv), consts(3), view(), table(RegisterClass::Sampler), mirror()]);
        let c = pack(&l, Bind::Compute);
        assert_eq!(sgprs(&c), [Some(4), Some(5), Some(8), Some(10), None]);
        let g = pack(&l, Bind::Graphics);
        assert_eq!(sgprs(&g), [Some(8), Some(9), Some(12), Some(14), None]);
        assert_eq!(g.placed[3].unwrap().slot, Slot::Table { sampler: true });
        assert_eq!(g.placed[1].unwrap().slot, Slot::Constants { count: 3 });
        assert_eq!(g.placed[2].unwrap().slot.dwords(), 2);
    }

    #[test]
    fn parameters_past_s31_are_left_out_with_everything_after_them() {
        let l = layout(vec![consts(22), view(), table(RegisterClass::Srv), mirror()]);
        // graphics: s8..s29 constants, descriptor s30 s31, the table has no room
        let g = pack(&l, Bind::Graphics);
        assert_eq!(sgprs(&g), [Some(8), Some(30), None, None]);
        let c = pack(&l, Bind::Compute);
        assert_eq!(sgprs(&c), [Some(4), Some(26), Some(28), None]);
        let tight = layout(vec![consts(24), table(RegisterClass::Srv)]);
        assert_eq!(sgprs(&pack(&tight, Bind::Graphics)), [Some(8), None]);
    }

    #[test]
    fn a_moved_table_is_not_synced_and_points_back_only_where_the_title_did_not_rebind() {
        let mut s = Shadow::default();
        s.set_table(Bind::Graphics, 0, 0x40);
        s.set_table(Bind::Graphics, 2, 0x40);
        s.set_table(Bind::Graphics, 1, 0x80);
        let _ = s.take_unsynced(Bind::Graphics);
        assert_eq!(s.retarget(Bind::Graphics, 0x40, 0x900), [0, 2]);
        assert!(s.take_unsynced(Bind::Graphics).is_empty(), "a moved table is not the title's to sync");
        s.set_table(Bind::Graphics, 2, 0x50);
        assert_eq!(s.retarget(Bind::Graphics, 0x900, 0x40), [0], "parameter 2 was bound again by the title");
        assert!(s.retarget(Bind::Compute, 0x40, 0x900).is_empty());
    }

    #[test]
    fn tables_set_since_the_last_draw_are_taken_once() {
        let mut s = Shadow::default();
        s.set_table(Bind::Graphics, 0, 0x40);
        s.set_table(Bind::Graphics, 1, 0x40);
        s.set_table(Bind::Graphics, 2, 0x80);
        s.set_table(Bind::Compute, 0, 0xC0);
        assert_eq!(s.take_unsynced(Bind::Graphics), [0x40, 0x80]);
        assert!(s.take_unsynced(Bind::Graphics).is_empty());
        assert_eq!(s.take_unsynced(Bind::Compute), [0xC0]);
    }

    #[test]
    fn a_handle_becomes_the_heap_index() {
        assert_eq!(heap_index(0x1000 + 5 * 32, 0x1000, 32), Some(5));
        assert_eq!(heap_index(0x1000, 0x1000, 32), Some(0));
        assert_eq!(heap_index(0x0FFF, 0x1000, 32), None);
        assert_eq!(heap_index(0x2000, 0x1000, 0), None);
        assert_eq!(heap_index(0x1_0000_1000, 0, 1), None, "an index past 32 bits");
    }

    fn heaps() -> [HeapRef; 2] {
        [HeapRef { sampler: false, gpu_start: 0x10_0000, increment: 32 }, HeapRef { sampler: true, gpu_start: 0x90_0000, increment: 16 }]
    }

    #[test]
    fn the_image_holds_constants_addresses_and_heap_indices_at_their_sgprs() {
        let l = layout(vec![table(RegisterClass::Srv), table(RegisterClass::Sampler), consts(2), view(), mirror()]);
        let sig = MirrorSig::from_layout(&l).unwrap();
        assert_eq!(sig.index, 4);
        let mut s = Shadow::default();
        s.set_heaps(&heaps());
        s.set_signature(Bind::Graphics, 0xA);
        s.set_table(Bind::Graphics, 0, 0x10_0000 + 7 * 32);
        s.set_table(Bind::Graphics, 1, 0x90_0000 + 3 * 16);
        s.set_constants(Bind::Graphics, 2, 1, &[0xBEEF]);
        s.set_constants(Bind::Graphics, 2, 0, &[0xAA]);
        s.set_address(Bind::Graphics, 3, 0x1234_5678_9ABC_DEF0);
        let image = s.image(Bind::Graphics, &sig);
        assert_eq!(&image[8..14], &[7, 3, 0xAA, 0xBEEF, 0x9ABC_DEF0, 0x1234_5678]);
        assert!(image[..8].iter().all(|&w| w == 0) && image[14..].iter().all(|&w| w == 0), "unused dwords are 0");
        // the same arguments for compute start at s4
        let mut s = Shadow::default();
        s.set_heaps(&heaps());
        s.set_table(Bind::Compute, 0, 0x10_0000 + 2 * 32);
        assert_eq!(s.image(Bind::Compute, &sig)[4], 2);
    }

    #[test]
    fn a_table_in_the_other_heap_is_found_and_an_unknown_handle_is_zero() {
        let l = layout(vec![table(RegisterClass::Srv), mirror()]);
        let sig = MirrorSig::from_layout(&l).unwrap();
        let mut s = Shadow::default();
        s.set_heaps(&heaps()[..1]);
        s.set_table(Bind::Compute, 0, 0x10_0000 + 9 * 32);
        assert_eq!(s.image(Bind::Compute, &sig)[4], 9);
        s.set_table(Bind::Compute, 0, 0x5);
        assert_eq!(s.image(Bind::Compute, &sig)[4], 0);
    }

    #[test]
    fn the_shadow_is_dirty_after_a_change_and_clean_after_the_image_was_taken() {
        let l = layout(vec![consts(1), mirror()]);
        let sig = MirrorSig::from_layout(&l).unwrap();
        let mut s = Shadow::default();
        assert!(s.take_image(Bind::Compute, &sig).is_none(), "nothing set yet");
        s.set_signature(Bind::Compute, 7);
        assert!(s.take_image(Bind::Compute, &sig).is_some(), "a new root signature needs its mirror");
        assert!(s.take_image(Bind::Compute, &sig).is_none(), "nothing changed");
        s.set_constants(Bind::Compute, 0, 0, &[5]);
        assert_eq!(s.take_image(Bind::Compute, &sig).map(|i| i[4]), Some(5));
        s.set_constants(Bind::Compute, 0, 0, &[5]);
        assert!(s.take_image(Bind::Compute, &sig).is_none(), "the same image is not written again");
        assert!(s.take_image(Bind::Graphics, &sig).is_none(), "the other pipeline kind is separate");
        s.set_heaps(&heaps());
        assert!(s.take_image(Bind::Compute, &sig).is_none(), "new heaps, same image");
        s.set_signature(Bind::Compute, 7);
        assert!(s.take_image(Bind::Compute, &sig).is_some(), "the same root signature set again clears the arguments");
        s.untake(Bind::Compute);
        assert!(s.take_image(Bind::Compute, &sig).is_some(), "a write that failed is tried again");
    }

    #[test]
    fn out_of_range_arguments_are_ignored() {
        let mut s = Shadow::default();
        s.set_constants(Bind::Compute, 0, 60, &[1; 8]);
        s.set_table(Bind::Compute, 200, 1);
        assert!(s.args[0].is_empty());
    }

    #[test]
    fn only_a_layout_with_the_mirror_parameter_makes_a_signature() {
        assert!(MirrorSig::from_layout(&layout(vec![view()])).is_none());
        assert!(!remember_signature(0x11, &layout(vec![view()])));
        assert!(remember_signature(0x22, &layout(vec![view(), mirror()])));
        assert_eq!(signature(0x22).map(|s| s.index), Some(1));
        assert!(signature(0x11).is_none());
    }

    #[test]
    fn prepare_follows_the_root_signature_of_the_list() {
        assert!(remember_signature(0x33, &layout(vec![consts(1), mirror()])));
        with_shadow(0x900, |s| s.set_signature(Bind::Graphics, 0x33));
        assert_eq!(prepare(0x900, Bind::Graphics).map(|(i, _)| i), Some(1));
        assert!(prepare(0x900, Bind::Graphics).is_none());
        assert!(prepare(0x900, Bind::Compute).is_none(), "no compute root signature");
        with_shadow(0x900, |s| s.set_constants(Bind::Graphics, 0, 0, &[3]));
        assert_eq!(prepare(0x900, Bind::Graphics).map(|(_, i)| i[8]), Some(3));
        prepare_failed(0x900, Bind::Graphics);
        assert!(prepare(0x900, Bind::Graphics).is_some());
        forget(0x900);
        assert!(prepare(0x900, Bind::Graphics).is_none());
    }
}
