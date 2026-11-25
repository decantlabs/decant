//! A null view for a table slot the title left all zero (`DECANT_D3D12X_ZERO_SLOT_NULL`, on by default,
//! `0` turns it off).
//!
//! On the console a descriptor of zero words reads as nothing: the title leaves a slot of its ring
//! zero when its shader does not use the view (the loop of the shader is guarded by a count from a
//! constant buffer). The layer writes its views into the host heap and leaves a tag in the title's
//! slot (`title_heap`); the table sync copies the host view of each tagged slot into the host slot
//! the shader reads and never looks at a slot without a tag. A slot of zero words therefore keeps
//! whatever view the host slot held before (a texture, a constant buffer view of an older table), and
//! the shader reads it as the kind it declares: on the AMD Linux driver a data read at address 0 that
//! ends in a ring timeout.
//!
//! Before the sync of a draw or dispatch, for each table offset the bound pipeline's shaders declare
//! (`kind_check`, kept per pipeline whether or not the check is on) whose title words are all zero,
//! the host slot gets a null view of the declared kind, once: until the slot is tagged or copied
//! over (`forget`), the slot is in [`Once`] and a draw only reads the title's words. This file holds
//! the pure parts: which offsets qualify, the once only set, the null view descriptions, the limit
//! of the log lines. `proxy/list_mirror.rs` writes the views.

use crate::kind_check::{Class, Dim, Want};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Lines logged before only a total every [`TOTAL_EVERY`] is.
pub const FIRST_LINES: u64 = 8;
/// A line with the total is logged every this many writes.
pub const TOTAL_EVERY: u64 = 1000;

static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether `DECANT_D3D12X_ZERO_SLOT_NULL` is anything but `0`, read once. On when not set.
#[inline]
pub fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_ZERO_SLOT_NULL").is_none_or(|v| v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// The null view a declared view gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Null {
    /// A constant buffer view of the zeroed fallback buffer (`fallback`): a null constant buffer view
    /// must still have a valid location on the PC (vkd3d-proton ignores a call with none, and the
    /// slot would keep its old view).
    Cbv,
    /// A null SRV of a shape (never [`Dim::Unknown`]).
    Srv(Dim),
    /// A null UAV of a shape (never [`Dim::Unknown`], never multisampled or a cube).
    Uav(Dim),
}

impl Null {
    /// For the log line.
    pub fn name(self) -> String {
        match self {
            Null::Cbv => "constant buffer (fallback buffer view)".into(),
            Null::Srv(d) => format!("null SRV {}", d.name()),
            Null::Uav(d) => format!("null UAV {}", d.name()),
        }
    }
}

/// The null view for what a shader declares, `None` when there is none: a typed view of a shape the
/// declaration does not say ([`Dim::Unknown`]), and shapes a PC UAV does not have.
pub fn null_of(want: &Want) -> Option<Null> {
    match want.class {
        Class::Cbv => Some(Null::Cbv),
        Class::Srv => (srv_desc(want.dim).is_some()).then_some(Null::Srv(want.dim)),
        Class::Uav => (uav_desc(want.dim).is_some()).then_some(Null::Uav(want.dim)),
    }
}

/// Whether the words of a title slot are all zero (and there are some).
pub fn zero(words: &[u64]) -> bool {
    !words.is_empty() && words.iter().all(|&w| w == 0)
}

/// What a draw does for one declared offset whose title words are `words`: the null view to write, or
/// `None` for a slot with something in it or a declaration without a null view.
pub fn decide(want: &Want, words: &[u64]) -> Option<Null> {
    if zero(words) { null_of(want) } else { None }
}

fn put(d: &mut [u8; 0x28], at: usize, v: u32) {
    d[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// The format of a null buffer view, as `view_raw` makes one of an empty buffer view (R32_UINT, one element,
/// no flags): the PC runtime takes it.
const BUFFER_FORMAT: u32 = 42;
/// The format of a null texture view (R8G8B8A8_UNORM, as the null views of `CreateSrvAt` have).
const TEXTURE_FORMAT: u32 = 28;
/// `D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING`.
const DEFAULT_MAPPING: u32 = 0x1688;

/// A `D3D12_SHADER_RESOURCE_VIEW_DESC` (0x28 bytes) of a null view of `dim`.
pub fn srv_desc(dim: Dim) -> Option<[u8; 0x28]> {
    let code = match dim {
        Dim::Buffer => 1,
        Dim::Tex1d => 2,
        Dim::Tex1dArray => 3,
        Dim::Tex2d => 4,
        Dim::Tex2dArray => 5,
        Dim::Tex2dMs => 6,
        Dim::Tex2dMsArray => 7,
        Dim::Tex3d => 8,
        Dim::Cube => 9,
        Dim::CubeArray => 10,
        Dim::Unknown => return None,
    };
    let mut d = [0u8; 0x28];
    put(&mut d, 0, if dim == Dim::Buffer { BUFFER_FORMAT } else { TEXTURE_FORMAT });
    put(&mut d, 4, code);
    put(&mut d, 8, DEFAULT_MAPPING);
    match dim {
        // FirstElement 0 at 0x10, NumElements at 0x18, stride 0, flags 0
        Dim::Buffer => put(&mut d, 0x18, 1),
        // MostDetailedMip 0 at 0x0c, MipLevels at 0x10
        Dim::Tex1d | Dim::Tex2d | Dim::Tex3d | Dim::Cube => put(&mut d, 0x10, 1),
        // ... and the array size at 0x18
        Dim::Tex1dArray | Dim::Tex2dArray | Dim::CubeArray => {
            put(&mut d, 0x10, 1);
            put(&mut d, 0x18, 1);
        }
        // no mip levels, the array size at 0x10
        Dim::Tex2dMsArray => put(&mut d, 0x10, 1),
        Dim::Tex2dMs | Dim::Unknown => {}
    }
    Some(d)
}

/// A `D3D12_UNORDERED_ACCESS_VIEW_DESC` (0x28 bytes) of a null view of `dim`; a UAV has no multisampled
/// or cube shape.
pub fn uav_desc(dim: Dim) -> Option<[u8; 0x28]> {
    let code = match dim {
        Dim::Buffer => 1,
        Dim::Tex1d => 2,
        Dim::Tex1dArray => 3,
        Dim::Tex2d => 4,
        Dim::Tex2dArray => 5,
        Dim::Tex3d => 8,
        _ => return None,
    };
    let mut d = [0u8; 0x28];
    put(&mut d, 0, if dim == Dim::Buffer { BUFFER_FORMAT } else { TEXTURE_FORMAT });
    put(&mut d, 4, code);
    match dim {
        // FirstElement at 8, NumElements at 0x10, stride 0, counter 0, flags 0
        Dim::Buffer => put(&mut d, 0x10, 1),
        // the array size (the depth of a 3D texture) at 0x10
        Dim::Tex1dArray | Dim::Tex2dArray | Dim::Tex3d => put(&mut d, 0x10, 1),
        _ => {}
    }
    Some(d)
}

/// The host slots written with a null view, and which: a slot is written once until it is tagged or
/// copied over.
#[derive(Default)]
pub struct Once {
    written: HashMap<usize, Null>,
}

impl Once {
    /// Whether host slot `host` has to get `null` now. Does not remember it: the caller writes the
    /// view first and then calls [`Once::mark`], so a list on another thread that looks in between
    /// still sees the slot as not written and writes the same view too (harmless), instead of
    /// drawing with the stale one.
    pub fn due(&self, host: usize, null: Null) -> bool {
        self.written.get(&host) != Some(&null)
    }

    /// The null view `null` is in host slot `host` now.
    pub fn mark(&mut self, host: usize, null: Null) {
        self.written.insert(host, null);
    }

    /// `n` slots of `increment` bytes from `host` were written by something else.
    pub fn forget(&mut self, host: usize, n: u32, increment: usize) {
        for i in 0..n as usize {
            self.written.remove(&(host + i * increment));
        }
    }

    pub fn len(&self) -> usize {
        self.written.len()
    }

    pub fn is_empty(&self) -> bool {
        self.written.is_empty()
    }
}

static ONCE: Mutex<Option<Once>> = Mutex::new(None);
/// The size of [`ONCE`], so that the hooks of the view calls cost one load while it is empty.
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Whether host slot `host` has to get `null` now. Write the view, then call [`mark`]. The lock is not
/// held across the write: the view calls reach `forget` through `title_heap`.
pub fn due(host: usize, null: Null) -> bool {
    ONCE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_none_or(|once| once.due(host, null))
}

/// The null view `null` was written into host slot `host`: it is not due again until forgotten.
pub fn mark(host: usize, null: Null) {
    let mut g = ONCE.lock().unwrap_or_else(|e| e.into_inner());
    let once = g.get_or_insert_with(Once::default);
    once.mark(host, null);
    HELD.store(once.len(), Ordering::Relaxed);
}

/// `n` host slots from `host` were tagged or copied over: they are written again if they go back to zero.
pub fn forget(host: usize, n: u32, increment: u32) {
    if HELD.load(Ordering::Relaxed) == 0 {
        return;
    }
    if let Some(once) = ONCE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        once.forget(host, n, increment as usize);
        HELD.store(once.len(), Ordering::Relaxed);
    }
}

/// All slots are forgotten (a heap was released).
pub fn clear() {
    if HELD.load(Ordering::Relaxed) == 0 {
        return;
    }
    *ONCE.lock().unwrap_or_else(|e| e.into_inner()) = None;
    HELD.store(0, Ordering::Relaxed);
}

static WRITES: AtomicU64 = AtomicU64::new(0);

/// What to log for the `n`th write: (its own line, the line with the total).
pub fn lines_for(n: u64) -> (bool, bool) {
    (n <= FIRST_LINES, n.is_multiple_of(TOTAL_EVERY))
}

/// Counts a write: (the number of it, its own line due, the total line due).
pub fn count() -> (u64, bool, bool) {
    let n = WRITES.fetch_add(1, Ordering::Relaxed) + 1;
    let (line, total) = lines_for(n);
    (n, line, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn want(class: Class, dim: Dim) -> Want {
        Want { dword: 4, slot: 0, class, dim }
    }

    fn word(d: &[u8; 0x28], at: usize) -> u32 {
        u32::from_le_bytes(d[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn only_an_all_zero_slot_with_a_declared_shape_qualifies() {
        let raw = want(Class::Srv, Dim::Buffer);
        assert_eq!(decide(&raw, &[0, 0, 0, 0]), Some(Null::Srv(Dim::Buffer)));
        assert_eq!(decide(&raw, &[0, 0, 0]), Some(Null::Srv(Dim::Buffer)));
        assert_eq!(decide(&raw, &[0, 0, 1, 0]), None, "a word of the slot is set");
        assert_eq!(decide(&raw, &crate::title_heap::tag_words(0x1000)), None, "a tagged slot");
        assert_eq!(decide(&raw, &[]), None, "no words known");
        assert_eq!(decide(&want(Class::Cbv, Dim::Unknown), &[0; 4]), Some(Null::Cbv));
        assert_eq!(decide(&want(Class::Uav, Dim::Tex1d), &[0; 4]), Some(Null::Uav(Dim::Tex1d)));
        // a typed view the declaration gives no shape for, and shapes a UAV does not have
        assert_eq!(decide(&want(Class::Srv, Dim::Unknown), &[0; 4]), None);
        assert_eq!(decide(&want(Class::Uav, Dim::Unknown), &[0; 4]), None);
        assert_eq!(decide(&want(Class::Uav, Dim::Cube), &[0; 4]), None);
        assert_eq!(decide(&want(Class::Uav, Dim::Tex2dMs), &[0; 4]), None);
    }

    #[test]
    fn a_slot_is_written_once_until_it_is_forgotten() {
        let mut once = Once::default();
        assert!(once.due(0x1000, Null::Cbv));
        once.mark(0x1000, Null::Cbv);
        assert!(!once.due(0x1000, Null::Cbv));
        assert!(!once.due(0x1000, Null::Cbv));
        assert!(once.due(0x1020, Null::Cbv), "another slot");
        // the same slot read as another kind by another pipeline is made that kind
        assert!(once.due(0x1000, Null::Srv(Dim::Buffer)));
        once.mark(0x1000, Null::Srv(Dim::Buffer));
        assert!(once.due(0x1000, Null::Cbv));
        once.mark(0x1000, Null::Cbv);
        once.mark(0x1020, Null::Cbv);
        // tagged or copied over: written again when it goes back to zero
        once.forget(0x1000, 2, 0x20);
        assert!(once.is_empty());
        assert!(once.due(0x1000, Null::Cbv));
        assert!(once.due(0x1020, Null::Cbv));
        once.mark(0x1000, Null::Cbv);
        once.mark(0x1020, Null::Cbv);
        // only the slots named are forgotten
        once.forget(0x1020, 1, 0x20);
        assert!(!once.due(0x1000, Null::Cbv));
        assert!(once.due(0x1020, Null::Cbv));
    }

    #[test]
    fn a_slot_is_due_until_its_view_is_written() {
        // Two lists on two threads: the first has checked the slot and not written its view yet. The
        // second must still find the slot due, or it skips the write and draws with the stale view
        // (the order that marked at the check, before the write, made the second answer `false` here).
        let mut once = Once::default();
        assert!(once.due(0x1000, Null::Cbv), "list one checks");
        assert!(once.due(0x1000, Null::Cbv), "list two checks before list one wrote: it writes the same view");
        once.mark(0x1000, Null::Cbv);
        assert!(!once.due(0x1000, Null::Cbv), "after the write the slot is done");
    }

    #[test]
    fn a_null_buffer_view_is_one_typed_element_of_no_resource() {
        let s = srv_desc(Dim::Buffer).unwrap();
        assert_eq!((word(&s, 0), word(&s, 4), word(&s, 8)), (42, 1, 0x1688));
        assert_eq!((word(&s, 0x10), word(&s, 0x14), word(&s, 0x18), word(&s, 0x1c), word(&s, 0x20)), (0, 0, 1, 0, 0));
        let u = uav_desc(Dim::Buffer).unwrap();
        assert_eq!((word(&u, 0), word(&u, 4)), (42, 1));
        assert_eq!((word(&u, 8), word(&u, 0x10), word(&u, 0x14), word(&u, 0x20)), (0, 1, 0, 0), "first, count, stride, flags");
        assert_eq!(&u[0x18..0x20], &[0; 8], "no counter");
    }

    #[test]
    fn a_null_texture_view_has_the_declared_dimension_and_one_mip_and_slice() {
        let dims = [(Dim::Tex1d, 2), (Dim::Tex1dArray, 3), (Dim::Tex2d, 4), (Dim::Tex2dArray, 5), (Dim::Tex2dMs, 6), (Dim::Tex2dMsArray, 7), (Dim::Tex3d, 8), (Dim::Cube, 9), (Dim::CubeArray, 10)];
        for (dim, code) in dims {
            let s = srv_desc(dim).unwrap();
            assert_eq!((word(&s, 0), word(&s, 4), word(&s, 8)), (28, code, 0x1688), "{dim:?}");
            if dim != Dim::Tex2dMs && dim != Dim::Tex2dMsArray {
                assert_eq!(word(&s, 0x10), 1, "{dim:?} mip levels");
            }
        }
        assert_eq!(word(&srv_desc(Dim::Tex2dArray).unwrap(), 0x18), 1, "array size");
        assert_eq!(word(&srv_desc(Dim::CubeArray).unwrap(), 0x18), 1, "cube count");
        assert_eq!(word(&srv_desc(Dim::Tex2dMsArray).unwrap(), 0x10), 1, "array size of a multisampled one");
        assert!(srv_desc(Dim::Unknown).is_none());
        for (dim, code) in [(Dim::Tex1d, 2), (Dim::Tex1dArray, 3), (Dim::Tex2d, 4), (Dim::Tex2dArray, 5), (Dim::Tex3d, 8)] {
            let u = uav_desc(dim).unwrap();
            assert_eq!((word(&u, 0), word(&u, 4)), (28, code), "{dim:?}");
        }
        assert_eq!(word(&uav_desc(Dim::Tex2dArray).unwrap(), 0x10), 1);
        assert_eq!(word(&uav_desc(Dim::Tex3d).unwrap(), 0x10), 1);
        assert!(uav_desc(Dim::Cube).is_none() && uav_desc(Dim::CubeArray).is_none() && uav_desc(Dim::Tex2dMs).is_none() && uav_desc(Dim::Unknown).is_none());
    }

    #[test]
    fn the_first_eight_writes_are_logged_then_a_total_every_thousand() {
        assert_eq!(lines_for(1), (true, false));
        assert_eq!(lines_for(8), (true, false));
        assert_eq!(lines_for(9), (false, false));
        assert_eq!(lines_for(1000), (false, true));
        assert_eq!(lines_for(2000), (false, true));
        assert_eq!(lines_for(2001), (false, false));
    }
}
