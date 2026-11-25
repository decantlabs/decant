//! Render target diagnostics behind `DECANT_D3D12X_PRESENT_PROBE=1`: which PC resource each render
//! target view was made for, how often each resource is bound and cleared, which render target
//! and which pipeline each draw ran with, how many pipelines the PC runtime refused, and how many
//! render target handles did not translate. The built in present prints the numbers next to a
//! readback of the presented resource, so a black picture can be told apart: nothing drawn,
//! drawn elsewhere, or drawn but not shown.
//!
//! `DECANT_D3D12X_PROBE_DUMP=<dir>` (see `present.rs`) additionally writes each presented and
//! target readback this probe already takes as an uncompressed picture file in `<dir>`.
//!
//! Off by default. Off, every hook here is one relaxed atomic load.
//!
//! Generic `AddRef`/`Release` by vtable slot (no typed COM binding, see `com_add_ref`/
//! `com_release`) keeps targets drawn this frame alive without needing `windows::*` types, so
//! this module has no OS-specific dependency and builds and tests on every host.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::Mutex;

static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether `DECANT_D3D12X_PRESENT_PROBE` is set to something other than 0, read once.
pub fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_PRESENT_PROBE").is_some_and(|v| v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

pub static DISPATCHES: AtomicU64 = AtomicU64::new(0);
pub static INDIRECT: AtomicU64 = AtomicU64::new(0);
pub static RT_SET_UNTRANSLATED: AtomicU64 = AtomicU64::new(0);
pub static RT_CLEAR_UNTRANSLATED: AtomicU64 = AtomicU64::new(0);
pub static RTV_UNTRANSLATED: AtomicU64 = AtomicU64::new(0);
/// Pipelines made and refused: other (compute), graphics with every stage replaced by a DXIL
/// override, graphics with a console shader kept.
static PSO: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];

/// The number of draws, and the kind and element count of the last one.
type DrawStats = (u64, &'static str, u32);

#[derive(Default)]
struct Maps {
    /// Host RTV handle to the PC resource it was made for.
    rtv: HashMap<usize, usize>,
    /// PC resource to the number of times it was bound as a render target.
    binds: HashMap<usize, u64>,
    /// PC resource to the number of clears of one of its views, and the last clear colour.
    clears: HashMap<usize, (u64, [f32; 4])>,
    /// Command list to its render target 0 (a PC resource, 0 for none) and its pipeline.
    lists: HashMap<usize, (usize, usize)>,
    /// (render target 0, pipeline or 0) to the number of draws and the kind and element count of
    /// the last one.
    draws: HashMap<(usize, usize), DrawStats>,
    /// PC resource to the number of copies, resolves and other writes outside draws into it.
    copies: HashMap<usize, u64>,
    /// PC pipeline to the content hashes of its shader stages, for the draws of the summary.
    shaders: HashMap<usize, String>,
    /// Command list to the draws recorded since its reset.
    recorded: HashMap<usize, u64>,
    /// Draws of lists handed to the PC queue, and lists handed.
    executed: (u64, u64),
    /// PC resource to its size/format/mips/flags, captured once at `rtv_made` (the resource is
    /// certainly live then). Nothing in this module calls `GetDesc()` on a resource later, when
    /// the title may already have released it.
    descs: HashMap<usize, String>,
    /// PC resource to the number of draws that bound it as render target 0 THIS frame
    /// (cleared at every present by `end_frame`). `most_drawn_other` picks from here, never
    /// from `binds`/`draws`, which are lifetime counts and may name a resource long released.
    frame_drawn: HashMap<usize, u64>,
    /// PC resource this module holds its own `AddRef` on (one per resource), taken the first
    /// time it is seen drawn in the current frame and released by `end_frame` or `forget`. Keeps
    /// the resource alive for the lifetime of `frame_drawn`'s bookkeeping, nothing more.
    frame_owned: Vec<usize>,
}

static MAPS: Mutex<Option<Maps>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Maps) -> R) -> R {
    let mut g = MAPS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Maps::default))
}

/// Generic `IUnknown::AddRef`/`Release` by vtable slot (1, 2): no typed COM binding needed, so
/// this module stays free of `windows::*` types and builds on every host. Same pattern as
/// `proxy::method` (slot calls by index).
///
/// # Safety
/// `p` is a non-zero pointer to a live COM object (vtable pointer, then data).
unsafe fn com_add_ref(p: usize) {
    // SAFETY: by the contract above.
    unsafe {
        let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(1));
        f(p);
    }
}

/// # Safety
/// As [`com_add_ref`].
unsafe fn com_release(p: usize) {
    // SAFETY: by the contract above.
    unsafe {
        let f: unsafe extern "system" fn(usize) -> u32 = std::mem::transmute(*(*(p as *const *const usize)).add(2));
        f(p);
    }
}

/// An owned reference to the resource [`most_drawn_other`] picked: its own `AddRef`, released on
/// drop. The caller may read it (e.g. for a readback) for as long as it is kept alive, even past
/// the next `end_frame` or a release of the title's own reference.
pub struct OwnedTarget(usize);

impl OwnedTarget {
    /// The raw PC resource pointer. Valid for the lifetime of this guard.
    pub fn raw(&self) -> usize {
        self.0
    }
}

impl Drop for OwnedTarget {
    fn drop(&mut self) {
        // SAFETY: `new` only ever wraps a pointer this module has just `AddRef`'d.
        unsafe { com_release(self.0) };
    }
}

pub fn count(counter: &AtomicU64) {
    if enabled() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// A pipeline was made (`ok`) or refused by the PC runtime. `full`: for a graphics pipeline,
/// whether every stage got a DXIL override.
pub fn pipeline(ok: bool, full: Option<bool>) {
    let kind = match full {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    count(&PSO[kind][usize::from(!ok)]);
}

/// Pipeline `pso` was made with the shader stages `stages` (names and content hashes).
pub fn pipeline_made(pso: usize, stages: impl FnOnce() -> String) {
    if enabled() && pso != 0 {
        let s = stages();
        with(|m| m.shaders.insert(pso, s));
    }
}

/// The resource drawn most THIS FRAME that is not `presented`, for the readback of the probe.
/// Picks only from `frame_drawn` (cleared every present by `end_frame`), never from the lifetime
/// `draws`/`binds` tables, so it never names a resource the title released frames ago. Returns an
/// owned reference (its own `AddRef`): the caller may use it for as long as it keeps the guard,
/// independent of this module's own bookkeeping reference.
pub fn most_drawn_other(presented: usize) -> Option<OwnedTarget> {
    with(|m| {
        let r = *m.frame_drawn.iter().filter(|&(&rt, _)| rt != 0 && rt != presented).max_by_key(|&(_, &n)| n)?.0;
        // SAFETY: every key of `frame_drawn` is a resource this module holds its own `AddRef`
        // on (`frame_owned`) for exactly this reason: it is still alive to `AddRef` again here.
        unsafe { com_add_ref(r) };
        Some(OwnedTarget(r))
    })
}

/// A render target view at host handle `host` was made for PC resource `res`. `desc` describes it
/// (size, format, mips, flags) read while it is certainly live (view creation needs a live
/// resource); computed lazily, only when the probe is on.
pub fn rtv_made(host: usize, res: usize, desc: impl FnOnce() -> String) {
    #[cfg(windows)]
    crate::capture::rtv_made(host, res);
    if enabled() {
        with(|m| {
            m.rtv.insert(host, res);
            if res != 0 {
                m.descs.insert(res, desc());
            }
        });
    }
}

/// Command list `list`'s render target 0 was drawn. Tracks it as drawn this frame (owned, see
/// `frame_owned`) so [`most_drawn_other`] can name it without reviving a lifetime-count pointer.
/// Does nothing for render target 0 (no target bound).
fn note_drawn_this_frame(m: &mut Maps, rt: usize) {
    if rt == 0 {
        return;
    }
    if !m.frame_drawn.contains_key(&rt) {
        // SAFETY: `rt` is the PC resource an RTV was made for (`rtv_made`), which the title's
        // proxy still holds a reference to while this draw runs.
        unsafe { com_add_ref(rt) };
        m.frame_owned.push(rt);
    }
    *m.frame_drawn.entry(rt).or_default() += 1;
}

/// Releases every `AddRef` this frame's bookkeeping took and clears it, ready for the next frame.
/// Call once per present, after any `most_drawn_other` of the frame being presented (that call
/// takes its own independent reference, so calling this first would not shorten its lifetime, but
/// calling it after keeps the ordering obvious: read the frame, then close it).
pub fn end_frame() {
    with(|m| {
        for r in m.frame_owned.drain(..) {
            // SAFETY: each entry of `frame_owned` is a reference `note_drawn_this_frame` took.
            unsafe { com_release(r) };
        }
        m.frame_drawn.clear();
    });
}

/// The title released PC resource `res`: forget it everywhere so no later diagnostic (summary,
/// `most_drawn_other`) can name it again. If this module held its own frame bookkeeping
/// `AddRef` on it, that reference is released now rather than at the next `end_frame`.
pub fn forget(res: usize) {
    if res == 0 || !enabled() {
        return;
    }
    with(|m| {
        m.rtv.retain(|_, v| *v != res);
        m.binds.remove(&res);
        m.clears.remove(&res);
        m.copies.remove(&res);
        m.descs.remove(&res);
        m.draws.retain(|(rt, _), _| *rt != res);
        for (_, rt) in m.lists.values_mut() {
            if *rt == res {
                *rt = 0;
            }
        }
        if m.frame_drawn.remove(&res).is_some()
            && let Some(i) = m.frame_owned.iter().position(|&r| r == res)
        {
            m.frame_owned.swap_remove(i);
            // SAFETY: that entry was a reference `note_drawn_this_frame` took.
            unsafe { com_release(res) };
        }
    });
}

/// Command list `list` bound the render target views at the host handles `hosts`.
pub fn bound(list: usize, hosts: &[usize]) {
    if !enabled() {
        return;
    }
    with(|m| {
        let mut first = 0;
        for (i, h) in hosts.iter().enumerate() {
            let res = m.rtv.get(h).copied().unwrap_or(0);
            if i == 0 {
                first = res;
            }
            *m.binds.entry(res).or_default() += 1;
        }
        m.lists.entry(list).or_default().0 = first;
    });
}

/// Command list `list` set pipeline `pso` (a PC object or null).
pub fn pipeline_set(list: usize, pso: usize) {
    if enabled() {
        with(|m| m.lists.entry(list).or_default().1 = pso);
    }
}

/// Command list `list` was reset with pipeline `pso` and has no render targets bound.
pub fn list_reset(list: usize, pso: usize) {
    if enabled() {
        with(|m| {
            m.lists.insert(list, (0, pso));
            m.recorded.insert(list, 0);
        });
    }
}

/// The PC command lists `lists` were handed to a PC queue.
pub fn executed(lists: &[usize]) {
    if enabled() {
        with(|m| {
            for l in lists {
                m.executed.0 += m.recorded.get(l).copied().unwrap_or(0);
                m.executed.1 += 1;
            }
        });
    }
}

/// Command list `list` recorded a draw of `kind` with `n` vertices or indices. Returns how many
/// draws its render target and pipeline pair has now, 0 when the probe is off.
pub fn draw(list: usize, kind: &'static str, n: u32) -> u64 {
    #[cfg(windows)]
    crate::capture::drew(list);
    if !enabled() {
        return 0;
    }
    with(|m| {
        let (rt, pso) = m.lists.get(&list).copied().unwrap_or_default();
        *m.recorded.entry(list).or_default() += 1;
        let n_now = {
            let e = m.draws.entry((rt, pso)).or_insert((0, kind, n));
            *e = (e.0 + 1, kind, n);
            e.0
        };
        note_drawn_this_frame(m, rt);
        n_now
    })
}

/// The render target 0 and the pipeline of command list `list`.
pub fn list_state(list: usize) -> (usize, usize) {
    with(|m| m.lists.get(&list).copied().unwrap_or_default())
}

/// Turns the probe on for the tests of other modules.
#[cfg(test)]
pub fn force_on() {
    STATE.store(2, Ordering::Relaxed);
}

/// A copy, resolve or other write outside a draw went into PC resource `dst`.
pub fn copied_into(dst: usize) {
    if enabled() {
        with(|m| *m.copies.entry(dst).or_default() += 1);
    }
}

/// The render target view at host handle `host` was cleared to `colour`.
pub fn cleared(host: usize, colour: [f32; 4]) {
    if !enabled() {
        return;
    }
    with(|m| {
        let res = m.rtv.get(&host).copied().unwrap_or(0);
        let e = m.clears.entry(res).or_insert((0, colour));
        e.0 += 1;
        e.1 = colour;
    });
}

/// One line of numbers about `presented` and the most bound resources. Resource descriptions come
/// only from `descs` (captured live at `rtv_made`), never from a fresh `GetDesc()` call here: a
/// resource named by the lifetime `binds`/`draws` tables may already be released by the time this
/// runs.
pub fn summary(presented: usize) -> String {
    with(|m| {
        let mut top: Vec<(usize, u64)> = m.binds.iter().map(|(r, n)| (*r, *n)).collect();
        top.sort_by_key(|a| std::cmp::Reverse(a.1));
        top.truncate(6);
        let top: Vec<String> = top.iter().map(|(r, n)| format!("{r:#x}:{n}")).collect();
        let mut draws: Vec<((usize, usize), DrawStats)> = m.draws.iter().map(|(k, n)| (*k, *n)).collect();
        draws.sort_by_key(|a| std::cmp::Reverse(a.1.0));
        draws.truncate(8);
        let draws: Vec<String> = draws
            .iter()
            .map(|((r, p), (n, kind, count))| {
                let pso = match (*p, m.shaders.get(p)) {
                    (0, _) => "no pipeline".to_owned(),
                    (p, Some(s)) => format!("pipeline {p:#x} ({s})"),
                    (p, None) => format!("pipeline {p:#x}"),
                };
                let desc = m.descs.get(r).map_or_else(|| "desc unknown, resource may be released".to_owned(), String::clone);
                format!("{r:#x} [{desc}] {pso} last {kind}({count}):{n}")
            })
            .collect();
        let mut copies: Vec<(usize, u64)> = m.copies.iter().map(|(r, n)| (*r, *n)).collect();
        copies.sort_by_key(|a| std::cmp::Reverse(a.1));
        copies.truncate(4);
        let copies: Vec<String> = copies.iter().map(|(r, n)| format!("{r:#x}:{n}")).collect();
        let (clears, colour) = m.clears.get(&presented).copied().unwrap_or_default();
        format!(
            "lists executed {}, their draws {}; draws by target {}, copies by target {} (presented {}), dispatches {}, indirect {}, pipelines made/refused: other {}/{}, graphics all replaced {}/{}, graphics console shader kept {}/{}; untranslated: {} sets {} clears {} views; {} views; presented {presented:#x} bound {} times, cleared {clears} times to {colour:?}; most bound {}",
            m.executed.1,
            m.executed.0,
            draws.join(" "),
            copies.join(" "),
            m.copies.get(&presented).copied().unwrap_or(0),
            DISPATCHES.load(Ordering::Relaxed),
            INDIRECT.load(Ordering::Relaxed),
            PSO[0][0].load(Ordering::Relaxed),
            PSO[0][1].load(Ordering::Relaxed),
            PSO[1][0].load(Ordering::Relaxed),
            PSO[1][1].load(Ordering::Relaxed),
            PSO[2][0].load(Ordering::Relaxed),
            PSO[2][1].load(Ordering::Relaxed),
            RT_SET_UNTRANSLATED.load(Ordering::Relaxed),
            RT_CLEAR_UNTRANSLATED.load(Ordering::Relaxed),
            RTV_UNTRANSLATED.load(Ordering::Relaxed),
            m.rtv.len(),
            m.binds.get(&presented).copied().unwrap_or(0),
            top.join(" ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    /// A synthetic COM object: just enough for `com_add_ref`/`com_release` to work on it, so the
    /// bookkeeping tests exercise the real `AddRef`/`Release` path instead of a plain integer.
    #[repr(C)]
    struct FakeObj {
        vtbl: *const usize,
        refs: AtomicI32,
    }

    extern "system" fn fake_add_ref(this: usize) -> u32 {
        // SAFETY: test fixture; `this` is always a live `FakeObj` leaked by `fake_resource`.
        (unsafe { &*(this as *const FakeObj) }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
    }

    extern "system" fn fake_release(this: usize) -> u32 {
        // SAFETY: as `fake_add_ref`.
        (unsafe { &*(this as *const FakeObj) }.refs.fetch_sub(1, Ordering::SeqCst) - 1) as u32
    }

    static FAKE_VTBL: std::sync::OnceLock<[usize; 3]> = std::sync::OnceLock::new();

    /// A fake resource with one reference (as if the title's proxy held it), leaked so its
    /// address stays stable for the test; returns the address and a way to read its refcount.
    fn fake_resource() -> (usize, &'static AtomicI32) {
        let vtbl = FAKE_VTBL.get_or_init(|| [0, fake_add_ref as *const () as usize, fake_release as *const () as usize]);
        let obj = Box::leak(Box::new(FakeObj { vtbl: vtbl.as_ptr(), refs: AtomicI32::new(1) }));
        (obj as *const FakeObj as usize, &obj.refs)
    }

    /// The tables of this module are process wide, so the tests that fill them take turns.
    static TEST_TURN: Mutex<()> = Mutex::new(());

    #[test]
    fn draws_are_counted_per_render_target_through_the_view_map() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        STATE.store(2, Ordering::Relaxed);
        let (res, _refs) = fake_resource();
        rtv_made(0x100, res, || format!("desc of {res:#x}"));
        list_reset(0x9, 0);
        bound(0x9, &[0x100, 0x200]);
        draw(0x9, "DrawInstanced", 3);
        pipeline_set(0x9, 0x77);
        draw(0x9, "DrawInstanced", 4);
        cleared(0x100, [0.0, 0.0, 0.0, 1.0]);
        copied_into(res);
        let s = summary(res);
        assert!(s.contains(&format!("presented {res:#x} bound 1 times, cleared 1 times to [0.0, 0.0, 0.0, 1.0]")), "{s}");
        assert!(s.contains(&format!("{res:#x} [desc of {res:#x}] no pipeline last DrawInstanced(3):1")), "{s}");
        assert!(s.contains(&format!("{res:#x} [desc of {res:#x}] pipeline 0x77 last DrawInstanced(4):1")), "{s}");
        assert!(s.contains(&format!("copies by target {res:#x}:1 (presented 1)")), "{s}");
        end_frame();
    }

    /// Made (RTV), drawn this frame (owned), released by the title (`forget`): the resource is
    /// no longer offered by `most_drawn_other`, and the bookkeeping `AddRef` it took is given
    /// back (no leak), matching the brief's "made, drawn, released, next frame -> not returned".
    /// `most_drawn_other` names only what was drawn THIS frame (not a lifetime count), gives the
    /// caller its own owned reference independent of the bookkeeping one, and `end_frame` clears
    /// the frame without disturbing a guard already handed out.
    ///
    /// One test, not several: `MAPS`/`frame_owned` are process-global statics the default test
    /// harness runs on separate threads, so two of these tests interleaving their own
    /// `draw`/`forget`/`end_frame` calls could release each other's bookkeeping references.
    #[test]
    fn frame_scoped_target_bookkeeping() {
        let _turn = TEST_TURN.lock().unwrap_or_else(|e| e.into_inner());
        STATE.store(2, Ordering::Relaxed);

        // made, drawn, released -> forgotten, not returned, no leaked reference.
        let (dead, dead_refs) = fake_resource();
        rtv_made(0x300, dead, || "target".to_owned());
        list_reset(0x1, 0);
        bound(0x1, &[0x300]);
        draw(0x1, "DrawIndexedInstanced", 6);
        assert_eq!(dead_refs.load(Ordering::SeqCst), 2, "draw takes its own bookkeeping reference");
        forget(dead);
        assert_eq!(dead_refs.load(Ordering::SeqCst), 1, "forget must release the bookkeeping AddRef");
        assert!(most_drawn_other(0).is_none(), "a forgotten resource is never offered again");
        end_frame();
        assert_eq!(dead_refs.load(Ordering::SeqCst), 1, "end_frame must not double release an already forgotten resource");

        // made, drawn, still alive -> named this frame with the caller's own, independent reference.
        let (live, live_refs) = fake_resource();
        rtv_made(0x400, live, || "other target".to_owned());
        list_reset(0x2, 0);
        bound(0x2, &[0x400]);
        draw(0x2, "DrawInstanced", 3);
        assert_eq!(live_refs.load(Ordering::SeqCst), 2, "one bookkeeping reference for the frame");
        let guard = most_drawn_other(0x999).expect("the only drawn target, not the presented one");
        assert_eq!(guard.raw(), live);
        assert_eq!(live_refs.load(Ordering::SeqCst), 3, "the guard holds its own, independent reference");
        end_frame();
        assert_eq!(live_refs.load(Ordering::SeqCst), 2, "end_frame released the frame's bookkeeping reference only");
        assert!(most_drawn_other(0).is_none(), "the next frame starts with nothing drawn");
        drop(guard);
        assert_eq!(live_refs.load(Ordering::SeqCst), 1, "dropping the guard released its own reference");
        forget(live);
    }
}
