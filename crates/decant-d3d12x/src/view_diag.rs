//! Descriptor diagnostics behind `DECANT_D3D12X_PRESENT_PROBE`: which view the layer wrote into each
//! host descriptor slot (made by a view call or copied from another slot), and for a few draws of
//! each render target and pipeline the tables of the user data mirror with the views their first
//! slots hold and the bytes of the same slots in the title's own heap memory. A draw that reads
//! black can so be told apart: a slot the layer never wrote (the title wrote the descriptor into
//! its heap memory itself), a slot past the clamped PC heap, or a view of a resource that is empty.
//!
//! Off by default. Off, every hook here is one relaxed atomic load. `DECANT_D3D12X_KIND_CHECK` (`kind_check`)
//! and `DECANT_D3D12X_FLICKER_PROBE` (`flicker_probe`) turn the record of the views on too, without the
//! draw descriptions.

use std::collections::HashMap;
use std::sync::Mutex;

/// One view the layer wrote into a host descriptor slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    /// The PC resource, 0 for a null view or a constant buffer view. Only a number: the resource
    /// may be released long before a draw is described (a GetDesc on it then crashed the title in
    /// its world load), so its shape is taken when the view is made.
    pub res: usize,
    /// Width, height, format and flags of `res` when the view was made, zeros for none.
    pub shape: (u64, u32, u32, u32),
    /// The view format and dimension (the first two words of the view description), 0 when none.
    pub format: u32,
    pub dim: u32,
    /// The title address of a view made at an address, 0 for a view of a resource object.
    pub va: u64,
    /// Which call made it.
    pub kind: &'static str,
}

/// A shader visible heap: where its slots start for the host CPU and in title memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heap {
    pub host_cpu: usize,
    pub increment: u32,
    pub host_count: u32,
    pub title_count: u32,
    /// The start of the title's own memory for the heap, 0 when the heap has none.
    pub title_base: usize,
}

#[derive(Default)]
struct Maps {
    views: HashMap<usize, View>,
    /// GPU start of a shader visible heap to the heap.
    heaps: HashMap<u64, Heap>,
    /// Draws described so far.
    described: u32,
    /// Title addresses of constant buffer views of described draws, read again at present.
    watched: Vec<u64>,
    /// Root signature to its layout, for the described draws.
    signatures: HashMap<usize, String>,
}

static MAPS: Mutex<Option<Maps>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Maps) -> R) -> R {
    let mut g = MAPS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(Maps::default))
}

pub fn enabled() -> bool {
    crate::rt_diag::enabled() || crate::kind_check::on() || crate::flicker_probe::on()
}

/// A shader visible heap with GPU start `gpu` was made.
pub fn heap_made(gpu: u64, heap: Heap) {
    if enabled() {
        with(|m| m.heaps.insert(gpu, heap));
    }
}

/// The layer wrote `view` into the host slot `host_cpu`.
pub fn view_made(host_cpu: usize, view: View) {
    if enabled() {
        with(|m| m.views.insert(host_cpu, view));
    }
}

/// The kind text of a copy of a view of `kind`: it keeps what the view was (`kind_check` reads the class).
fn copy_of(kind: &'static str) -> &'static str {
    match kind {
        "srv" | "copy srv" => "copy srv",
        "uav" | "copy uav" => "copy uav",
        "cbv" | "cbv copy" | "copy cbv" => "copy cbv",
        "srv at" | "copy srv at" => "copy srv at",
        "uav at" | "copy uav at" => "copy uav at",
        _ => "copy",
    }
}

/// `n` slots of `increment` bytes were copied from `src` to `dst`.
pub fn copied(dst: usize, src: usize, n: u32, increment: u32) {
    if !enabled() || increment == 0 {
        return;
    }
    with(|m| {
        for k in 0..n as usize {
            let off = k * increment as usize;
            match m.views.get(&(src + off)).copied() {
                Some(v) => m.views.insert(dst + off, View { kind: copy_of(v.kind), ..v }),
                None => m.views.remove(&(dst + off)),
            };
        }
    });
}

/// Whether the `n`th draw of a render target and pipeline pair (or dispatch of a pipeline) is one
/// the probes describe: the first, the 40th, two late ones for a view that stays for minutes, then every 2048th so a slot that
/// goes black minutes after the load has a binding record from near that present.
pub fn picked(n: u64) -> bool {
    matches!(n, 1 | 40 | 800 | 1600) || (n > 1600 && n.is_multiple_of(2048))
}

/// Whether the next draw of a render target and pipeline pair, the `n`th, is described: a
/// [`picked`] one, at most 600 in total.
pub fn wanted(n: u64) -> bool {
    // the descriptor kind check (`kind_check`) only records views, it does not describe draws
    if !crate::rt_diag::enabled() || !picked(n) {
        return false;
    }
    with(|m| {
        m.described += 1;
        m.described <= 600
    })
}

/// The heap whose GPU start is `gpu`.
pub fn heap(gpu: u64) -> Option<Heap> {
    with(|m| m.heaps.get(&gpu).copied())
}

/// The view the layer last wrote into `host_cpu`.
pub fn view(host_cpu: usize) -> Option<View> {
    with(|m| m.views.get(&host_cpu).copied())
}

/// Root signature `object` was made with `layout`.
pub fn signature_made(object: usize, layout: String) {
    with(|m| m.signatures.insert(object, layout));
}

/// The layout of root signature `object`, once per signature.
pub fn signature_once(object: usize) -> Option<String> {
    with(|m| m.signatures.remove(&object))
}

/// Keeps the title address `va` of a constant buffer view to read it again at present.
pub fn watch(va: u64) {
    with(|m| {
        if !m.watched.contains(&va) {
            m.watched.push(va);
            if m.watched.len() > 8 {
                m.watched.remove(0);
            }
        }
    });
}

/// The first 16 words at each watched address now, for the present probe line. Only addresses of
/// readable title memory are watched (the caller checked them).
#[allow(unsafe_code)]
pub fn watched_now() -> String {
    with(|m| {
        m.watched
            .iter()
            .map(|va| {
                // SAFETY: the address was readable title memory when it was watched, and title memory
                // stays mapped while the title runs.
                let words: Vec<u32> = (0..16).map(|w| unsafe { std::ptr::read_unaligned((*va as usize + w * 4) as *const u32) }).collect();
                format!("{va:#x}: {words:x?}")
            })
            .collect::<Vec<_>>()
            .join(", ")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_draws_are_sampled_at_a_steady_pace() {
        assert!([1, 40, 800, 1600, 2048, 4096, 40960].iter().all(|&n| picked(n)));
        assert!(![0, 2, 1601, 2047, 2049, 4097].iter().any(|&n| picked(n)));
    }

    #[test]
    fn copies_carry_views_and_clear_slots_the_source_lacks() {
        crate::rt_diag::force_on();
        let v = View { res: 0x50, shape: (64, 64, 28, 0), format: 28, dim: 4, va: 0, kind: "srv" };
        view_made(0x1000, v);
        view_made(0x2020, v);
        copied(0x2000, 0x1000, 2, 0x20);
        assert_eq!(view(0x2000), Some(View { kind: "copy srv", ..v }));
        assert_eq!(view(0x2020), None);
    }
}
