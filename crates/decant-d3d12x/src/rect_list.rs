//! The console's rectangle list topology (`RECT_LIST`, 0x11), which PC D3D12 does not have. Every
//! three vertices of a rectangle list are three corners of a rectangle and the GPU supplies the
//! fourth. A PC triangle list over the same vertices covers only half of each rectangle (a retail
//! title's full screen blits filled a single triangle of the screen). For a draw fed by vertex buffer
//! 0 the layer reads the vertices from title memory, appends the missing triangle of every rectangle
//! and draws six vertices per rectangle from a copy in upload memory.
//!
//! The missing corner lies opposite the right angle. Each 32 bit lane of it is the sum of the two
//! neighbouring corners minus the right angle corner. A lane that is identical in all three corners
//! is copied unchanged (a packed colour, say), every other lane is treated as a float. Packed lanes
//! that differ between corners are therefore not interpolated correctly, which the blits of the
//! titles seen so far do not need. The position is the first two floats of a vertex.

use std::collections::HashMap;
use std::sync::Mutex;

/// Vertices that make up one rectangle as the title submits it.
pub const IN: usize = 3;
/// Vertices that make up the two triangles drawn for one rectangle.
pub const OUT: usize = 6;

/// What a list knows about its vertex buffer 0: the title address, size and stride, and the view
/// that was passed on to the PC runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vb0 {
    pub va: u64,
    pub size: u32,
    pub stride: u32,
    pub view: [u64; 2],
}

/// The input assembler state of one command list that matters for rectangle lists.
#[derive(Default)]
struct Input {
    is_rect_topology: bool,
    vertex_buffer: Option<Vb0>,
}

static INPUTS: Mutex<Option<HashMap<usize, Input>>> = Mutex::new(None);

fn update<R>(list: usize, f: impl FnOnce(&mut Input) -> R) -> R {
    let mut guard = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(HashMap::new).entry(list).or_default())
}

/// The topology of `list` changed to a rectangle list (`rect`) or to something else.
pub fn set_rect(list: usize, rect: bool) {
    update(list, |input| input.is_rect_topology = rect);
}

/// `list` bound vertex buffer 0. `None` means no buffer, or one the layer cannot read.
pub fn set_vb0(list: usize, vb0: Option<Vb0>) {
    update(list, |input| input.vertex_buffer = vb0);
}

/// The vertex buffer 0 of `list`, but only while the list is set to draw rectangle lists.
pub fn rect_vb0(list: usize) -> Option<Vb0> {
    let guard = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
    let input = guard.as_ref()?.get(&list)?;
    input.is_rect_topology.then_some(input.vertex_buffer).flatten()
}

/// `list` was reset or released, so its input state is dropped.
pub fn forget(list: usize) {
    if let Some(all) = INPUTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        all.remove(&list);
    }
}

/// Dword `index` of a vertex.
fn dword(vertex: &[u8], index: usize) -> u32 {
    let at = index * 4;
    u32::from_le_bytes(vertex[at..at + 4].try_into().unwrap_or_default())
}

fn position(vertex: &[u8]) -> [f32; 2] {
    [f32::from_bits(dword(vertex, 0)), f32::from_bits(dword(vertex, 1))]
}

/// Which of the three corners sits at the right angle: the one where the two edges leaving it are
/// closest to perpendicular, judged by the absolute cosine between them.
fn right_angle_corner(corners: [&[u8]; 3]) -> usize {
    let points = corners.map(position);
    let abs_cosine = |at: usize| {
        let origin = points[at];
        let edge = |to: [f32; 2]| [to[0] - origin[0], to[1] - origin[1]];
        let (a, b) = (edge(points[(at + 1) % 3]), edge(points[(at + 2) % 3]));
        let length = (a[0].hypot(a[1]) * b[0].hypot(b[1])).max(f32::MIN_POSITIVE);
        ((a[0] * b[0] + a[1] * b[1]) / length).abs()
    };
    let mut best = 0;
    for at in 1..3 {
        if abs_cosine(at).total_cmp(&abs_cosine(best)).is_lt() {
            best = at;
        }
    }
    best
}

/// The corner opposite `apex`, built lane by lane from the apex and its two neighbours.
fn fourth_corner(apex: &[u8], left: &[u8], right: &[u8]) -> Vec<u8> {
    (0..apex.len() / 4)
        .flat_map(|lane| {
            let (o, a, b) = (dword(apex, lane), dword(left, lane), dword(right, lane));
            let value = if o == a && a == b { o } else { (f32::from_bits(a) + f32::from_bits(b) - f32::from_bits(o)).to_bits() };
            value.to_le_bytes()
        })
        .collect()
}

/// Turns the rectangles in `bytes` (vertices of `stride` bytes, three per rectangle) into the
/// vertices of two triangles each. The first triangle is the rectangle as given. The second one is
/// the fourth corner followed by the two corners next to the right angle, in the order that keeps the
/// winding of the first. `None` when the stride cannot hold a position, is not a whole number of
/// dwords, or when `bytes` is not a whole number of rectangles.
pub fn expand(bytes: &[u8], stride: usize) -> Option<Vec<u8>> {
    let rect_bytes = stride * IN;
    if stride < 8 || !stride.is_multiple_of(4) || bytes.is_empty() || !bytes.len().is_multiple_of(rect_bytes) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / IN * OUT);
    for rect in bytes.chunks_exact(rect_bytes) {
        let corners = [&rect[..stride], &rect[stride..stride * 2], &rect[stride * 2..]];
        let apex = right_angle_corner(corners);
        let (next, prev) = (corners[(apex + 1) % 3], corners[(apex + 2) % 3]);
        out.extend_from_slice(rect);
        // The corners go around as 0, 1, 2 and the half turn about the centre swaps the other two.
        out.extend_from_slice(&fourth_corner(corners[apex], next, prev));
        out.extend_from_slice(prev);
        out.extend_from_slice(next);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f32, y: f32, packed: u32, u: f32) -> Vec<u8> {
        [x.to_bits(), y.to_bits(), packed, u.to_bits()].into_iter().flat_map(u32::to_le_bytes).collect()
    }

    /// Twice the signed area of the triangle in the first three vertices of `bytes`.
    fn orientation(bytes: &[u8], stride: usize) -> f32 {
        let [a, b, c] = [0, 1, 2].map(|i| position(&bytes[i * stride..]));
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
    }

    #[test]
    fn full_screen_rectangle_gets_its_fourth_corner_and_keeps_the_winding() {
        let rect = [vertex(-1.0, 1.0, 0xFFFF_FFFF, 0.0), vertex(-1.0, -1.0, 0xFFFF_FFFF, 0.0), vertex(1.0, 1.0, 0xFFFF_FFFF, 1.0)].concat();
        let out = expand(&rect, 16).unwrap();
        assert_eq!(out.len(), OUT * 16);
        assert_eq!(out[..48], rect[..], "the original triangle comes first");
        assert_eq!(out[48..64], vertex(1.0, -1.0, 0xFFFF_FFFF, 1.0)[..], "opposite the right angle at the upper left");
        assert_eq!(orientation(&out[..48], 16).signum(), orientation(&out[48..], 16).signum());
    }

    #[test]
    fn right_angle_can_sit_at_any_corner() {
        for turn in 0..3 {
            let corners = [vertex(0.0, 0.0, 1, 0.0), vertex(10.0, 5.0, 1, 1.0), vertex(10.0, 0.0, 1, 1.0)];
            let rect: Vec<u8> = (0..3).flat_map(|i| corners[(i + turn) % 3].clone()).collect();
            let out = expand(&rect, 16).unwrap();
            assert_eq!(out[48..64], vertex(0.0, 5.0, 1, 0.0)[..], "turn {turn}");
            assert_eq!(orientation(&out[..48], 16).signum(), orientation(&out[48..], 16).signum(), "turn {turn}");
        }
    }

    #[test]
    fn differing_packed_lanes_are_treated_as_floats_and_equal_ones_are_kept() {
        let rect = [vertex(0.0, 0.0, 7, 0.5), vertex(0.0, 2.0, 7, 0.5), vertex(2.0, 0.0, 7, 1.5)].concat();
        let out = expand(&rect, 16).unwrap();
        assert_eq!(dword(&out[48..64], 2), 7);
        assert_eq!(f32::from_bits(dword(&out[48..64], 3)), 1.5);
    }

    #[test]
    fn bad_sizes_are_refused() {
        assert_eq!(expand(&[0u8; 40], 16), None, "a partial rectangle");
        assert_eq!(expand(&[0u8; 30], 10), None, "a stride that is no whole number of dwords");
        assert_eq!(expand(&[0u8; 12], 4), None, "a stride without room for a position");
        assert_eq!(expand(&[], 16), None, "no vertices");
    }

    #[test]
    fn a_list_reports_its_buffer_only_while_the_rectangle_topology_is_set() {
        let vb = Vb0 { va: 0x1000, size: 96, stride: 16, view: [0x2000, (16 << 32) | 96] };
        set_vb0(0x91, Some(vb));
        assert_eq!(rect_vb0(0x91), None);
        set_rect(0x91, true);
        assert_eq!(rect_vb0(0x91), Some(vb));
        set_rect(0x91, false);
        assert_eq!(rect_vb0(0x91), None);
        set_rect(0x91, true);
        forget(0x91);
        assert_eq!(rect_vb0(0x91), None);
    }
}
