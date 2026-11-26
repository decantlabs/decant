//! Immediate samplers of directly translated shaders. A console program can build a
//! sampler from four constant words instead of reading it from memory. PC shaders have no such
//! thing, so the layer gives each distinct set of words an index and keeps the PC sampler at
//! `SamplerDescriptorHeap[HEAP_BASE + index]` of every shader visible sampler heap. The translator
//! asks [`index_of`] for the index and emits that heap read (this works for comparison samplers
//! too, and needs no root signature change beyond `SAMPLER_HEAP_DIRECTLY_INDEXED`).
//!
//! The registry is append only and kept in `immediate-samplers.bin` beside the DXIL cache
//! (`direct_cache`), so an index means the same sampler in every run and cached DXIL stays valid.
//! The file is a 8 byte header (`SXSM`, format version) and 16 bytes per sampler in index order.
//!
//! Every shader visible sampler heap the title creates under `DECANT_D3D12X_DIRECT` is made
//! [`HEAP_SIZE`] long (the PC maximum). Its tail from [`HEAP_BASE`] on belongs to the layer and is
//! filled from the registry, at creation and whenever a new sampler is registered. A title heap
//! that already needs more than [`HEAP_BASE`] descriptors keeps its size and gets no tail (logged).
//! A list on which the title binds view heaps without a sampler heap gets a sampler heap of the
//! layer added (`proxy`).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::PathBuf;
use std::sync::{Mutex, RwLock};

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
            self.0 = self.0.wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0 ^ u64::from(i)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;
type FastSet<K> = HashSet<K, BuildHasherDefault<FastHasher>>;

/// The size of a shader visible sampler heap under the direct path (`D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE`).
pub const HEAP_SIZE: u32 = 2048;
/// How many immediate samplers fit in the tail.
pub const CAPACITY: u32 = 256;
/// The first descriptor of the layer's tail: the index a translated shader adds to the registry index.
pub const HEAP_BASE: u32 = HEAP_SIZE - CAPACITY;

const MAGIC: &[u8; 4] = b"SXSM";
const FORMAT: u32 = 1;
const HEADER: usize = 8;

/// The four console sampler words.
pub type Words = [u32; 4];

/// Words to index, append only, optionally backed by a file.
#[derive(Default)]
pub struct Registry {
    words: Vec<Words>,
    index: FastMap<Words, u32>,
    file: Option<PathBuf>,
}

impl Registry {
    /// A registry backed by `file`, with what the file already holds. A file that does not check
    /// is set aside (renamed to `.bad`) and the registry starts empty, because cached DXIL that
    /// used its indices would point at the wrong samplers: those cache files then miss anyway when
    /// the translator version changes, and are the price of a damaged file.
    pub fn open(file: Option<PathBuf>) -> Registry {
        let mut r = Registry { file, ..Registry::default() };
        let Some(path) = r.file.clone() else { return r };
        let Ok(bytes) = std::fs::read(&path) else { return r };
        let ok = bytes.len() >= HEADER && &bytes[..4] == MAGIC && bytes[4..8] == FORMAT.to_le_bytes() && (bytes.len() - HEADER).is_multiple_of(16);
        if !ok {
            let _ = std::fs::rename(&path, path.with_extension("bad"));
            return r;
        }
        for chunk in bytes[HEADER..].as_chunks::<16>().0 {
            let w: Words = std::array::from_fn(|i| u32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]));
            if !r.index.contains_key(&w) && (r.words.len() as u32) < CAPACITY {
                r.index.insert(w, r.words.len() as u32);
                r.words.push(w);
            }
        }
        r
    }

    /// The index of `words`, registered (and appended to the file) when new. `None` when the tail
    /// is full. The second value is true for a new entry.
    pub fn index_of(&mut self, words: Words) -> Option<(u32, bool)> {
        if let Some(i) = self.index.get(&words) {
            return Some((*i, false));
        }
        let i = self.words.len() as u32;
        if i >= CAPACITY {
            return None;
        }
        self.index.insert(words, i);
        self.words.push(words);
        if let Some(path) = &self.file {
            let _ = append(path, &words);
        }
        Some((i, true))
    }

    /// Every registered sampler in index order.
    pub fn all(&self) -> &[Words] {
        &self.words
    }
}

fn append(path: &std::path::Path, words: &Words) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let fresh = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut out = Vec::with_capacity(HEADER + 16);
    if fresh {
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
    }
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    f.write_all(&out)
}

/// The file of the registry beside the DXIL cache, none when the cache is off. The unit tests of
/// this crate keep the registry of their process in memory, so a test run never writes into the
/// registry file of the user's real cache.
pub fn default_file() -> Option<PathBuf> {
    if cfg!(test) {
        return None;
    }
    crate::direct_cache::DiskCache::from_env().map(|c| c.dir().join("immediate-samplers.bin"))
}

/// A decoded console sampler: the fields of a `D3D12_SAMPLER_DESC` and what it cannot say.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoded {
    /// `D3D12_FILTER`.
    pub filter: u32,
    /// `D3D12_TEXTURE_ADDRESS_MODE` of U, V and W.
    pub address: [u32; 3],
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    /// `D3D12_COMPARISON_FUNC`.
    pub comparison: u32,
    pub border_color: [f32; 4],
    pub min_lod: f32,
    pub max_lod: f32,
    /// `FORCE_UNNORMALIZED`: the sampler takes texel coordinates. D3D12 has no such sampler, so
    /// the translator is told ([`Entry`]) and divides the coordinates by the size itself.
    pub force_unnormalized: bool,
}

impl Decoded {
    /// The `D3D12_SAMPLER_DESC` (0x34 bytes).
    pub fn to_desc(&self) -> [u8; 0x34] {
        let mut d = [0u8; 0x34];
        let mut put = |at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(0, self.filter);
        put(4, self.address[0]);
        put(8, self.address[1]);
        put(0xC, self.address[2]);
        put(0x10, self.mip_lod_bias.to_bits());
        put(0x14, self.max_anisotropy);
        put(0x18, self.comparison);
        for (i, c) in self.border_color.iter().enumerate() {
            put(0x1C + i * 4, c.to_bits());
        }
        put(0x2C, self.min_lod.to_bits());
        put(0x30, self.max_lod.to_bits());
        d
    }
}

/// A bit or value of the words that the `D3D12_SAMPLER_DESC` cannot express: a name and the value
/// that was seen. Each distinct one is logged once.
type Gap = (&'static str, u32);

/// Decodes four console sampler words after the GFX10 sampler word layout, returning what the
/// description cannot express as well. Pure: nothing is logged.
pub fn decode_words(words: Words) -> (Decoded, Vec<Gap>) {
    let [w0, w1, w2, w3] = words;
    let mut gaps: Vec<Gap> = Vec::new();

    // Address modes (`SQ_TEX_CLAMP`) to `D3D12_TEXTURE_ADDRESS_MODE`.
    let mut address = [0u32; 3];
    for (i, a) in address.iter_mut().enumerate() {
        let mode = (w0 >> (i * 3)) & 7;
        *a = match mode {
            0 => 1,
            1 => 2,
            2 => 3,
            3 => 5,
            6 => 4,
            // Half border and once border modes have no D3D12 value: the nearest one.
            4 | 7 => {
                gaps.push(("address mode", mode));
                4
            }
            _ => {
                gaps.push(("address mode", mode));
                5
            }
        };
    }

    let ratio = ((w0 >> 9) & 7).min(4);
    let compare = (w0 >> 12) & 7;
    let force_unnormalized = w0 & (1 << 15) != 0;
    if force_unnormalized {
        gaps.push(("FORCE_UNNORMALIZED", 1));
    }
    for (bit, name) in [(19, "MC_COORD_TRUNC"), (20, "FORCE_DEGAMMA"), (27, "TRUNC_COORD"), (28, "DISABLE_CUBE_WRAP"), (31, "SKIP_DEGAMMA")] {
        if w0 & (1 << bit) != 0 {
            gaps.push((name, 1));
        }
    }
    let filter_mode = (w0 >> 29) & 3;

    let mag = (w2 >> 20) & 3;
    let min = (w2 >> 22) & 3;
    let z_filter = (w2 >> 24) & 3;
    let mip_filter = (w2 >> 26) & 3;
    if z_filter != 0 {
        gaps.push(("Z_FILTER", z_filter));
    }

    // Reduction: comparison wins over min and max (D3D12 has one reduction per filter).
    let reduction = if compare != 0 {
        1
    } else {
        match filter_mode {
            1 => 2,
            2 => 3,
            _ => 0,
        }
    };
    let filter = if min >= 2 || mag >= 2 {
        0x55 | reduction << 7
    } else {
        // POINT_ANISO_ADJ (3) is a point mip filter, LINEAR (2) the only linear one.
        let mip = u32::from(mip_filter == 2);
        (min & 1) << 4 | (mag & 1) << 2 | mip | reduction << 7
    };

    // LOD range, unsigned 4.8, and bias, signed 6.8.
    let min_raw = w1 & 0xFFF;
    let max_raw = (w1 >> 12) & 0xFFF;
    let min_lod = min_raw as f32 / 256.0;
    let max_lod = if mip_filter == 0 {
        // No mips: only the base level is read, as rts0 does.
        min_lod
    } else if max_raw == 0xFFF || max_raw == 15 * 256 {
        f32::MAX
    } else {
        (max_raw as f32 / 256.0).max(min_lod)
    };
    let bias_raw = ((w2 & 0x3FFF) << 18) as i32 >> 18;
    let mip_lod_bias = (bias_raw as f32 / 256.0).clamp(-16.0, 15.99);

    let border_color = match w3 >> 30 {
        1 => [0.0, 0.0, 0.0, 1.0],
        2 => [1.0; 4],
        3 => {
            // The custom colour table of the console runtime is unknown.
            gaps.push(("BORDER_COLOR_TYPE register, pointer", w3 & 0xFFF));
            [0.0; 4]
        }
        _ => [0.0; 4],
    };

    let decoded = Decoded { filter, address, mip_lod_bias, max_anisotropy: 1 << ratio, comparison: compare + 1, border_color, min_lod, max_lod, force_unnormalized };
    (decoded, gaps)
}

/// What the layer knows about one registered sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The index of the sampler, to add to [`HEAP_BASE`].
    pub index: u32,
    /// The sampler has `FORCE_UNNORMALIZED` set: the translator divides texel coordinates itself.
    pub force_unnormalized: bool,
}

/// The `D3D12_SAMPLER_DESC` (0x34 bytes) for four console sampler words. What D3D12 cannot
/// express is logged once per distinct value and otherwise ignored.
pub fn decode(words: Words) -> [u8; 0x34] {
    let (decoded, gaps) = decode_words(words);
    if !gaps.is_empty() {
        static SEEN: Mutex<Option<FastSet<Gap>>> = Mutex::new(None);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(Default::default);
        for gap in gaps {
            if seen.insert(gap) {
                #[cfg(windows)]
                crate::log::xs_log!("direct shaders: sampler {:08X} {:08X} {:08X} {:08X}: {} {} has no D3D12 equivalent", words[0], words[1], words[2], words[3], gap.0, gap.1);
            }
        }
    }
    decoded.to_desc()
}

/// The number of descriptors to create a sampler heap with: `count` raised to [`HEAP_SIZE`] when
/// the title's part fits below [`HEAP_BASE`], else `count` (no tail). Only shader visible sampler
/// heaps (type 1, flag 1) are raised.
pub fn heap_count(heap_type: u32, flags: u32, count: u32) -> u32 {
    if heap_type == 1 && flags & 1 != 0 && count <= HEAP_BASE { HEAP_SIZE } else { count }
}

/// A sampler heap with the layer's tail: where to write descriptor `HEAP_BASE + i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tail {
    /// The PC device that made the heap.
    pub device: usize,
    /// The CPU handle of descriptor 0.
    pub cpu_start: usize,
    /// The sampler descriptor increment.
    pub increment: u32,
}

impl Tail {
    /// The CPU handle of registry entry `i`.
    pub fn handle(&self, i: u32) -> usize {
        self.cpu_start + (HEAP_BASE + i) as usize * self.increment as usize
    }
}

/// Writes one sampler: `(device, sampler description, CPU handle)`.
pub type Write = fn(usize, &[u8; 0x34], usize);

/// The registry of the process and the heaps it fills.
struct Global {
    registry: Registry,
    /// Sampler heaps with a tail.
    tails: FastMap<usize, Tail>,
    /// Sampler heaps without one (too big).
    others: FastSet<usize>,
    /// How samplers are written, from the first heap.
    write: Option<Write>,
}

static GLOBAL: RwLock<Option<Global>> = RwLock::new(None);

fn with_write<R>(f: impl FnOnce(&mut Global) -> R) -> R {
    let mut g = GLOBAL.write().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(|| Global { registry: Registry::open(default_file()), tails: FastMap::default(), others: FastSet::default(), write: None }))
}

#[inline]
fn force_unnormalized_of(words: Words) -> bool {
    words[0] & (1 << 15) != 0
}

/// The entry of `words` for the translator (heap index offset from [`HEAP_BASE`], and whether the
/// sampler is unnormalised), registered when new and then written to every sampler heap with a
/// tail. `None` when the tail is full.
pub fn index_of(words: Words) -> Option<Entry> {
    {
        let g = GLOBAL.read().unwrap_or_else(|e| e.into_inner());
        if let Some(g) = g.as_ref()
            && let Some(&index) = g.registry.index.get(&words)
        {
            return Some(Entry { index, force_unnormalized: force_unnormalized_of(words) });
        }
    }
    with_write(|g| {
        let (i, fresh) = g.registry.index_of(words)?;
        if let (true, Some(write)) = (fresh, g.write) {
            let desc = decode(words);
            for t in g.tails.values() {
                write(t.device, &desc, t.handle(i));
            }
        }
        Some(Entry { index: i, force_unnormalized: force_unnormalized_of(words) })
    })
}

/// A sampler heap `heap` was made, with a tail or without (`None`). A tail is filled with every
/// registered sampler by `write`, which also writes the ones registered later.
pub fn heap_created(heap: usize, tail: Option<Tail>, write: Write) {
    with_write(|g| {
        g.write = Some(write);
        let Some(tail) = tail else {
            g.others.insert(heap);
            return;
        };
        for (i, w) in g.registry.all().iter().enumerate() {
            write(tail.device, &decode(*w), tail.handle(i as u32));
        }
        g.tails.insert(heap, tail);
    });
}

/// Whether `heap` is a sampler heap made under the direct path.
pub fn is_sampler_heap(heap: usize) -> bool {
    let g = GLOBAL.read().unwrap_or_else(|e| e.into_inner());
    g.as_ref().is_some_and(|g| g.tails.contains_key(&heap) || g.others.contains(&heap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_cache::tests::TempDir;

    #[test]
    fn indices_are_append_only_and_survive_a_restart() {
        let tmp = TempDir::new("samplers");
        let file = tmp.0.join("immediate-samplers.bin");
        let mut r = Registry::open(Some(file.clone()));
        assert_eq!(r.index_of([1, 2, 3, 4]), Some((0, true)));
        assert_eq!(r.index_of([5, 6, 7, 8]), Some((1, true)));
        assert_eq!(r.index_of([1, 2, 3, 4]), Some((0, false)));
        let mut again = Registry::open(Some(file.clone()));
        assert_eq!(again.all(), [[1, 2, 3, 4], [5, 6, 7, 8]]);
        assert_eq!(again.index_of([5, 6, 7, 8]), Some((1, false)));
        assert_eq!(again.index_of([9, 9, 9, 9]), Some((2, true)));
        assert_eq!(std::fs::metadata(&file).expect("file").len(), (HEADER + 3 * 16) as u64);
        // A damaged file is set aside, not read.
        std::fs::write(&file, b"nonsense").expect("write");
        assert!(Registry::open(Some(file.clone())).all().is_empty());
        assert!(file.with_extension("bad").is_file());
    }

    #[test]
    fn the_tail_is_full_at_its_capacity() {
        let mut r = Registry::open(None);
        for i in 0..CAPACITY {
            assert_eq!(r.index_of([i, 0, 0, 0]), Some((i, true)));
        }
        assert_eq!(r.index_of([CAPACITY, 0, 0, 0]), None);
    }

    #[test]
    fn only_shader_visible_sampler_heaps_that_fit_get_a_tail() {
        assert_eq!(heap_count(1, 1, 16), HEAP_SIZE);
        assert_eq!(heap_count(1, 1, HEAP_BASE), HEAP_SIZE);
        assert_eq!(heap_count(1, 1, HEAP_BASE + 1), HEAP_BASE + 1);
        assert_eq!(heap_count(1, 0, 16), 16, "not shader visible");
        assert_eq!(heap_count(0, 1, 16), 16, "a view heap");
        let t = Tail { device: 1, cpu_start: 0x1000, increment: 32 };
        assert_eq!(t.handle(2), 0x1000 + (HEAP_BASE as usize + 2) * 32);
    }

    /// Builds words field by field.
    struct W([u32; 4]);

    impl W {
        /// Wrap, no compare, trilinear, all mips.
        fn new() -> W {
            W([0, 0xFFF << 12, 1 << 20 | 1 << 22 | 2 << 26, 0]).filters(1, 1, 2)
        }
        fn address(mut self, u: u32, v: u32, w: u32) -> W {
            self.0[0] = self.0[0] & !0x1FF | u | v << 3 | w << 6;
            self
        }
        fn ratio(mut self, r: u32) -> W {
            self.0[0] = self.0[0] & !(7 << 9) | r << 9;
            self
        }
        fn compare(mut self, c: u32) -> W {
            self.0[0] = self.0[0] & !(7 << 12) | c << 12;
            self
        }
        fn bit0(mut self, bit: u32) -> W {
            self.0[0] |= 1 << bit;
            self
        }
        fn mode(mut self, m: u32) -> W {
            self.0[0] = self.0[0] & !(3 << 29) | m << 29;
            self
        }
        fn lod(mut self, min: u32, max: u32) -> W {
            self.0[1] = min | max << 12;
            self
        }
        fn bias(mut self, raw: i32) -> W {
            self.0[2] = self.0[2] & !0x3FFF | (raw as u32 & 0x3FFF);
            self
        }
        fn filters(mut self, mag: u32, min: u32, mip: u32) -> W {
            self.0[2] = self.0[2] & !(0xF << 20) & !(3 << 26) | mag << 20 | min << 22 | mip << 26;
            self
        }
        fn border(mut self, t: u32, ptr: u32) -> W {
            self.0[3] = t << 30 | ptr;
            self
        }
        fn d(&self) -> Decoded {
            decode_words(self.0).0
        }
        fn gaps(&self) -> Vec<Gap> {
            decode_words(self.0).1
        }
    }

    #[test]
    fn filters() {
        // (mag, min, mip) to the D3D12 filter.
        for ((mag, min, mip), want) in [
            ((0, 0, 0), 0x00),
            ((0, 0, 1), 0x00),
            ((0, 0, 2), 0x01),
            ((1, 0, 2), 0x05),
            ((0, 1, 2), 0x11),
            ((1, 1, 0), 0x14),
            ((1, 1, 1), 0x14),
            ((1, 1, 2), 0x15),
            ((1, 1, 3), 0x14),
            ((2, 0, 2), 0x55),
            ((0, 3, 0), 0x55),
            ((3, 3, 2), 0x55),
        ] {
            assert_eq!(W::new().filters(mag, min, mip).d().filter, want, "mag {mag} min {min} mip {mip}");
        }
    }

    #[test]
    fn reductions_and_comparison_filters() {
        assert_eq!(W::new().d().filter, 0x15);
        assert_eq!(W::new().mode(1).d().filter, 0x115, "minimum");
        assert_eq!(W::new().mode(2).d().filter, 0x195, "maximum");
        assert_eq!(W::new().mode(3).d().filter, 0x15, "reserved mode");
        assert_eq!(W::new().mode(2).compare(3).d().filter, 0x95, "comparison beats the reduction");
        assert_eq!(W::new().filters(3, 3, 2).ratio(4).compare(7).d().filter, 0xD5, "comparison anisotropic");
        assert_eq!(W::new().filters(0, 0, 0).compare(1).d().filter, 0x80);
    }

    #[test]
    fn address_modes() {
        // SQ_TEX_CLAMP value to D3D12 address mode (wrap 1, mirror 2, clamp 3, border 4, mirror once 5).
        for (mode, want) in [(0, 1), (1, 2), (2, 3), (3, 5), (4, 4), (5, 5), (6, 4), (7, 4)] {
            // The three axes decode on their own.
            assert_eq!(W::new().address(mode, 0, 0).d().address, [want, 1, 1]);
            assert_eq!(W::new().address(0, mode, 0).d().address, [1, want, 1]);
            assert_eq!(W::new().address(0, 0, mode).d().address, [1, 1, want]);
        }
        assert!(W::new().address(0, 1, 2).gaps().is_empty());
        assert!(W::new().address(3, 6, 0).gaps().is_empty());
        assert_eq!(W::new().address(4, 5, 7).gaps(), [("address mode", 4), ("address mode", 5), ("address mode", 7)]);
    }

    #[test]
    fn compare_functions() {
        for f in 0..8 {
            let d = W::new().compare(f).d();
            assert_eq!(d.comparison, f + 1);
            assert_eq!(d.filter & 0x80 != 0, f != 0);
        }
    }

    #[test]
    fn anisotropy_levels() {
        for (ratio, want) in [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16), (5, 16), (7, 16)] {
            assert_eq!(W::new().filters(3, 3, 2).ratio(ratio).d().max_anisotropy, want, "ratio {ratio}");
        }
    }

    #[test]
    fn lod_edges() {
        assert_eq!((W::new().lod(0, 0).d().min_lod, W::new().lod(0, 0).d().max_lod), (0.0, 0.0));
        let d = W::new().lod(256, 8 * 256).d();
        assert_eq!((d.min_lod, d.max_lod), (1.0, 8.0));
        let d = W::new().lod(0xFFF, 0xFFF).d();
        assert_eq!((d.min_lod, d.max_lod), (4095.0 / 256.0, f32::MAX), "all mips");
        assert_eq!(W::new().lod(0, 15 * 256).d().max_lod, f32::MAX);
        assert_eq!(W::new().lod(3 * 256, 2 * 256).d().max_lod, 3.0, "max never below min");
        assert_eq!(W::new().filters(1, 1, 0).lod(3 * 256 + 128, 0xFFF).d().max_lod, 3.5, "no mips: base level");
        assert_eq!(W::new().lod(1, 0xFFF).d().min_lod, 1.0 / 256.0);
    }

    #[test]
    fn lod_bias_is_signed_and_clamped() {
        for (raw, want) in [(0, 0.0), (256, 1.0), (-384, -1.5), (1, 1.0 / 256.0), (-1, -1.0 / 256.0), (0x1FFF, 15.99), (-0x2000, -16.0), (0xF00, 15.0)] {
            assert_eq!(W::new().bias(raw).d().mip_lod_bias, want, "raw {raw}");
        }
    }

    #[test]
    fn border_colours() {
        for (t, want) in [(0, [0.0; 4]), (1, [0.0, 0.0, 0.0, 1.0]), (2, [1.0; 4]), (3, [0.0; 4])] {
            assert_eq!(W::new().address(6, 6, 6).border(t, 0).d().border_color, want, "type {t}");
        }
        assert!(W::new().border(2, 0).gaps().is_empty());
        assert_eq!(W::new().border(3, 0x2A).gaps(), [("BORDER_COLOR_TYPE register, pointer", 0x2A)]);
    }

    #[test]
    fn bits_without_an_equivalent_are_reported_not_fatal() {
        let mut w = W::new().bit0(27).bit0(28).bit0(19).bit0(20).bit0(31);
        w.0[2] |= 1 << 24;
        let names: Vec<_> = w.gaps().iter().map(|g| g.0).collect();
        assert_eq!(names, ["MC_COORD_TRUNC", "FORCE_DEGAMMA", "TRUNC_COORD", "DISABLE_CUBE_WRAP", "SKIP_DEGAMMA", "Z_FILTER"]);
        assert_eq!(w.d().filter, 0x15, "none of them changes the description");
        // Quality hints are ignored without a log: ANISO_THRESHOLD, ANISO_BIAS, PERF_MIP, ANISO_OVERRIDE.
        let mut hints = W::new().filters(3, 3, 2).ratio(4);
        hints.0[0] |= 2 << 16 | 4 << 21;
        hints.0[1] |= 10 << 24;
        hints.0[2] |= 1 << 29;
        assert!(hints.gaps().is_empty());
    }

    #[test]
    fn force_unnormalized_is_recorded_in_the_entry() {
        let plain = W::new().filters(1, 1, 0);
        let unnormalized = W::new().filters(1, 1, 0).bit0(15);
        assert!(unnormalized.d().force_unnormalized && !plain.d().force_unnormalized);
        assert_eq!(unnormalized.gaps(), [("FORCE_UNNORMALIZED", 1)]);
        assert_eq!(unnormalized.d().to_desc(), plain.d().to_desc(), "the description itself does not change");
        let a = index_of(unnormalized.0).expect("room");
        assert!(a.force_unnormalized);
        assert!(!index_of(plain.0).expect("room").force_unnormalized);
        assert_eq!(index_of(unnormalized.0), Some(a), "same words, same entry");
    }

    /// Words as radeonsi writes them (`si_create_sampler_state` into `ac_build_sampler_descriptor`,
    /// GFX10), worked out by hand from the field table, and the sampler they stand for.
    #[test]
    fn radeonsi_states() {
        let f = |d: &[u8; 0x34], at: usize| u32::from_le_bytes(d[at..at + 4].try_into().expect("4 bytes"));
        let fl = |d: &[u8; 0x34], at: usize| f32::from_bits(f(d, at));
        // Trilinear, wrap, all mips: word 1 = 0xFFF << 12, mag 1 << 20, min 1 << 22, mip 2 << 26.
        let d = decode([0, 0x00FF_F000, 0x0850_0000, 0]);
        assert_eq!((f(&d, 0), f(&d, 4), f(&d, 8), f(&d, 0xC), f(&d, 0x14), f(&d, 0x18)), (0x15, 1, 1, 1, 1, 1));
        assert_eq!((fl(&d, 0x2C), fl(&d, 0x30)), (0.0, f32::MAX));
        // Point, clamp to edge, TRUNC_COORD: word 0 = 2 | 2 << 3 | 2 << 6 | 1 << 27, no mips.
        let d = decode([0x0800_0092, 0, 0, 0]);
        assert_eq!((f(&d, 0), f(&d, 4), f(&d, 8), f(&d, 0xC), fl(&d, 0x30)), (0, 3, 3, 3, 0.0));
        // 16x anisotropic wrap: word 0 = ratio 4 << 9 | threshold 2 << 16 | bias 4 << 21 = 0x0082_0800,
        // word 1 = all mips | PERF_MIP 10 << 24, word 2 = XY filters 3, mip 2, ANISO_OVERRIDE 1 << 29.
        let d = decode([0x0082_0800, 0x0AFF_F000, 0x2000_0000 | 0x08F0_0000, 0]);
        assert_eq!((f(&d, 0), f(&d, 0x14)), (0x55, 16));
        // Shadow PCF: compare LESSEQUAL 3 << 12, clamp to border 0x1B6, bilinear (1 << 20 | 1 << 22), opaque black.
        let d = decode([0x31B6, 0, 0x0050_0000, 0x4000_0000]);
        assert_eq!((f(&d, 0), f(&d, 4), f(&d, 0x18)), (0x94, 4, 4));
        assert_eq!(fl(&d, 0x28), 1.0, "alpha of opaque black");
        // LOD bias -1.5 (0x3E80 as signed 14 bit), min LOD 1 (0x100), max LOD 8 (0x800 << 12), trilinear.
        let d = decode([0, 0x0080_0100, 0x0850_3E80, 0]);
        assert_eq!((fl(&d, 0x10), fl(&d, 0x2C), fl(&d, 0x30)), (-1.5, 1.0, 8.0));
    }

    #[test]
    fn decoded_samplers_are_valid_descriptions() {
        let d = decode([0; 4]);
        assert_eq!(u32::from_le_bytes(d[4..8].try_into().expect("4 bytes")), 1, "address mode WRAP");
        assert_eq!(f32::from_bits(u32::from_le_bytes(d[0x30..0x34].try_into().expect("4 bytes"))), 0.0, "no mips: base level only");
    }

    #[test]
    fn concurrent_index_of_and_heap_queries_agree() {
        let w_norm = W::new().address(1, 2, 0).0;
        let w_unnorm = W::new().address(1, 2, 0).bit0(15).0;
        let expected_norm = index_of(w_norm).expect("room");
        let expected_unnorm = index_of(w_unnorm).expect("room");
        assert!(!expected_norm.force_unnormalized);
        assert!(expected_unnorm.force_unnormalized);
        fn noop_write(_device: usize, _desc: &[u8; 0x34], _cpu: usize) {}
        let heap = 0x7777_0000usize;
        heap_created(heap, None, noop_write);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..32 {
                        assert_eq!(index_of(w_norm), Some(expected_norm));
                        assert_eq!(index_of(w_unnorm), Some(expected_unnorm));
                        assert!(is_sampler_heap(heap));
                    }
                });
            }
        });
    }
}
