//! The descriptor kind check behind `DECANT_D3D12X_KIND_CHECK=1` (off by default): at every draw,
//! dispatch and indirect execution, after the table sync, each table the bound pipeline's shaders
//! read is compared with what the host slots hold. A shader reads a view at the offset its own
//! binding table names (a constant buffer, a buffer or a texture, read only or writable), the title
//! packs each table for that shader, and a slot that holds another kind is read as a view of the
//! wrong type: on a GPU that is undefined, on the AMD Linux driver a ring timeout. The vkd3d-proton
//! descriptor QA build reports it, but it needs a special build and does not exist on Windows. This
//! gives the same signal from the layer.
//!
//! What a shader declares per offset is kept per pipeline when the pipeline is made:
//! * a direct pipeline: from the console container of each stage (its binding table and the `PSV0`
//!   resources, see [`views_of_program`]), read without translating, so a pipeline taken from the
//!   disk cache has them too;
//! * a DXIL override: from the `table_views` of its `<hash>.json` (`decant_shader_dxil_cache`), which
//!   names what the override reads at each offset. An override built before that field is not
//!   checked, and it is said once.
//!
//! What a slot holds comes from `view_diag` (the view the layer wrote, or copied, into the host
//! slot; the switch turns that record on) and from the title's own words of the slot (the layer's
//! tag, zero when the title never wrote the slot, and the table sync then leaves the slot alone).
//!
//! On a mismatch one line is logged, for the first [`PER_KEY`] of each (pipeline, parameter,
//! offset), and a total every [`TOTAL_EVERY`]. Off, every hook is one relaxed atomic load.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Mismatches logged for each (pipeline, parameter, offset).
pub const PER_KEY: u32 = 4;
/// A line with the total of all mismatches is logged every this many.
pub const TOTAL_EVERY: u64 = 1000;

static STATE: AtomicU8 = AtomicU8::new(0);

/// Whether `DECANT_D3D12X_KIND_CHECK` is set to something other than `0`, read once.
#[inline]
pub fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_KIND_CHECK").is_some_and(|v| v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Whether the declared kinds of each pipeline are kept: the check is on, or the zero slot null view
/// (`zero_null`) needs them. Only the comparison is behind [`on`].
#[inline]
pub fn tracking() -> bool {
    on() || crate::zero_null::on()
}

fn say(line: &str) {
    #[cfg(windows)]
    crate::log::xs_log!("{line}");
    #[cfg(not(windows))]
    let _ = line;
}

/// The class of a view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    Cbv,
    Srv,
    Uav,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Cbv => "constant buffer",
            Class::Srv => "SRV",
            Class::Uav => "UAV",
        }
    }
}

/// The shape of a view: a buffer, a texture of a dimension, or not known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Unknown,
    Buffer,
    Tex1d,
    Tex1dArray,
    Tex2d,
    Tex2dArray,
    Tex2dMs,
    Tex2dMsArray,
    Tex3d,
    Cube,
    CubeArray,
}

impl Dim {
    pub fn name(self) -> &'static str {
        match self {
            Dim::Unknown => "shape unknown",
            Dim::Buffer => "buffer",
            Dim::Tex1d => "texture 1D",
            Dim::Tex1dArray => "texture 1D array",
            Dim::Tex2d => "texture 2D",
            Dim::Tex2dArray => "texture 2D array",
            Dim::Tex2dMs => "texture 2D MS",
            Dim::Tex2dMsArray => "texture 2D MS array",
            Dim::Tex3d => "texture 3D",
            Dim::Cube => "texture cube",
            Dim::CubeArray => "texture cube array",
        }
    }

    fn is_texture(self) -> bool {
        !matches!(self, Dim::Unknown | Dim::Buffer)
    }

    /// The dimension of a DXIL resource kind (`PSV0`): 1 to 9 are textures, 10 to 12 buffers.
    pub fn from_dxil_kind(kind: u32) -> Dim {
        match kind {
            1 => Dim::Tex1d,
            2 => Dim::Tex2d,
            3 => Dim::Tex2dMs,
            4 => Dim::Tex3d,
            5 => Dim::Cube,
            6 => Dim::Tex1dArray,
            7 => Dim::Tex2dArray,
            8 => Dim::Tex2dMsArray,
            9 => Dim::CubeArray,
            10..=12 => Dim::Buffer,
            _ => Dim::Unknown,
        }
    }

    /// The dimension of a view description (`D3D12_SRV_DIMENSION` and `D3D12_UAV_DIMENSION`).
    pub fn from_view_dimension(class: Class, dimension: u32) -> Dim {
        match (class, dimension) {
            (_, 1) => Dim::Buffer,
            (_, 2) => Dim::Tex1d,
            (_, 3) => Dim::Tex1dArray,
            (_, 4) => Dim::Tex2d,
            (_, 5) => Dim::Tex2dArray,
            (Class::Srv, 6) => Dim::Tex2dMs,
            (Class::Srv, 7) => Dim::Tex2dMsArray,
            (Class::Srv, 8) => Dim::Tex3d,
            (Class::Srv, 9) => Dim::Cube,
            (Class::Srv, 10) => Dim::CubeArray,
            (Class::Uav, 8) => Dim::Tex3d,
            _ => Dim::Unknown,
        }
    }

    fn from_report(kind: &str) -> Option<Dim> {
        Some(match kind {
            "typed_buffer" | "raw_buffer" | "structured_buffer" => Dim::Buffer,
            "texture1d" => Dim::Tex1d,
            "texture2d" => Dim::Tex2d,
            "texture3d" => Dim::Tex3d,
            "texture_cube" => Dim::Cube,
            "texture1d_array" => Dim::Tex1dArray,
            "texture2d_array" => Dim::Tex2dArray,
            "texture_cube_array" => Dim::CubeArray,
            "cbuffer" => Dim::Unknown,
            _ => return None,
        })
    }

    /// Whether a shader that declares `self` can read a view of `found`: an unknown side matches.
    fn fits(self, found: Dim) -> bool {
        self == Dim::Unknown || found == Dim::Unknown || self == found
    }
}

/// What a shader reads at one offset of one of its tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Want {
    /// The mirror dword that holds the heap index of the table.
    pub dword: u8,
    /// The slot in the table (the offset in 32 byte descriptors).
    pub slot: u32,
    pub class: Class,
    /// The shape; [`Dim::Unknown`] for a constant buffer and for a typed view that is a buffer or a texture.
    pub dim: Dim,
}

impl Want {
    pub fn describe(&self) -> String {
        match self.class {
            Class::Cbv => "constant buffer".into(),
            c => format!("{} {}", c.name(), self.dim.name()),
        }
    }
}

/// How a shader stage's declarations were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The console container of a stage the direct path translated.
    Direct,
    /// The `table_views` of the json of a DXIL override.
    Override,
}

/// What one shader stage of a pipeline reads from its tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageDecl {
    /// `VS`, `PS`, `CS` and so on.
    pub stage: String,
    /// The content hash of the console bytecode, as the override files are named.
    pub hash: u64,
    pub source: Source,
    pub views: Vec<Want>,
}

/// The stages of a pipeline that are checked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipeDecl {
    pub graphics: bool,
    pub stages: Vec<StageDecl>,
}

/// The class and shape the `PSV0` record of a binding table entry declares.
fn dim_of_psv(class: Class, psv_type: u32, psv_kind: u32) -> Dim {
    if class == Class::Cbv {
        return Dim::Unknown;
    }
    match Dim::from_dxil_kind(psv_kind) {
        Dim::Unknown => match psv_type {
            // raw and structured SRVs, raw, structured and counter UAVs
            4 | 5 | 7 | 8 | 9 => Dim::Buffer,
            _ => Dim::Unknown,
        },
        d => d,
    }
}

/// What the console container of a stage reads at each offset of its tables: the entries of its
/// binding table at 32 byte slots, with the shape its `PSV0` resources declare.
pub fn views_of_program(program: &decant_shader::StageProgram) -> Vec<Want> {
    use decant_shader::RegisterClass as C;
    let mut out: Vec<Want> = Vec::new();
    for b in &program.binding_table {
        let class = match b.class {
            C::Srv => Class::Srv,
            C::Uav => Class::Uav,
            C::Cbv => Class::Cbv,
            C::Sampler => continue,
        };
        if !b.word_offset.is_multiple_of(8) {
            continue;
        }
        let declared = program.resources.iter().find(|r| r.class == b.class && (r.lower..=r.upper).contains(&b.register));
        let dim = declared.map_or(Dim::Unknown, |d| dim_of_psv(class, d.psv_type, d.psv_kind));
        let want = Want { dword: b.sgpr, slot: b.word_offset / 8, class, dim };
        if !out.contains(&want) {
            out.push(want);
        }
    }
    out
}

static OVERRIDES: Mutex<Option<HashMap<u64, Arc<Vec<Want>>>>> = Mutex::new(None);
static PIPES: Mutex<Option<HashMap<usize, Arc<PipeDecl>>>> = Mutex::new(None);
/// Pipelines kept at most. The layer learns of no release of a pipeline (the stand in of `lazy_pso` and
/// `pso_diag` is never freed, a PC pipeline is released by the title straight to the runtime), so the map
/// is bounded instead; one entry is a few hundred bytes.
const MAX_PIPES: usize = 1 << 15;

/// Keeps `decl` for `pso`. At the bound the map starts over: a list that has the pipeline bound keeps its
/// own `Arc`, and a pipeline bound later that was dropped counts as one this does not know (no check).
fn keep(map: &mut HashMap<usize, Arc<PipeDecl>>, pso: usize, decl: Arc<PipeDecl>, max: usize) {
    if map.len() >= max && !map.contains_key(&pso) {
        map.clear();
    }
    map.insert(pso, decl);
}
/// The pipeline last set on each list, graphics first, compute second, with its PC pointer.
type Bound = [Option<(usize, Arc<PipeDecl>)>; 2];
static LISTS: Mutex<Option<HashMap<usize, Bound>>> = Mutex::new(None);
static LIMITS: Mutex<Option<Limiter>> = Mutex::new(None);
static TOTAL: AtomicU64 = AtomicU64::new(0);
static UNCHECKED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);

/// The views of a `table_views` list; entries of a class or kind this does not know are left out.
pub fn wants_of_report(views: &[crate::dxil_override::TableViewReport]) -> Vec<Want> {
    let mut out: Vec<Want> = Vec::new();
    for v in views {
        let class = match v.class.as_str() {
            "srv" => Class::Srv,
            "uav" => Class::Uav,
            "cbv" => Class::Cbv,
            _ => continue,
        };
        let (Ok(dword), Some(dim)) = (u8::try_from(v.dword), Dim::from_report(&v.kind)) else { continue };
        let want = Want { dword, slot: v.slot, class, dim };
        if !out.contains(&want) {
            out.push(want);
        }
    }
    out
}

/// The override whose console bytecode has content hash `hash` (its json is `file_hash`) reads `views`.
/// Kept while declared kinds are tracked ([`tracking`]).
pub fn override_report(hash: u64, _file_hash: u64, views: &[crate::dxil_override::TableViewReport]) {
    if !tracking() {
        return;
    }
    let wants = wants_of_report(views);
    OVERRIDES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(hash, Arc::new(wants));
}

/// A stage of a pipeline as the creation saw it.
pub struct StageIn {
    pub stage: &'static str,
    pub hash: u64,
    /// Whether a DXIL override replaced it.
    pub overridden: bool,
}

/// The declarations of the stages of a pipeline made with DXIL overrides.
pub fn override_stages(stages: &[StageIn]) -> Vec<StageDecl> {
    let map = OVERRIDES.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for s in stages.iter().filter(|s| s.overridden) {
        match map.as_ref().and_then(|m| m.get(&s.hash)) {
            Some(views) if !views.is_empty() => out.push(StageDecl { stage: s.stage.to_string(), hash: s.hash, source: Source::Override, views: views.to_vec() }),
            _ => {
                let first = UNCHECKED.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(s.hash);
                if first && on() {
                    say(&format!("kind check: override {:016x} ({}) has no table_views in its json (built before the field, or it reads no table), not checked", s.hash, s.stage));
                }
            }
        }
    }
    out
}

/// Pipeline `pso` (the PC object) was made, with the DXIL overrides of its stages.
pub fn pipeline_made(pso: usize, graphics: bool, stages: &[StageIn]) {
    if !tracking() || pso == 0 {
        return;
    }
    let decl = PipeDecl { graphics, stages: override_stages(stages) };
    keep(PIPES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new), pso, Arc::new(decl), MAX_PIPES);
}

/// Pipeline `pso` was made with the direct shaders `stages`.
pub fn direct_made(pso: usize, graphics: bool, stages: Vec<StageDecl>) {
    if !tracking() || pso == 0 {
        return;
    }
    keep(PIPES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new), pso, Arc::new(PipeDecl { graphics, stages }), MAX_PIPES);
}

/// List `list` set (or was reset with) the PC pipeline `pso`, 0 for none.
pub fn bound(list: usize, pso: usize) {
    if !tracking() {
        return;
    }
    let decl = PIPES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&pso).cloned());
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let slots = lists.get_or_insert_with(HashMap::new).entry(list).or_default();
    match decl {
        Some(d) => {
            let at = usize::from(!d.graphics);
            slots[at] = Some((pso, d));
        }
        // a pipeline this does not know: the kind is not known either, so both are dropped
        None => *slots = [None, None],
    }
}

/// List `list` was reset: no pipeline is bound.
pub fn reset(list: usize) {
    if tracking()
        && let Some(m) = LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut()
    {
        m.remove(&list);
    }
}

/// The pipeline of the kind `graphics` last set on `list`, with its pointer.
pub fn pipeline_of(list: usize, graphics: bool) -> Option<(usize, Arc<PipeDecl>)> {
    LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&list)?[usize::from(!graphics)].clone()
}

/// What a host slot holds, as far as the layer knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    pub class: Class,
    /// A null view (of no resource).
    pub null: bool,
    pub dim: Dim,
    pub kind: &'static str,
}

impl Held {
    /// From a `view_diag` record: its kind text (`srv`, `uav at`, `cbv copy`, `copy srv`, ...), the
    /// resource (0 for none), the address and the format and dimension words of the view description.
    pub fn from_view(kind: &'static str, res: usize, va: u64, dimension: u32) -> Option<Held> {
        let class = if kind.contains("cbv") {
            Class::Cbv
        } else if kind.contains("uav") {
            Class::Uav
        } else if kind.contains("srv") {
            Class::Srv
        } else {
            return None;
        };
        let null = if class == Class::Cbv { va == 0 } else { res == 0 };
        let dim = if class == Class::Cbv { Dim::Unknown } else { Dim::from_view_dimension(class, dimension) };
        Some(Held { class, null, dim, kind })
    }

    fn describe(&self) -> String {
        let null = if self.null { "null " } else { "" };
        match self.class {
            Class::Cbv => format!("{null}constant buffer ({})", self.kind),
            c => format!("{null}{} {} ({})", c.name(), self.dim.name(), self.kind),
        }
    }
}

/// A host slot of a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The view the layer wrote there, `None` when it never wrote the slot (or its record is not of a view).
    pub held: Option<Held>,
    /// Whether the title's own words of the slot are all zero, `None` when they are not known.
    pub title_zero: Option<bool>,
}

/// Whether a shader that declares `want` is served by `slot`; if not, what the slot holds, as text.
pub fn compare(want: &Want, slot: &Slot) -> Option<String> {
    let Some(held) = slot.held else {
        return Some("never written".into());
    };
    if held.class != want.class {
        return Some(held.describe());
    }
    if want.class != Class::Cbv {
        let (w, h) = (want.dim, held.dim);
        let buffer_vs_texture = (w == Dim::Buffer && h.is_texture()) || (w.is_texture() && h == Dim::Buffer);
        if buffer_vs_texture || !w.fits(h) {
            return Some(held.describe());
        }
    }
    if slot.title_zero == Some(true) && !held.null {
        return Some(format!("{}, the title's words of the slot are all zero (the table sync leaves the slot as it was)", held.describe()));
    }
    None
}

/// One view of a stage that its slot does not serve.
#[derive(Debug, PartialEq, Eq)]
pub struct Mismatch<'a> {
    pub stage: &'a StageDecl,
    pub want: &'a Want,
    pub found: String,
}

/// Every view of `decl` whose slot does not serve it. `slot_of(dword, slot)` is what the host slot
/// holds, `None` when it cannot be looked at (the table is not in a view heap).
pub fn mismatches<'a>(decl: &'a PipeDecl, mut slot_of: impl FnMut(u8, u32) -> Option<Slot>) -> Vec<Mismatch<'a>> {
    let mut out = Vec::new();
    for stage in &decl.stages {
        for want in &stage.views {
            if let Some(found) = slot_of(want.dword, want.slot).and_then(|s| compare(want, &s)) {
                out.push(Mismatch { stage, want, found });
            }
        }
    }
    out
}

/// What a mismatch counts as when it is logged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    /// Log the line (the `n`th of its key).
    Line(u32),
    /// Over the limit of its key.
    Skip,
}

/// The per key limit of the lines.
#[derive(Default)]
pub struct Limiter {
    seen: HashMap<(usize, u32, u32), u32>,
}

impl Limiter {
    pub fn due(&mut self, pso: usize, param: u32, offset: u32) -> Due {
        let n = self.seen.entry((pso, param, offset)).or_insert(0);
        *n += 1;
        if *n <= PER_KEY { Due::Line(*n) } else { Due::Skip }
    }
}

/// Counts a mismatch of (`pso`, `param`, `offset`): whether its line is logged, and the line with the
/// total when it is a multiple of [`TOTAL_EVERY`].
pub fn count(pso: usize, param: u32, offset: u32) -> (Due, Option<String>) {
    let due = LIMITS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Limiter::default).due(pso, param, offset);
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    (due, total.is_multiple_of(TOTAL_EVERY).then(|| format!("kind check: {total} mismatches so far")))
}

/// Logs a line.
pub fn log(line: &str) {
    say(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn want(dword: u8, slot: u32, class: Class, dim: Dim) -> Want {
        Want { dword, slot, class, dim }
    }

    fn view(kind: &'static str, res: usize, va: u64, dimension: u32) -> Option<Held> {
        Held::from_view(kind, res, va, dimension)
    }

    fn slot(held: Option<Held>) -> Slot {
        Slot { held, title_zero: Some(false) }
    }

    #[test]
    fn a_buffer_read_where_a_constant_buffer_sits_is_a_mismatch() {
        let w = want(9, 5, Class::Srv, Dim::Buffer);
        let cbv = slot(view("cbv", 0, 0x1000, 0));
        assert_eq!(compare(&w, &cbv).as_deref(), Some("constant buffer (cbv)"));
        let copy = slot(view("copy cbv", 0, 0x1000, 0));
        assert!(compare(&w, &copy).is_some());
        let buffer = slot(view("srv", 0x10, 0, 1));
        assert_eq!(compare(&w, &buffer), None);
    }

    #[test]
    fn a_constant_buffer_is_served_by_a_constant_buffer_only() {
        let w = want(8, 2, Class::Cbv, Dim::Unknown);
        assert_eq!(compare(&w, &slot(view("cbv", 0, 0x40, 0))), None);
        assert_eq!(compare(&w, &slot(view("cbv copy", 0, 0x40, 0))), None);
        assert!(compare(&w, &slot(view("srv", 0x10, 0, 1))).unwrap().contains("SRV buffer"));
        assert!(compare(&w, &slot(view("uav", 0x10, 0, 4))).unwrap().contains("UAV texture 2D"));
    }

    #[test]
    fn buffers_and_textures_do_not_stand_for_each_other_and_classes_differ() {
        let buffer_srv = want(4, 0, Class::Srv, Dim::Buffer);
        let texture_srv = want(4, 1, Class::Srv, Dim::Tex2d);
        let texture = slot(view("srv", 0x10, 0, 4));
        let buffer = slot(view("srv", 0x10, 0, 1));
        assert!(compare(&buffer_srv, &texture).unwrap().contains("texture 2D"));
        assert_eq!(compare(&buffer_srv, &buffer), None);
        assert!(compare(&texture_srv, &buffer).unwrap().contains("buffer"));
        assert_eq!(compare(&texture_srv, &texture), None);
        // a writable buffer where a read only one is declared
        assert!(compare(&buffer_srv, &slot(view("uav", 0x10, 0, 1))).unwrap().contains("UAV buffer"));
        // a texture of another dimension
        assert!(compare(&texture_srv, &slot(view("srv", 0x10, 0, 8))).unwrap().contains("texture 3D"));
        // a shape the declaration does not say, or the view description does not
        assert_eq!(compare(&want(4, 0, Class::Srv, Dim::Unknown), &texture), None);
        assert_eq!(compare(&texture_srv, &slot(view("srv", 0x10, 0, 0))), None);
    }

    #[test]
    fn a_slot_never_written_or_stale_behind_zero_title_words_is_a_mismatch() {
        let w = want(4, 0, Class::Srv, Dim::Buffer);
        assert_eq!(compare(&w, &slot(None)).as_deref(), Some("never written"));
        // zero words and a view the sync would not touch: the F4 case (a texture where a buffer is wanted, and a same kind view)
        let zero = |held| Slot { held, title_zero: Some(true) };
        assert!(compare(&w, &zero(view("srv", 0x10, 0, 4))).unwrap().contains("texture 2D"));
        assert!(compare(&w, &zero(view("srv", 0x10, 0, 1))).unwrap().contains("all zero"));
        // zero words and a null view of the right kind: both read zeros
        assert_eq!(compare(&w, &zero(view("srv at", 0, 0x2000, 1))), None);
        // title words not known
        assert_eq!(compare(&w, &Slot { held: view("srv", 0x10, 0, 1), title_zero: None }), None);
    }

    #[test]
    fn a_view_record_is_read_by_its_kind_text() {
        let h = view("copy uav", 0x10, 0, 8).unwrap();
        assert_eq!((h.class, h.dim, h.null), (Class::Uav, Dim::Tex3d, false));
        assert!(view("uav at", 0, 0x5000, 1).unwrap().null);
        assert!(view("cbv", 0, 0, 0).unwrap().null);
        assert!(!view("cbv copy", 0, 0x100, 0).unwrap().null);
        assert_eq!(view("rtv", 0, 0, 0), None);
        assert_eq!(Dim::from_view_dimension(Class::Srv, 9), Dim::Cube);
        assert_eq!(Dim::from_view_dimension(Class::Uav, 9), Dim::Unknown);
    }

    #[test]
    fn every_view_of_every_stage_is_compared_with_its_own_slot() {
        let decl = PipeDecl {
            graphics: false,
            stages: vec![StageDecl {
                stage: "CS".into(),
                hash: 0xabc,
                source: Source::Override,
                views: vec![want(4, 0, Class::Srv, Dim::Buffer), want(4, 3, Class::Srv, Dim::Buffer), want(4, 5, Class::Uav, Dim::Buffer), want(8, 0, Class::Cbv, Dim::Unknown)],
            }],
        };
        let found = mismatches(&decl, |dword, s| match (dword, s) {
            (4, 0) => Some(slot(view("srv", 0x10, 0, 1))),
            (4, 3) | (4, 5) => Some(slot(view("cbv", 0, 0x100, 0))),
            // not in a view heap
            _ => None,
        });
        let at: Vec<(u8, u32)> = found.iter().map(|m| (m.want.dword, m.want.slot)).collect();
        assert_eq!(at, [(4, 3), (4, 5)]);
        assert_eq!(found[0].stage.hash, 0xabc);
        assert_eq!(found[0].found, "constant buffer (cbv)");
    }

    #[test]
    fn the_first_four_of_each_key_are_logged_and_the_total_is_counted() {
        let mut l = Limiter::default();
        let seen: Vec<Due> = (0..6).map(|_| l.due(0x10, 2, 5)).collect();
        assert_eq!(seen, [Due::Line(1), Due::Line(2), Due::Line(3), Due::Line(4), Due::Skip, Due::Skip]);
        assert_eq!(l.due(0x10, 2, 3), Due::Line(1), "another offset");
        assert_eq!(l.due(0x10, 3, 5), Due::Line(1), "another parameter");
        assert_eq!(l.due(0x20, 2, 5), Due::Line(1), "another pipeline");
    }

    #[test]
    fn the_pipeline_map_is_bounded_and_keeps_what_is_bound() {
        let decl = || Arc::new(PipeDecl { graphics: true, stages: Vec::new() });
        let mut map = HashMap::new();
        for pso in 1..=4 {
            keep(&mut map, pso, decl(), 4);
        }
        assert_eq!(map.len(), 4);
        keep(&mut map, 2, decl(), 4);
        assert_eq!(map.len(), 4, "a pipeline made again takes its own entry");
        keep(&mut map, 5, decl(), 4);
        assert_eq!(map.len(), 1, "at the bound the map starts over with the new pipeline");
        assert!(map.contains_key(&5));
        for pso in 10..10_000 {
            keep(&mut map, pso, decl(), 4);
        }
        assert!(map.len() <= 4);
    }

    #[test]
    fn a_report_names_the_views_of_an_override() {
        let report: crate::dxil_override::BindingReport = serde_json::from_str(
            r#"{"registers":[],"table_views":[{"dword":9,"slot":5,"class":"srv","kind":"raw_buffer"},
                {"dword":9,"slot":5,"class":"srv","kind":"raw_buffer"},
                {"dword":8,"slot":0,"class":"uav","kind":"texture2d"},
                {"dword":8,"slot":1,"class":"cbv","kind":"cbuffer"},
                {"dword":8,"slot":2,"class":"sampler","kind":"sampler"}]}"#,
        )
        .unwrap();
        let wants = wants_of_report(&report.table_views);
        assert_eq!(wants, [want(9, 5, Class::Srv, Dim::Buffer), want(8, 0, Class::Uav, Dim::Tex2d), want(8, 1, Class::Cbv, Dim::Unknown)]);
        let old: crate::dxil_override::BindingReport = serde_json::from_str(r#"{"registers":[]}"#).unwrap();
        assert!(old.table_views.is_empty());
    }

    #[test]
    fn the_psv_record_of_a_console_container_gives_the_shape() {
        assert_eq!(dim_of_psv(Class::Srv, 4, 0), Dim::Buffer, "raw SRV in the short record");
        assert_eq!(dim_of_psv(Class::Srv, 3, 0), Dim::Unknown, "a typed view is a buffer or a texture");
        assert_eq!(dim_of_psv(Class::Uav, 6, 4), Dim::Tex3d);
        assert_eq!(dim_of_psv(Class::Uav, 8, 12), Dim::Buffer);
        assert_eq!(dim_of_psv(Class::Cbv, 2, 13), Dim::Unknown);
    }
}
