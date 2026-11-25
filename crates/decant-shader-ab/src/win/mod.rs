//! Everything that needs Windows: DXC, the D3D12 device and the runner of a side.
#![allow(unsafe_code)]

pub mod device;
pub mod dxc;
pub mod exec;
pub mod root;

use std::path::Path;

use crate::compare::{Output, Tolerance};
use crate::model::{Reflection, Stage};
use crate::pairs::Outcome;
use crate::plan::{Options, Plan, build};
use device::{Gpu, open_device};
use dxc::Dxc;

/// Shader model of HLSL inputs without an explicit profile, `6_6` by default.
pub fn hlsl_profile(stage: Stage, model: &str) -> String {
    format!("{}_{model}", stage.short())
}

/// Reads a side: a DXIL container (`.dxil`, `.cso`, `.dxbc`, anything else), or HLSL (`.hlsl`,
/// optionally `file.hlsl@entry`, entry `main` by default) compiled for `stage` with DXC.
pub fn load_shader(dxc: &Dxc, spec: &str, stage: Option<Stage>, model: &str) -> Result<Vec<u8>, String> {
    let (path, entry) = match spec.rsplit_once('@') {
        Some((p, e)) if p.to_ascii_lowercase().ends_with(".hlsl") => (p, e),
        _ => (spec, "main"),
    };
    if path.to_ascii_lowercase().ends_with(".hlsl") {
        let source = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let stage = stage.ok_or_else(|| format!("{path}: HLSL needs a stage (vs, ps or cs)"))?;
        dxc.compile(&source, entry, &hlsl_profile(stage, model), &[])
    } else {
        std::fs::read(path).map_err(|e| format!("{path}: {e}"))
    }
}

/// Whether an error says the device is gone: removed, hung, reset, or a GPU that stopped answering.
pub fn device_lost(error: &str) -> bool {
    ["device removed", "did not finish", "0x887A0005", "0x887A0006", "0x887A0007", "0x887A0020"].iter().any(|w| error.contains(w))
}

/// A device that is opened when first needed and opened again after it was lost.
#[derive(Default)]
pub struct Session {
    gpu: Option<Gpu>,
    /// `warp` or `hardware` to choose the device here instead of through `DECANT_SHADER_DEVICE`.
    pub kind: Option<String>,
    /// The adapter of the last device.
    pub adapter: String,
}

impl Session {
    /// A session on the device `kind` (`warp` or `hardware`).
    pub fn on(kind: &str) -> Session {
        Session { kind: Some(kind.to_string()), ..Session::default() }
    }

    /// Runs one side. When the device is removed or hangs, it is dropped so that the next side
    /// gets a fresh one. A side whose device was found lost is run once more on a fresh device,
    /// because a loss can surface only in the work that follows the side that caused it.
    pub fn side(&mut self, plan: &Plan, which: usize, container: &[u8], feeder: Option<&[u8]>) -> Result<Vec<Output>, String> {
        let first = self.attempt(plan, which, container, feeder);
        match first {
            Err(e) if device_lost(&e) => self.attempt(plan, which, container, feeder).map_err(|again| if device_lost(&again) { again } else { format!("{again} (first attempt: {e})") }),
            other => other,
        }
    }

    fn attempt(&mut self, plan: &Plan, which: usize, container: &[u8], feeder: Option<&[u8]>) -> Result<Vec<Output>, String> {
        if self.gpu.is_none() {
            let g = match &self.kind {
                Some(k) => device::open_device_kind(k),
                None => open_device(),
            }
            .ok_or("no D3D12 device with Shader Model 6.0")?;
            self.adapter = g.name.clone();
            self.gpu = Some(g);
        }
        let gpu = self.gpu.as_ref().expect("opened above");
        let result = exec::run_side(gpu, plan, which, container, feeder);
        if gpu.removed().is_some() || result.as_ref().is_err_and(|e| device_lost(e)) {
            self.gpu = None;
        }
        result
    }
}

/// The two sides of a pair after reading and reflecting: each its container and reflection, or why
/// it could not be read.
pub struct Prepared {
    pub plan: Plan,
    pub containers: [Result<Vec<u8>, String>; 2],
    pub feeders: [Option<Result<Vec<u8>, String>>; 2],
}

/// Reads, reflects and plans a pair. `Err` when neither side can be reflected or the plan fails.
pub fn prepare(dxc: &Dxc, a: &str, b: &str, stage: Option<Stage>, root: Option<&Path>, model: &str, options: Options) -> Result<Prepared, String> {
    let mut repaired = Vec::new();
    let mut load = |spec: &str, side: &str| -> Result<(Vec<u8>, Reflection), String> {
        let mut c = load_shader(dxc, spec, stage, model)?;
        let r = dxc.reflect(&c)?;
        // A handle index outside its binding range removes a WARP device: run the repaired module.
        match dxc.repair_binding_indices(&c) {
            Ok(Some((fixed, n))) => {
                c = fixed;
                repaired.push(format!("{side}: {n} createHandleFromBinding indices outside their range, rewritten to the register"));
            }
            Ok(None) => {}
            Err(e) => repaired.push(format!("{side}: binding indices outside their range could not be rewritten: {e}")),
        }
        Ok((c, r))
    };
    let (la, lb) = (load(a, "A"), load(b, "B"));
    let (ra, rb) = match (&la, &lb) {
        (Ok((_, x)), Ok((_, y))) => (x.clone(), y.clone()),
        (Ok((_, x)), Err(_)) => (x.clone(), x.clone()),
        (Err(_), Ok((_, y))) => (y.clone(), y.clone()),
        (Err(x), Err(y)) => return Err(format!("A: {x} | B: {y}")),
    };
    let root = match root {
        Some(p) => Some(root::deserialize(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)?),
        None => None,
    };
    let mut plan = build(ra, rb, stage, root, options)?;
    plan.notes.extend(repaired);
    let feeders = [0, 1].map(|i| plan.sides[i].feeder.as_ref().map(|src| dxc.compile(src, "main", "vs_6_0", &[]).map_err(|e| format!("feeder vertex shader: {e}"))));
    let containers = [la.map(|x| x.0), lb.map(|x| x.0)];
    Ok(Prepared { plan, containers, feeders })
}

/// Runs side `which` of a prepared pair in this process.
pub fn run_prepared_side(session: &mut Session, p: &Prepared, which: usize) -> Result<Vec<Output>, String> {
    let container = p.containers[which].as_ref().map_err(|e| e.clone())?;
    let feeder = match &p.feeders[which] {
        Some(Ok(f)) => Some(f.as_slice()),
        Some(Err(e)) => return Err(e.clone()),
        None => None,
    };
    session.side(&p.plan, which, container, feeder)
}

/// Runs a whole pair in this process, A first, then B on the same device or a fresh one.
#[allow(clippy::too_many_arguments)]
pub fn run_pair(session: &mut Session, dxc: &Dxc, hash: &str, a: &str, b: &str, stage: Option<Stage>, root: Option<&Path>, model: &str, options: Options, tol: Tolerance) -> Outcome {
    let p = match prepare(dxc, a, b, stage, root, model, options) {
        Ok(p) => p,
        Err(e) => {
            let mut o = Outcome::skipped(hash, e.clone());
            if let Some((x, y)) = e.split_once(" | B: ") {
                o.verdict = crate::compare::Verdict::BothFailed;
                o.a_error = Some(x.trim_start_matches("A: ").to_string());
                o.b_error = Some(y.to_string());
                o.notes.clear();
            }
            return o;
        }
    };
    let ra = run_prepared_side(session, &p, 0);
    let rb = run_prepared_side(session, &p, 1);
    Outcome::from_sides(hash, &p.plan, ra, rb, tol)
}
