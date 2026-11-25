//! The census: what a corpus of programs contains and how far the translator gets with each.
//!
//! It counts, over every program:
//! * instructions by name and by timing class, and the encoding forms with an extra word,
//! * features of the program shape (branches, loops, the ways EXEC is saved and restored),
//! * the first refusal of the whole translation, by key (the "first blocker"),
//! * every lifter refusal of the program, by key, so that a missing instruction is counted for
//!   each program that needs it and not only for the programs it blocks first.
//!
//! Work packages name the keys they remove, and the report is how that is measured.

pub mod ab;
pub mod pipeline;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::api::{CountingHost, Options, RootSignature, Stage, StageProgram};
use crate::cfg::{self, Term};
use crate::decode::{Form, Operand, Reg};
use crate::tables::{Class, Op};
use crate::{exec, lift};

/// A count with one example program.
#[derive(Clone, Debug, Default)]
pub struct Tally {
    pub count: u64,
    pub example: String,
    /// Position of the example program, so merged tallies keep the first one in corpus order.
    order: usize,
}

impl Tally {
    fn add(&mut self, example: &str, order: usize) {
        if self.count == 0 || order < self.order {
            self.example = example.to_owned();
            self.order = order;
        }
        self.count += 1;
    }

    fn merge(&mut self, other: Tally) {
        if other.count > 0 && (self.count == 0 || other.order < self.order) {
            self.example = other.example;
            self.order = other.order;
        }
        self.count += other.count;
    }
}

fn merge_map<K: Ord>(a: &mut BTreeMap<K, u64>, b: BTreeMap<K, u64>) {
    for (k, n) in b {
        *a.entry(k).or_default() += n;
    }
}

fn merge_tallies<K: Ord>(a: &mut BTreeMap<K, Tally>, b: BTreeMap<K, Tally>) {
    for (k, t) in b {
        a.entry(k).or_default().merge(t);
    }
}

/// The counts of a corpus.
#[derive(Debug, Default)]
pub struct Census {
    pub programs: u64,
    pub by_stage: BTreeMap<Stage, u64>,
    pub translated: BTreeMap<Stage, u64>,
    pub instructions: u64,
    pub ops: BTreeMap<Op, u64>,
    pub classes: BTreeMap<Class, u64>,
    /// Instructions with an extra word or modifiers, by kind.
    pub forms: BTreeMap<&'static str, u64>,
    /// Programs with a shape feature.
    pub features: BTreeMap<&'static str, Tally>,
    /// Programs by the key of the refusal that stops their translation.
    pub first_blocker: BTreeMap<String, Tally>,
    /// Programs by every lifter refusal key they contain.
    pub lift_gaps: BTreeMap<String, Tally>,
    /// Programs the lifter carries completely (every reachable instruction has a rule).
    pub fully_lifted: u64,
    /// Per program id: `translated` or the key of its first blocker.
    pub outcomes: BTreeMap<String, String>,
}

impl Census {
    /// Counts a container that does not decode.
    pub fn add_undecodable(&mut self, id: &str) {
        self.outcomes.insert(id.to_owned(), "container".to_owned());
        self.programs += 1;
        self.first_blocker
            .entry("container".to_owned())
            .or_default()
            .count += 1;
    }

    /// Adds the counts of another census, made over other programs.
    pub fn merge(&mut self, o: Census) {
        self.programs += o.programs;
        self.instructions += o.instructions;
        self.fully_lifted += o.fully_lifted;
        self.outcomes.extend(o.outcomes);
        merge_map(&mut self.by_stage, o.by_stage);
        merge_map(&mut self.translated, o.translated);
        merge_map(&mut self.ops, o.ops);
        merge_map(&mut self.classes, o.classes);
        merge_map(&mut self.forms, o.forms);
        merge_tallies(&mut self.features, o.features);
        merge_tallies(&mut self.first_blocker, o.first_blocker);
        merge_tallies(&mut self.lift_gaps, o.lift_gaps);
    }

    /// Adds one program. `order` is its position in the corpus (it decides which program is
    /// named as the example of a key), `id` names it in the per program outcomes of a report
    /// (see `corpus::Entry::id`). Returns the refusal key when it is not translated.
    pub fn add(&mut self, order: usize, id: &str, program: &StageProgram) -> Option<String> {
        self.programs += 1;
        *self.by_stage.entry(program.stage).or_default() += 1;
        let name = program.name.as_str();
        if let Ok(graph) = cfg::build(&program.code) {
            let mut features: BTreeSet<&'static str> = BTreeSet::new();
            if graph.blocks.len() > 1 {
                features.insert("branches");
            }
            for (i, b) in graph.blocks.iter().enumerate() {
                let back = match b.term {
                    Term::End => false,
                    Term::Jump(t) => t <= i,
                    Term::Branch { taken, fall, .. } => taken <= i || fall <= i,
                };
                if back {
                    features.insert("loops");
                }
            }
            let masks = exec::analyse(&graph).ok();
            let mut gaps: BTreeSet<String> = BTreeSet::new();
            let mut scratch = masks
                .as_ref()
                .map(|m| lift::Lifter::new(program.stage, m).with_relative(&graph));
            for insn in graph.insns() {
                self.instructions += 1;
                *self.ops.entry(insn.op).or_default() += 1;
                *self.classes.entry(insn.op.info().class).or_default() += 1;
                match insn.form {
                    Form::Sdwa => *self.forms.entry("sdwa").or_default() += 1,
                    Form::Dpp16 => *self.forms.entry("dpp16").or_default() += 1,
                    Form::Dpp8 => *self.forms.entry("dpp8").or_default() += 1,
                    Form::AsVop3 => {
                        *self
                            .forms
                            .entry("vop3 form of a one word instruction")
                            .or_default() += 1
                    }
                    Form::Plain => {}
                }
                if insn.src.iter().any(|s| matches!(s, Operand::Lit(_))) {
                    *self.forms.entry("literal").or_default() += 1;
                }
                // The two code shapes for leaving a masked region, and the other mask idioms.
                let exec_dst = insn.dst == Operand::Reg(Reg::ExecLo);
                match insn.op {
                    Op::s_mov_b64 if exec_dst => features.insert("exec restore by s_mov_b64"),
                    Op::s_or_b64 if exec_dst => features.insert("exec restore by s_or_b64"),
                    Op::s_xor_b64 if exec_dst => features.insert("exec else by s_xor_b64"),
                    Op::s_andn2_b64 if exec_dst => features.insert("exec cleared by s_andn2_b64"),
                    Op::s_wqm_b64 => features.insert("whole quad mode"),
                    Op::s_setpc_b64 | Op::s_getpc_b64 => features.insert("far branch code"),
                    Op::s_barrier => features.insert("barrier"),
                    Op::s_sendmsg | Op::s_sendmsghalt => features.insert("sendmsg"),
                    Op::v_readfirstlane_b32 | Op::v_readlane_b32_e64 | Op::v_writelane_b32_e64 => {
                        features.insert("lane read or write")
                    }
                    Op::s_bcnt1_i32_b64
                    | Op::s_ff1_i32_b64
                    | Op::s_ff0_i32_b64
                    | Op::s_flbit_i32_b64 => features.insert("mask counted or searched"),
                    op if exec::saveexec(op) => features.insert("saveexec"),
                    _ => false,
                };
                if let Some(cx) = scratch.as_mut()
                    && let Err(r) = lift::lift(cx, insn)
                {
                    gaps.insert(r.key());
                }
                if let Some(cx) = scratch.as_mut() {
                    cx.take();
                }
            }
            for f in features {
                self.features.entry(f).or_default().add(name, order);
            }
            if gaps.is_empty() && masks.is_some() {
                self.fully_lifted += 1;
            }
            for g in gaps {
                self.lift_gaps.entry(g).or_default().add(name, order);
            }
        }
        match crate::translate_probe(
            std::slice::from_ref(program),
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
        ) {
            Ok(_) => {
                *self.translated.entry(program.stage).or_default() += 1;
                self.outcomes.insert(id.to_owned(), "translated".to_owned());
                None
            }
            Err(r) => {
                let key = r.key();
                self.outcomes.insert(id.to_owned(), key.clone());
                self.first_blocker
                    .entry(key.clone())
                    .or_default()
                    .add(name, order);
                Some(key)
            }
        }
    }

    /// The report. `top` limits each list.
    pub fn report(&self, top: usize) -> String {
        self.render(top, false)
    }

    /// Only the counts and the two key lists (first blocker, lifter gaps).
    pub fn summary(&self, top: usize) -> String {
        self.render(top, true)
    }

    fn render(&self, top: usize, quiet: bool) -> String {
        let mut out = String::new();
        let w = &mut out;
        writeln!(
            w,
            "programs {}  instructions reached {}",
            self.programs, self.instructions
        )
        .ok();
        for (stage, n) in &self.by_stage {
            writeln!(
                w,
                "  {stage:?}: {n} programs, {} translated to HLSL",
                self.translated.get(stage).copied().unwrap_or(0)
            )
            .ok();
        }
        writeln!(
            w,
            "programs whose every instruction has a lifter rule: {}",
            self.fully_lifted
        )
        .ok();
        if quiet {
            self.write_lists(w, top, false);
            return out;
        }
        writeln!(w, "\ninstructions by class:").ok();
        let mut classes: Vec<_> = self.classes.iter().collect();
        classes.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (class, n) in classes {
            writeln!(w, "  {n:>10}  {class:?}").ok();
        }
        writeln!(
            w,
            "\ndistinct instructions: {}. Most frequent:",
            self.ops.len()
        )
        .ok();
        let mut ops: Vec<_> = self.ops.iter().collect();
        ops.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (op, n) in ops.iter().take(top) {
            writeln!(w, "  {n:>10}  {}", op.name()).ok();
        }
        writeln!(w, "\nencoding forms:").ok();
        for (form, n) in &self.forms {
            writeln!(w, "  {n:>10}  {form}").ok();
        }
        self.write_lists(w, top, true);
        out
    }

    fn write_lists(&self, w: &mut String, top: usize, features: bool) {
        let list = |w: &mut String, title: &str, map: &BTreeMap<String, Tally>| {
            writeln!(w, "\n{title}").ok();
            let mut rows: Vec<_> = map.iter().collect();
            rows.sort_by_key(|(_, t)| std::cmp::Reverse(t.count));
            for (key, t) in rows.iter().take(top) {
                writeln!(w, "  {:>7}  {key}  (for example {})", t.count, t.example).ok();
            }
            if rows.len() > top {
                writeln!(w, "  ... and {} more keys", rows.len() - top).ok();
            }
        };
        if features {
            writeln!(w, "\nprograms with a shape feature:").ok();
            for (f, t) in &self.features {
                writeln!(w, "  {:>7}  {f}", t.count).ok();
            }
        }
        list(
            w,
            "first blocker of each program that is not translated (programs by refusal key):",
            &self.first_blocker,
        );
        list(
            w,
            "lifter gaps (programs that contain the instruction or form, by refusal key):",
            &self.lift_gaps,
        );
    }
}

/// Every lifter refusal key of one program, sorted: each instruction or form the lifter has no
/// rule for, once. This is the complete list that the census counts per key.
pub fn lifter_gaps(program: &StageProgram) -> Vec<String> {
    let Ok(graph) = cfg::build(&program.code) else {
        return vec!["cfg".to_owned()];
    };
    let Ok(masks) = exec::analyse(&graph) else {
        return vec!["exec".to_owned()];
    };
    let mut lifter = lift::Lifter::new(program.stage, &masks).with_relative(&graph);
    let mut gaps = BTreeSet::new();
    for insn in graph.insns() {
        if let Err(r) = lift::lift(&mut lifter, insn) {
            gaps.insert(r.key());
        }
        lifter.take();
    }
    gaps.into_iter().collect()
}

/// A count with one example, as stored in a report file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Keyed {
    pub count: u64,
    pub example: String,
}

/// The census as a file: stable keys, so two runs can be compared.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub format: u32,
    /// What was counted: the corpus and the filters, to warn when two reports are not comparable.
    pub selection: String,
    pub programs: u64,
    pub instructions: u64,
    pub fully_lifted: u64,
    pub by_stage: BTreeMap<String, u64>,
    pub translated: BTreeMap<String, u64>,
    pub classes: BTreeMap<String, u64>,
    pub ops: BTreeMap<String, u64>,
    pub forms: BTreeMap<String, u64>,
    pub features: BTreeMap<String, Keyed>,
    pub first_blocker: BTreeMap<String, Keyed>,
    pub lift_gaps: BTreeMap<String, Keyed>,
    /// Per program id: `translated` or the key of its first blocker. Pipelines (see
    /// [`pipeline`]) are listed here too, as `pipeline VS:<hash>+PS:<hash>`.
    pub outcomes: BTreeMap<String, String>,
    #[serde(default)]
    pub pipelines: u64,
    #[serde(default)]
    pub pipelines_translated: u64,
    #[serde(default)]
    pub pipeline_blockers: BTreeMap<String, Keyed>,
}

fn keyed<K: ToString>(m: &BTreeMap<K, Tally>) -> BTreeMap<String, Keyed> {
    m.iter()
        .map(|(k, t)| {
            (
                k.to_string(),
                Keyed {
                    count: t.count,
                    example: t.example.clone(),
                },
            )
        })
        .collect()
}

fn named<K: std::fmt::Debug>(m: &BTreeMap<K, u64>) -> BTreeMap<String, u64> {
    m.iter().map(|(k, n)| (format!("{k:?}"), *n)).collect()
}

impl Census {
    /// The counts as a [`Report`]. `selection` describes the corpus and filters.
    pub fn to_report(&self, selection: &str) -> Report {
        Report {
            format: 1,
            selection: selection.to_owned(),
            programs: self.programs,
            instructions: self.instructions,
            fully_lifted: self.fully_lifted,
            by_stage: named(&self.by_stage),
            translated: named(&self.translated),
            classes: named(&self.classes),
            ops: self
                .ops
                .iter()
                .map(|(o, n)| (o.name().to_owned(), *n))
                .collect(),
            forms: self
                .forms
                .iter()
                .map(|(k, n)| ((*k).to_owned(), *n))
                .collect(),
            features: keyed(&self.features),
            first_blocker: keyed(&self.first_blocker),
            lift_gaps: keyed(&self.lift_gaps),
            outcomes: self.outcomes.clone(),
            ..Report::default()
        }
    }
}

/// 1234567 as `1,234,567`.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl Report {
    /// The standing progress line of one list: translated per stage against `old`, then the top
    /// blockers with their change. `label` names the list.
    pub fn progress_line(&self, old: Option<&Report>, label: &str, top: usize) -> String {
        let short = |stage: &str| match stage {
            "Vertex" => "VS",
            "Pixel" => "PS",
            "Compute" => "CS",
            "Mesh" => "MS",
            "Geometry" => "GS",
            _ => "??",
        };
        let mut parts = Vec::new();
        let order = ["Vertex", "Pixel", "Compute", "Mesh", "Geometry"];
        for stage in order {
            let Some(n) = self.by_stage.get(stage) else {
                continue;
            };
            let now = self.translated.get(stage).copied().unwrap_or(0);
            let of = grouped(*n);
            parts.push(match old {
                Some(o) => {
                    let before = o.translated.get(stage).copied().unwrap_or(0);
                    format!("{} {before} -> {now} of {of}", short(stage))
                }
                None => format!("{} {now} of {of}", short(stage)),
            });
        }
        let mut rows: Vec<(&String, u64)> = self
            .first_blocker
            .iter()
            .map(|(k, t)| (k, t.count))
            .collect();
        rows.sort_by_key(|(k, n)| (std::cmp::Reverse(*n), (*k).clone()));
        let was = |k: &String| old.and_then(|o| o.first_blocker.get(k)).map(|t| t.count);
        let mut out = format!("{label}: {}", parts.join(", "));
        if let Some((k, n)) = rows.first() {
            out.push_str(&format!(", top blocker now {k} {n}"));
            if let Some(w) = was(k) {
                out.push_str(&format!(" (was {w})"));
            }
        }
        out.push('\n');
        for (k, n) in rows.iter().take(top) {
            let change = match was(k) {
                Some(w) if w != *n => format!(" ({:+})", *n as i64 - w as i64),
                Some(_) => String::new(),
                None if old.is_some() => " (new)".to_owned(),
                None => String::new(),
            };
            out.push_str(&format!("  {n:>6}{change}  {k}\n"));
        }
        out
    }

    pub fn with_selection(mut self, selection: &str) -> Report {
        self.format = 1;
        self.selection = selection.to_owned();
        self
    }

    /// Adds the result of a pipeline census.
    pub fn with_pipelines(mut self, p: &pipeline::PipelineCensus) -> Report {
        self.pipelines = p.pipelines;
        self.pipelines_translated = p.translated;
        self.pipeline_blockers = keyed(&p.blockers);
        self.outcomes.extend(
            p.outcomes
                .iter()
                .map(|(id, o)| (format!("pipeline {id}"), o.clone())),
        );
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a report is plain data")
    }

    pub fn from_json(text: &str) -> Result<Report, String> {
        let r: Report = serde_json::from_str(text).map_err(|e| format!("report: {e}"))?;
        if r.format != 1 {
            return Err(format!("report: format {} is not known", r.format));
        }
        Ok(r)
    }

    /// What changed from `old` to `self`: counts per stage, keys added, removed and moved,
    /// programs gained and lost. `top` limits each list.
    pub fn diff(&self, old: &Report, top: usize) -> String {
        let mut out = String::new();
        let w = &mut out;
        if old.selection != self.selection {
            writeln!(
                w,
                "note: the reports count different things ({:?} against {:?})",
                old.selection, self.selection
            )
            .ok();
        }
        let num = |w: &mut String, what: &str, a: u64, b: u64| {
            if a != b {
                writeln!(w, "  {what}: {a} -> {b} ({:+})", b as i64 - a as i64).ok();
            }
        };
        writeln!(w, "delta to the stored report:").ok();
        num(w, "programs", old.programs, self.programs);
        for (stage, n) in &self.by_stage {
            let before = old.translated.get(stage).copied().unwrap_or(0);
            let now = self.translated.get(stage).copied().unwrap_or(0);
            num(w, &format!("{stage} translated (of {n})"), before, now);
        }
        num(
            w,
            "every instruction has a lifter rule",
            old.fully_lifted,
            self.fully_lifted,
        );
        num(
            w,
            "instructions reached",
            old.instructions,
            self.instructions,
        );
        num(w, "pipelines", old.pipelines, self.pipelines);
        num(
            w,
            "pipelines translated",
            old.pipelines_translated,
            self.pipelines_translated,
        );
        for (title, a, b) in [
            ("first blocker", &old.first_blocker, &self.first_blocker),
            ("lifter gaps", &old.lift_gaps, &self.lift_gaps),
            (
                "pipeline first blocker",
                &old.pipeline_blockers,
                &self.pipeline_blockers,
            ),
        ] {
            let mut rows: Vec<(String, u64, u64)> = Vec::new();
            for k in a.keys().chain(b.keys().filter(|k| !a.contains_key(*k))) {
                let before = a.get(k).map_or(0, |t| t.count);
                let now = b.get(k).map_or(0, |t| t.count);
                if before != now {
                    rows.push((k.clone(), before, now));
                }
            }
            if rows.is_empty() {
                continue;
            }
            let gone = rows.iter().filter(|r| r.2 == 0).count();
            let new = rows.iter().filter(|r| r.1 == 0).count();
            writeln!(
                w,
                "{title}: {} keys changed, {gone} removed, {new} added",
                rows.len()
            )
            .ok();
            rows.sort_by_key(|r| std::cmp::Reverse((r.2 as i64 - r.1 as i64).unsigned_abs()));
            for (k, before, now) in rows.iter().take(top) {
                let tag = if *now == 0 {
                    "removed"
                } else if *before == 0 {
                    "added"
                } else {
                    "changed"
                };
                writeln!(
                    w,
                    "  {tag:>7}  {k}: {before} -> {now} ({:+})",
                    *now as i64 - *before as i64
                )
                .ok();
            }
            if rows.len() > top {
                writeln!(w, "  ... and {} more", rows.len() - top).ok();
            }
        }
        let (mut gained, mut lost) = (Vec::new(), Vec::new());
        let mut moves: BTreeMap<(String, String), u64> = BTreeMap::new();
        for (name, now) in &self.outcomes {
            let Some(before) = old.outcomes.get(name) else {
                continue;
            };
            if before == now {
                continue;
            }
            match (before.as_str(), now.as_str()) {
                (_, "translated") => gained.push(name),
                ("translated", _) => lost.push(name),
                _ => {
                    *moves.entry((before.clone(), now.clone())).or_default() += 1;
                }
            }
        }
        let fresh = self
            .outcomes
            .keys()
            .filter(|k| !old.outcomes.contains_key(*k))
            .count();
        let vanished = old
            .outcomes
            .keys()
            .filter(|k| !self.outcomes.contains_key(*k))
            .count();
        if fresh + vanished > 0 {
            writeln!(
                w,
                "programs only in the new report: {fresh}, only in the old one: {vanished}"
            )
            .ok();
        }
        for (title, names) in [
            ("programs gained (now translated)", &gained),
            ("programs lost (no longer translated)", &lost),
        ] {
            if names.is_empty() {
                continue;
            }
            writeln!(w, "{title}: {}", names.len()).ok();
            for n in names.iter().take(top) {
                writeln!(w, "  {n}").ok();
            }
            if names.len() > top {
                writeln!(w, "  ... and {} more", names.len() - top).ok();
            }
        }
        if !moves.is_empty() {
            let mut rows: Vec<_> = moves.into_iter().collect();
            rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            writeln!(w, "first blocker moved (programs, old -> new):").ok();
            for ((a, b), n) in rows.iter().take(top) {
                writeln!(w, "  {n:>7}  {a} -> {b}").ok();
            }
            if rows.len() > top {
                writeln!(w, "  ... and {} more kinds of move", rows.len() - top).ok();
            }
        }
        if out.lines().count() == 1 {
            out.push_str("  no change\n");
        }
        out
    }
}
