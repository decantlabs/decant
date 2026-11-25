//! Named refusals. A program the translator cannot carry yet is refused with a reason whose
//! [`Refusal::key`] is stable, so the census can count programs by their first blocker and a
//! work package can name the keys it removes.
//!
//! Adding a variant is allowed for every pass. Renaming a key is not: census reports compare them.

use std::fmt;

use crate::api::Stage;
use crate::tables::Op;

/// Why a program is not translated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Refusal {
    pub stage: Option<Stage>,
    /// The word index of the instruction that caused it, when one did.
    pub pc: Option<u32>,
    pub kind: RefusalKind,
    /// Free text for people. Not part of the key.
    pub detail: String,
}

/// The reason, by pass.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RefusalKind {
    /// The code does not start with the six word signature, or is empty.
    Container(&'static str),
    /// A word that is no instruction.
    Decode,
    /// An instruction the lifter has no rule for yet.
    Op(Op),
    /// An instruction the lifter knows, in a form it does not handle yet: `sdwa`, `dpp`, `opsel`,
    /// `omod`, `clamp`, `literal64`, ...
    Form(Op, &'static str),
    /// Control flow: `indirect-jump` (a computed jump that is no far jump), `depth` (nesting
    /// deeper than the HLSL compiler reads).
    ControlFlow(&'static str),
    /// Lane masks: `mask-as-data`, `cross-lane`, `mixed-pair`, ...
    Exec(&'static str),
    /// Resource tracing and placement: `untraced-descriptor`, `dynamic-table-offset`,
    /// `no-mirror`, `class-unknown`, ...
    Resource(&'static str),
    /// Stage rules: `stage-unsupported`, `export-target`, `input-layout`, ...
    Stage(&'static str),
    /// The HLSL compiler rejected the text. Always a translator bug.
    Compile,
}

impl Refusal {
    pub fn new(kind: RefusalKind, detail: impl Into<String>) -> Refusal {
        Refusal {
            stage: None,
            pc: None,
            kind,
            detail: detail.into(),
        }
    }

    pub fn at(mut self, pc: u32) -> Refusal {
        self.pc = Some(pc);
        self
    }

    pub fn in_stage(mut self, stage: Stage) -> Refusal {
        self.stage = Some(stage);
        self
    }

    /// The stable key: `op:v_fma_f32`, `form:v_add_f32:sdwa`, `cfg:depth`, `resource:no-mirror`.
    pub fn key(&self) -> String {
        match &self.kind {
            RefusalKind::Container(what) => format!("container:{what}"),
            RefusalKind::Decode => "decode".to_owned(),
            RefusalKind::Op(op) => format!("op:{}", op.name()),
            RefusalKind::Form(op, what) => format!("form:{}:{what}", op.name()),
            RefusalKind::ControlFlow(what) => format!("cfg:{what}"),
            RefusalKind::Exec(what) => format!("exec:{what}"),
            RefusalKind::Resource(what) => format!("resource:{what}"),
            RefusalKind::Stage(what) => format!("stage:{what}"),
            RefusalKind::Compile => "compile".to_owned(),
        }
    }

    /// The key without the instruction name, for the summary by pass: `op`, `form:sdwa`, `cfg:depth`.
    pub fn class(&self) -> String {
        match &self.kind {
            RefusalKind::Op(op) => format!("op ({:?})", op.info().class),
            RefusalKind::Form(_, what) => format!("form:{what}"),
            _ => self.key(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key())?;
        if let Some(pc) = self.pc {
            write!(f, " at word {pc}")?;
        }
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        Ok(())
    }
}

impl std::error::Error for Refusal {}
