//! The census: counting in parts and merging gives the counts of counting in one piece.

use decant_shader::api::{Stage, StageProgram};
use decant_shader::asm::*;
use decant_shader::census::{Census, Report};
use decant_shader::tables::Op;

fn program(name: &str, stage: Stage, build: impl FnOnce(&mut Asm)) -> StageProgram {
    let mut a = Asm::new();
    build(&mut a);
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        name: name.to_owned(),
        ..StageProgram::new(stage, a.finish())
    }
}

fn corpus() -> Vec<StageProgram> {
    vec![
        program("a", Stage::Pixel, |a| {
            a.push(sop1(Op::s_mov_b32, s(0), int(1)));
        }),
        program("b", Stage::Pixel, |a| {
            a.push(refused());
        }),
        program("c", Stage::Vertex, |a| {
            a.push(refused());
        }),
    ]
}

#[test]
fn merged_parts_equal_the_whole() {
    let programs = corpus();
    let mut whole = Census::default();
    for (i, p) in programs.iter().enumerate() {
        whole.add(i, &p.name, p);
    }
    // The second part is counted first and merged into the first part: the examples must still
    // be the earliest programs.
    let mut first = Census::default();
    first.add(0, "a", &programs[0]);
    let mut second = Census::default();
    second.add(1, "b", &programs[1]);
    second.add(2, "c", &programs[2]);
    let mut merged = Census::default();
    merged.merge(second);
    merged.merge(first);
    assert_eq!(whole.report(40), merged.report(40));
    assert_eq!(whole.summary(40), merged.summary(40));
}

#[test]
fn summary_is_shorter_than_the_report() {
    let mut c = Census::default();
    for (i, p) in corpus().iter().enumerate() {
        c.add(i, &p.name, p);
    }
    assert!(c.summary(40).len() < c.report(40).len());
    assert!(c.summary(40).contains("lifter gaps"));
}

#[test]
fn report_round_trips_and_diffs() {
    let programs = corpus();
    let mut before = Census::default();
    before.add(0, "a", &programs[0]);
    before.add(1, "b", &programs[1]);
    let old = Report::from_json(&before.to_report("t").to_json()).unwrap();
    let mut after = Census::default();
    after.add(0, "a", &programs[0]);
    // The same id with a different program: it was refused before and is translated now.
    after.add(1, "b", &programs[0]);
    let new = after.to_report("t");
    let text = new.diff(&old, 10);
    assert!(text.contains("programs gained"), "{text}");
    assert!(text.contains("  b\n"), "{text}");
    assert!(new.diff(&new, 10).contains("no change"));
}
