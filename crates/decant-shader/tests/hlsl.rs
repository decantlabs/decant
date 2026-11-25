//! The HLSL printer: golden text of small programs, and the corpus measurement.
//!
//! `corpus_compile` is ignored by default. Run it on the title's archive:
//! `DECANT_SHADER_CORPUS=D:\title\shaders.pack cargo test --release -p decant-shader --test hlsl -- --ignored --nocapture`
//! It translates every program, picks `DECANT_SHADER_SAMPLE` (default 1000) translated ones with a fixed
//! stride, compiles them with DXC and prints the output size, the compile time and the count of
//! `compile` refusals. `DECANT_SHADER_DXC_ARGS` adds extra DXC arguments.

mod common;

use std::path::Path;
use std::time::{Duration, Instant};

use decant_shader::api::{Compiler, CountingHost, Options, RootSignature};
use decant_shader::corpus;

#[test]
#[ignore = "needs the game archive and dxc.exe"]
fn corpus_compile() {
    let Some(path) = std::env::var_os("DECANT_SHADER_CORPUS") else {
        eprintln!("DECANT_SHADER_CORPUS not set, skipped");
        return;
    };
    let Some(dxc) = common::find_dxc() else {
        eprintln!("dxc.exe not found, skipped");
        return;
    };
    let want: usize = std::env::var("DECANT_SHADER_SAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    type Text = (usize, String, String, Vec<String>);
    let visit = |acc: &mut (Vec<Text>, usize), e: corpus::Entry| {
        acc.1 += 1;
        let Ok(p) = &e.program else { return };
        if let Ok(t) = decant_shader::translate(
            std::slice::from_ref(p),
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
        ) {
            let s = &t.stages[0];
            acc.0.push((
                e.index,
                s.hlsl.clone(),
                s.profile.clone(),
                s.dxc_args.clone(),
            ));
        }
    };
    let (_, parts) = corpus::par_corpus(
        Path::new(&path),
        &corpus::Selection::default(),
        0,
        || (Vec::new(), 0usize),
        visit,
    )
    .expect("corpus");
    let mut texts: Vec<Text> = Vec::new();
    let mut total = 0usize;
    for (part, seen) in parts {
        texts.extend(part);
        total += seen;
    }
    // The threads give their programs back in no order: the sample must not depend on it.
    texts.sort_by_key(|t| t.0);
    let stride = (texts.len() / want).max(1);
    let sample: Vec<_> = texts.iter().step_by(stride).take(want).collect();
    let extra: Vec<String> = std::env::var("DECANT_SHADER_DXC_ARGS")
        .map(|v| v.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    if let Some(dir) = std::env::var_os("DECANT_SHADER_DUMP") {
        for (index, text, ..) in &sample {
            std::fs::write(Path::new(&dir).join(format!("{index}.hlsl")), text).expect("dump");
        }
    }
    let (mut bytes, mut lines, mut dxil) = (0usize, 0usize, 0usize);
    let (mut spent, mut failed, mut slowest) = (Duration::ZERO, 0usize, (Duration::ZERO, 0usize));
    for (index, text, profile, args) in &sample {
        let mut args = args.clone();
        args.extend(extra.iter().cloned());
        bytes += text.len();
        lines += text.lines().count();
        let t = Instant::now();
        let r = dxc.compile(text, "main", profile, &args);
        let d = t.elapsed();
        spent += d;
        if d > slowest.0 {
            slowest = (d, *index);
        }
        match r {
            Ok(o) => dxil += o.len(),
            Err(m) => {
                failed += 1;
                if failed <= 100 {
                    eprintln!(
                        "compile refusal at {index}: {}",
                        m.lines().take(2).collect::<Vec<_>>().join(" | ")
                    );
                }
            }
        }
    }
    let n = sample.len().max(1);
    println!(
        "programs {total}, translated {}, sampled {}",
        texts.len(),
        sample.len()
    );
    println!(
        "hlsl bytes {bytes} ({} per program), lines {lines}",
        bytes / n
    );
    println!("dxil bytes {dxil}");
    println!(
        "dxc time {:.1} s ({:.1} ms per program), slowest {:.1} ms at {}",
        spent.as_secs_f64(),
        spent.as_secs_f64() * 1000.0 / n as f64,
        slowest.0.as_secs_f64() * 1000.0,
        slowest.1
    );
    println!("compile refusals {failed}");
}

#[test]
fn ir_dump_golden() {
    use decant_shader::ir::{Bin, Expr, Function, Guard, Node, Stmt, Var};
    let set = |dst, value, guard| Stmt::Set { dst, value, guard };
    let f = Function {
        body: vec![
            Node::Block(vec![set(
                Var::S(4),
                Expr::bin(Bin::Add, Expr::var(Var::S(2)), Expr::U32(0x10)),
                Guard::Always,
            )]),
            Node::If {
                cond: Expr::var(Var::Scc),
                then: vec![Node::Block(vec![set(
                    Var::V(0),
                    Expr::select(Expr::var(Var::Vcc), Expr::U32(1), Expr::var(Var::S(4))),
                    Guard::Exec,
                )])],
                els: vec![Node::Break],
            },
            Node::Return,
        ],
        resources: Vec::new(),
        groupshared_dwords: 0,
        lanes: 64,
    };
    assert_eq!(
        f.to_string(),
        "s4 = (s2 + 0x10)\nif scc {\n  @exec v0 = (vcc ? 1 : s4)\n} else {\n  break\n}\nreturn\n"
    );
}

/// Prints a function with no inputs and outputs and returns the body of `main`.
fn print_body(f: &decant_shader::ir::Function) -> String {
    use decant_shader::hlsl::{self, Context};
    use decant_shader::stage::Interface;
    let iface = Interface::default();
    let text = hlsl::print(
        f,
        &Context {
            iface: &iface,
            mirror: None,
            header: &[],
            code: &[],
        },
    )
    .expect("print");
    if let Some(dxc) = common::find_dxc() {
        dxc.compile(&text, "main", "ps_6_6", &["-Gis".to_owned()])
            .unwrap_or_else(|m| panic!("dxc: {m}\n{text}"));
    }
    text
}

#[test]
fn wide_values_print_with_wide_operators() {
    use decant_shader::ir::{Bin, Expr, Function, Guard, Node, Stmt, Un, Var};
    let wide = |v| Expr::un(Un::U32ToU64, Expr::var(v));
    let set = |dst, value| Stmt::Set {
        dst,
        value,
        guard: Guard::Always,
    };
    let f = Function {
        body: vec![Node::Block(vec![
            set(
                Var::S(0),
                Expr::un(
                    Un::U64ToU32,
                    Expr::bin(Bin::Shl, wide(Var::S(2)), Expr::var(Var::S(4))),
                ),
            ),
            set(
                Var::S(1),
                Expr::un(
                    Un::U64ToU32,
                    Expr::bin(Bin::AShr, wide(Var::S(2)), Expr::U32(40)),
                ),
            ),
            set(Var::Scc, Expr::bin(Bin::SLt, wide(Var::S(2)), Expr::U64(5))),
        ])],
        resources: Vec::new(),
        groupshared_dwords: 0,
        lanes: 64,
    };
    let text = print_body(&f);
    assert!(text.contains("<< (s4 & 63u)"), "{text}");
    assert!(
        text.contains("int64_t(uint64_t(s2)) >> (40u & 63u)"),
        "{text}"
    );
    assert!(
        text.contains("int64_t(uint64_t(s2)) < int64_t(5ull)"),
        "{text}"
    );
}
