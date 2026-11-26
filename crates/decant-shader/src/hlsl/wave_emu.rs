//! A 64 lane console wave on a host whose waves have 32 lanes (`WaveMode::Emulated`).
//!
//! A console compute program that looks at other lanes of a 64 lane wave is printed with
//! `[WaveSize(64)]` (`print`). A GPU whose waves have at most 32 lanes refuses that pipeline:
//! measured on an RTX 4080, `D3D12_FEATURE_D3D12_OPTIONS1.WaveLaneCountMax` 32, the runtime answers
//! `E_INVALIDARG` and the debug layer says "Shader wants wave lane count of 64 but maximum wave lane
//! count for device is 32" (message 622).
//!
//! Here one console wave is 64 threads of the group in group index order, and the operations across
//! lanes go through group shared memory. That the console hardware gives a group's threads to its
//! waves in this order is an assumption that was not measured on a console.
//!
//! * `WaveReadLaneAt(v, lane)`: every thread stores `v`, a group barrier, each thread reads the slot
//!   of `lane` in its own 64 thread wave, a barrier before the slots are written again.
//! * `WaveActiveBallot(b)`: every 32 thread half sets its bits in a word with an atomic OR between
//!   two barriers; a lane reads the two words of its wave (`.x` lanes 0 to 31, `.y` 32 to 63, as the
//!   printer expects of a 64 lane ballot).
//! * `WaveGetLaneIndex()` is the group index modulo 64, `WaveGetLaneCount()` is 64.
//!
//! A group barrier must be reached by every thread of the group, so the form is taken only when
//! every call that synchronises is reached by all of them: such a call stands in the top level of
//! its function (not under an `if`, a loop or a `switch`), not right of `&&`, `||` or `?`, and no
//! `return` or `discard` comes before it. The calls that synchronise are the two intrinsics above
//! and the functions that call them, transitively. The group size must be a whole number of 64
//! thread waves. Anything else is refused with the reason (`stage:wave64-on-32-lane-host`), and the
//! pipeline is then known as refused instead of being handed to a driver that cannot make it.
//! Programs that read lanes inside branches or loops are therefore refused until a fuller form
//! exists.

use std::collections::BTreeSet;

/// Intrinsics that synchronise in the emulated form, and their replacement.
const SYNC: [(&str, &str); 2] = [("WaveReadLaneAt", "decant_emu_read"), ("WaveActiveBallot", "decant_emu_ballot")];
/// Intrinsics that become plain values.
const FREE: [(&str, &str); 2] = [("WaveGetLaneIndex()", "decant_emu_lane()"), ("WaveGetLaneCount()", "64u")];
/// Wave intrinsics the form knows (anything else named `Wave...(` is refused).
const KNOWN: [&str; 4] = ["WaveReadLaneAt", "WaveActiveBallot", "WaveGetLaneIndex", "WaveGetLaneCount"];

/// The text of `text` with its 64 lane wave emulated, or why it cannot be. A text without
/// `[WaveSize(64)]` comes back as it is.
pub fn emulate_wave64(text: &str) -> Result<String, String> {
    if !text.lines().any(|l| l.trim() == "[WaveSize(64)]") {
        return Ok(text.to_owned());
    }
    let threads = group_size(text).ok_or("no [numthreads] line")?;
    if threads == 0 || !threads.is_multiple_of(64) {
        return Err(format!("a group of {threads} threads is not a whole number of 64 lane waves"));
    }
    for name in wave_calls(text) {
        if !KNOWN.contains(&name.as_str()) {
            return Err(format!("{name} has no emulated form"));
        }
    }
    let functions = functions(text);
    let main = functions.iter().find(|f| f.name == "main").ok_or("no main function")?;
    let gi = group_index_name(&main.header).ok_or("main does not take SV_GroupIndex")?;

    // The functions that synchronise: they call a synchronising intrinsic, or such a function.
    let mut syncing: BTreeSet<String> = BTreeSet::new();
    loop {
        let before = syncing.len();
        for f in &functions {
            if f.name != "main" && f.body.iter().any(|(_, line)| !sync_positions(line, &syncing).is_empty()) {
                syncing.insert(f.name.clone());
            }
        }
        if syncing.len() == before {
            break;
        }
    }
    // Only what main reaches runs: a helper the printer wrote but nobody calls does not count.
    let mut reached: BTreeSet<String> = BTreeSet::from(["main".to_owned()]);
    loop {
        let before = reached.len();
        let mut found = Vec::new();
        for f in functions.iter().filter(|f| reached.contains(&f.name)) {
            for g in &functions {
                if f.body.iter().any(|(_, line)| calls(line, &g.name)) {
                    found.push(g.name.clone());
                }
            }
        }
        reached.extend(found);
        if reached.len() == before {
            break;
        }
    }
    for f in functions.iter().filter(|f| reached.contains(&f.name) && (f.name == "main" || syncing.contains(&f.name))) {
        check_uniform(f, &syncing)?;
    }

    // The new text: the attribute goes, the preamble comes before the first function, main
    // notes its group index first.
    let mut out = String::with_capacity(text.len() + 2048);
    let first_function = functions.first().map_or(0, |f| f.first_line);
    for (i, line) in text.lines().enumerate() {
        if i == first_function {
            out.push_str(&preamble(threads));
        }
        if line.trim() == "[WaveSize(64)]" {
            continue;
        }
        let mut line = line.to_owned();
        for (from, to) in SYNC {
            line = replace_call(&line, from, to);
        }
        for (from, to) in FREE {
            line = line.replace(from, to);
        }
        out.push_str(&line);
        out.push('\n');
        if i == main.header_line {
            out.push_str(&format!("    decant_emu_gi = {gi};\n"));
        }
    }
    Ok(out)
}

/// The total thread count of the `[numthreads(x, y, z)]` line.
fn group_size(text: &str) -> Option<u32> {
    let line = text.lines().map(str::trim).find(|l| l.starts_with("[numthreads("))?;
    let inner = line.strip_prefix("[numthreads(")?.strip_suffix(")]")?;
    inner.split(',').map(|n| n.trim().parse::<u32>().ok()).try_fold(1u32, |acc, n| acc.checked_mul(n?))
}

/// Every distinct `Wave...(` call name of the text (the `[WaveSize(` attribute is not a call).
fn wave_calls(text: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for (at, _) in text.match_indices("Wave") {
        if at > 0 && is_ident(text.as_bytes()[at - 1]) {
            continue;
        }
        let rest = &text[at..];
        let len = rest.bytes().take_while(|b| is_ident(*b)).count();
        let name = &rest[..len];
        if rest[len..].starts_with('(') && name != "WaveSize" && !(at > 0 && text.as_bytes()[at - 1] == b'[') {
            names.insert(name.to_owned());
        }
    }
    names
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// A function of the text: its name, the index of its first line and of the line with its `{`,
/// that line, and its body lines with the brace depth before each (1 is the top level of the body).
struct Function {
    name: String,
    first_line: usize,
    header_line: usize,
    header: String,
    body: Vec<(u32, String)>,
}

/// The functions of a printed text. A function starts at depth 0 on a line that has a name, `(`
/// and ends with `{` (or holds the whole function, `{ ... }`, on one line).
fn functions(text: &str) -> Vec<Function> {
    let mut out: Vec<Function> = Vec::new();
    let mut depth: i64 = 0;
    let mut open: Option<Function> = None;
    let mut attributes_from: Option<usize> = None;
    for (i, line) in text.lines().enumerate() {
        let opens = line.matches('{').count() as i64;
        let closes = line.matches('}').count() as i64;
        let trimmed = line.trim_start();
        if depth == 0 && open.is_none() {
            if trimmed.starts_with('[') {
                attributes_from.get_or_insert(i);
            } else if let Some(name) = function_name(trimmed).filter(|_| opens > 0) {
                let f = Function { name, first_line: attributes_from.unwrap_or(i), header_line: i, header: line.to_owned(), body: Vec::new() };
                attributes_from = None;
                if opens == closes {
                    // the whole function on one line: its body is that line, at the top level
                    let mut f = f;
                    f.body.push((1, line.to_owned()));
                    out.push(f);
                } else {
                    open = Some(f);
                }
                depth += opens - closes;
                continue;
            } else if !trimmed.is_empty() && !trimmed.starts_with("//") {
                attributes_from = None;
            }
        }
        if let Some(f) = open.as_mut() {
            f.body.push((depth.max(0) as u32, line.to_owned()));
        }
        depth += opens - closes;
        if depth <= 0 {
            depth = 0;
            if let Some(f) = open.take() {
                out.push(f);
            }
        }
    }
    out
}

/// The name of a function whose definition starts `line` (`uint name(`, `void main(`), if it does.
fn function_name(line: &str) -> Option<String> {
    let paren = line.find('(')?;
    let head = &line[..paren];
    let mut words = head.split_whitespace();
    let ty = words.next()?;
    let name = words.next()?;
    if words.next().is_some() || matches!(ty, "cbuffer" | "struct" | "if" | "for" | "while" | "return" | "static" | "groupshared") {
        return None;
    }
    (name.bytes().all(is_ident) && ty.bytes().all(|b| is_ident(b) || b == b'<' || b == b'>')).then(|| name.to_owned())
}

/// The name of main's `SV_GroupIndex` parameter.
fn group_index_name(header: &str) -> Option<String> {
    let at = header.find(": SV_GroupIndex")?;
    let before = header[..at].trim_end();
    let start = before.rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(0, |p| p + 1);
    Some(before[start..].to_owned())
}

/// The byte positions in `line` of calls that synchronise: the intrinsics of [`SYNC`] and the
/// functions of `syncing`.
fn sync_positions(line: &str, syncing: &BTreeSet<String>) -> Vec<usize> {
    let mut at = Vec::new();
    let names = SYNC.iter().map(|(n, _)| (*n).to_owned()).chain(syncing.iter().cloned());
    for name in names {
        for (p, _) in line.match_indices(&format!("{name}(")) {
            if p == 0 || !is_ident(line.as_bytes()[p - 1]) {
                at.push(p);
            }
        }
    }
    at
}

/// Whether every synchronising call of `f` is reached by every thread that enters `f`.
fn check_uniform(f: &Function, syncing: &BTreeSet<String>) -> Result<(), String> {
    let mut exit_before = false;
    for (depth, line) in &f.body {
        let positions = sync_positions(line, syncing);
        let defines = function_name(line.trim_start()).is_some_and(|n| n == f.name);
        // the definition line of a one line function names the function itself
        let positions: Vec<usize> = if defines {
            let skip = line.find('{').unwrap_or(0);
            positions.into_iter().filter(|p| *p > skip).collect()
        } else {
            positions
        };
        if !positions.is_empty() {
            let trimmed = line.trim_start();
            if *depth != 1 {
                return Err(format!("{} reads other lanes inside a branch or a loop: {}", f.name, trimmed.trim()));
            }
            let body = if defines { &line[line.find('{').map_or(0, |p| p + 1)..] } else { line.as_str() };
            let body_trimmed = body.trim_start();
            if ["if", "else", "for", "while", "do", "switch", "case"].iter().any(|k| starts_with_word(body_trimmed, k)) {
                return Err(format!("{} reads other lanes in a condition: {}", f.name, trimmed.trim()));
            }
            if defines && body.matches('{').count() > 0 {
                return Err(format!("{} reads other lanes inside a block", f.name));
            }
            let first = *positions.iter().min().unwrap_or(&0);
            if ["&&", "||", "?"].iter().any(|op| line[..first].contains(op)) {
                return Err(format!("{} reads other lanes right of &&, || or ?: {}", f.name, trimmed.trim()));
            }
            if exit_before {
                return Err(format!("{} reads other lanes after a return or discard", f.name));
            }
        }
        if contains_word(line, "return") && positions.is_empty() || contains_word(line, "discard") {
            exit_before = true;
        }
    }
    Ok(())
}

/// Whether `line` calls the function `name` (a whole identifier followed by `(`), outside the
/// definition of `name` itself.
fn calls(line: &str, name: &str) -> bool {
    let body = if function_name(line.trim_start()).is_some_and(|n| n == name) { &line[line.find('{').map_or(line.len(), |p| p + 1)..] } else { line };
    body.match_indices(&format!("{name}(")).any(|(p, _)| p == 0 || !is_ident(body.as_bytes()[p - 1]))
}

fn starts_with_word(text: &str, word: &str) -> bool {
    text.starts_with(word) && text[word.len()..].bytes().next().is_none_or(|b| !is_ident(b))
}

fn contains_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(p, _)| {
        (p == 0 || !is_ident(text.as_bytes()[p - 1])) && text[p + word.len()..].bytes().next().is_none_or(|b| !is_ident(b))
    })
}

/// `line` with every call of `from` (a whole identifier followed by `(`) renamed to `to`.
fn replace_call(line: &str, from: &str, to: &str) -> String {
    let pattern = format!("{from}(");
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(p) = rest.find(&pattern) {
        let whole = p == 0 || !is_ident(rest.as_bytes()[p - 1]);
        out.push_str(&rest[..p]);
        out.push_str(if whole { to } else { from });
        out.push('(');
        rest = &rest[p + pattern.len()..];
    }
    out.push_str(rest);
    out
}

/// The emulation's declarations for a group of `threads` threads.
fn preamble(threads: u32) -> String {
    let words = threads / 32;
    format!(
        "// A 64 lane console wave on 32 lane host waves: 64 threads of the group in group index order, lanes exchanged through group shared memory (wave_emu).
static uint decant_emu_gi;
groupshared uint decant_emu_x[{threads}];
groupshared uint decant_emu_m[{words}];
uint decant_emu_lane() {{ return decant_emu_gi & 63u; }}
uint decant_emu_read(uint v, uint lane) {{
    decant_emu_x[decant_emu_gi] = v;
    GroupMemoryBarrierWithGroupSync();
    uint r = decant_emu_x[(decant_emu_gi & ~63u) | (lane & 63u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}}
int decant_emu_read(int v, uint lane) {{ return int(decant_emu_read(uint(v), lane)); }}
float decant_emu_read(float v, uint lane) {{ return asfloat(decant_emu_read(asuint(v), lane)); }}
bool decant_emu_read(bool v, uint lane) {{ return decant_emu_read(v ? 1u : 0u, lane) != 0u; }}
uint4 decant_emu_ballot(bool b) {{
    uint w = decant_emu_gi >> 5u;
    if ((decant_emu_gi & 31u) == 0u) {{ decant_emu_m[w] = 0u; }}
    GroupMemoryBarrierWithGroupSync();
    if (b) {{ InterlockedOr(decant_emu_m[w], 1u << (decant_emu_gi & 31u)); }}
    GroupMemoryBarrierWithGroupSync();
    uint first = (decant_emu_gi & ~63u) >> 5u;
    uint4 r = uint4(decant_emu_m[first], decant_emu_m[first + 1u], 0u, 0u);
    GroupMemoryBarrierWithGroupSync();
    return r;
}}
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOP: &str = "uint decant_read_lane(uint value, uint lane) { return WaveReadLaneAt(value, lane); }
[WaveSize(64)]
[numthreads(8, 8, 1)]
void main(uint3 decant_tid : SV_GroupThreadID, uint decant_gi : SV_GroupIndex) {
    uint v8 = decant_gi * 3u;
    uint s32 = decant_read_lane(v8, 32u);
    uint4 b = WaveActiveBallot(v8 > 4u);
    uint n = WaveGetLaneCount();
}
";

    #[test]
    fn a_top_level_lane_read_of_a_64_thread_group_is_emulated() {
        let out = emulate_wave64(TOP).expect("emulated");
        assert!(!out.contains("[WaveSize(64)]"), "{out}");
        assert!(out.contains("[numthreads(8, 8, 1)]"), "{out}");
        assert!(out.contains("return decant_emu_read(value, lane);"), "{out}");
        assert!(out.contains("uint4 b = decant_emu_ballot(v8 > 4u);"), "{out}");
        assert!(out.contains("uint n = 64u;"), "{out}");
        assert!(out.contains("    decant_emu_gi = decant_gi;\n"), "{out}");
        assert!(out.contains("groupshared uint decant_emu_x[64];"), "{out}");
        // the preamble comes before the first function, so every helper is declared before use
        assert!(out.find("uint decant_emu_read(uint v").unwrap() < out.find("uint decant_read_lane(").unwrap(), "{out}");
    }

    #[test]
    fn a_text_without_wave_size_64_is_kept() {
        let text = TOP.replace("[WaveSize(64)]", "[WaveSize(32)]");
        assert_eq!(emulate_wave64(&text).unwrap(), text);
    }

    #[test]
    fn a_lane_read_under_a_branch_is_refused() {
        let text = TOP.replace("    uint s32 = decant_read_lane(v8, 32u);", "    if (v8 > 2u) {\n        v8 = decant_read_lane(v8, 32u);\n    }");
        let why = emulate_wave64(&text).unwrap_err();
        assert!(why.contains("inside a branch or a loop"), "{why}");
    }

    #[test]
    fn a_lane_read_right_of_a_short_circuit_is_refused() {
        let text = TOP.replace("uint s32 = decant_read_lane(v8, 32u);", "bool s32 = v8 > 1u && decant_read_lane(v8, 32u) > 2u;");
        let why = emulate_wave64(&text).unwrap_err();
        assert!(why.contains("right of"), "{why}");
    }

    #[test]
    fn a_lane_read_after_an_early_return_is_refused() {
        let text = TOP.replace("    uint s32 = decant_read_lane(v8, 32u);", "    if (v8 == 0u) return;\n    uint s32 = decant_read_lane(v8, 32u);");
        let why = emulate_wave64(&text).unwrap_err();
        assert!(why.contains("after a return"), "{why}");
    }

    #[test]
    fn a_group_that_is_no_whole_number_of_waves_is_refused() {
        let text = TOP.replace("[numthreads(8, 8, 1)]", "[numthreads(8, 4, 1)]");
        assert!(emulate_wave64(&text).unwrap_err().contains("32 threads"));
    }

    #[test]
    fn an_unknown_wave_intrinsic_is_refused() {
        let text = TOP.replace("uint n = WaveGetLaneCount();", "uint n = WaveActiveSum(v8);");
        assert!(emulate_wave64(&text).unwrap_err().contains("WaveActiveSum"));
    }

    #[test]
    fn a_helper_nobody_calls_does_not_count() {
        let text = TOP.replace(
            "[WaveSize(64)]",
            "uint decant_unused(uint value, uint lane) {\n    if (lane > 3u) {\n        return WaveReadLaneAt(value, lane);\n    }\n    return value;\n}\n[WaveSize(64)]",
        );
        assert!(emulate_wave64(&text).is_ok());
    }

    #[test]
    fn a_helper_that_reads_lanes_inside_a_branch_is_refused() {
        let text = TOP.replace(
            "uint decant_read_lane(uint value, uint lane) { return WaveReadLaneAt(value, lane); }",
            "uint decant_read_lane(uint value, uint lane) {\n    if (lane > 3u) {\n        return WaveReadLaneAt(value, lane);\n    }\n    return value;\n}",
        );
        assert!(emulate_wave64(&text).unwrap_err().contains("decant_read_lane"));
    }
}
