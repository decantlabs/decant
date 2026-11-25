//! `aco_opcode_crosscheck`: compares the opcode tables of the translator's GFX10 decoder (the
//! `*_OPCODE_LIST[]` tables of the translator's source checkout) with the GFX10 numbering
//! of the ACO compiler of Mesa.
//!
//! The ACO numbers come from the `aco_opcodes.cpp` that Mesa generates during its build from
//! `aco_opcodes.py`. That file lists every instruction, so nothing has to be executed: the arrays
//! of `instr_info` are indexed by the same position (the instructions sorted by name) and hold the
//! GFX10 number, the name and the format. Mesa is only read at run time, nothing of it is copied.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::path::Path;

use regex::Regex;
use decant_shader_analysis::NL;

/// The decoder tables and the format of ACO that each one is compared with.
pub const LISTS: [(&str, &str); 18] = [
    ("VOP2_OPCODE_LIST", "VOP2"),
    ("VOP1_OPCODE_LIST", "VOP1"),
    ("VOPC_OPCODE_LIST", "VOPC"),
    ("VOP3_OPCODE_LIST", "VOP3"),
    ("VOP3P_OPCODE_LIST", "VOP3P"),
    ("SOP2_OPCODE_LIST", "SOP2"),
    ("SOP1_OPCODE_LIST", "SOP1"),
    ("SOPC_OPCODE_LIST", "SOPC"),
    ("SOPK_OPCODE_LIST", "SOPK"),
    ("SOPP_OPCODE_LIST", "SOPP"),
    ("SMEM_OPCODE_LIST", "SMEM"),
    ("MUBUF_OPCODE_LIST", "MUBUF"),
    ("MTBUF_OPCODE_LIST", "MTBUF"),
    ("FLAT_OPCODE_LIST", "FLAT"),
    ("DS_OPCODE_LIST", "DS"),
    ("MIMG_SAMPLE_OPCODE_LIST", "MIMG"),
    ("MIMG_GATHER_OPCODE_LIST", "MIMG"),
    ("MIMG_ATOMIC_OPCODE_LIST", "MIMG"),
];

/// GFX10 names of ACO for instructions that the decoder names otherwise.
pub const RENAMES: [(&str, &str); 11] = [
    ("v_addc_u32", "v_addc_co_u32"),
    ("v_sub_co_ci_u32", "v_subb_co_u32"),
    ("v_subrev_co_ci_u32", "v_subbrev_co_u32"),
    ("v_add_i32", "v_add_co_u32"),
    ("v_sub_i32", "v_sub_co_u32"),
    ("v_subrev_i32", "v_subrev_co_u32"),
    ("s_add_i32", "s_addk_i32"),
    ("s_endpgm", "s_endpgm_ordered_ps_done"),
    ("v_fma_f32", "v_fma_mix_f32"),
    ("v_mad_mixlo_f16", "v_fma_mixlo_f16"),
    ("v_mad_mixhi_f16", "v_fma_mixhi_f16"),
];

/// The instructions of ACO that have a GFX10 number: format, then number, then the names.
pub type Maps = HashMap<String, BTreeMap<i64, Vec<String>>>;

/// The verdict on one entry of a decoder table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Identique,
    Alias,
    Absent,
    Ecart,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Identique => "identique",
            Verdict::Alias => "alias",
            Verdict::Absent => "absent",
            Verdict::Ecart => "ECART",
        }
    }
}

/// The text of `text` between the braces that follow `instr_info = `, split into the groups at
/// the first level, comments removed. A group is the text between its braces.
fn groups(text: &str) -> Result<Vec<String>, String> {
    let start = text.find("instr_info").ok_or("aco_opcodes.cpp: pas de instr_info")?;
    let rest = &text[start..];
    let open = rest.find('{').ok_or("aco_opcodes.cpp: pas d'accolade apres instr_info")?;
    let (mut depth, mut in_string, mut in_comment) = (0usize, false, false);
    let mut chars = rest[open + 1..].chars().peekable();
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    while let Some(c) = chars.next() {
        if in_comment {
            if c == '\n' {
                in_comment = false;
                current.push(c);
            }
            continue;
        }
        if in_string {
            if depth >= 1 {
                current.push(c);
            }
            in_string = c != '"';
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                if depth >= 1 {
                    current.push(c);
                }
            }
            '/' if chars.peek() == Some(&'/') => in_comment = true,
            '{' => {
                depth += 1;
                if depth > 1 {
                    current.push(c);
                }
            }
            '}' if depth == 0 => return Ok(out),
            '}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(std::mem::take(&mut current));
                } else {
                    current.push(c);
                }
            }
            _ if depth >= 1 => current.push(c),
            _ => {}
        }
    }
    Err("aco_opcodes.cpp: instr_info n'est pas ferme".to_owned())
}

/// The items of a flat group, separated by commas.
fn items(group: &str) -> Vec<&str> {
    group.split(',').map(str::trim).filter(|s| !s.is_empty()).collect()
}

/// Reads the instructions of a generated `aco_opcodes.cpp`: the third numeric array is GFX10, the
/// first array of quoted strings holds the names and the array after it the formats. The numbers
/// of the other generations are ignored, as are instructions without a number (missing or
/// negative). `GLOBAL` and `SCRATCH` count as `FLAT`, and the compares, VOP2 and VOP1 are also
/// present in VOP3 at offsets 0x000, 0x100 and 0x180.
pub fn read_maps(generated: &str) -> Result<Maps, String> {
    let groups = groups(generated)?;
    let gfx10 = groups.get(2).ok_or("aco_opcodes.cpp: le tableau GFX10 manque")?;
    let names_at = groups.iter().position(|g| items(g).first().is_some_and(|s| s.starts_with('"'))).ok_or("aco_opcodes.cpp: le tableau des noms manque")?;
    let formats = groups.get(names_at + 1).ok_or("aco_opcodes.cpp: le tableau des formats manque")?;
    let (numbers, names, formats) = (items(gfx10), items(&groups[names_at]), items(formats));
    if numbers.len() != names.len() || formats.len() != names.len() {
        return Err(format!("aco_opcodes.cpp: {} numeros GFX10, {} noms et {} formats", numbers.len(), names.len(), formats.len()));
    }
    let mut maps = Maps::new();
    for ((number, name), format) in numbers.iter().zip(&names).zip(&formats) {
        let number: i64 = number.parse().map_err(|_| format!("aco_opcodes.cpp: numero GFX10 illisible {number:?}"))?;
        if number < 0 {
            continue;
        }
        let format = format.rsplit("::").next().unwrap_or(format);
        let format = if format == "GLOBAL" || format == "SCRATCH" { "FLAT" } else { format };
        maps.entry(format.to_owned()).or_default().entry(number).or_default().push(name.trim_matches('"').to_owned());
    }
    let mut vop3 = maps.remove("VOP3").unwrap_or_default();
    for (format, base) in [("VOPC", 0x000), ("VOP2", 0x100), ("VOP1", 0x180)] {
        for (number, names) in maps.get(format).into_iter().flatten() {
            vop3.entry(base + number).or_default().extend(names.iter().cloned());
        }
    }
    maps.insert("VOP3".to_owned(), vop3);
    Ok(maps)
}

/// The other names under which ACO may know the instruction `name` for GFX10.
fn aliases(name: &str) -> Vec<String> {
    let renamed = RENAMES.iter().find(|(from, _)| *from == name).map_or(name, |(_, to)| to);
    let ne = Regex::new(r"_ne_(i|u)").expect("static pattern");
    let t = Regex::new(r"_t_(i|u)").expect("static pattern");
    vec![
        renamed.to_owned(),
        name.replace("_nc_", "_"),
        ne.replace_all(name, "_lg_${1}").into_owned(),
        t.replace_all(name, "_tru_${1}").into_owned(),
        name.replace("s_cmp_", "s_cmpk_"),
        name.replace("flat_", "global_"),
    ]
}

/// `identique`, `alias` or nothing (the caller then reports an `ECART`).
fn same(ours: &str, theirs: &[String]) -> Option<Verdict> {
    let norm: Vec<&str> = theirs.iter().map(|t| t.strip_suffix("_e64").unwrap_or(t)).collect();
    if ours == "s_waitcnt" && theirs.iter().any(|t| t.starts_with("s_waitcnt_")) {
        return Some(Verdict::Alias);
    }
    if norm.contains(&ours) {
        return Some(if theirs.iter().any(|t| t == ours) { Verdict::Identique } else { Verdict::Alias });
    }
    aliases(ours).iter().any(|a| norm.contains(&a.as_str())).then_some(Verdict::Alias)
}

/// The `{number, Opcode::NAME` entries of the table `table` of one decoder source.
pub fn entries(source: &str, table: &str) -> Vec<(u64, String)> {
    let list = Regex::new(&format!(r"(?s){table}\[\]\s*=\s*\{{(.*?)\n\}};")).expect("static pattern");
    let entry = Regex::new(r"\{\s*0x([0-9a-fA-F]+)u?\s*,\s*Opcode::([A-Z0-9_]+)").expect("static pattern");
    let Some(body) = list.captures(source) else {
        return Vec::new();
    };
    entry.captures_iter(&body[1]).filter_map(|c| Some((u64::from_str_radix(&c[1], 16).ok()?, c[2].to_lowercase()))).collect()
}

/// The count of each verdict.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Counts {
    pub identique: usize,
    pub alias: usize,
    pub absent: usize,
    pub ecart: usize,
}

/// Compares the tables of the decoder sources with `maps`: the lines of the verdicts that need a
/// look (`ECART`, `absent`, `alias`) and the counts.
pub fn compare(sources: &[String], maps: &Maps) -> (String, Counts) {
    let mut out = String::new();
    let mut counts = Counts::default();
    for text in sources {
        for (table, format) in LISTS {
            for (number, ours) in entries(text, table) {
                let theirs = i64::try_from(number).ok().and_then(|n| maps.get(format)?.get(&n));
                let verdict = match theirs {
                    None => Verdict::Absent,
                    Some(t) if t.is_empty() => Verdict::Absent,
                    Some(t) => same(&ours, t).unwrap_or(Verdict::Ecart),
                };
                match verdict {
                    Verdict::Identique => counts.identique += 1,
                    Verdict::Alias => counts.alias += 1,
                    Verdict::Absent => counts.absent += 1,
                    Verdict::Ecart => counts.ecart += 1,
                }
                if verdict != Verdict::Identique {
                    let names = theirs.map_or_else(|| "-".to_owned(), |t| t.join(", "));
                    write!(out, "{:9} {format:6} 0x{number:03x} {ours:30} ACO GFX10 : {names}{NL}", verdict.name()).unwrap();
                }
            }
        }
    }
    (out, counts)
}

/// The whole tool: the report to print (lines and the final count) and whether an `ECART` exists.
pub fn run(decode: &Path, generated: &Path) -> Result<(String, bool), String> {
    let generated = std::fs::read(generated).map_err(|e| format!("{}: {e}", generated.display()))?;
    let maps = read_maps(&String::from_utf8_lossy(&generated))?;
    let mut files: Vec<_> = std::fs::read_dir(decode)
        .map_err(|e| format!("{}: {e}", decode.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "cpp"))
        .collect();
    files.sort();
    let mut sources = Vec::new();
    for file in &files {
        let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
        sources.push(String::from_utf8_lossy(&bytes).into_owned());
    }
    let (mut text, c) = compare(&sources, &maps);
    write!(text, "identique={} alias={} absent={} ECART={}{NL}", c.identique, c.alias, c.absent, c.ecart).unwrap();
    Ok((text, c.ecart > 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An invented generated file: five instructions sorted by name, six numeric arrays, the
    /// atomic bits, the names, the formats and a class array. Nothing of Mesa is in it.
    fn generated() -> String {
        let numbers = |a: [i32; 5]| a.iter().map(|n| format!("      {n},\n")).collect::<String>();
        let names = ["s_demo_nop", "v_cmp_demo_i32", "v_demo_add", "v_demo_cvt", "v_demo_mov"];
        let formats = ["SOPP", "VOPC", "VOP2", "VOP1", "VOP1"];
        let mut s = String::from("namespace aco {\n\nextern const aco::Info instr_info = {\n");
        for a in [[1, -1, 3, 4, 5], [1, 2, 3, 4, 5], [7, 0x0a, 0x03, -1, 0x01], [9; 5], [9; 5], [9; 5]] {
            s += &format!("   {{\n{}   }},\n", numbers(a));
        }
        s += "   std::bitset<5>(\"00100\"),\n   {\n";
        for n in names {
            s += &format!("      \"{n}\",\n");
        }
        s += "   },\n   {\n";
        for f in formats {
            s += &format!("      aco::Format::{f},\n");
        }
        s += "   },\n   {\n      instr_class::alu,\n   },\n   {\n      { // s_demo_nop\n         0,\n         {\n         },\n      },\n   },\n};\n}\n";
        s
    }

    fn names(maps: &Maps, format: &str, number: i64) -> Vec<String> {
        maps.get(format).and_then(|m| m.get(&number)).cloned().unwrap_or_default()
    }

    #[test]
    fn reads_the_gfx10_array_with_name_and_format() {
        let maps = read_maps(&generated()).unwrap();
        assert_eq!(names(&maps, "SOPP", 7), ["s_demo_nop"]);
        assert_eq!(names(&maps, "VOPC", 0x0a), ["v_cmp_demo_i32"]);
        assert_eq!(names(&maps, "VOP2", 3), ["v_demo_add"]);
        assert_eq!(names(&maps, "VOP1", 1), ["v_demo_mov"]);
    }

    #[test]
    fn skips_negative_numbers_and_mirrors_into_vop3() {
        let maps = read_maps(&generated()).unwrap();
        assert!(!maps["VOP1"].contains_key(&-1));
        assert_eq!(maps["VOP1"].len(), 1);
        assert_eq!(names(&maps, "VOP3", 0x00a), ["v_cmp_demo_i32"]);
        assert_eq!(names(&maps, "VOP3", 0x103), ["v_demo_add"]);
        assert_eq!(names(&maps, "VOP3", 0x181), ["v_demo_mov"]);
    }

    #[test]
    fn global_and_scratch_count_as_flat() {
        let text = generated().replace("aco::Format::SOPP", "aco::Format::GLOBAL").replace("aco::Format::VOP1,\n      aco::Format::VOP1", "aco::Format::SCRATCH,\n      aco::Format::SCRATCH");
        let maps = read_maps(&text).unwrap();
        assert_eq!(names(&maps, "FLAT", 7), ["s_demo_nop"]);
        assert_eq!(names(&maps, "FLAT", 1), ["v_demo_mov"]);
        assert!(!maps.contains_key("GLOBAL") && !maps.contains_key("SCRATCH"));
    }

    #[test]
    fn rejects_files_that_are_not_readable() {
        assert!(read_maps("int x;").is_err());
        assert!(read_maps(&generated().replace("      \"v_demo_mov\",
", "")).unwrap_err().contains("noms"));
        assert!(read_maps(&generated().replacen("      10,
", "      None,
", 1)).unwrap_err().contains("illisible"));
    }

    #[test]
    fn crlf_files_read_the_same() {
        let text = generated().replace('\n', "\r\n");
        assert_eq!(read_maps(&text).unwrap(), read_maps(&generated()).unwrap());
    }

    fn decode() -> String {
        "static const Entry VOP2_OPCODE_LIST[] = {\n    {0x003u, Opcode::V_DEMO_ADD},\n    {0x004, Opcode::V_DEMO_MISSING},\n    {0x003, Opcode::V_DEMO_OTHER},\n};\n\
         static const Entry VOPC_OPCODE_LIST[] = {\n    { 0x0a, Opcode::V_CMP_DEMO_I32 },\n};\n"
            .to_owned()
    }

    #[test]
    fn finds_the_entries_of_a_table() {
        assert_eq!(entries(&decode(), "VOP2_OPCODE_LIST"), [(3, "v_demo_add".to_owned()), (4, "v_demo_missing".to_owned()), (3, "v_demo_other".to_owned())]);
        assert_eq!(entries(&decode(), "VOPC_OPCODE_LIST"), [(0x0a, "v_cmp_demo_i32".to_owned())]);
        assert!(entries(&decode(), "DS_OPCODE_LIST").is_empty());
    }

    #[test]
    fn gives_each_verdict() {
        let maps = read_maps(&generated()).unwrap();
        let (text, c) = compare(&[decode()], &maps);
        assert_eq!(c, Counts { identique: 2, alias: 0, absent: 1, ecart: 1 });
        let line = |verdict: &str, number: &str, name: &str, theirs: &str| format!("{verdict:<9} VOP2   {number} {name:<30} ACO GFX10 : {theirs}{NL}");
        assert_eq!(text, line("absent", "0x004", "v_demo_missing", "-") + &line("ECART", "0x003", "v_demo_other", "v_demo_add"));
    }

    #[test]
    fn aliases_follow_the_rules() {
        let theirs = |n: &str| vec![n.to_owned()];
        assert_eq!(same("v_demo", &theirs("v_demo")), Some(Verdict::Identique));
        assert_eq!(same("v_demo", &theirs("v_demo_e64")), Some(Verdict::Alias));
        assert_eq!(same("v_add_nc_u32", &theirs("v_add_u32")), Some(Verdict::Alias));
        assert_eq!(same("v_cmp_ne_u32", &theirs("v_cmp_lg_u32")), Some(Verdict::Alias));
        assert_eq!(same("v_cmp_t_i32", &theirs("v_cmp_tru_i32")), Some(Verdict::Alias));
        assert_eq!(same("s_cmp_eq_i32", &theirs("s_cmpk_eq_i32")), Some(Verdict::Alias));
        assert_eq!(same("flat_load_dword", &theirs("global_load_dword")), Some(Verdict::Alias));
        assert_eq!(same("s_endpgm", &theirs("s_endpgm_ordered_ps_done")), Some(Verdict::Alias));
        assert_eq!(same("s_waitcnt", &theirs("s_waitcnt_vscnt")), Some(Verdict::Alias));
        assert_eq!(same("v_other", &theirs("v_demo")), None);
    }

    #[test]
    fn run_reports_the_exit_state() {
        let dir = std::env::temp_dir().join(format!("aco_crosscheck_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.cpp"), decode()).unwrap();
        std::fs::write(dir.join("ignored.h"), "VOP2_OPCODE_LIST[] = {\n{0x1, Opcode::X},\n};\n").unwrap();
        std::fs::write(dir.join("aco_opcodes.cpp"), generated()).unwrap();
        let (text, ecart) = run(&dir, &dir.join("aco_opcodes.cpp")).unwrap();
        assert!(ecart);
        assert!(text.ends_with(&format!("identique=2 alias=0 absent=1 ECART=1{NL}")));
        assert!(run(&dir.join("none"), &dir.join("aco_opcodes.cpp")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
