//! `decant_shader_instrument.py`: puts a temporary trace into `ResourceTracking.cpp` of the modified
//! translator checkout, or takes it out again. `on` keeps a `.decant_bak` copy, `off` restores it as
//! it was. Nothing else in the file should change between the two.
//!
//! Only the text edit is here. The inserted lines and the anchors they go after are the ones of
//! the script, which belongs to this project. The translator file itself is GPL code: it is only
//! read and rewritten in the user's own checkout, never copied into this repository.

use std::path::{Path, PathBuf};

/// The file the script edits, under `DECANT_TRANSLATOR_SRC`.
pub fn target(src: &Path) -> PathBuf {
    src.join("graphics/shader/recompiler/ir/passes/ResourceTracking.cpp")
}

/// `p.with_suffix(".cpp.decant_bak")`.
pub fn backup(path: &Path) -> PathBuf {
    path.with_extension("cpp.decant_bak")
}

const NM: &str = "const auto nm = [](const Inst* i) { return i ? std::string(ValueOpcodeName(i->GetOpcode())) : std::string(\"imm\"); };";
const NAMESPACE: &str = "namespace {\n";
const DECANT_TRACE: &str = "namespace {\nstatic bool DecantTrace(int line) { if (std::getenv(\"DECANT_SHADER_TRACE\")) std::fprintf(stderr, \" DECANT_TREE at%d;\", line); return false; }\n";
const BAD_READ: &str = "\t\t\t    (dword != 0u && table_mask != nullptr && mask != *table_mask)) {\n";
const BAD_READ_HEAD: &str = "\t\t\t\tif (std::getenv(\"DECANT_SHADER_TRACE\")) { ";
const BAD_READ_BODY: &str = "\n\t\t\t\t\tconst auto* o1 = read->Arg(1).Resolve().TryInstruction();\n\t\t\t\t\tstd::string t = \" DECANT_TREE bad h=\" + nm(current_handle) + \" off=\" + nm(o1) + \"(\";\n\t\t\t\t\tif (o1) for (uint32_t a = 0; a < o1->NumArgs() && a < 3; ++a) { const auto* q = o1->Arg(a).Resolve().TryInstruction(); t += nm(q); if (q) { t += \"[\"; for (uint32_t b = 0; b < q->NumArgs() && b < 3; ++b) { const auto* z = q->Arg(b).Resolve().TryInstruction(); t += nm(z); if (z && (z->GetOpcode() == ValueOpcode::SelectU32 || z->GetOpcode() == ValueOpcode::Phi)) { t += \"{\"; for (uint32_t c = 0; c < z->NumArgs() && c < 3; ++c) t += nm(z->Arg(c).Resolve().TryInstruction()) + \",\"; t += \"}\"; } t += \",\"; } t += \"]\"; } t += \" \"; }\n\t\t\t\t\tt += \") avail=\" + std::to_string((int)(current_handle != nullptr && AvailableBefore(*read, handle))) + \" kind=\" + std::to_string((int)memory->kind);\n\t\t\t\t\tstd::fprintf(stderr, \"%s;\", t.c_str()); }\n";
const HEAP: &str = "\t\t\t\t\t\tif (sampler || indirect.workgroup_axis != UINT32_MAX || table_stride != 32u ||";
const HEAP_TRACE: &str = "\t\t\t\t\t\tif (std::getenv(\"DECANT_SHADER_TRACE\")) { std::string t = \" DECANT_TREE heap\";\n\t\t\t\t\t\t\tif (table_stride != 32u) t += \" stride\" + std::to_string(table_stride); if ((table_base & 31u) != 0u) t += \" base\"; if (address == nullptr) t += \" noaddr\";\n\t\t\t\t\t\t\tif (address != nullptr) for (const auto& use: address->Uses()) { const auto op = use.user->GetOpcode(); if (std::ranges::find(plan.reads, use.user) == plan.reads.end() && op != ValueOpcode::IEqual32 && op != ValueOpcode::Phi && op != ValueOpcode::LoadAddressU32) t += \" addr_\" + std::string(ValueOpcodeName(op)); }\n\t\t\t\t\t\t\tfor (const auto& use: handle.Uses()) { const auto op = use.user->GetOpcode(); if (!(ImageOpcodeInfoOf(op).access == ImageAccess::None || op == ValueOpcode::ImageSampleRaw || op == ValueOpcode::ImageQueryDimensions || op == ValueOpcode::ImageRead)) t += \" img_\" + std::string(ValueOpcodeName(op)); }\n\t\t\t\t\t\t\tstd::fprintf(stderr, \"%s;\", t.c_str()); }\n";
const INDEXED: &str = "\t\t\tif (indexed) {\n\t\t\t\tif (handle.NumArgs() != 8u) return DecantTrace(__LINE__);";
const INDEXED_TRACE: &str = "\t\t\tif (indexed && std::getenv(\"DECANT_SHADER_TRACE\")) {\n\t\t\t\tstd::function<std::string(Value, int)> tree = [&](Value v, int d) -> std::string { v = v.Resolve(); uint32_t im = 0; if (ImmediateU32(v, im)) return \"#\"; const auto* x = v.TryInstruction(); if (!x) return \"?\"; std::string r(ValueOpcodeName(x->GetOpcode())); if (d >= 4 || x->NumArgs() == 0) return r; r += \"(\"; for (uint32_t k = 0; k < x->NumArgs() && k < 3; ++k) r += tree(x->Arg(k), d + 1) + \",\"; return r + \")\"; };\n\t\t\t\tstd::fprintf(stderr, \" DECANT_TREE word%u/%u %s;\", dword, handle.NumArgs(), tree(descriptor.dwords[dword], 0).c_str());\n\t\t\t}\n";
const FUNCTIONAL: &str = "#include <functional>";
const CSTDLIB: &str = "#include <cstdlib>\n";
const CSTDLIB_FUNCTIONAL: &str = "#include <cstdlib>\n#include <functional>\n";

/// `between(a, b, fn)`: applies `edit` to the text from the first `a` up to the next `b`.
fn between(s: &str, a: &str, b: &str, edit: impl Fn(&str) -> String) -> Result<String, String> {
    let i = s.find(a).ok_or_else(|| format!("ValueError: substring not found: {a:?}"))?;
    let j = s[i..].find(b).map(|j| i + j).ok_or_else(|| format!("ValueError: substring not found: {b:?}"))?;
    Ok(format!("{}{}{}", &s[..i], edit(&s[i..j]), &s[j..]))
}

fn once(s: &str, anchor: &str) -> Result<(), String> {
    if s.matches(anchor).count() == 1 { Ok(()) } else { Err(format!("AssertionError: {anchor:?} is not found exactly once")) }
}

/// The instrumented text of the file, with `\n` line ends.
pub fn instrument(text: &str) -> Result<String, String> {
    let trace = |t: &str| t.replace("return false;", "return DecantTrace(__LINE__);");
    let mut s = between(text, "\tbool MatchDescriptorReads(Inst& handle", "\tbool SetExactTableMaximum(Value key", trace)?;
    s = between(&s, "\tbool TryMakeTableDescriptor(Inst& handle", "\tbool MatchPackedScalarKey(Value key", trace)?;
    s = s.replacen(NAMESPACE, DECANT_TRACE, 1);
    once(&s, BAD_READ)?;
    s = s.replace(BAD_READ, &format!("{BAD_READ}{BAD_READ_HEAD}{NM}{BAD_READ_BODY}"));
    once(&s, HEAP)?;
    s = s.replace(HEAP, &format!("{HEAP_TRACE}{HEAP}"));
    once(&s, INDEXED)?;
    s = s.replace(INDEXED, &format!("{INDEXED_TRACE}{INDEXED}"));
    if !s.contains(FUNCTIONAL) {
        s = s.replacen(CSTDLIB, CSTDLIB_FUNCTIONAL, 1);
    }
    Ok(s)
}

/// `on`: keeps the backup, then writes the instrumented file. As in the script, the backup is
/// written before the anchors are checked, and the file is read as UTF 8 in text mode and
/// written back with the platform's line ends.
pub fn on(path: &Path) -> Result<&'static str, String> {
    let bak = backup(path);
    if bak.exists() {
        return Err("AssertionError: deja instrumente".to_owned());
    }
    std::fs::copy(path, &bak).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8(bytes).map_err(|e| format!("UnicodeDecodeError: {e}"))?;
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let s = instrument(&text)?;
    std::fs::write(path, decant_shader_core::py::native_newlines(&s)).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok("instrumente")
}

/// `off`: puts the backup back and removes it.
pub fn off(path: &Path) -> Result<&'static str, String> {
    let bak = backup(path);
    std::fs::copy(&bak, path).map_err(|e| format!("{}: {e}", bak.display()))?;
    std::fs::remove_file(&bak).map_err(|e| format!("{}: {e}", bak.display()))?;
    Ok("restaure")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An invented file with the anchors the script looks for, nothing of the translator.
    pub fn sample() -> String {
        [
            "#include <cstdlib>\n",
            "namespace {\n",
            "\tbool MatchDescriptorReads(Inst& handle) {\n\t\tif (x) return false;\n\t\t\t    (dword != 0u && table_mask != nullptr && mask != *table_mask)) {\n\t\treturn false;\n\t}\n",
            "\tbool SetExactTableMaximum(Value key) { return false; }\n",
            "\tbool TryMakeTableDescriptor(Inst& handle) {\n\t\t\tif (indexed) {\n\t\t\t\tif (handle.NumArgs() != 8u) return false;\n",
            "\t\t\t\t\t\tif (sampler || indirect.workgroup_axis != UINT32_MAX || table_stride != 32u ||\n\t}\n",
            "\tbool MatchPackedScalarKey(Value key) { return false; }\n",
            "}\n",
        ]
        .concat()
    }

    #[test]
    fn traces_go_in_both_functions_only() {
        let s = instrument(&sample()).unwrap();
        assert_eq!(s.matches("return DecantTrace(__LINE__);").count(), 3);
        assert!(s.contains("Value key) { return false; }"), "the other functions keep their returns");
        assert!(s.starts_with(CSTDLIB_FUNCTIONAL));
        assert_eq!(s.matches("DECANT_TREE").count(), 4);
        assert!(s.contains(&format!("{BAD_READ}{BAD_READ_HEAD}{NM}")));
        assert!(instrument("namespace {\n").is_err());
    }

    #[test]
    fn on_and_off_restore_the_bytes() {
        let dir = std::env::temp_dir().join(format!("decant-shader-instrument-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("ResourceTracking.cpp");
        let original = sample().replace('\n', "\r\n").into_bytes();
        std::fs::write(&file, &original).unwrap();
        assert_eq!(on(&file).unwrap(), "instrumente");
        assert!(on(&file).is_err(), "a second on is refused");
        let written = std::fs::read_to_string(&file).unwrap();
        assert_eq!(written, decant_shader_core::py::native_newlines(&instrument(&sample()).unwrap()));
        assert_eq!(off(&file).unwrap(), "restaure");
        assert_eq!(std::fs::read(&file).unwrap(), original);
        assert!(!backup(&file).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
