//! `rapport_imports.md` and `rapport_imports.json`, worded and laid out as the Python auditor
//! writes them.

use serde_json::{Map, Value, json};

use crate::audit::{Audit, Dll};
use crate::host::Host;
use crate::pe::Func;
use crate::pyfmt::{float, round2, text};

pub const GROUPS: [(&str, &str); 4] = [
    ("PC", "Presentes sur ce PC"),
    ("paquet", "Livrees avec le jeu"),
    ("absente", "Absentes du PC (a reimplementer)"),
    ("shim", "Fournies par nos DLL (--shims)"),
];

fn group(origin: &str) -> &'static str {
    match origin {
        "absente" => "absente",
        "paquet" => "paquet",
        "shim" => "shim",
        _ => "PC",
    }
}

/// Imported and unresolved function counts per group, in `GROUPS` order.
pub fn totals(a: &Audit) -> [(usize, usize); 4] {
    let mut out = [(0, 0); 4];
    for (_, d) in &a.dlls {
        let i = GROUPS.iter().position(|(g, _)| *g == group(d.origin)).unwrap();
        out[i].0 += d.funcs.len();
        out[i].1 += missing(d);
    }
    out
}

fn missing(d: &Dll) -> usize {
    d.funcs.iter().filter(|(_, ok)| !ok).count()
}

/// DLLs absent from the PC first, then those of our own folder, then those in the package, then the rest, each by name.
fn ordered(a: &Audit) -> Vec<&(String, Dll)> {
    let rank = |o: &str| match o {
        "absente" => 0,
        "shim" => 1,
        "paquet" => 2,
        _ => 3,
    };
    let mut out: Vec<&(String, Dll)> = a.dlls.iter().collect();
    out.sort_by(|(ka, a), (kb, b)| (rank(a.origin), ka).cmp(&(rank(b.origin), kb)));
    out
}

pub fn exes(a: &Audit) -> Vec<&str> {
    a.files.iter().map(|(rel, _)| rel.as_str()).filter(|rel| rel.to_lowercase().ends_with(".exe")).collect()
}

pub fn markdown(a: &Audit, host: &Host) -> String {
    let exes = exes(a);
    let mut l: Vec<String> = vec!["# Audit des imports - build Xbox GDK".into(), String::new()];
    l.push(format!("Dossier analyse : `{}`", text(&a.root)));
    l.push(format!("Fichiers PE lus : {} ({} .exe, {} .dll)", a.files.len(), exes.len(), a.files.len() - exes.len()));
    if exes.is_empty() {
        l.extend(["".into(), "> **Aucun .exe dans le dossier** : ce rapport ne couvre que les DLL livrees avec le jeu.".into(), "> Relancer avec un executable pour couvrir ses imports.".into()]);
    }
    l.extend(["", "## Resume", "", "| Origine des DLL importees | Fonctions importees | Introuvables |", "|---|---:|---:|"].map(String::from));
    let sums = totals(a);
    for ((g, name), (count, miss)) in GROUPS.iter().zip(sums) {
        if *g == "shim" && !a.shims {
            continue;
        }
        l.push(format!("| {name} | {count} | {miss} |"));
    }
    l.push(format!("| **Total** | **{}** | **{}** |", sums.iter().map(|s| s.0).sum::<usize>(), sums.iter().map(|s| s.1).sum::<usize>()));

    l.extend(["", "## DLL importees", "", "| DLL | Origine | Fonctions | Introuvables | Importee par | Role |", "|---|---|---:|---:|---|---|"].map(String::from));
    for (_, d) in ordered(a) {
        let users: Vec<&String> = d.users.iter().collect();
        let mut shown = users.iter().take(3).map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        if users.len() > 3 {
            shown += &format!(" (+{})", users.len() - 3);
        }
        let origin = format!("{}{}", d.origin, if d.delay { " (differe)" } else { "" });
        l.push(format!("| `{}` | {origin} | {} | {} | {shown} |  |", d.name, d.funcs.len(), missing(d)));
    }

    l.extend(["", "## Fonctions a fournir", "", "Tout ce que ce PC ne sait pas resoudre : DLL absentes, et fonctions manquantes dans des DLL presentes.", ""].map(String::from));
    let mut todo = 0;
    for (_, d) in ordered(a) {
        let mut absent: Vec<&Func> = d.funcs.iter().filter(|(_, ok)| !ok).map(|(f, _)| f).collect();
        if absent.is_empty() {
            continue;
        }
        absent.sort_by_key(|f| f.label());
        todo += absent.len();
        l.push(format!("### `{}` - {} fonction(s), origine : {}", d.name, absent.len(), d.origin));
        l.push(String::new());
        for f in absent {
            let pretty = match f {
                Func::Name(n) => host.demangle(n),
                Func::Ordinal(_) => None,
            };
            l.push(format!("- `{}`{}", f.label(), pretty.map_or(String::new(), |p| format!(" : `{p}`"))));
        }
        l.push(String::new());
    }
    if todo == 0 {
        l.extend(["Rien : tous les imports sont resolus.".into(), String::new()]);
    }

    l.extend(["## Fichiers analyses", "", "| Fichier | Taille (Mo) | DLL importees | Fonctions importees |", "|---|---:|---:|---:|"].map(String::from));
    let mut by_name: Vec<_> = a.files.iter().collect();
    by_name.sort_by(|x, y| x.0.cmp(&y.0));
    for (rel, f) in by_name {
        l.push(format!("| `{rel}` | {} | {} | {} |", float(round2(f.size_mib)), f.dll_count, f.func_count));
    }
    for (rel, f) in a.files.iter().filter(|(rel, _)| exes.contains(&rel.as_str())) {
        l.extend(["".into(), format!("### Sections de `{rel}`"), String::new(), "| Section | Taille (Mo) | Code | Entropie |".into(), "|---|---:|---|---:|".into()]);
        let rows = f.sections.as_deref().unwrap_or_default();
        for s in rows {
            l.push(format!("| `{}` | {} | {} | {} |", s.name, float(round2(s.size_mib)), if s.executable { "oui" } else { "" }, float(round2(s.entropy))));
        }
        if round2(f.size_mib) > 50.0 && f.func_count < 100 {
            l.extend([
                "".into(),
                "> Attention : tres peu d'imports pour un executable de cette taille. Il est probablement".into(),
                "> protege ou compresse (les vrais imports sont resolus au demarrage) : la liste ci-dessus est incomplete.".into(),
            ]);
        }
        if rows.iter().any(|s| s.executable && round2(s.entropy) > 7.5) {
            l.extend(["".into(), "> Attention : une section de code a une entropie superieure a 7,5 : code chiffre ou compresse.".into()]);
        }
    }
    if !a.errors.is_empty() {
        l.extend(["", "## Fichiers illisibles", ""].map(String::from));
        for (rel, e) in &a.errors {
            l.push(format!("- `{rel}` : {e}"));
        }
    }
    l.extend([
        "".into(),
        "## Limites".into(),
        String::new(),
        "- Les fonctions chargees a l'execution par `LoadLibrary` / `GetProcAddress` ne sont pas vues.".into(),
        "- Une fonction presente sur PC peut se comporter autrement que sur la console.".into(),
        String::new(),
    ]);
    l.join("\n")
}

fn number(x: f64) -> Value {
    serde_json::Number::from_f64(round2(x)).map_or(Value::Null, Value::Number)
}

pub fn json(a: &Audit) -> String {
    let mut files = Map::new();
    for (rel, f) in &a.files {
        let mut entry = Map::new();
        entry.insert("taille_mo".into(), number(f.size_mib));
        entry.insert("machine".into(), json!(format!("{:#x}", f.machine)));
        entry.insert("dll_importees".into(), json!(f.dll_count));
        entry.insert("fonctions_importees".into(), json!(f.func_count));
        if let Some(rows) = &f.sections {
            let rows: Vec<Value> = rows.iter().map(|s| json!({"name": s.name, "taille_mo": number(s.size_mib), "executable": s.executable, "entropie": number(s.entropy)})).collect();
            entry.insert("sections".into(), Value::Array(rows));
        }
        files.insert(rel.clone(), Value::Object(entry));
    }
    let errors: Map<String, Value> = a.errors.iter().map(|(rel, e)| (rel.clone(), json!(e))).collect();
    let mut dlls: Vec<&(String, Dll)> = a.dlls.iter().collect();
    dlls.sort_by(|x, y| x.0.cmp(&y.0));
    let mut dll_map = Map::new();
    for (key, d) in dlls {
        let mut funcs: Vec<(String, bool)> = d.funcs.iter().map(|(f, ok)| (f.label(), *ok)).collect();
        funcs.sort_by(|x, y| x.0.cmp(&y.0));
        let funcs: Map<String, Value> = funcs.into_iter().map(|(k, ok)| (k, json!(ok))).collect();
        dll_map.insert(key.clone(), json!({"nom": d.name, "origine": d.origin, "differe": d.delay, "importee_par": d.users.iter().collect::<Vec<_>>(), "fonctions": funcs}));
    }
    let payload = json!({"dossier": text(&a.root), "fichiers": files, "illisibles": errors, "dll": dll_map});
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, serde_json::ser::PrettyFormatter::with_indent(b" "));
    serde::Serialize::serialize(&payload, &mut ser).expect("JSON values always serialise");
    String::from_utf8(out).expect("serde_json writes UTF 8")
}
