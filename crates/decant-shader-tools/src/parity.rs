//! The tools on the invented inputs of `synthetic.rs`. The expected texts are what the Python
//! scripts print for the same inputs, written by `examples/synthetic_inputs.rs`.
//!
//! The folder level checks depend on the order the directory lists the modules, which `glob`
//! and `read_dir` both take from the system. The expected texts were taken on Windows, so those
//! checks run there.

use std::path::{Path, PathBuf};

use decant_shader_analysis::NL;
use decant_shader_analysis::random::Random;

use crate::synthetic::{Module, blit_image, texture_image};
use crate::{capabilities, hlsl, images, reflect};

fn lines(text: &str) -> String {
    text.replace('\n', NL)
}

/// A fresh folder under the system temporary folder, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("decant-shader-tools-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn reflection_of_the_sample_module() {
    let r = reflect::reflect(&Module::reflect_sample().bytes()).unwrap();
    assert_eq!(
        r.py().json(true),
        concat!(
            r#"{"version":66816,"capabilities":[1,1,5347],"resources":["#,
            r#"{"set":1,"binding":3,"count":4,"name":"tex","kind":"combined_image","dim":"2d","depth":0,"arrayed":false,"ms":false,"sampled_type":"float","format":0},"#,
            r#"{"set":0,"binding":0,"count":0,"name":"","kind":"storage_image","dim":"2d","depth":0,"arrayed":true,"ms":false,"sampled_type":"uint","format":4},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"uniform_texel_buffer","dim":"buffer","depth":0,"arrayed":false,"ms":false,"sampled_type":"sint","format":0},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"uniform_buffer"},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"storage_buffer"},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"sampler"},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"combined_image","dim":"?","depth":"?","arrayed":true,"ms":true,"sampled_type":"?","format":"?"},"#,
            r#"{"set":0,"binding":0,"count":1,"name":"","kind":"unknown"}],"#,
            r#""push_size":128,"inputs":[{"location":2,"builtin":null}],"outputs":[{"location":null,"builtin":true}]}"#
        )
    );
}

#[cfg(windows)]
#[test]
fn capabilities_of_the_synthetic_measurement() {
    let dir = Scratch::new("caps");
    crate::synthetic::write_measurement(&dir.0.join("measure")).unwrap();
    let folder = dir.0.join("measure").to_string_lossy().into_owned();
    let report = capabilities::folder(&folder).unwrap();
    assert_eq!(
        report.replacen(&folder, "measure", 1),
        lines(
            "measure: 24 modules
   capacite     1                                                          24 (100.0 %)
   capacite  5347 PhysicalStorageBufferAddresses (memoire par adresse)     17 (70.8 %)
   capacite  5301 RuntimeDescriptorArray                                   13 (54.2 %)
   capacite    11                                                          12 (50.0 %)
   capacite  4427 Int64/autres                                             12 (50.0 %)
   capacite    64                                                           9 (37.5 %)
   capacite     6                                                           5 (20.8 %)
   capacite  5302                                                           3 (12.5 %)
   bindings les plus frequents : [(3, 10), (6, 6), (7, 5), (0, 4), (4, 4), (1, 2), (2, 2), (5, 2), (8, 2)]
"
        )
    );
}

#[cfg(windows)]
#[test]
fn hlsl_probe_draws_the_python_sample() {
    let dir = Scratch::new("hlsl");
    let measure = dir.0.join("measure");
    crate::synthetic::write_measurement(&measure).unwrap();
    let mut random = Random::seed(21);
    let short = |list: Vec<PathBuf>| -> Vec<String> {
        list.iter().map(|p| p.strip_prefix(&measure).unwrap().to_string_lossy().replace('/', "\\")).collect()
    };
    assert_eq!(short(hlsl::draw(&mut random, &measure, 6).unwrap()), ["1-00048\\0.spv", "2-00096\\0.spv", "2-00096\\8.spv", "2-00096\\6.spv", "1-00048\\4.spv", "2-00096\\10.spv"]);
    assert_eq!(short(hlsl::draw(&mut random, &measure, 6).unwrap()), ["1-00048\\1.spv", "2-00096\\10.spv", "2-00096\\2.spv", "1-00048\\0.spv", "2-00096\\7.spv", "2-00096\\5.spv"]);
    let all = short(hlsl::draw(&mut random, &measure, 30).unwrap());
    assert_eq!(all.len(), 24, "a count above the modules takes them all");
    assert_eq!(all[..3], ["1-00048\\2.spv", "0-00000\\0.spv", "2-00096\\8.spv"]);
    // The report counts the outcomes of a converter in sample order, on eight workers.
    let jobs: Vec<(PathBuf, String, usize)> = (0..20).map(|i| (Path::new(&format!("{i}")).to_path_buf(), "ps_6_6".to_owned(), i)).collect();
    let outcome = |_: &Path, _: &str, i: usize| -> Result<hlsl::Outcome, String> {
        Ok(match i % 4 {
            0 => ("ok".into(), String::new()),
            1 | 2 => ("dxc".into(), format!("reason {}", i % 3)),
            _ => ("spirv-cross".into(), "nullptr".into()),
        })
    };
    let results = hlsl::convert_all(&jobs, &outcome).unwrap();
    let sample: Vec<PathBuf> = jobs.iter().map(|j| j.0.clone()).collect();
    assert_eq!(
        hlsl::report("m", "ps_6_6", &sample, &results),
        lines("== m (ps_6_6) : 5 / 20 convertis jusqu'au DXIL\n      5  spirv-cross: nullptr\n      4  dxc: reason 2\n      3  dxc: reason 1\n      3  dxc: reason 0\n")
    );
}

#[test]
fn image_checks_print_what_python_prints() {
    let check = |v: images::Verdict| (v.text, v.ok);
    assert_eq!(
        check(images::blit(&blit_image([1.0; 4], false), None).unwrap()),
        (lines("composantes : 16384, non dessinees : 0\necart maximal : 0.000000 en None\negales au demi-flottant attendu (arrondi vers zero) : 16384 (100.0 %)\nRESULTAT : CONFORME\n"), true)
    );
    assert_eq!(
        check(images::blit(&blit_image([1.0, 1.0, 0.5, 1.0], false), None).unwrap()),
        (lines("composantes : 16384, non dessinees : 0\necart maximal : 0.373047 en (63, 63, 2, 0.373046875, 0.74609375)\negales au demi-flottant attendu (arrondi vers zero) : 12288 (75.0 %)\nRESULTAT : ECART\n"), false)
    );
    assert_eq!(
        check(images::blit(&blit_image([1.0; 4], true), None).unwrap()),
        (lines("composantes : 16384, non dessinees : 4\necart maximal : 1.997070 en (0, 0, 3, -1.0, 0.9970703125)\negales au demi-flottant attendu (arrondi vers zero) : 16380 (100.0 %)\nRESULTAT : ECART\n"), false)
    );
    assert!(images::blit(&blit_image([0.5, 2.0, 1.0, 0.25], false), Some([0.5, 2.0, 1.0, 0.25])).unwrap().ok);
    assert_eq!(
        check(images::texture(&texture_image(false, 0.0), false).unwrap()),
        (lines("composantes verifiees : 16384, pixels ignores (frontiere de texel) : 0\necart maximal : 0.000191 en (63, 2, 1, 0.548828125, 0.5490196078431373)\negales au demi-flottant attendu : 16384 (100.0 %)\nRESULTAT : CONFORME\n"), true)
    );
    assert_eq!(
        check(images::texture(&texture_image(true, 0.0), true).unwrap()),
        (lines("composantes verifiees : 16384, pixels ignores (frontiere de texel) : 0\necart maximal : 0.000345 en (32, 32, 2, 0.587890625, 0.5882352941176471)\negales au demi-flottant attendu : 16384 (100.0 %)\nRESULTAT : CONFORME\n"), true)
    );
    assert_eq!(
        check(images::texture(&texture_image(false, 0.01), false).unwrap()),
        (lines("composantes verifiees : 16384, pixels ignores (frontiere de texel) : 0\necart maximal : 0.010000 en (0, 0, 3, 1.0099999904632568, 1.0)\negales au demi-flottant attendu : 0 (0.0 %)\nRESULTAT : ECART\n"), false)
    );
}
