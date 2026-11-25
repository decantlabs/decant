//! The six tools on the invented corpus of `synthetic.rs`, from container bytes to printed text.
//! The expected texts are what the Python scripts printed for the same shaders with the stand in
//! decoder (`examples/stand_in_decoder.rs`) placed where they expect `xbox_decode_test_xbox.exe`.

use decant_extract_shaders::{Shader, container};
use serde_json::Value;

use crate::metadata::{Summary, analyse, outputs_row};
use crate::{NL, load_blob, load_fssm, pixel, synthetic, tables, vertex};

/// Expected text written with LF, compared in the line end the tools write.
fn lines(text: &str) -> String {
    text.replace('\n', NL)
}

struct Corpus {
    items: Vec<Shader>,
    rows: Vec<String>,
    meta: Vec<Value>,
    summary: String,
}

fn corpus() -> Corpus {
    let items: Vec<Shader> = synthetic::corpus().into_iter().map(|(name, blob)| Shader::from_bytes(format!("{name}_Wrapped"), blob)).collect();
    let mut summary = Summary::default();
    let mut rows = Vec::new();
    for (id, item) in items.iter().enumerate() {
        let blob = load_blob(item).unwrap();
        let s = analyse(&blob, container::gpu_code(&blob).as_deref()).unwrap();
        summary.add(&s);
        rows.push(s.row(id, &item.name));
    }
    let meta = rows.iter().map(|r| serde_json::from_str(r).unwrap()).collect();
    Corpus { summary: summary.line(), items, rows, meta }
}

fn listing(name: &str) -> Result<String, String> {
    Ok(crate::decoder::text_mode(synthetic::listing(name).as_bytes()))
}

#[test]
fn metadata_and_outputs() {
    let c = corpus();
    assert_eq!(c.items.len(), 59);
    assert_eq!(
        c.summary,
        "shaders 59 wave Counter({('pixel', 0, 4294967295): 36, ('vertex', 0, 4294967295): 18, ('compute', 32, 32): 1, \
         ('compute', 64, 64): 1, ('compute', 0, 0): 1, ('mesh', 0, 4294967295): 1, ('geometry', 0, 4294967295): 1}) \
         resources Counter({('pixel', 't'): 36, ('pixel', 'b'): 36, ('vertex', 'b'): 18, ('vertex', 't'): 18, ('pixel', 's'): 4, \
         ('compute', 'u'): 3, ('compute', 't'): 3, ('vertex', 'u'): 2, ('mesh', 't'): 1}) \
         fssm_kind {'pixel': Counter({2: 36}), 'vertex': Counter({2: 18}), 'compute': Counter({2: 2, 3: 1}), 'mesh': Counter({2: 1}), \
         'geometry': Counter({2: 1})} \
         top_threads [(('compute', (8, 8, 1)), 1), (('compute', (64, 1, 1)), 1), (('compute', (4, 4, 4)), 1), (('mesh', (128, 1, 1)), 1)] \
         correlation size (compute) {0: [(316, 3), (32, 1), (80, 1)], 1: [(316, 2), (32, 1), (84, 1)], 2: [(64, 1), (84, 1), (316, 1)]} \
         wave candidates {32: [(48, 1), (0, 0), (4, 0), (8, 0)], 64: [(80, 1), (316, 1), (0, 0), (4, 0)]}"
    );
    let expected = [
        (1, r#"{"id":1,"name":"PS_Shared_Wrapped","stage":"pixel","wave_bounds":[0,4294967295],"wave":null,"threads":null,"mesh_limits":[0,0],"inputs":[["SV_Position",0,0,15,0,1,3,0,0],["TEXCOORD",1,1,3,3,0,3,1,0],["TEXCOORD",2,1,12,4,0,3,0,0],["COLOR",3,2,15,0,0,3,1,0],["NORMAL",4,3,7,7,0,3,0,0]],"bindings":[["t",0,2,0,3,1,0],["b",1,1,0,2,1,0],["s",0,0,0,1,1,0]],"fssm_kind":2,"heap_flags":1}"#),
        (4, r#"{"id":4,"name":"VS_Linked_Wrapped","stage":"vertex","wave_bounds":[0,4294967295],"wave":null,"threads":null,"mesh_limits":[0,0],"inputs":[["POSITION",0,0,7,7,0,3,0,0],["TEXCOORD",1,1,3,1,0,3,1,0],["SV_VertexID",2,2,1,1,6,3,0,0],["SV_InstanceID",3,3,1,1,8,3,1,0]],"bindings":[["b",0,0,0,2,1,0],["t",0,0,0,3,1,0],["u",1,1,0,6,1,0]],"fssm_kind":2,"heap_flags":1}"#),
        (6, r#"{"id":6,"name":"CS_Wave32_Wrapped","stage":"compute","wave_bounds":[32,32],"wave":32,"threads":[8,8,1],"mesh_limits":[0,0],"inputs":[["TEXCOORD",0,1,3,1,0,3,0,0],["SV_Position",1,0,15,0,1,3,1,0]],"bindings":[["u",0,3,0,7,1,0],["t",0,0,0,4,1,0]],"fssm_kind":2,"heap_flags":3}"#),
        (9, r#"{"id":9,"name":"MS_Mesh_Wrapped","stage":"mesh","wave_bounds":[0,4294967295],"wave":null,"threads":[128,1,1],"mesh_limits":[64,126],"inputs":[["TEXCOORD",0,1,3,1,0,3,0,0],["SV_Position",1,0,15,0,1,3,1,0]],"bindings":[["t",2,5,0,3,1,0]],"fssm_kind":2,"heap_flags":0}"#),
        (10, r#"{"id":10,"name":"GS_Empty_Wrapped","stage":"geometry","wave_bounds":[0,4294967295],"wave":null,"threads":null,"mesh_limits":[0,0],"inputs":[],"bindings":[],"fssm_kind":2,"heap_flags":1}"#),
    ];
    for (id, row) in expected {
        assert_eq!(c.rows[id], row);
    }
    let output = |id: usize| outputs_row(id, &c.items[id].name, &load_blob(&c.items[id]).unwrap()).unwrap();
    assert_eq!(output(4), r#"{"id":4,"name":"VS_Linked_Wrapped","stage":"vertex","outputs":[["SV_Position",0,0,15,0,1,3,0,0],["TEXCOORD",1,1,3,12,0,3,1,0]]}"#);
    assert_eq!(output(10), r#"{"id":10,"name":"GS_Empty_Wrapped","stage":"geometry","outputs":[]}"#);
}

#[test]
fn fssm_tables_and_bindings() {
    let c = corpus();
    let mut out = String::new();
    for n in [1usize, 4] {
        let fs = load_fssm(&c.items[n]).unwrap();
        tables::dump(&mut out, &fs, n as i64, &c.items[n].name, &tables::binding_pairs(&c.meta[n]).unwrap()).unwrap();
    }
    assert_eq!(
        out,
        lines(
            "== 1 PS_Shared_Wrapped  liaisons PSV0 : [('t', 0), ('b', 1), ('s', 0)]
   table A a 0x15c (2 entree(s)), table B a 0x164 (4 entree(s)), nom a 0x1b4
   A: 11 22
   B: 300 4 0 0 20804 302 0 1 0 10a08 401 2 0 0 4 123 1 7 9 0
== 4 VS_Linked_Wrapped  liaisons PSV0 : [('b', 0), ('t', 0), ('u', 1)]
   table A a 0x15c (2 entree(s)), table B a 0xdeadbeefdeadbeef (0 entree(s)), nom a 0x164
   A: 11 22
   B: \n"
        )
    );
    let mut out = String::new();
    for n in [1usize, 3] {
        out.push_str(&format!("== {n} {}{NL}", c.items[n].name));
        for r in tables::records(&load_fssm(&c.items[n]).unwrap()).unwrap() {
            out.push_str(&format!("{}{NL}", tables::record_line(&r)));
        }
    }
    let table = "   t0    table s8  octet 0x0010  classe 2  bas 4  extra 0x0
   b1    table s10 octet 0x0000  classe 1  bas 8  extra 0x0
   s0    table s0  octet 0x0008  classe 0  bas 4  extra 0x0
   ?1237    table s0  octet 0x0004  classe 0  bas 0  extra 0x9
";
    assert_eq!(out, lines(&format!("== 1 PS_Shared_Wrapped\n{table}== 3 PS_Twin_Wrapped\n{table}")));
}

#[test]
fn bindings_verification() {
    let c = corpus();
    let names: Vec<&str> = c.items.iter().map(|i| i.name.as_str()).collect();
    let records_of = |n: usize| tables::records(&load_fssm(&c.items[n])?);
    let report = tables::verify(&names, 40, &records_of, &mut listing).unwrap();
    assert_eq!(
        report,
        lines(
            "    28  classe  0 table s8  : entrees
     0  classe  0 table s8  : retrouvees avec decalage x1
     0  classe  0 table s8  : retrouvees avec decalage x4
     1  classe  1 table s10 : entrees
     0  classe  1 table s10 : retrouvees avec decalage x1
     0  classe  1 table s10 : retrouvees avec decalage x4
    19  classe  1 table s9  : entrees
     0  classe  1 table s9  : retrouvees avec decalage x1
     0  classe  1 table s9  : retrouvees avec decalage x4
     1  classe  2 table s8  : entrees
     0  classe  2 table s8  : retrouvees avec decalage x1
     0  classe  2 table s8  : retrouvees avec decalage x4
    49  entrees de table
    29  shaders
genres : {'t': 29, 'b': 20}
"
        )
    );
    assert!(tables::verify(&names, 999, &records_of, &mut listing).is_err(), "a sample larger than the candidates raises");
}

#[test]
fn vertex_abi() {
    let c = corpus();
    // Batch results for the vertex shaders: every fourth without T, then one other failure, then
    // two that must be linked.
    let rows: Vec<Value> = c
        .meta
        .iter()
        .filter(|m| m["stage"] == "vertex")
        .enumerate()
        .map(|(i, m)| match i % 4 {
            0 => serde_json::json!({"name": m["name"]}),
            1 => serde_json::json!({"name": m["name"], "T": ["FAIL", "other reason"]}),
            _ => serde_json::json!({"name": m["name"], "T": ["FAIL", "routine must be linked first"]}),
        })
        .collect();
    let mut file = Vec::new();
    let report = vertex::run(&c.meta, &rows, &mut listing, &mut |line| {
        file.push(line.to_owned());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        report,
        lines(
            "    0  A. vivants == composantes utilisees
    0  B. vivants inclus dans les composantes declarees
    2  C. composantes utilisees toutes vivantes
    1  sans S_SWAPPC en tete
    7  total
valeurs systeme (elements, v0-v3 vivants, emplacements 4+4r vivants) :
    1  ((('SV_Position', 0), ('SV_VertexID', 6)), (1,), (4, 5, 6, 7))
    1  ((('SV_Position', 0),), (3,), (4, 6))
    1  ((('SV_Position', 0), ('SV_VertexID', 6)), (), (5,))
    1  ((('SV_Position', 0),), (0, 3), ())
    1  ((('SV_Position', 0),), (), (5,))
    1  ((('SV_Position', 0), ('SV_VertexID', 6)), (2,), (5, 7))
    1  ((('SV_Position', 0),), (3,), (7,))
ECART ('VS_Gen2_Wrapped', [21, 22], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0], ['COLOR', 1, 1, 12, 0, 0, 3, 1, 0], ['NORMAL', 2, 2, 3, 0, 0, 3, 0, 0], ['SV_VertexID', 3, 6, 1, 1, 6, 3, 1, 0]])
ECART ('VS_Gen5_Wrapped', [12, 13, 21, 22, 23], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0]])
ECART ('VS_Gen14_Wrapped', [8, 15, 23], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0], ['COLOR', 1, 1, 12, 0, 0, 3, 1, 0], ['NORMAL', 2, 2, 3, 0, 0, 3, 0, 0], ['TANGENT', 3, 3, 15, 3, 0, 3, 1, 0], ['TEXCOORD', 4, 4, 7, 0, 0, 3, 0, 0], ['SV_VertexID', 5, 6, 1, 1, 6, 3, 1, 0]])
ECART ('VS_Gen17_Wrapped', [11, 13], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0], ['COLOR', 1, 1, 7, 1, 0, 3, 1, 0], ['NORMAL', 2, 2, 12, 0, 0, 3, 0, 0]])
ECART ('VS_Gen29_Wrapped', [13, 18], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0], ['COLOR', 1, 1, 7, 1, 0, 3, 1, 0], ['NORMAL', 2, 2, 12, 0, 0, 3, 0, 0], ['TANGENT', 3, 3, 3, 3, 0, 3, 1, 0], ['TEXCOORD', 4, 4, 15, 0, 0, 3, 0, 0]])
ECART ('VS_Gen38_Wrapped', [21], [['SV_Position', 0, 0, 15, 0, 1, 3, 0, 0], ['COLOR', 1, 1, 12, 0, 0, 3, 1, 0], ['NORMAL', 2, 2, 3, 0, 0, 3, 0, 0], ['TANGENT', 3, 3, 15, 3, 0, 3, 1, 0], ['SV_VertexID', 4, 6, 1, 1, 6, 3, 0, 0]])
"
        )
    );
    assert_eq!(file.len(), 7);
    assert_eq!(
        file[..2],
        [
            r#"{"name": "VS_Gen2_Wrapped", "live": [1, 4, 5, 6, 7, 11, 21, 22], "inputs": [["SV_Position", 0, 0, 15, 0, 1, 3, 0, 0], ["COLOR", 1, 1, 12, 0, 0, 3, 1, 0], ["NORMAL", 2, 2, 3, 0, 0, 3, 0, 0], ["SV_VertexID", 3, 6, 1, 1, 6, 3, 1, 0]]}"#,
            r#"{"name": "VS_Gen5_Wrapped", "live": [3, 4, 6, 12, 13, 21, 22, 23], "inputs": [["SV_Position", 0, 0, 15, 0, 1, 3, 0, 0]]}"#,
        ]
    );
}

#[test]
fn pixel_inputs() {
    let c = corpus();
    let first = "ECART ('PS_Gen7_Wrapped', [3], [('SV_Position', 0, 0, 0, 1), ('COLOR', 1, 1, 3, 0), ('NORMAL', 2, 2, 3, 0)])\n";
    // Five draws take the set based branch of random.sample, thirty the pool based one.
    assert_eq!(
        pixel::run(&c.meta, 5, &mut listing).unwrap(),
        lines(&format!(
            "    1  A'. et composantes incluses dans le masque lu
    4  A. attributs = entrees lues, dans l ordre
    2  B. attributs = toutes les entrees declarees
    5  shaders
{first}"
        ))
    );
    assert_eq!(
        pixel::run(&c.meta, 30, &mut listing).unwrap(),
        lines(&format!(
            "    5  A'. et composantes incluses dans le masque lu
    8  A. attributs = entrees lues, dans l ordre
    6  B. attributs = toutes les entrees declarees
   29  shaders
{first}ECART ('PS_Gen25_Wrapped', [1], [('SV_Position', 0, 0, 0, 1)])
ECART ('PS_Shared_Wrapped', [1], [('SV_Position', 0, 0, 0, 1), ('TEXCOORD', 1, 1, 3, 0), ('TEXCOORD', 2, 1, 4, 0), ('COLOR', 3, 2, 0, 0), ('NORMAL', 4, 3, 7, 0)])
ECART ('PS_Gen0_Wrapped', [2], [('SV_Position', 0, 0, 0, 1)])
ECART ('PS_Gen19_Wrapped', [1, 2], [('SV_Position', 0, 0, 0, 1), ('COLOR', 1, 1, 3, 0), ('NORMAL', 2, 2, 3, 0), ('TANGENT', 3, 3, 1, 0), ('TEXCOORD', 4, 4, 0, 0)])
"
        ))
    );
}
