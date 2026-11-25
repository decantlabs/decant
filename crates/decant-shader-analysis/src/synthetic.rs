//! Invented shader containers for the tests and the stand in decoder. Nothing here comes from a game file: each
//! container is assembled field by field from the layouts the scripts read.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;

/// A signature element: name, register, mask, read or written mask, system value.
pub type Sig = (&'static str, u32, u8, u8, u32);

/// One invented shader container.
#[derive(Clone, Debug)]
pub struct Container {
    /// The DXIL program type: 0 pixel, 1 vertex, 5 compute, 13 mesh and so on.
    pub stage: u32,
    /// The stage byte of `PSV0`, the pack code of the stage unless a test breaks it.
    pub psv_stage: u8,
    pub wave: [u32; 2],
    pub threads: [u32; 3],
    pub mesh_limits: [u16; 2],
    /// `PSV0` resources: type, space, lower, upper, shape, flags.
    pub bindings: Vec<[u32; 6]>,
    pub inputs: Vec<Sig>,
    pub outputs: Vec<Sig>,
    pub sfi0: Vec<u8>,
    pub fssm_kind: u32,
    pub code: Vec<u32>,
    /// The FSSM table A words and table B records, placed after the code.
    pub table_a: Vec<u32>,
    pub table_b: Option<Vec<[u32; 5]>>,
    /// Extra FSSM header words, as `(position, value)`.
    pub header_words: Vec<(usize, u32)>,
    /// Parts left out of the container.
    pub omit: Vec<&'static str>,
    /// No XBCH entry, so no FSSM block.
    pub no_fssm: bool,
}

fn stage_code(stage: u32) -> u8 {
    match stage {
        0..=5 => stage as u8,
        13 => 13,
        _ => 255,
    }
}

impl Container {
    pub fn new(stage: u32) -> Container {
        Container {
            stage,
            psv_stage: stage_code(stage),
            wave: [0, u32::MAX],
            threads: [0, 0, 0],
            mesh_limits: [0, 0],
            bindings: Vec::new(),
            inputs: vec![("TEXCOORD", 1, 3, 1, 0), ("SV_Position", 0, 15, 0, 1)],
            outputs: vec![("SV_Target", 0, 15, 15, 0)],
            sfi0: vec![0, 0, 0, 2, 0, 0, 0, 0],
            fssm_kind: 2,
            code: vec![0xBE80_0080, 0xBF81_0000, 0xBF9F_0000],
            table_a: vec![0x11, 0x22],
            table_b: Some(Vec::new()),
            header_words: Vec::new(),
            omit: Vec::new(),
            no_fssm: false,
        }
    }

    pub fn with_binding(mut self, kind: u32, lower: u32, upper: u32) -> Container {
        self.bindings.push([kind, 0, lower, upper, 1, 0]);
        self
    }

    pub fn with_threads(mut self, threads: [u32; 3]) -> Container {
        self.threads = threads;
        self
    }

    pub fn with_wave(mut self, wave: u32) -> Container {
        self.wave = [wave, wave];
        self
    }

    pub fn without(mut self, part: &'static str) -> Container {
        self.omit.push(part);
        self
    }

    fn psv0(&self) -> Vec<u8> {
        let ri = 52u32;
        let mut info = vec![0u8; ri as usize];
        info[12..14].copy_from_slice(&self.mesh_limits[0].to_le_bytes());
        info[14..16].copy_from_slice(&self.mesh_limits[1].to_le_bytes());
        info[16..20].copy_from_slice(&self.wave[0].to_le_bytes());
        info[20..24].copy_from_slice(&self.wave[1].to_le_bytes());
        info[24] = self.psv_stage;
        for (i, t) in self.threads.iter().enumerate() {
            info[36 + 4 * i..40 + 4 * i].copy_from_slice(&t.to_le_bytes());
        }
        let mut out = ri.to_le_bytes().to_vec();
        out.extend(info);
        out.extend((self.bindings.len() as u32).to_le_bytes());
        if !self.bindings.is_empty() {
            // A stride above 24, as later PSV0 versions have, with the extra words left zero.
            out.extend(28u32.to_le_bytes());
            for b in &self.bindings {
                out.extend(b.iter().flat_map(|w| w.to_le_bytes()));
                out.extend(0u32.to_le_bytes());
            }
        }
        out
    }

    /// The FSSM block: a 0x150 byte header, the code with its sub header, table A, table B and
    /// a name. Table positions count from byte 16 of the block.
    pub fn fssm(&self) -> Vec<u8> {
        let mut fs = vec![0u8; 0x150];
        fs[..4].copy_from_slice(b"FSSM");
        let put32 = |fs: &mut Vec<u8>, at: usize, v: u32| fs[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put64 = |fs: &mut Vec<u8>, at: usize, v: u64| fs[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put32(&mut fs, 16, self.fssm_kind);
        put64(&mut fs, 0xC0, 0x150);
        put64(&mut fs, 0xC8, 4 * self.code.len() as u64);
        for (i, t) in self.threads.iter().enumerate() {
            if self.stage == 5 {
                put32(&mut fs, 0x13C + 4 * i, *t);
            }
        }
        fs.extend(0xA0u8..0xB0);
        fs.extend(self.code.iter().flat_map(|w| w.to_le_bytes()));
        let a = fs.len() as u64 - 16;
        fs.extend(self.table_a.iter().flat_map(|w| w.to_le_bytes()));
        let b = fs.len() as u64 - 16;
        let records = self.table_b.clone().unwrap_or_default();
        for r in &records {
            fs.extend(r.iter().flat_map(|w| w.to_le_bytes()));
        }
        let name_at = fs.len() as u64 - 16;
        fs.extend(b"main\0\0\0\0");
        put64(&mut fs, 0xE8, a);
        put64(&mut fs, 0xF0, if self.table_b.is_some() { b } else { 0xDEAD_BEEF_DEAD_BEEF });
        put32(&mut fs, 0xF8, self.table_a.len() as u32);
        put32(&mut fs, 0xFC, records.len() as u32);
        put64(&mut fs, 0x120, name_at);
        for &(at, v) in &self.header_words {
            put32(&mut fs, at, v);
        }
        fs
    }

    fn dxil(&self) -> Vec<u8> {
        let mut body = ((self.stage << 16) | 0x66).to_le_bytes().to_vec();
        body.extend(b"\0\0\0\0padXDAT");
        let x = body.len() - 4;
        body.extend(1u32.to_le_bytes());
        body.extend(0u32.to_le_bytes());
        let part = x + 24;
        let kind: u32 = if self.no_fssm { 0x0C } else { 0x0D };
        body.extend(kind.to_le_bytes());
        body.extend((part as u32).to_le_bytes());
        body.extend(0u32.to_le_bytes());
        let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
        z.write_all(&self.fssm()).unwrap();
        let z = z.finish().unwrap();
        body.extend(b"XBCH\0\0\0\0");
        body.extend((z.len() as u64).to_le_bytes());
        body.extend(0u64.to_le_bytes());
        body.extend(z);
        body
    }

    /// The DXBC container.
    pub fn build(&self) -> Vec<u8> {
        let sig = |elements: &[Sig]| {
            let mut out = Vec::new();
            out.extend((elements.len() as u32).to_le_bytes());
            out.extend(8u32.to_le_bytes());
            let names_at = 8 + 32 * elements.len();
            let mut names = Vec::new();
            for (i, (name, register, mask, used, system)) in elements.iter().enumerate() {
                let words = [i as u32 % 2, (names_at + names.len()) as u32, i as u32, *system, 3, *register];
                out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
                out.extend([*mask, *used, 0, 0]);
                out.extend(0u32.to_le_bytes());
                names.extend(name.as_bytes());
                names.push(0);
            }
            out.extend(names);
            out
        };
        let parts: Vec<(&str, Vec<u8>)> = [
            ("SFI0", self.sfi0.clone()),
            ("ISG1", sig(&self.inputs)),
            ("OSG1", sig(&self.outputs)),
            ("PSV0", self.psv0()),
            ("DXIL", self.dxil()),
        ]
        .into_iter()
        .filter(|(tag, _)| !self.omit.contains(tag))
        .collect();
        let mut blob = b"DXBC".to_vec();
        blob.extend([0u8; 16]);
        blob.extend(1u32.to_le_bytes());
        blob.extend(0u32.to_le_bytes());
        blob.extend((parts.len() as u32).to_le_bytes());
        let mut at = 32 + 4 * parts.len();
        for (_, body) in &parts {
            blob.extend((at as u32).to_le_bytes());
            at += 8 + body.len().next_multiple_of(4);
        }
        for (tag, body) in &parts {
            blob.extend(tag.as_bytes());
            blob.extend((body.len() as u32).to_le_bytes());
            blob.extend(body);
            blob.resize(blob.len().next_multiple_of(4), 0);
        }
        let total = blob.len() as u32;
        blob[24..28].copy_from_slice(&total.to_le_bytes());
        blob
    }
}

/// The fixed set of invented shaders the parity comparison and the tests share.
pub fn corpus() -> Vec<(String, Vec<u8>)> {
    let mut list: Vec<(&'static str, Container)> = Vec::new();
    let pixel = Container::new(0).with_binding(3, 0, 2).with_binding(2, 1, 1).with_binding(1, 0, 0);
    list.push(("PS_Plain", pixel.clone()));
    let mut shared = pixel.clone();
    shared.inputs = vec![
        ("SV_Position", 0, 15, 0, 1),
        ("TEXCOORD", 1, 3, 3, 0),
        ("TEXCOORD", 1, 12, 4, 0),
        ("COLOR", 2, 15, 0, 0),
        ("NORMAL", 3, 7, 7, 0),
    ];
    shared.table_b = Some(vec![[0x300, 4, 0, 0, (2 << 16) | (8 << 8) | 4], [0x302, 0, 1, 0, (1 << 16) | (10 << 8) | 8], [0x401, 2, 0, 0, 4], [0x123, 1, 7, 9, 0]]);
    list.push(("PS_Shared", shared.clone()));
    list.push(("PS_Twin", pixel.clone()));
    list.push(("PS_Twin", shared.clone()));
    let mut vertex = Container::new(1).with_binding(2, 0, 0).with_binding(3, 0, 0).with_binding(6, 1, 1);
    vertex.inputs = vec![("POSITION", 0, 7, 7, 0), ("TEXCOORD", 1, 3, 1, 0), ("SV_VertexID", 2, 1, 1, 6), ("SV_InstanceID", 3, 1, 1, 8)];
    vertex.outputs = vec![("SV_Position", 0, 15, 0, 1), ("TEXCOORD", 1, 3, 12, 0)];
    vertex.table_b = None;
    list.push(("VS_Linked", vertex.clone()));
    let mut vertex2 = vertex.clone();
    vertex2.inputs = vec![("POSITION", 0, 15, 15, 0), ("BINORMAL", 1, 15, 3, 0)];
    list.push(("VS_Other", vertex2));
    let mut compute = Container::new(5).with_threads([8, 8, 1]).with_wave(32).with_binding(7, 0, 3).with_binding(4, 0, 0);
    compute.header_words = vec![(0x20, 8), (0x30, 32), (0x40, 1), (0x44, 64)];
    compute.sfi0 = vec![0, 0, 0, 6];
    list.push(("CS_Wave32", compute.clone()));
    let mut compute64 = compute.clone().with_threads([64, 1, 1]).with_wave(64);
    compute64.header_words = vec![(0x50, 64), (0x54, 1)];
    compute64.fssm_kind = 3;
    list.push(("CS_Wave64", compute64));
    list.push(("CS_Any", compute.with_threads([4, 4, 4]).with_wave(0)));
    let mut mesh = Container::new(13).with_threads([128, 1, 1]).with_binding(3, 2, 5);
    mesh.mesh_limits = [64, 126];
    mesh.sfi0 = vec![0, 0];
    list.push(("MS_Mesh", mesh));
    let mut geometry = Container::new(2);
    geometry.outputs = vec![];
    geometry.inputs = vec![];
    list.push(("GS_Empty", geometry));
    let mut out: Vec<(String, Vec<u8>)> = list.into_iter().map(|(name, c)| (name.to_owned(), c.build())).collect();
    // A generated batch, so the random draws and the counts have something to choose from.
    const SEMANTICS: [&str; 4] = ["TEXCOORD", "COLOR", "NORMAL", "TANGENT"];
    for i in 0..48u32 {
        let pixel = i % 3 != 2;
        let mut c = Container::new(if pixel { 0 } else { 1 }).with_binding(3, i % 4, i % 4 + i % 3).with_binding(2, 0, 0);
        let mut inputs: Vec<Sig> = vec![("SV_Position", 0, 15, 0, 1)];
        for r in 1..1 + i % 5 {
            let mask = [3u8, 15, 7, 12][(i + r) as usize % 4];
            let used = mask & [0u8, 1, 3, 15][((i * r) % 4) as usize];
            inputs.push((SEMANTICS[(r % 4) as usize], r, mask, used, 0));
        }
        if !pixel && i % 2 == 0 {
            inputs.push(("SV_VertexID", 6, 1, 1, 6));
        }
        c.inputs = inputs;
        c.table_b = Some((0..i % 4).map(|k| [[0x300, 0x302, 0x401][(k % 3) as usize], k + i % 3, k, 0, ((k % 3) << 16) | ((8 + k % 4) << 8) | 4]).collect());
        out.push((format!("{}_Gen{i}", if pixel { "PS" } else { "VS" }), c.build()));
    }
    out
}

/// The invented listing the stand in decoder prints for a shader name: a pseudo random mix, from
/// a hash of the name, of the instruction forms the scripts look for (`S_SWAPPC`, branches,
/// vector reads and writes, `V_INTERP`, scalar table loads and register copies).
pub fn listing(name: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut next = |n: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 29;
        h % n
    };
    let mut lines = vec![format!("listing for {name}"), "0x00000000: signature 0".to_owned()];
    let mut pc: u64 = 4;
    let emit = |lines: &mut Vec<String>, pc: &mut u64, text: String| {
        lines.push(format!("0x{pc:08x}: {text}"));
        *pc += 4;
    };
    if next(5) != 0 {
        emit(&mut lines, &mut pc, "S_SWAPPC_B64 s[30:31], s[0:1]".to_owned());
    }
    for _ in 0..next(4) {
        emit(&mut lines, &mut pc, format!("S_MOV_B32 s{}, s{}", 16 + next(8), 6 + next(12)));
    }
    for _ in 0..1 + next(14) {
        let kind = next(10);
        let (a, b, c) = (next(24), next(24), next(24));
        let text = match kind {
            0 => format!("V_ADD_F32 v{a}, v{b}, v{c}"),
            1 => format!("V_FMAC_F32 v{a}, v{b}, v{c}"),
            2 => format!("BUFFER_LOAD_DWORDX{} v{a}, v{b}, s[0:3], 0 offen", 2 + next(3)),
            3 => format!("V_CMP_GT_F32 vcc, v{b}, v{c}"),
            4 => format!("S_CBRANCH_SCC0 0x{:08x}", pc + 4 * (1 + next(3))),
            5 => format!("V_INTERP_P1_F32 v{a}, v{b}, {}, {}", next(4), next(4)),
            6 => format!("S_LOAD_DWORDX{} s[4:7], s{}, null ; offset={} ", 2 << next(3), 8 + next(4), 4 * next(4)),
            7 => format!("V_MOV_B32 v{a}, v{b}"),
            8 => format!("V_INTERP_MOV_F32 v{a}, p0, {}, {}", next(3), next(3)),
            _ => format!("exp pos0 v{a}, v{b}, v{c}, v{a} done"),
        };
        emit(&mut lines, &mut pc, text);
    }
    emit(&mut lines, &mut pc, "S_ENDPGM".to_owned());
    lines.iter().map(|l| format!("{l}\n")).collect()
}
