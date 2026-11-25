//! Invented inputs for the tests and for comparing the Rust tools with the Python scripts
//! (`examples/synthetic_inputs.rs`): SPIR-V modules assembled word by word, measurement folders
//! made of them, and bench images, correct ones and deliberately wrong ones. Nothing here comes
//! from a game file or from the translator.

use std::path::Path;

/// A SPIR-V module under construction: a header, then instructions in the order added.
#[derive(Clone, Debug, Default)]
pub struct Module {
    body: Vec<u32>,
}

impl Module {
    pub fn new() -> Module {
        Module::default()
    }

    /// Any instruction: its opcode and operands, the word count computed.
    pub fn op(mut self, op: u32, operands: &[u32]) -> Module {
        self.body.push(((operands.len() as u32 + 1) << 16) | op);
        self.body.extend(operands);
        self
    }

    pub fn capability(self, c: u32) -> Module {
        self.op(17, &[c])
    }

    /// `OpDecorate target Binding value`.
    pub fn binding(self, target: u32, value: u32) -> Module {
        self.op(71, &[target, 33, value])
    }

    /// `OpFunction`, which ends the declarations the capability count reads.
    pub fn function(self) -> Module {
        self.op(54, &[1, 2, 0, 3])
    }

    /// `OpName target "text"`, the string zero ended and padded to whole words.
    pub fn name(self, target: u32, text: &str) -> Module {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        let mut operands = vec![target];
        operands.extend(bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)));
        self.op(5, &operands)
    }

    pub fn words(&self) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0500, 0, 100, 0];
        words.extend(&self.body);
        words
    }

    pub fn bytes(&self) -> Vec<u8> {
        self.words().iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A module with one resource of every kind the reflection tells apart, push constants, an
    /// input and an output.
    pub fn reflect_sample() -> Module {
        let (uc, uniform, input, output, push) = (0, 2, 1, 3, 9);
        Module::new()
            .capability(1)
            .capability(1)
            .capability(5347)
            .name(10, "tex")
            .op(71, &[10, 34, 1])
            .op(71, &[10, 33, 3])
            .op(71, &[18, 2])
            .op(71, &[21, 3])
            .op(71, &[36, 30, 2])
            .op(71, &[38, 11])
            .op(22, &[2, 32])
            .op(21, &[3, 32, 1])
            .op(21, &[4, 32, 0])
            .op(25, &[5, 2, 1, 0, 0, 0, 1, 0])
            .op(27, &[6, 5])
            .op(43, &[4, 7, 4])
            .op(28, &[8, 6, 7])
            .op(32, &[9, uc, 8])
            .op(59, &[9, 10, uc])
            .op(25, &[11, 4, 1, 0, 1, 0, 2, 4])
            .op(29, &[12, 11])
            .op(32, &[13, uc, 12])
            .op(59, &[13, 14, uc])
            .op(25, &[15, 3, 5, 0, 0, 0, 1, 0])
            .op(32, &[16, uc, 15])
            .op(59, &[16, 17, uc])
            .op(30, &[18, 2])
            .op(32, &[19, uniform, 18])
            .op(59, &[19, 20, uniform])
            .op(30, &[21, 4, 4])
            .op(32, &[22, uniform, 21])
            .op(59, &[22, 23, uniform])
            .op(26, &[24])
            .op(32, &[25, uc, 24])
            .op(59, &[25, 26, uc])
            .op(27, &[27, 99])
            .op(32, &[28, uc, 27])
            .op(59, &[28, 29, uc])
            .op(32, &[30, uc, 2])
            .op(59, &[30, 31, uc])
            .op(30, &[32, 2])
            .op(32, &[33, push, 32])
            .op(59, &[33, 34, push])
            .op(32, &[35, input, 2])
            .op(59, &[35, 36, input])
            .op(32, &[37, output, 2])
            .op(59, &[37, 38, output])
            .function()
    }
}

/// The modules of the synthetic measurement: some variety of capabilities and bindings, chosen
/// from the index, plus the reflection sample.
pub fn measurement_module(batch: usize, index: usize) -> Module {
    let n = batch * 7 + index;
    let mut m = Module::new().capability(1);
    for (k, cap) in [5347u32, 5301, 4427, 11, 64, 5302, 6].iter().enumerate() {
        if (n >> k) & 1 == 1 || (n + k).is_multiple_of(5) {
            m = m.capability(*cap);
        }
    }
    for b in 0..n % 4 {
        m = m.binding(40 + b as u32, ((n * 3 + b) % 9) as u32);
    }
    if n.is_multiple_of(6) { Module::reflect_sample() } else { m.function().capability(4427) }
}

/// Writes the synthetic measurement under `folder`: three batch folders and a stray file.
pub fn write_measurement(folder: &Path) -> std::io::Result<()> {
    for batch in 0..3 {
        let dir = folder.join(format!("{batch}-{:05}", batch * 48));
        std::fs::create_dir_all(&dir)?;
        for index in 0..(5 + batch * 3) {
            std::fs::write(dir.join(format!("{index}.spv")), measurement_module(batch, index).bytes())?;
        }
        std::fs::write(dir.join("batch.tsv"), b"B\t0\n")?;
    }
    std::fs::write(folder.join("results.jsonl"), b"{}\n")
}

/// The interpolated colour of the bench triangle at a pixel, times `factor`.
fn blit_want(x: usize, y: usize, c: usize, factor: [f64; 4]) -> f64 {
    const COLORS: [[f64; 4]; 3] = [[1.00, 0.00, 0.25, 1.00], [0.00, 1.00, 0.50, 0.50], [0.50, 0.25, 1.00, 0.75]];
    let ndc_x = -1.0 + 2.0 * (x as f64 + 0.5) / 64.0;
    let ndc_y = -1.0 + 2.0 * (y as f64 + 0.5) / 64.0;
    let (b1, b2) = ((ndc_x + 1.0) / 4.0, (ndc_y + 1.0) / 4.0);
    let b0 = 1.0 - b1 - b2;
    (b0 * COLORS[0][c] + b1 * COLORS[1][c] + b2 * COLORS[2][c]) * factor[c]
}

/// A blit image as the console would write it for `factor`: each value rounded to a half float
/// toward zero. With `undrawn`, one corner pixel is left at the clear value -1.
pub fn blit_image(factor: [f64; 4], undrawn: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 * 64 * 16);
    for y in 0..64 {
        for x in 0..64 {
            for c in 0..4 {
                let v = if undrawn && x == 0 && y == 0 { -1.0 } else { crate::pyfmt::half_rtz(blit_want(x, y, c, factor)).unwrap() };
                out.extend((v as f32).to_le_bytes());
            }
        }
    }
    out
}

/// A texture image: the nearest texel at the coordinate of the chosen mode, as half floats toward
/// zero, plus `offset` on every value.
pub fn texture_image(linked: bool, offset: f64) -> Vec<u8> {
    const COLORS: [[f64; 4]; 3] = [[1.00, 0.00, 0.25, 1.00], [0.00, 1.00, 0.50, 0.50], [0.50, 0.25, 1.00, 0.75]];
    let mut out = Vec::with_capacity(64 * 64 * 16);
    for y in 0..64 {
        for x in 0..64 {
            let b1 = (2.0 * (x as f64 + 0.5) / 64.0) / 4.0;
            let b2 = (2.0 * (y as f64 + 0.5) / 64.0) / 4.0;
            let b0 = 1.0 - b1 - b2;
            let (u, v) = if linked {
                (2.0 * b1, 2.0 * b2)
            } else {
                (b0 * COLORS[0][0] + b1 * COLORS[1][0] + b2 * COLORS[2][0], b0 * COLORS[0][1] + b1 * COLORS[1][1] + b2 * COLORS[2][1])
            };
            let (i, j) = (((u * 4.0).trunc() as i64).clamp(0, 3), ((v * 4.0).trunc() as i64).clamp(0, 3));
            let texel = [(40 * i + 10) as f64 / 255.0, (60 * j + 20) as f64 / 255.0, (15 * (i + 4 * j)) as f64 / 255.0, 1.0];
            for w in texel {
                let value = crate::pyfmt::half_rtz(f64::from(w as f32)).unwrap() + offset;
                out.extend((value as f32).to_le_bytes());
            }
        }
    }
    out
}
