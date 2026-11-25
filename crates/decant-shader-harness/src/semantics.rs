//! The SEMANTIC test of SPIR-V operations (binary `decant_shader_semantics`): checks that the
//! operations our translated modules rely on COMPUTE the right value on the driver, not only that
//! they pass spirv-val.
//!
//! Each case is a small fragment shader WRITTEN HERE (from scratch, no third party source) that
//! computes a known integer result with ONE targeted operation (bit field, shift, bitwise,
//! arithmetic, division and modulo, conversion, selection, bit count and reverse) on immediate
//! constants, then writes its four bytes to the four channels. The bench `decant_shader_run` runs the
//! shader, the tool reads the f32 output back and compares it with the value computed here with
//! 32 bit semantics. Nothing here comes from title files.

/// The operation of one case: a SPIR-V instruction, or one of the two composite cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Spirv(&'static str),
    /// u32 to float and back.
    ConvertRoundTrip,
    /// `a == b ? t : e`.
    SelectEqual,
}

/// One case: name, operation, signed result type, constant operands (as u32 bit patterns) and the
/// expected result.
pub struct Case {
    pub name: &'static str,
    pub op: Op,
    pub signed: bool,
    pub operands: &'static [u32],
    pub expected: u32,
}

const A: u32 = 0xABCD_1234;
const B: u32 = 0x0F0F_0F0F;

fn bitfield_sextract(base: u32, offset: u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let v = (base >> offset) & ((1 << count) - 1);
    if v & (1 << (count - 1)) != 0 { v.wrapping_sub(1 << count) } else { v }
}

fn bitfield_insert(base: u32, insert: u32, offset: u32, count: u32) -> u32 {
    let mask = if count == 0 { 0 } else { ((1u32 << count) - 1) << offset };
    (base & !mask) | ((insert << offset) & mask)
}

/// Extend the list to add a case.
pub fn cases() -> Vec<Case> {
    use Op::Spirv as S;
    let c = |name, op, signed, operands, expected| Case { name, op, signed, operands, expected };
    vec![
        c("IAdd", S("OpIAdd"), false, &[A, B], A.wrapping_add(B)),
        c("ISub", S("OpISub"), false, &[A, B], A.wrapping_sub(B)),
        c("IMul", S("OpIMul"), false, &[0x0001_2345, 0x37], 0x0001_2345u32.wrapping_mul(0x37)),
        c("UDiv", S("OpUDiv"), false, &[A, 0x101], A / 0x101),
        c("UMod", S("OpUMod"), false, &[A, 0x101], A % 0x101),
        // -1000 / 7 rounds towards 0.
        c("SDiv_neg", S("OpSDiv"), true, &[(-1000i32) as u32, 7], (-142i32) as u32),
        c("ShiftLeft", S("OpShiftLeftLogical"), false, &[0x0000_0F0F, 5], 0x0000_0F0F << 5),
        c("ShiftRightLog", S("OpShiftRightLogical"), false, &[0xF000_0000, 4], 0xF000_0000 >> 4),
        c("ShiftRightArith", S("OpShiftRightArithmetic"), true, &[0xF000_0000, 4], ((0xF000_0000u32 as i32) >> 4) as u32),
        c("BitwiseAnd", S("OpBitwiseAnd"), false, &[A, B], A & B),
        c("BitwiseOr", S("OpBitwiseOr"), false, &[A, B], A | B),
        c("BitwiseXor", S("OpBitwiseXor"), false, &[A, B], A ^ B),
        c("BitCount", S("OpBitCount"), false, &[A], A.count_ones()),
        c("BitReverse", S("OpBitReverse"), false, &[A], A.reverse_bits()),
        c("BitFieldUExtract", S("OpBitFieldUExtract"), false, &[A, 4, 8], (A >> 4) & 0xFF),
        c("BitFieldSExtract", S("OpBitFieldSExtract"), true, &[0x0000_00F0, 4, 4], bitfield_sextract(0xF0, 4, 4)),
        c("BitFieldInsert", S("OpBitFieldInsert"), false, &[A, 0xFF, 8, 8], bitfield_insert(A, 0xFF, 8, 8)),
        c("ConvRoundTrip", Op::ConvertRoundTrip, false, &[1_234_567], 1_234_567),
        c("Select", Op::SelectEqual, false, &[7, 7, 111, 222], 111),
    ]
}

/// Declares each constant once, in the order of first use.
struct Constants<'a> {
    lines: &'a mut Vec<String>,
    names: Vec<((bool, u32), String)>,
}

impl Constants<'_> {
    fn get(&mut self, signed: bool, value: u32) -> String {
        if let Some((_, name)) = self.names.iter().find(|(k, _)| *k == (signed, value)) {
            return name.clone();
        }
        let n = self.names.len();
        let (name, line) = if signed {
            (format!("%ci{n}"), format!("%ci{n} = OpConstant %i32 {}", value as i32))
        } else {
            (format!("%cu{n}"), format!("%cu{n} = OpConstant %u32 {value}"))
        };
        self.lines.push(line);
        self.names.push(((signed, value), name.clone()));
        name
    }
}

/// The SPIR-V assembly of a complete fragment module for `case`. The four bytes of the u32 result
/// go to the four channels (each 0 to 255, exact in f32), so any 32 bit pattern comes back,
/// negative signed results included.
pub fn module_text(case: &Case) -> String {
    let mut lines: Vec<String> = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "OpEntryPoint Fragment %main \"main\" %color",
        "OpExecutionMode %main OriginUpperLeft",
        "OpDecorate %color Location 0",
        "%void = OpTypeVoid",
        "%fnty = OpTypeFunction %void",
        "%float = OpTypeFloat 32",
        "%u32 = OpTypeInt 32 0",
        "%i32 = OpTypeInt 32 1",
        "%bool = OpTypeBool",
        "%v4f = OpTypeVector %float 4",
        "%ptr = OpTypePointer Output %v4f",
        "%color = OpVariable %ptr Output",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut body = vec!["%main = OpFunction %void None %fnty".to_owned(), "%entry = OpLabel".to_owned()];
    let mut k = Constants { lines: &mut lines, names: Vec::new() };
    let o = case.operands;
    let mut signed = case.signed;
    let ty = if signed { "%i32" } else { "%u32" };
    match case.op {
        Op::ConvertRoundTrip => {
            body.push(format!("%rf = OpConvertUToF %float {}", k.get(false, o[0])));
            body.push("%r = OpConvertFToU %u32 %rf".into());
            signed = false;
        }
        Op::SelectEqual => {
            let (a, b) = (k.get(false, o[0]), k.get(false, o[1]));
            body.push(format!("%cond = OpIEqual %bool {a} {b}"));
            let (t, e) = (k.get(false, o[2]), k.get(false, o[3]));
            body.push(format!("%r = OpSelect %u32 %cond {t} {e}"));
            signed = false;
        }
        Op::Spirv(op @ ("OpBitFieldUExtract" | "OpBitFieldSExtract")) => {
            let (base, offset, count) = (k.get(signed, o[0]), k.get(false, o[1]), k.get(false, o[2]));
            body.push(format!("%r = {op} {ty} {base} {offset} {count}"));
        }
        Op::Spirv(op @ "OpBitFieldInsert") => {
            let (base, insert, offset, count) = (k.get(signed, o[0]), k.get(signed, o[1]), k.get(false, o[2]), k.get(false, o[3]));
            body.push(format!("%r = {op} {ty} {base} {insert} {offset} {count}"));
        }
        Op::Spirv(op @ ("OpBitCount" | "OpBitReverse")) => {
            body.push(format!("%r = {op} {ty} {}", k.get(signed, o[0])));
        }
        Op::Spirv(op @ ("OpShiftLeftLogical" | "OpShiftRightLogical" | "OpShiftRightArithmetic")) => {
            let (value, shift) = (k.get(signed, o[0]), k.get(false, o[1]));
            body.push(format!("%r = {op} {ty} {value} {shift}"));
        }
        Op::Spirv(op) => {
            let (a, b) = (k.get(signed, o[0]), k.get(signed, o[1]));
            body.push(format!("%r = {op} {ty} {a} {b}"));
        }
    }
    let mut result = "%r";
    if signed {
        body.push("%ru = OpBitcast %u32 %r".into());
        result = "%ru";
    }
    for (n, offset) in [0u32, 8, 16, 24].into_iter().enumerate() {
        let (offset, width) = (k.get(false, offset), k.get(false, 8));
        body.push(format!("%b{n} = OpBitFieldUExtract %u32 {result} {offset} {width}"));
        body.push(format!("%fb{n} = OpConvertUToF %float %b{n}"));
    }
    body.push("%res = OpCompositeConstruct %v4f %fb0 %fb1 %fb2 %fb3".into());
    body.extend(["OpStore %color %res", "OpReturn", "OpFunctionEnd"].map(str::to_owned));
    lines.extend(body);
    lines.join("\n") + "\n"
}

/// The four bytes of the first drawn pixel (one not left at the clear value -1), as a u32.
pub fn read_pixel(image: &[u8]) -> Option<u32> {
    let values: Vec<f32> = image.as_chunks::<4>().0.iter().map(|w| f32::from_le_bytes(*w)).collect();
    let pixel = values.as_chunks::<4>().0.iter().find(|p| **p != [-1.0, -1.0, -1.0, -1.0])?;
    Some(pixel.iter().enumerate().map(|(k, v)| (v.round_ties_even() as i64 as u32) << (8 * k)).fold(0, u32::wrapping_add))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_values_follow_32_bit_semantics() {
        let get = |name| cases().into_iter().find(|c| c.name == name).unwrap().expected;
        assert_eq!(get("SDiv_neg"), 0xFFFF_FF72);
        assert_eq!(get("ShiftRightArith"), 0xFF00_0000);
        assert_eq!(get("BitFieldSExtract"), 0xFFFF_FFFF);
        assert_eq!(get("BitFieldInsert"), 0xABCD_FF34);
        assert_eq!(get("BitCount"), 15);
        assert_eq!(get("BitReverse"), 0x2C48_B3D5);
        assert_eq!(get("IAdd"), 0xBADC_2143);
    }

    #[test]
    fn the_module_declares_each_constant_once() {
        let cases = cases();
        let insert = cases.iter().find(|c| c.name == "BitFieldInsert").unwrap();
        let text = module_text(insert);
        assert!(text.contains("%cu0 = OpConstant %u32 2882343476\n%cu1 = OpConstant %u32 255\n%cu2 = OpConstant %u32 8\n%cu3 = OpConstant %u32 0\n"));
        assert!(text.contains("%r = OpBitFieldInsert %u32 %cu0 %cu1 %cu2 %cu2\n"));
        assert!(text.contains("%b0 = OpBitFieldUExtract %u32 %r %cu3 %cu2\n"));
        let sdiv = module_text(cases.iter().find(|c| c.name == "SDiv_neg").unwrap());
        assert!(sdiv.contains("%ci0 = OpConstant %i32 -1000\n%ci1 = OpConstant %i32 7\n"));
        assert!(sdiv.contains("%r = OpSDiv %i32 %ci0 %ci1\n%ru = OpBitcast %u32 %r\n"));
        assert!(sdiv.ends_with("OpStore %color %res\nOpReturn\nOpFunctionEnd\n"));
    }

    #[test]
    fn the_first_drawn_pixel_gives_the_result() {
        let mut image = Vec::new();
        for v in [-1.0f32, -1.0, -1.0, -1.0, 0x34 as f32, 0x12 as f32, 0xCD as f32, 0xAB as f32] {
            image.extend(v.to_le_bytes());
        }
        assert_eq!(read_pixel(&image), Some(0xABCD_1234));
        assert_eq!(read_pixel(&(-1.0f32).to_le_bytes().repeat(8)), None);
    }
}
