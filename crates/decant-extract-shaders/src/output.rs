//! The files the extractor writes, byte for byte as `extract_shaders.py` writes them.
//!
//! `shaders_gpu.pack`: `XSGC`, version 1, shader count, then per shader a u16 name length, the
//! Latin 1 name, a u8 stage code, the 16 byte sub header, a u32 word count and the words.
//! `shaders.csv`: semicolon separated, CRLF line ends, as Python's `csv` module writes them.

use std::io::{self, Write};

use crate::fssm::Code;

/// Stage codes in the pack. Anything else is 255.
pub fn stage_code(stage: &str) -> u8 {
    match stage {
        "pixel" => 0,
        "vertex" => 1,
        "geometry" => 2,
        "hull" => 3,
        "domain" => 4,
        "compute" => 5,
        "mesh" => 13,
        _ => 255,
    }
}

pub const CSV_HEADER: [&str; 8] = ["nom", "type", "position", "dxbc_octets", "fssm_octets", "code_mots", "remplissage_mots", "sous_entete"];

pub fn pack_header(out: &mut impl Write, count: u32) -> io::Result<()> {
    out.write_all(b"XSGC")?;
    out.write_all(&1u32.to_le_bytes())?;
    out.write_all(&count.to_le_bytes())
}

/// One pack record. Names are Latin 1, as the archive stores them.
pub fn pack_record(out: &mut impl Write, name: &str, stage: &str, code: &Code) -> io::Result<()> {
    let latin1: Vec<u8> = name.chars().map(|c| u8::try_from(u32::from(c)).unwrap_or(b'?')).collect();
    out.write_all(&(latin1.len() as u16).to_le_bytes())?;
    out.write_all(&latin1)?;
    out.write_all(&[stage_code(stage)])?;
    out.write_all(&code.sub_header)?;
    out.write_all(&(code.words.len() as u32).to_le_bytes())?;
    for word in &code.words {
        out.write_all(&word.to_le_bytes())?;
    }
    Ok(())
}

/// A field as Python's `csv` writes it with `;`: quoted only when it holds `;`, `"`, CR or LF.
fn field(text: &str) -> String {
    if text.contains([';', '"', '\r', '\n']) { format!("\"{}\"", text.replace('"', "\"\"")) } else { text.to_owned() }
}

pub fn csv_row(out: &mut impl Write, fields: &[String]) -> io::Result<()> {
    let line: Vec<String> = fields.iter().map(|f| field(f)).collect();
    out.write_all(line.join(";").as_bytes())?;
    out.write_all(b"\r\n")
}

/// Lowercase hex, as Python's `bytes.hex()`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_rows_match_the_python_layout() {
        let code = Code { sub_header: [0xAB; 16], words: vec![1, 2], padding: 3 };
        let mut pack = Vec::new();
        pack_header(&mut pack, 1).unwrap();
        pack_record(&mut pack, "PS_É", "pixel", &code).unwrap();
        let mut expected = b"XSGC\x01\0\0\0\x01\0\0\0\x04\0PS_\xc9\x00".to_vec();
        expected.extend([0xAB; 16]);
        expected.extend([2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(pack, expected);
        let mut csv = Vec::new();
        csv_row(&mut csv, &["a;b".into(), "x\"y".into(), "plain".into()]).unwrap();
        assert_eq!(csv, b"\"a;b\";\"x\"\"y\";plain\r\n");
        assert_eq!(stage_code("library"), 255);
    }
}
