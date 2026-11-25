//! Minimal PE pre-flight for a staged title executable: `MZ`, then `PE\0\0` at `e_lfanew`, AMD64
//! machine, PE32+ optional header, Windows GUI or CUI subsystem. Header bytes only: sections and
//! imports are not parsed, and nothing here concludes about encryption or DRM (an encrypted image
//! passes and is still unrunnable).

use std::io::Read;
use std::path::Path;

/// The headers live in the first pages of the image. 64 KiB covers every `e_lfanew` and optional
/// header a normal linker emits.
const HEADER_READ: u64 = 64 * 1024;

fn rd16(d: &[u8], at: usize) -> u32 {
    u32::from(u16::from_le_bytes([d[at], d[at + 1]]))
}

fn rd32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

/// C `printf("%#x")`: no prefix for zero.
fn hex(v: u32) -> String {
    if v == 0 { "0".to_owned() } else { format!("{v:#x}") }
}

/// The check on an image prefix in memory (the header area).
pub fn check_pe_image_from_memory(data: &[u8]) -> Result<(), String> {
    let size = data.len();
    if size < 0x40 {
        return Err(format!("truncated: {size} bytes is too small for a DOS header"));
    }
    if &data[..2] != b"MZ" {
        return Err("no MZ signature".to_owned());
    }
    let pe = rd32(data, 0x3c) as usize;
    if pe < 0x40 || pe + 24 > size {
        return Err("PE header offset is out of range".to_owned());
    }
    if &data[pe..pe + 4] != b"PE\0\0" {
        return Err("no PE\\0\\0 signature at e_lfanew".to_owned());
    }
    // COFF file header: Machine at +4, SizeOfOptionalHeader at +20.
    let machine = rd16(data, pe + 4);
    if machine != 0x8664 {
        return Err(format!("machine {} is not AMD64 (0x8664)", hex(machine)));
    }
    // Optional header: magic at +0, Subsystem at +68 (PE32+ layout). The Subsystem field spans
    // bytes 68 and 69, hence the 70 byte floor.
    let opt_size = rd16(data, pe + 20) as usize;
    if opt_size < 70 || pe + 24 + opt_size > size {
        return Err("optional header is truncated".to_owned());
    }
    let opt = pe + 24;
    let magic = rd16(data, opt);
    if magic != 0x20b {
        return Err(format!("optional header magic {} is not PE32+ (0x20b)", hex(magic)));
    }
    // 2 = Windows GUI (games), 3 = Windows CUI (console tools).
    let subsystem = rd16(data, opt + 68);
    if subsystem != 2 && subsystem != 3 {
        return Err(format!("subsystem {} is not Windows GUI/CUI", hex(subsystem)));
    }
    Ok(())
}

/// The check on the first 64 KiB of a file. Errors name the file.
pub fn check_pe_image(path: &Path) -> Result<(), String> {
    let shown = path.display();
    let file = std::fs::File::open(path).map_err(|_| format!("cannot open {shown}"))?;
    let mut head = Vec::new();
    match file.take(HEADER_READ).read_to_end(&mut head) {
        Ok(n) if n > 0 => {}
        _ => return Err(format!("cannot read {shown}")),
    }
    check_pe_image_from_memory(&head).map_err(|reason| format!("{shown}: {reason}"))
}
