//! Builders for invented test inputs.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;

pub fn zlib(data: &[u8]) -> Vec<u8> {
    let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
    z.write_all(data).unwrap();
    z.finish().unwrap()
}

/// A DXBC container with one DXIL part: version word, then an XDAT table whose XBCH entry holds
/// `gpu` as a zlib stream.
pub fn dxbc(version: u32, gpu: &[u8]) -> Vec<u8> {
    let packed = zlib(gpu);
    let mut body = version.to_le_bytes().to_vec();
    body.extend([0u8; 4]);
    let x = body.len();
    body.extend(b"XDAT");
    body.extend(1u32.to_le_bytes());
    body.extend([0u8; 4]);
    let part = x + 12 + 12 + 8;
    body.extend(0x0Du32.to_le_bytes());
    body.extend((part as u32).to_le_bytes());
    body.extend(((24 + packed.len()) as u32).to_le_bytes());
    body.extend([0u8; 8]);
    body.extend(b"XBCH");
    body.extend([0u8; 4]);
    body.extend((packed.len() as u64).to_le_bytes());
    body.extend([0u8; 8]);
    body.extend(&packed);
    let mut blob = b"DXBC".to_vec();
    blob.extend([0u8; 24]);
    blob.extend(1u32.to_le_bytes());
    blob.extend(36u32.to_le_bytes());
    blob.extend(b"DXIL");
    blob.extend((body.len() as u32).to_le_bytes());
    blob.extend(body);
    blob
}
