//! Reading `shaders_gpu.pack` back, for the shader research tools (`decant_shader_batch.py`,
//! `decant_shader_one.py`). The layout is the one [`crate::output`] writes.
//!
//! The tools copy whole records into smaller packs, so a record keeps its raw bytes: name length,
//! name, stage code, sub header, word count and words, exactly as they are in the file.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The 12 byte header: signature, version and record count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub signature: [u8; 4],
    pub version: u32,
    pub count: u32,
}

impl Header {
    /// `XSGC` version 1, the only layout the tools accept.
    pub fn is_current(&self) -> bool {
        self.signature == *b"XSGC" && self.version == 1
    }
}

/// One record of the pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    /// Position of the record in the pack, from 0.
    pub index: usize,
    /// The name, read as Latin 1.
    pub name: String,
    /// The stage code of [`crate::output::stage_code`].
    pub stage: u8,
    /// The whole record as stored, ready to copy into another pack.
    pub bytes: Vec<u8>,
}

/// Bytes between the name and the words: stage code, sub header, word count.
const TAIL: usize = 21;

/// Reads the header and the records for which `keep(index, name, stage)` holds. The code of the
/// other records is skipped without reading it.
pub fn read(path: &Path, mut keep: impl FnMut(usize, &str, u8) -> bool) -> io::Result<(Header, Vec<Record>)> {
    let mut file = BufReader::new(File::open(path)?);
    let mut head = [0u8; 12];
    file.read_exact(&mut head)?;
    let header = Header {
        signature: [head[0], head[1], head[2], head[3]],
        version: u32::from_le_bytes([head[4], head[5], head[6], head[7]]),
        count: u32::from_le_bytes([head[8], head[9], head[10], head[11]]),
    };
    let mut records = Vec::new();
    for index in 0..header.count as usize {
        let mut length = [0u8; 2];
        file.read_exact(&mut length)?;
        let mut name = vec![0u8; usize::from(u16::from_le_bytes(length))];
        file.read_exact(&mut name)?;
        let mut tail = [0u8; TAIL];
        file.read_exact(&mut tail)?;
        let words = u32::from_le_bytes([tail[17], tail[18], tail[19], tail[20]]) as usize;
        let text = latin1(&name);
        if keep(index, &text, tail[0]) {
            let mut bytes = Vec::with_capacity(2 + name.len() + TAIL + words * 4);
            bytes.extend_from_slice(&length);
            bytes.extend_from_slice(&name);
            bytes.extend_from_slice(&tail);
            let start = bytes.len();
            bytes.resize(start + words * 4, 0);
            file.read_exact(&mut bytes[start..])?;
            records.push(Record { index, name: text, stage: tail[0], bytes });
        } else {
            file.seek(SeekFrom::Current(words as i64 * 4))?;
        }
    }
    Ok((header, records))
}

/// A pack made of the given raw records, as the tools write it: header, then the records in order.
pub fn write<'a>(records: impl ExactSizeIterator<Item = &'a [u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    crate::output::pack_header(&mut out, records.len() as u32).expect("writing to memory");
    for record in records {
        out.extend_from_slice(record);
    }
    out
}

/// Latin 1 bytes as text, one character per byte.
pub fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fssm::Code;
    use crate::output::{pack_header, pack_record};

    #[test]
    fn records_read_back_as_written() {
        let dir = std::env::temp_dir().join(format!("extract-shaders-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.pack");
        let mut pack = Vec::new();
        pack_header(&mut pack, 3).unwrap();
        let one = Code { sub_header: [7; 16], words: vec![1, 2, 3], padding: 0 };
        let two = Code { sub_header: [9; 16], words: vec![], padding: 0 };
        pack_record(&mut pack, "PS_É", "pixel", &one).unwrap();
        let second = pack.len();
        pack_record(&mut pack, "VS", "vertex", &two).unwrap();
        let third = pack.len();
        pack_record(&mut pack, "CS", "compute", &one).unwrap();
        std::fs::write(&path, &pack).unwrap();

        let (header, records) = read(&path, |_, name, stage| name != "VS" && stage != 3).unwrap();
        assert!(header.is_current());
        assert_eq!(header.count, 3);
        assert_eq!(records.len(), 2);
        assert_eq!((records[0].index, records[0].name.as_str(), records[0].stage), (0, "PS_É", 0));
        assert_eq!(records[0].bytes, pack[12..second]);
        assert_eq!((records[1].index, records[1].stage), (2, 5));
        assert_eq!(records[1].bytes, pack[third..]);

        let copy = write(records.iter().map(|r| r.bytes.as_slice()));
        assert_eq!(&copy[..12], b"XSGC\x01\0\0\0\x02\0\0\0");
        assert_eq!(copy.len(), 12 + records[0].bytes.len() + records[1].bytes.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn truncated_packs_are_errors() {
        let dir = std::env::temp_dir().join(format!("extract-shaders-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("b.pack");
        std::fs::write(&path, b"XSGC\x01\0\0\0\x01\0\0\0\x02\0A").unwrap();
        assert!(read(&path, |_, _, _| true).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
