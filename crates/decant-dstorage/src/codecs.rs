//! The decoders of the CPU path (the former `codecs.cpp`): zlib, GDeflate tile streams and BCPack.
//!
//! zlib is decoded here with `flate2`, so it needs nothing external. GDeflate needs libdeflate
//! with the NVIDIA additions (see `libdeflate.rs`): without it GDeflate requests on the CPU path
//! answer `E_NOTIMPL`. BCPack is decoded by the `bcpack` crate, linked in.
#![allow(unsafe_code)]

#[cfg(decant_gdeflate)]
use crate::libdeflate::TILE_BYTES;
use crate::log::storage_log;
use crate::decant::{E_NOTIMPL, ERROR_INVALID_DATA, HRESULT, S_OK, codec, hresult_from_win32};

/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: the data does not decode to the size asked.
const BAD: HRESULT = hresult_from_win32(ERROR_INVALID_DATA);

/// A BCPack stream of `mode` (1 = BC1 ... 7 = BC7, as `DSTORAGE_BCPACK_MODE`) to exactly
/// `output.len()` bytes of blocks. BC2 and unknown modes are not implemented.
fn bcpack(mode: u32, source: &[u8], output: &mut [u8]) -> HRESULT {
    let Some(format) = decant_bcpack::Format::from_mode(mode) else {
        decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("BCPack mode {mode}"));
        return E_NOTIMPL;
    };
    match decant_bcpack::decode(format, source, output) {
        Ok(report) => {
            if report.inconsistent > 0 {
                storage_log(&format!("BCPack mode={mode} input={} output={} inconsistent blocks={}
", source.len(), output.len(), report.inconsistent));
            }
            S_OK
        }
        Err(error) => {
            storage_log(&format!("BCPack mode={mode} input={} output={} failed: {error}
", source.len(), output.len()));
            BAD
        }
    }
}

/// zlib stream to exactly `output.len()` bytes, the whole input consumed.
fn zlib(source: &[u8], output: &mut [u8]) -> HRESULT {
    thread_local! {
        static DECOMPRESS: std::cell::RefCell<flate2::Decompress> = std::cell::RefCell::new(flate2::Decompress::new(true));
    }
    DECOMPRESS.with(|cell| {
        let mut decoder = cell.borrow_mut();
        decoder.reset(true);
        let finished = matches!(decoder.decompress(source, output, flate2::FlushDecompress::Finish), Ok(flate2::Status::StreamEnd));
        if finished && decoder.total_in() == source.len() as u64 && decoder.total_out() == output.len() as u64 { S_OK } else { BAD }
    })
}

/// A little endian `u32` at `at`.
#[cfg(decant_gdeflate)]
fn word(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

#[cfg(decant_gdeflate)]
fn tiles(data: &[u8], out: &mut [u8]) -> HRESULT {
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    // Keep small requests on the caller and cap the number of threads per request.
    let workers = cores.min(8).min((out.len() / (4 * TILE_BYTES)).max(1));
    tiles_with_workers(data, out, workers)
}

/// Microsoft tile stream: entry zero is the last compressed tile length, subsequent entries are
/// offsets relative to the end of the table.
#[cfg(decant_gdeflate)]
fn tiles_with_workers(data: &[u8], out: &mut [u8], workers: usize) -> HRESULT {
    use crate::decant::E_OUTOFMEMORY;

    if data.len() < 12 || data[0] != 4 || data[1] != 251 {
        return BAD;
    }
    let count = usize::from(u16::from_le_bytes([data[2], data[3]]));
    let tail = (out.len() % TILE_BYTES) as u32;
    if count == 0 || count != out.len().div_ceil(TILE_BYTES) || word(data, 4) != (1 | (tail << 2)) {
        return BAD;
    }
    let header = 8 + 4 * count;
    if data.len() <= header {
        return BAD;
    }
    let payload = &data[header..];
    let mut pages = Vec::with_capacity(count);
    let mut start = 0;
    for i in 1..count {
        let end = word(data, 8 + 4 * i) as usize;
        if end <= start || end >= payload.len() {
            return BAD;
        }
        pages.push(&payload[start..end]);
        start = end;
    }
    if payload.len() - start != word(data, 8) as usize {
        return BAD;
    }
    pages.push(&payload[start..]);

    let workers = workers.min(count);
    if workers == 1 {
        return decode_pages(&pages, out);
    }
    let pages = pages.as_slice();
    std::thread::scope(move |scope| {
        let mut remaining = pages;
        let mut out = out;
        let mut handles = Vec::with_capacity(workers - 1);
        for workers_left in (2..=workers).rev() {
            let count = remaining.len().div_ceil(workers_left);
            let batch;
            let target;
            (batch, remaining) = remaining.split_at(count);
            (target, out) = out.split_at_mut(count * TILE_BYTES);
            let Ok(handle) = std::thread::Builder::new().spawn_scoped(scope, move || decode_pages(batch, target)) else {
                return E_OUTOFMEMORY;
            };
            handles.push(handle);
        }
        let last = decode_pages(remaining, out);
        let mut result = S_OK;
        for handle in handles {
            let status = handle.join().expect("GDeflate worker panicked");
            if result == S_OK {
                result = status;
            }
        }
        if result == S_OK { last } else { result }
    })
}

#[cfg(decant_gdeflate)]
struct CachedGdeflateDecoder(*mut std::ffi::c_void);

#[cfg(decant_gdeflate)]
impl Drop for CachedGdeflateDecoder {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: allocated by `libdeflate_alloc_gdeflate_decompressor`, freed once on thread exit.
            unsafe { crate::libdeflate::libdeflate_free_gdeflate_decompressor(self.0) };
        }
    }
}

#[cfg(decant_gdeflate)]
fn decode_pages(pages: &[&[u8]], out: &mut [u8]) -> HRESULT {
    use crate::libdeflate::{InPage, SUCCESS, libdeflate_alloc_gdeflate_decompressor, libdeflate_gdeflate_decompress};
    use crate::decant::E_OUTOFMEMORY;

    // The native library initializes its process-wide GDeflate dispatch pointer on first decode.
    static DISPATCH: std::sync::Once = std::sync::Once::new();
    thread_local! {
        static DECODER: std::cell::RefCell<Option<CachedGdeflateDecoder>> = const { std::cell::RefCell::new(None) };
    }

    DECODER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            // SAFETY: no precondition.
            let ptr = unsafe { libdeflate_alloc_gdeflate_decompressor() };
            if ptr.is_null() {
                return E_OUTOFMEMORY;
            }
            *slot = Some(CachedGdeflateDecoder(ptr));
        }
        let decoder = slot.as_ref().expect("initialized above").0;
        let mut result = S_OK;
        for (i, (source, target)) in pages.iter().zip(out.chunks_mut(TILE_BYTES)).enumerate() {
            let mut decode = || {
                let mut page = InPage { data: source.as_ptr().cast(), nbytes: source.len() };
                let mut written = 0;
                // SAFETY: the page describes the input slice and target is writable for its length.
                let status = unsafe { libdeflate_gdeflate_decompress(decoder, &mut page, 1, target.as_mut_ptr().cast(), target.len(), &mut written) };
                status == SUCCESS && written == target.len()
            };
            let mut first = None;
            if i == 0 {
                DISPATCH.call_once(|| first = Some(decode()));
            }
            if !first.unwrap_or_else(decode) {
                result = BAD;
                break;
            }
        }
        result
    })
}

/// Without libdeflate there is no GDeflate decoder.
#[cfg(not(decant_gdeflate))]
fn tiles(_data: &[u8], _out: &mut [u8]) -> HRESULT {
    decant_log::calls::not_implemented("GDeflate decode (built without libdeflate)");
    E_NOTIMPL
}

/// Decodes `size` bytes at `source` into exactly `capacity` bytes at `output`.
///
/// # Safety
/// `source` is readable for `size` bytes and `output` writable for `capacity` bytes, and they do not overlap.
pub unsafe fn decode(kind: u32, bc_mode: u32, source: *const u8, size: u32, output: *mut u8, capacity: u32) -> HRESULT {
    if kind != codec::GDEFLATE && kind != codec::ZLIB && kind != codec::BCPACK {
        return E_NOTIMPL;
    }
    // SAFETY: the caller guarantees both ranges, see above.
    let (input, out) = unsafe { (std::slice::from_raw_parts(source, size as usize), std::slice::from_raw_parts_mut(output, capacity as usize)) };
    match kind {
        codec::GDEFLATE => tiles(input, out),
        codec::ZLIB => zlib(input, out),
        _ => bcpack(bc_mode, input, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[cfg(decant_gdeflate)]
    #[test]
    fn gdeflate_tiles_are_byte_identical() {
        use crate::libdeflate::gdeflate_compress;

        let raw: Vec<u8> = (0..TILE_BYTES * 17 + 321)
            .map(|i| ((i / TILE_BYTES) ^ (i % 251)) as u8)
            .collect();
        let packed = gdeflate_compress(&raw, 1).expect("compresses");
        let mut out = vec![0; raw.len()];
        let mut serial = vec![0; raw.len()];
        for (source, target) in raw.chunks(TILE_BYTES).zip(serial.chunks_mut(TILE_BYTES)) {
            let single = gdeflate_compress(source, 1).expect("compresses one tile");
            assert_eq!(tiles(&single, target), S_OK);
        }
        assert_eq!(tiles(&packed, &mut out), S_OK);
        assert_eq!(out, serial);
        assert_eq!(out, raw);
        serial.fill(0);
        assert_eq!(tiles_with_workers(&packed, &mut serial, 1), S_OK);
        assert_eq!(serial, raw);
        for workers in [2, 4, 8] {
            out.fill(0);
            assert_eq!(tiles_with_workers(&packed, &mut out, workers), S_OK);
            assert_eq!(out, serial, "{workers} workers");
        }
    }

    #[cfg(decant_gdeflate)]
    #[test]
    #[ignore = "run explicitly in release mode to measure GDeflate scaling"]
    fn gdeflate_scaling() {
        use crate::libdeflate::gdeflate_compress;
        use std::time::Instant;

        // Mix compressible runs with deterministic noise. Prepare the stream outside the timer.
        let mut state = 1u32;
        let raw: Vec<u8> = (0..256 * TILE_BYTES)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                if i % 256 < 160 { (i / 256) as u8 } else { state as u8 }
            })
            .collect();
        let packed = gdeflate_compress(&raw, 1).expect("compresses");
        let mut out = vec![0; raw.len()];
        let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
        let mut baseline = 0.0;
        println!("threads\tMB/s\tspeedup");
        for workers in [1, 2, 4, 8].into_iter().filter(|&workers| workers <= cores) {
            assert_eq!(tiles_with_workers(&packed, &mut out, workers), S_OK);
            let start = Instant::now();
            let mut runs = 0;
            while start.elapsed().as_secs_f64() < 1.0 {
                assert_eq!(tiles_with_workers(&packed, &mut out, workers), S_OK);
                std::hint::black_box(&out);
                runs += 1;
            }
            let rate = f64::from(runs) * raw.len() as f64 / (start.elapsed().as_secs_f64() * 1_000_000.0);
            assert_eq!(out, raw);
            if workers == 1 {
                baseline = rate;
            }
            println!("{workers}\t{rate:.1}\t{:.2}x", rate / baseline);
        }
    }

    fn zlib_of(data: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(data).expect("compresses");
        encoder.finish().expect("finishes")
    }

    #[test]
    fn zlib_must_fill_the_output_exactly_and_use_all_the_input() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i * 7) as u8).collect();
        let packed = zlib_of(&data);
        let mut out = vec![0u8; data.len()];
        assert_eq!(zlib(&packed, &mut out), S_OK);
        assert_eq!(out, data);
        let mut bigger = vec![0u8; data.len() + 1];
        assert_eq!(zlib(&packed, &mut bigger), BAD);
        let mut smaller = vec![0u8; data.len() - 1];
        assert_eq!(zlib(&packed, &mut smaller), BAD);
        let mut trailing = packed.clone();
        trailing.push(0);
        assert_eq!(zlib(&trailing, &mut out), BAD);
        assert_eq!(zlib(&packed[..packed.len() - 3], &mut out), BAD);
        assert_eq!(zlib(&[1, 2, 3], &mut out), BAD);
        let mut corrupt = packed.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xFF; // the Adler 32 check
        assert_eq!(zlib(&corrupt, &mut out), BAD);
    }

    #[test]
    fn zlib_reuses_thread_local_decompressor_across_calls() {
        for len in [128usize, 4096, 65536, 1024] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13 + i / 7) as u8).collect();
            let packed = zlib_of(&data);
            let mut out = vec![0u8; len];
            assert_eq!(zlib(&packed, &mut out), S_OK);
            assert_eq!(out, data);
        }
        let mut bad_out = [0u8; 64];
        assert_eq!(zlib(&[0x78, 0x9c, 0x00], &mut bad_out), BAD);
        let good = b"reusable zlib state check";
        let packed = zlib_of(good);
        let mut out = vec![0u8; good.len()];
        assert_eq!(zlib(&packed, &mut out), S_OK);
        assert_eq!(out, good);
    }

    #[test]
    fn bcpack_decodes_in_process() {
        let blocks: Vec<u8> = (0..4096u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8).collect();
        for (mode, format) in [(1, decant_bcpack::Format::Bc1), (3, decant_bcpack::Format::Bc3), (4, decant_bcpack::Format::Bc4), (5, decant_bcpack::Format::Bc5)] {
            let packed = decant_bcpack::encode::pack_plain(format, &blocks, 100);
            let mut out = vec![0u8; blocks.len()];
            assert_eq!(bcpack(mode, &packed, &mut out), S_OK, "mode {mode}");
            assert_eq!(out, blocks, "mode {mode}");
            assert_eq!(bcpack(mode, &packed[..packed.len() / 2], &mut out), BAD, "mode {mode}, cut short");
        }
        let mut out = vec![0u8; 64];
        assert_eq!(bcpack(2, &[0; 16], &mut out), E_NOTIMPL, "BC2 has no packed form");
        assert_eq!(bcpack(9, &[0; 16], &mut out), E_NOTIMPL);
        assert_eq!(bcpack(1, &[0; 16], &mut out[..12]), BAD, "not a whole number of blocks");
    }

    #[test]
    fn other_codecs_are_not_implemented() {
        let source = [0u8; 4];
        let mut out = [0u8; 4];
        // SAFETY: both buffers are valid for their lengths and do not overlap.
        assert_eq!(unsafe { decode(codec::COPY, 0, source.as_ptr(), 4, out.as_mut_ptr(), 4) }, E_NOTIMPL);
        // SAFETY: as above.
        assert_eq!(unsafe { decode(9, 0, source.as_ptr(), 4, out.as_mut_ptr(), 4) }, E_NOTIMPL);
    }
}
