//! The calls into libdeflate with the NVIDIA GDeflate additions, which `build.rs` compiles and
//! links only when `DECANT_GDEFLATE_ROOT` names a checkout of microsoft/DirectStorage's `GDeflate`
//! folder (the same variable the former `build_storage.cmd` read). Without it, this module is
//! empty and GDeflate requests on the CPU path answer `E_NOTIMPL`. The compressors are used by
//! the test programs only.
#![cfg(decant_gdeflate)]
#![allow(unsafe_code)]

use std::ffi::c_void;

/// `LIBDEFLATE_SUCCESS`.
pub const SUCCESS: i32 = 0;

/// `struct libdeflate_gdeflate_in_page`.
#[repr(C)]
pub struct InPage {
    pub data: *const c_void,
    pub nbytes: usize,
}

/// `struct libdeflate_gdeflate_out_page`.
#[repr(C)]
pub struct OutPage {
    pub data: *mut c_void,
    pub nbytes: usize,
}

unsafe extern "C" {
    pub fn libdeflate_alloc_gdeflate_decompressor() -> *mut c_void;
    pub fn libdeflate_gdeflate_decompress(decompressor: *mut c_void, pages: *mut InPage, npages: usize, out: *mut c_void, avail: usize, actual: *mut usize) -> i32;
    pub fn libdeflate_free_gdeflate_decompressor(decompressor: *mut c_void);

    pub fn libdeflate_alloc_compressor(level: i32) -> *mut c_void;
    pub fn libdeflate_zlib_compress_bound(compressor: *mut c_void, bytes: usize) -> usize;
    pub fn libdeflate_zlib_compress(compressor: *mut c_void, input: *const c_void, bytes: usize, out: *mut c_void, avail: usize) -> usize;
    pub fn libdeflate_free_compressor(compressor: *mut c_void);

    pub fn libdeflate_alloc_gdeflate_compressor(level: i32) -> *mut c_void;
    pub fn libdeflate_gdeflate_compress_bound(compressor: *mut c_void, bytes: usize, npages: *mut usize) -> usize;
    pub fn libdeflate_gdeflate_compress(compressor: *mut c_void, input: *const c_void, bytes: usize, pages: *mut OutPage, npages: usize) -> usize;
    pub fn libdeflate_free_gdeflate_compressor(compressor: *mut c_void);
}

/// The size of a GDeflate tile.
pub const TILE_BYTES: usize = 65536;

/// The GDeflate tile stream of the reference compressor (`GDeflate::Compress`, one thread): an
/// eight byte header, a table of tile offsets (entry zero holds the size of the last tile), then
/// the tiles. For the test programs.
pub fn gdeflate_compress(data: &[u8], level: i32) -> Option<Vec<u8>> {
    if data.is_empty() || data.len() > TILE_BYTES * 0xFFFF {
        return None;
    }
    // SAFETY: the compressor is used by this function only, and freed before it returns.
    let compressor = unsafe { libdeflate_alloc_gdeflate_compressor(level) };
    if compressor.is_null() {
        return None;
    }
    let mut npages = 0usize;
    // SAFETY: a null compressor asks for the bound over every compressor, `npages` is writable.
    let scratch_size = unsafe { libdeflate_gdeflate_compress_bound(std::ptr::null_mut(), TILE_BYTES, &mut npages) };
    let mut scratch = vec![0u8; scratch_size];
    let tile_count = data.len().div_ceil(TILE_BYTES);
    let mut tiles: Vec<Vec<u8>> = Vec::with_capacity(tile_count);
    for chunk in data.chunks(TILE_BYTES) {
        let mut page = OutPage { data: scratch.as_mut_ptr().cast(), nbytes: scratch_size };
        // SAFETY: one page whose buffer holds `scratch_size` bytes, and `chunk` is readable.
        let written = unsafe { libdeflate_gdeflate_compress(compressor, chunk.as_ptr().cast(), chunk.len(), &mut page, 1) };
        if written == 0 {
            // SAFETY: allocated above, freed once.
            unsafe { libdeflate_free_gdeflate_compressor(compressor) };
            return None;
        }
        tiles.push(scratch[..page.nbytes].to_vec());
    }
    // SAFETY: allocated above, freed once.
    unsafe { libdeflate_free_gdeflate_compressor(compressor) };
    let count = tiles.len();
    let last = data.len() - (count - 1) * TILE_BYTES;
    let tail = if last < TILE_BYTES { last as u32 } else { 0 };
    let mut out = vec![4u8, 4 ^ 0xFF];
    out.extend_from_slice(&(count as u16).to_le_bytes());
    out.extend_from_slice(&(1 | (tail << 2)).to_le_bytes());
    let mut position = 0u32;
    for (i, tile) in tiles.iter().enumerate() {
        let entry = if i == 0 { tiles[count - 1].len() as u32 } else { position };
        out.extend_from_slice(&entry.to_le_bytes());
        position += tile.len() as u32;
    }
    for tile in &tiles {
        out.extend_from_slice(tile);
    }
    Some(out)
}

/// The zlib stream of libdeflate at `level`. For the test programs.
pub fn zlib_compress(data: &[u8], level: i32) -> Option<Vec<u8>> {
    // SAFETY: the compressor is used by this function only, and freed before it returns.
    let compressor = unsafe { libdeflate_alloc_compressor(level) };
    if compressor.is_null() {
        return None;
    }
    // SAFETY: the bound of this compressor for `data.len()` bytes.
    let mut out = vec![0u8; unsafe { libdeflate_zlib_compress_bound(compressor, data.len()) }];
    // SAFETY: `data` and `out` are readable and writable for their lengths.
    let written = unsafe { libdeflate_zlib_compress(compressor, data.as_ptr().cast(), data.len(), out.as_mut_ptr().cast(), out.len()) };
    // SAFETY: allocated above, freed once.
    unsafe { libdeflate_free_compressor(compressor) };
    out.truncate(written);
    (written != 0).then_some(out)
}
