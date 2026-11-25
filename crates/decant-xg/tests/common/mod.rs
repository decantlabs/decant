//! Helpers of the vector tests: a C style client of the exports and of the texture computer table.
#![allow(unsafe_code, dead_code)]

use std::sync::{Mutex, MutexGuard};

pub use decant_texture::xg::resource_tiling::{PackedMipInfo, SubresourceTiling};
use xg_xs::abi::{E_INVALIDARG, E_NOTIMPL, Hresult, S_OK, XgTileShape};
use xg_xs::computer::TextureComputer;
use xg_xs::exports::*;

pub const INVALIDARG: Hresult = E_INVALIDARG;
pub const NOTIMPL: Hresult = E_NOTIMPL;
pub const OK: Hresult = S_OK;

static SERIAL: Mutex<()> = Mutex::new(());

/// The library state is process wide: tests that depend on it run one after another, from the default hardware version.
pub fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    XGSetHardwareVersion(0);
    XGEnableExclusiveLocking(0);
    guard
}

/// A `D3D12_RESOURCE_DESC` of 0x38 bytes.
#[derive(Clone, Copy)]
pub struct D3d {
    pub dimension: u32,
    pub width: u64,
    pub height: u32,
    pub depth: u16,
    pub mips: u16,
    pub format: u32,
    pub samples: u32,
    pub layout: u32,
    pub flags: u32,
}

impl D3d {
    /// A 2D texture of one slice, one mip, one sample, layout 0 and no flags.
    pub fn tex2d(width: u64, height: u32, format: u32) -> D3d {
        D3d { dimension: 3, width, height, depth: 1, mips: 1, format, samples: 1, layout: 0, flags: 0 }
    }

    pub fn bytes(&self) -> [u8; 0x38] {
        let mut b = [0u8; 0x38];
        b[0..4].copy_from_slice(&self.dimension.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&self.width.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&self.height.to_le_bytes());
        b[0x1C..0x1E].copy_from_slice(&self.depth.to_le_bytes());
        b[0x1E..0x20].copy_from_slice(&self.mips.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&self.format.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&self.samples.to_le_bytes());
        b[0x2C..0x30].copy_from_slice(&self.layout.to_le_bytes());
        b[0x30..0x34].copy_from_slice(&self.flags.to_le_bytes());
        b
    }
}

/// A texture computer held through its table, as a C client does.
pub struct Computer(pub *mut TextureComputer);

impl Drop for Computer {
    fn drop(&mut self) {
        if !self.0.is_null() {
            self.release();
        }
    }
}

type AddRef = unsafe extern "system" fn(*mut TextureComputer) -> u32;
type GetLayout = unsafe extern "system" fn(*mut TextureComputer, *mut u8) -> Hresult;
type GetU64 = unsafe extern "system" fn(*mut TextureComputer) -> u64;
type GetMip = unsafe extern "system" fn(*mut TextureComputer, u32, u32) -> u64;
type GetTexel = unsafe extern "system" fn(*mut TextureComputer, u32, u32, u64, u32, u32, u32, *mut u32) -> u64;
type Copy = unsafe extern "system" fn(*mut TextureComputer, *mut u8, u32, u32, *mut u8, u32, u32) -> Hresult;
type Unknown6 = unsafe extern "system" fn(*mut TextureComputer, usize, usize, usize, usize, usize, usize) -> Hresult;
type Tiling = unsafe extern "system" fn(*mut TextureComputer, *mut u32, *mut PackedMipInfo, *mut XgTileShape, *mut u32, u32, *mut SubresourceTiling) -> Hresult;
type View = unsafe extern "system" fn(*mut TextureComputer, u32, *mut u8) -> Hresult;

impl Computer {
    /// The method of slot `i`.
    unsafe fn slot<F: std::marker::Copy>(&self, i: usize) -> F {
        // SAFETY: the object starts with the pointer to its table of 12 methods.
        unsafe {
            let table = *(self.0 as *const *const *const ());
            std::mem::transmute_copy::<*const (), F>(&*table.add(i))
        }
    }

    pub fn create(desc: &D3d) -> Result<Computer, Hresult> {
        let bytes = desc.bytes();
        let mut out: *mut TextureComputer = std::ptr::null_mut();
        // SAFETY: valid pointers.
        let hr = unsafe { XGCreateTextureComputer(&bytes, &mut out) };
        if hr == OK { Ok(Computer(out)) } else { Err(hr) }
    }

    pub fn add_ref(&self) -> u32 {
        unsafe { self.slot::<AddRef>(0)(self.0) }
    }

    pub fn release(&self) -> u32 {
        unsafe { self.slot::<AddRef>(1)(self.0) }
    }

    pub fn record(&self) -> Vec<u8> {
        let mut r = vec![0xAAu8; 0x1B40];
        assert_eq!(unsafe { self.slot::<GetLayout>(2)(self.0, r.as_mut_ptr()) }, OK);
        r
    }

    pub fn size(&self) -> u64 {
        unsafe { self.slot::<GetU64>(3)(self.0) }
    }

    pub fn align(&self) -> u64 {
        unsafe { self.slot::<GetU64>(4)(self.0) }
    }

    pub fn mip_offset(&self, plane: u32, mip: u32) -> u64 {
        unsafe { self.slot::<GetMip>(5)(self.0, plane, mip) }
    }

    pub fn element(&self, mip: u32, x: u64, y: u32, z: u32) -> u64 {
        self.texel(0, mip, x, y, z, 0)
    }

    pub fn texel(&self, plane: u32, mip: u32, x: u64, y: u32, z: u32, sample: u32) -> u64 {
        let mut bit = 0xFFFF_FFFFu32;
        let v = unsafe { self.slot::<GetTexel>(6)(self.0, plane, mip, x, y, z, sample, &mut bit) };
        assert_eq!(bit, 0, "the bit offset is zeroed");
        v
    }

    pub fn copy_into(&self, resource: &mut [u8], subresource: u32, src: &[u8], row_pitch: u32, slice_pitch: u32) -> Hresult {
        unsafe { self.slot::<Copy>(8)(self.0, resource.as_mut_ptr(), 0, subresource, src.as_ptr() as *mut u8, row_pitch, slice_pitch) }
    }

    pub fn copy_from(&self, dst: &mut [u8], subresource: u32, resource: &[u8], row_pitch: u32, slice_pitch: u32) -> Hresult {
        unsafe { self.slot::<Copy>(9)(self.0, dst.as_mut_ptr(), 0, subresource, resource.as_ptr() as *mut u8, row_pitch, slice_pitch) }
    }

    pub fn slot7(&self) -> Hresult {
        unsafe { self.slot::<Unknown6>(7)(self.0, 0, 0, 0, 0, 0, 0) }
    }

    /// Slot 10 with every output: (result, total tiles, packed mip info, tile shape, the tilings written).
    pub fn tiling(&self, first: u32, room: u32) -> (Hresult, u32, PackedMipInfo, XgTileShape, Vec<SubresourceTiling>) {
        let (mut tiles, mut packed, mut shape, mut count) = (0xAAAA_AAAAu32, PackedMipInfo::default(), XgTileShape::default(), room);
        let mut out = vec![SubresourceTiling { width_in_tiles: 0xAAAA, height_in_tiles: 0xAAAA, depth_in_tiles: 0xAAAA, start_tile: 0xAAAA }; room as usize];
        let hr = unsafe { self.slot::<Tiling>(10)(self.0, &mut tiles, &mut packed, &mut shape, &mut count, first, out.as_mut_ptr()) };
        if hr == OK {
            assert!(count <= room);
            out.truncate(count as usize);
        }
        (hr, tiles, packed, shape, out)
    }

    /// Slot 10 with every output null.
    pub fn tiling_null(&self) -> Hresult {
        let null = std::ptr::null_mut::<u8>();
        unsafe { self.slot::<Tiling>(10)(self.0, null.cast(), null.cast(), null.cast(), null.cast(), 0, null.cast()) }
    }

    pub fn slot11(&self) -> Hresult {
        unsafe { self.slot::<View>(11)(self.0, 0, std::ptr::null_mut()) }
    }

    pub fn null_layout(&self) -> Hresult {
        unsafe { self.slot::<GetLayout>(2)(self.0, std::ptr::null_mut()) }
    }
}

/// Reads fields of an `XG_RESOURCE_LAYOUT` record.
pub struct Record(pub Vec<u8>);

impl Record {
    pub fn u32(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }

    pub fn u64(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.0[at..at + 8].try_into().unwrap())
    }

    pub fn size(&self) -> u64 {
        self.u64(0)
    }

    pub fn align(&self) -> u64 {
        self.u64(8)
    }

    pub fn mips(&self) -> u32 {
        self.u32(0x10)
    }

    pub fn planes(&self) -> u32 {
        self.u32(0x14)
    }

    pub fn dimension(&self) -> u32 {
        self.u32(0x1B38)
    }

    fn plane_base(plane: usize) -> usize {
        0x18 + plane * 0x6C8
    }

    pub fn plane_usage(&self, p: usize) -> u32 {
        self.u32(Self::plane_base(p))
    }

    pub fn plane_size(&self, p: usize) -> u64 {
        self.u64(Self::plane_base(p) + 8)
    }

    pub fn plane_slice(&self, p: usize) -> u64 {
        self.u64(Self::plane_base(p) + 0x10)
    }

    pub fn plane_offset(&self, p: usize) -> u64 {
        self.u64(Self::plane_base(p) + 0x18)
    }

    pub fn plane_align(&self, p: usize) -> u64 {
        self.u64(Self::plane_base(p) + 0x20)
    }

    pub fn bpe(&self) -> u32 {
        self.u32(Self::plane_base(0) + 0x28)
    }

    pub fn first_tail(&self) -> u32 {
        self.u32(Self::plane_base(0) + 0x2C)
    }

    pub fn tail_count(&self) -> u32 {
        self.u32(Self::plane_base(0) + 0x30)
    }

    /// Field at `at` of level `i` of plane `p`.
    pub fn level_u32(&self, p: usize, i: usize, at: usize) -> u32 {
        self.u32(Self::plane_base(p) + 0x38 + i * 0x70 + at)
    }

    pub fn level_u64(&self, p: usize, i: usize, at: usize) -> u64 {
        self.u64(Self::plane_base(p) + 0x38 + i * 0x70 + at)
    }

    pub fn offset(&self, i: usize) -> u64 {
        self.level_u64(0, i, 0x08)
    }

    pub fn padded(&self, i: usize) -> (u32, u32, u32) {
        (self.level_u32(0, i, 0x24), self.level_u32(0, i, 0x28), self.level_u32(0, i, 0x2C))
    }

    pub fn mode(&self, i: usize) -> u32 {
        self.level_u32(0, i, 0x48)
    }

    pub fn in_tail(&self, i: usize) -> bool {
        self.level_u32(0, i, 0x64) == 1
    }
}
