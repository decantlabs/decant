//! Result codes, constants and structures of `AcpHal.dll` (`docs/CONSOLE_LIBS.md`, section 2).

pub type Hresult = i32;

pub const S_OK: Hresult = 0;
pub const E_NOTIMPL: Hresult = 0x8000_4001_u32 as i32;
pub const E_FAIL: Hresult = 0x8000_4005_u32 as i32;
pub const E_UNEXPECTED: Hresult = 0x8000_FFFF_u32 as i32;
pub const E_OUTOFMEMORY: Hresult = 0x8007_000E_u32 as i32;
pub const E_INVALIDARG: Hresult = 0x8007_0057_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)`.
pub const E_NOT_SUPPORTED: Hresult = 0x8007_0032_u32 as i32;
/// Returned by `SubmitCommand` when the object is not connected.
pub const HAL_S_PENDING: Hresult = 0x8000_000E_u32 as i32;

pub const APU_E_HEAP_ALREADY_CREATED: Hresult = 0x8ACA_0001_u32 as i32;
pub const APU_E_HEAP_NOT_CREATED: Hresult = 0x8ACA_0002_u32 as i32;
pub const SHAPE_E_ALREADY_ALLOCATED: Hresult = 0x8AC8_0001_u32 as i32;
pub const SHAPE_E_NOT_ALLOCATED: Hresult = 0x8AC8_0002_u32 as i32;
pub const HAL_E_QUEUE_FULL: Hresult = 0x8AC8_0003_u32 as i32;
pub const HAL_E_ALREADY_CONNECTED: Hresult = 0x8AC8_0009_u32 as i32;
pub const HAL_E_NOT_CONNECTED: Hresult = 0x8AC8_000A_u32 as i32;
pub const SHAPE_E_BUSY: Hresult = 0x8AC8_0010_u32 as i32;
pub const HAL_E_UNSUPPORTED: Hresult = 0x8AC8_0F0E_u32 as i32;

/// Heap sizes and addresses.
pub const DEFAULT_HEAP1_SIZE: u32 = 0x400_0000;
pub const DEFAULT_HEAP2_SIZE: u32 = 0x11_4000;
pub const HEAP2_EXTRA: u32 = 0x2000;
pub const INTERNAL_HEAP_SIZE: u32 = 0x1000;
pub const HEAP1_APU_BASE: u32 = 0x100_0000;
pub const HEAP_HOST_ALIGN: usize = 0x1_0000;
pub const MIN_ALIGNMENT: u32 = 256;

/// `APU_HEAP_STATE`: five UINT32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApuHeapState {
    pub free_bytes: u32,
    pub used_bytes: u32,
    pub overhead_bytes: u32,
    pub largest_free_block: u32,
    pub allocations: u32,
}

/// Limits and element sizes of the six SHAPE context arrays, in the order of `SHAPE_CONTEXTS`.
pub const SHAPE_MAX_COUNTS: [u32; 6] = [0x1000, 0x1000, 0x1000, 0x1000, 0x200, 0xE00];
pub const SHAPE_ELEMENT_SIZES: [u32; 6] = [0x20, 0xA0, 0x20, 0x10, 0x40, 0x20];
pub const SHAPE_ALIGNMENTS: [u32; 6] = [0x20, 0x10, 0x20, 0x10, 0x40, 0x40];
pub const XMA_CONTEXT_COUNT_MAX: u32 = 0x200;
pub const DMA_CONTEXT_COUNT_MAX: u32 = 0x1000;

/// `SHAPE_CONTEXTS`, 0x60 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeContexts {
    /// SRC, EQ and compressor, filter and volume, DMA, XMA, PCM.
    pub counts: [u32; 6],
    pub pointers: [u64; 6],
    pub apu_addresses: [u32; 6],
}

/// One SHAPE audio frame is 128 sample frames, 2.667 ms at 48 kHz.
pub const FRAME_SAMPLES: u32 = 128;
pub const SAMPLE_RATE: u32 = 48_000;

/// Message types.
pub const MSG_AUDIO_FRAME_START: u32 = 0x01;
pub const MSG_FLOWGRAPH_COMPLETED: u32 = 0x02;
pub const MSG_SRC_BLOCKED: u32 = 0x04;
pub const MSG_DMA_BLOCKED: u32 = 0x08;
pub const MSG_COMMAND_COMPLETED: u32 = 0x10;
pub const MSG_FLOWGRAPH_TERMINATED: u32 = 0x20;
pub const MSG_ERROR: u32 = 0x40;
pub const MSG_DISCONNECTED: u32 = 0x80;

/// `ACP_MESSAGE`, 28 bytes (the 64 bit field is at offset 0x10, unaligned in the 28 byte image).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcpMessage {
    pub kind: u32,
    pub clock: u32,
    pub a: u32,
    pub b: u64,
    pub c: u32,
}

pub const MESSAGE_SIZE: usize = 28;

impl AcpMessage {
    /// The 28 byte image: type, 0, clock, a, b, c.
    pub fn to_bytes(&self) -> [u8; MESSAGE_SIZE] {
        let mut out = [0u8; MESSAGE_SIZE];
        out[0..4].copy_from_slice(&self.kind.to_le_bytes());
        out[8..12].copy_from_slice(&self.clock.to_le_bytes());
        out[12..16].copy_from_slice(&self.a.to_le_bytes());
        out[16..24].copy_from_slice(&self.b.to_le_bytes());
        out[24..28].copy_from_slice(&self.c.to_le_bytes());
        out
    }
}

/// Command types.
pub const CMD_RUN_FLOWGRAPH: u32 = 0x01;
pub const CMD_CONFIGURE: u32 = 0x02;
pub const CMD_XMA_ENABLE: u32 = 0x05;
pub const CMD_XMA_ENABLE_MASK: u32 = 0x06;
pub const CMD_XMA_DISABLE: u32 = 0x07;
pub const CMD_XMA_DISABLE_MASK: u32 = 0x08;
pub const CMD_XMA_UPDATE: u32 = 0x0E;
pub const CMD_DMA_CONSUMED: u32 = 0x10;
pub const CMD_XMA_OUTPUT_CONSUMED: u32 = 0x12;
pub const CMD_XMA_SET_INPUT: u32 = 0x13;
pub const CMD_TYPE_MAX: u32 = 0x1B;

/// Size of one command ring slot and of one message ring slot.
pub const COMMAND_SLOT_SIZE: u32 = 0x10C;
pub const MESSAGE_SLOT_SIZE: u32 = 0x100;
pub const DEFAULT_MESSAGE_SLOTS: u32 = 256;
pub const MAX_CONNECTED: u32 = 4;
