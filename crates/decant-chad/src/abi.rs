//! Constants, result codes and the structures of `chad.dll` (`docs/CONSOLE_LIBS.md` 4.1 and 4.2).

pub type Hresult = i32;

pub const S_OK: Hresult = 0;
pub const E_POINTER: Hresult = 0x8000_4003_u32 as i32;
pub const E_INVALIDARG: Hresult = 0x8007_0057_u32 as i32;
pub const E_OUTOFMEMORY: Hresult = 0x8007_000E_u32 as i32;

pub const XDSP_E_NO_SLOT: Hresult = 0x8AC8_10A0_u32 as i32;
pub const XDSP_E_BUSY: Hresult = 0x8AC8_10A1_u32 as i32;
pub const XDSP_E_TOO_MANY_CONTEXTS: Hresult = 0x8AC8_10A2_u32 as i32;
pub const XDSP_E_CONTEXTS_ACTIVE: Hresult = 0x8AC8_10A4_u32 as i32;
pub const XDSP_E_TOO_MANY_CLIENTS: Hresult = 0x8AC8_10A5_u32 as i32;
pub const XDSP_E_TOO_MANY_ACTIVATIONS: Hresult = 0x8AC8_10A6_u32 as i32;
pub const XDSP_E_IR_SIZE: Hresult = 0x8AC8_10A7_u32 as i32;

/// Process types.
pub const PROCESS_CONVOLUTION: u32 = 1;
pub const PROCESS_FFT: u32 = 2;
pub const PROCESS_INVERSE_FFT: u32 = 3;

/// Activation flags.
pub const FLAG_DEINTERLEAVED: u32 = 2;
pub const FLAG_POLAR: u32 = 4;
pub const FLAG_STEREO_IR: u32 = 8;

/// Limits.
pub const MEMORY_MIN: u32 = 0x1_0000;
pub const MEMORY_MAX: u32 = 0x1F40_0000;
pub const MAX_ACTIVATIONS: u32 = 0x80;
pub const MAX_STREAMS: u32 = 0x100;
pub const MAX_PRIORITY: u32 = 2;
pub const MAX_CLIENTS: usize = 8;
pub const MAX_CONTEXTS: u32 = 0x100;
pub const MAX_OUTSTANDING: u32 = 8;
pub const MAX_PARTITIONS_ROUNDED: usize = 0x800;
pub const MIN_PARTITIONS: usize = 9;
/// Bit pattern of the first float of a metadata partition of an impulse response.
pub const METADATA_MARKER: u32 = 0xFFFF_ACE1;
/// A group of this many commands shares one envelope.
pub const ENVELOPE_GROUP: u32 = 8;

/// Size of the status array in UINT32: submitted, completed, (2 unused here), sticky error at index 3.
pub const STATUS_WORDS: usize = 4;

/// `XDSP_ACTIVATION_PARAMETERS`, 0x1C bytes read at the offsets of the specification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActivationParameters {
    pub process_type: u32,
    pub flags: u32,
    pub block: u32,
    pub channels: u32,
    /// Impulse response length in floats.
    pub ir_length: u32,
    pub ir_pointer: u64,
}

pub const ACTIVATION_SIZE: usize = 0x1C;

impl ActivationParameters {
    pub fn from_bytes(b: &[u8; ACTIVATION_SIZE]) -> Self {
        let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        ActivationParameters { process_type: u(0), flags: u(4), block: u(8), channels: u(0xC), ir_length: u(0x10), ir_pointer: u64::from(u(0x14)) | u64::from(u(0x18)) << 32 }
    }

    pub fn to_bytes(&self) -> [u8; ACTIVATION_SIZE] {
        let mut b = [0u8; ACTIVATION_SIZE];
        for (i, v) in [self.process_type, self.flags, self.block, self.channels, self.ir_length].iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[0x14..0x1C].copy_from_slice(&self.ir_pointer.to_le_bytes());
        b
    }
}

/// The `XDSP_ACTIVATION_PARAMETERS` image as the caller passes it.
pub type RawActivationParameters = [u8; ACTIVATION_SIZE];

/// `XDSP_COMMAND`, 0x20 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XdspCommand {
    pub input_gain: f32,
    pub output_gain: f32,
    pub input: u64,
    pub output: u64,
    /// FFT only: block buffer multiplier, may be 0.
    pub multiplier: u64,
}

/// Opaque handles: the address identifies the object.
#[repr(C)]
pub struct XdspClient {
    _token: u64,
}

#[repr(C)]
pub struct XdspContext {
    _token: u64,
}
