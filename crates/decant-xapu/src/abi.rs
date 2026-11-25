//! Constants, result codes and the binary layouts of `xapu.dll` (`docs/CONSOLE_LIBS.md`, section 3).

pub type Hresult = i32;
/// `XAPU_HANDLE`, pointer sized.
pub type Handle = usize;

pub const S_OK: Hresult = 0;
pub const E_POINTER: Hresult = 0x8000_4003_u32 as i32;
pub const E_INVALIDARG: Hresult = 0x8007_0057_u32 as i32;
pub const E_OUTOFMEMORY: Hresult = 0x8007_000E_u32 as i32;
pub const E_FAIL: Hresult = 0x8000_4005_u32 as i32;

pub const XAPU_E_QUEUE_FULL: Hresult = 0x8AC8_1000_u32 as i32;
pub const XAPU_E_NO_RESULT: Hresult = 0x8AC8_1001_u32 as i32;
pub const XAPU_E_INVALID_COMMAND: Hresult = 0x8AC8_1002_u32 as i32;
pub const XAPU_E_BAD_STREAM: Hresult = 0x8AC8_1003_u32 as i32;
pub const XAPU_E_NO_STREAMS: Hresult = 0x8AC8_1004_u32 as i32;
pub const XAPU_E_STREAM_ACTIVE: Hresult = 0x8AC8_1005_u32 as i32;
pub const XAPU_E_BUFFER_TOO_SMALL: Hresult = 0x8AC8_1006_u32 as i32;
pub const XAPU_E_PROCESSING_ERROR: Hresult = 0x8AC8_1007_u32 as i32;
pub const XAPU_E_BAD_OPUS_PACKET: Hresult = 0x8AC8_1008_u32 as i32;

/// Process types.
pub const PROCESS_DECODE_OPUS: u32 = 1;
pub const PROCESS_DECODE_OPUS_CONVERT: u32 = 2;
pub const PROCESS_CONVERT_PCM: u32 = 3;

/// Connection options.
pub const OPTION_DEINTERLEAVED_INPUT: u32 = 2;
pub const OPTION_DEINTERLEAVED_OUTPUT: u32 = 4;
pub const OPTION_RESULT_EVENT: u32 = 8;
pub const OPTION_MASK: u32 = 0xF;

/// Command types.
pub const COMMAND_START: u32 = 1;
pub const COMMAND_STOP: u32 = 2;
pub const COMMAND_DATA: u32 = 3;
pub const COMMAND_RESET: u32 = 5;

/// Command options.
pub const COMMAND_OPTION_FLUSH: u32 = 2;
pub const COMMAND_OPTION_RESET_DECODER: u32 = 4;

/// Result state bits.
pub const STATE_OUTPUT_READY: u32 = 1;
pub const STATE_NEED_MORE_INPUT: u32 = 2;
pub const STATE_MORE_OUTPUT: u32 = 4;
pub const STATE_SATURATION: u32 = 8;

pub const MAX_STREAMS: u32 = 500;
pub const MAX_QUEUE_DEPTH: u32 = 16;
pub const MAX_REGION_SIZE: u32 = 0x2000_0000;
pub const REGION_ALIGN: usize = 0x1_0000;
pub const COMMAND_SIZE: usize = 0x40;
pub const COMMAND_READ_SIZE: usize = 0x38;
pub const RESULT_SIZE: usize = 0x50;

/// `XAPU_CONNECT_PARAMS`, 6 UINT32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConnectParams {
    pub process_type: u32,
    pub stream_count: u32,
    pub queue_depth: u32,
    pub region_size: u32,
    pub frame_count: u32,
    pub options: u32,
}

/// `XAPU_CONNECTION_REGION`, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ConnectionRegion {
    pub memory: *mut u8,
    pub result_event: usize,
}

/// `XAPU_COMMAND`, 0x40 bytes read through typed accessors at the offsets of the specification.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Command(pub [u8; COMMAND_SIZE]);

impl Default for Command {
    fn default() -> Self {
        Command([0; COMMAND_SIZE])
    }
}

impl Command {
    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes([self.0[off], self.0[off + 1], self.0[off + 2], self.0[off + 3]])
    }
    fn f32_at(&self, off: usize) -> f32 {
        f32::from_bits(self.u32_at(off))
    }
    fn u64_at(&self, off: usize) -> u64 {
        u64::from(self.u32_at(off)) | u64::from(self.u32_at(off + 4)) << 32
    }
    fn set_u32(&mut self, off: usize, v: u32) {
        self.0[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    pub fn kind(&self) -> u32 {
        self.u32_at(0)
    }
    pub fn stream(&self) -> u32 {
        self.u32_at(4)
    }
    pub fn options(&self) -> u32 {
        self.u32_at(0xC)
    }
    /// Start: channel count.
    pub fn channels(&self) -> u32 {
        self.u32_at(0x18)
    }
    /// Data: pitch ramp per output frame.
    pub fn pitch_ramp(&self) -> f32 {
        self.f32_at(0x18)
    }
    /// Data: target pitch.
    pub fn target_pitch(&self) -> f32 {
        self.f32_at(0x1C)
    }
    /// Data: first frame (decode).
    pub fn first_frame(&self) -> u32 {
        self.u32_at(0x20)
    }
    /// Data: frame count.
    pub fn frame_count(&self) -> u32 {
        self.u32_at(0x24)
    }
    /// Start: start pitch.
    pub fn start_pitch(&self) -> f32 {
        self.f32_at(0x28)
    }
    /// Data: input length in bytes.
    pub fn input_length(&self) -> u32 {
        self.u32_at(0x28)
    }
    /// Start: length of the processing buffer (4 bytes used).
    pub fn processing_buffer_length(&self) -> u32 {
        self.u32_at(0x2C)
    }
    /// Data: input pointer (unaligned 64 bit).
    pub fn input_pointer(&self) -> u64 {
        self.u64_at(0x2C)
    }
    /// Data: output capacity in frames.
    pub fn output_capacity(&self) -> u32 {
        self.u32_at(0x34)
    }
    /// Data: output pointer.
    pub fn output_pointer(&self) -> u64 {
        self.u64_at(0x38)
    }

    /// A command image with the common fields set, for tests and callers that build commands.
    pub fn new(kind: u32, stream: u32, options: u32) -> Command {
        let mut c = Command::default();
        c.set_u32(0, kind);
        c.set_u32(4, stream);
        c.set_u32(0xC, options);
        c
    }

    pub fn with_start(mut self, channels: u32, pitch: f32, processing_buffer: u32) -> Command {
        self.set_u32(0x18, channels);
        self.set_u32(0x28, pitch.to_bits());
        self.set_u32(0x2C, processing_buffer);
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_data(mut self, ramp: f32, target: f32, frames: u32, input_len: u32, input: u64, capacity: u32, output: u64) -> Command {
        self.set_u32(0x18, ramp.to_bits());
        self.set_u32(0x1C, target.to_bits());
        self.set_u32(0x24, frames);
        self.set_u32(0x28, input_len);
        self.0[0x2C..0x34].copy_from_slice(&input.to_le_bytes());
        self.set_u32(0x34, capacity);
        self.0[0x38..0x40].copy_from_slice(&output.to_le_bytes());
        self
    }
}

/// `XAPU_RESULT`, 0x50 bytes with typed accessors.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct XapuResult(pub [u8; RESULT_SIZE]);

impl Default for XapuResult {
    fn default() -> Self {
        XapuResult([0; RESULT_SIZE])
    }
}

impl XapuResult {
    fn u32_at(&self, off: usize) -> u32 {
        u32::from_le_bytes([self.0[off], self.0[off + 1], self.0[off + 2], self.0[off + 3]])
    }
    pub fn kind(&self) -> u32 {
        self.u32_at(0)
    }
    pub fn stream(&self) -> u32 {
        self.u32_at(4)
    }
    pub fn options(&self) -> u32 {
        self.u32_at(0xC)
    }
    pub fn state(&self) -> u32 {
        self.u32_at(0x18)
    }
    /// Frames left in the converter, frames consumed, 0, input frames not yet converted.
    pub fn state_data(&self) -> [u16; 4] {
        let mut out = [0u16; 4];
        for (i, v) in out.iter_mut().enumerate() {
            *v = u16::from_le_bytes([self.0[0x1C + 2 * i], self.0[0x1D + 2 * i]]);
        }
        out
    }
    pub fn execution_time_us(&self) -> u64 {
        u64::from(self.u32_at(0x24)) | u64::from(self.u32_at(0x28)) << 32
    }
    pub fn code(&self) -> Hresult {
        self.u32_at(0x2C) as i32
    }
    pub fn frames_written(&self) -> u32 {
        self.u32_at(0x30)
    }
    pub fn bytes_consumed(&self) -> u32 {
        self.u32_at(0x34)
    }
    pub fn pitch(&self) -> f32 {
        f32::from_bits(self.u32_at(0x40))
    }
}
