//! Shared, safe parts of the DecantShader GPU benches `decant_shader_run` (Vulkan) and `vkd3d_test`
//! (Direct3D 12 through vkd3d-proton).
//!
//! Everything that touches the GPU lives in the binaries, in one module each that allows
//! `unsafe`. This library holds what can be checked without a GPU: the command line of
//! `decant_shader_run`, the `--layout` file and its checks, and the text the C runtime would print
//! (`%.4f`, `atoi`, `strtoul`, line endings), so that both benches print byte for byte what the
//! C++ printed.

pub mod cfmt;
pub mod layout;
pub mod run_args;

#[cfg(test)]
mod tests;

/// A failure that ends the program: the message goes to stderr, then the process exits with
/// `code`. The codes are the ones of the C++ benches (1 usage, 2 file, 3 loading, 4 GPU call,
/// 5 skipped resource or GPU timeout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exit {
    pub code: i32,
    pub message: String,
}

impl Exit {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Reads a whole input file, failing as the C++ `ReadFile` did (`lecture impossible`, exit 2).
pub fn read_file(path: &std::ffi::OsStr) -> Result<Vec<u8>, Exit> {
    std::fs::read(path).map_err(|_| {
        Exit::new(
            2,
            format!("lecture impossible : {}", path.to_string_lossy()),
        )
    })
}

/// Writes `text` (lines ending in `\n`) to stdout as the C runtime does in text mode, then
/// flushes, so that output and exit stay in the C++ order.
pub fn out(text: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(cfmt::text_mode(text).as_bytes());
    let _ = stdout.flush();
}

/// Same as [`out`] for stderr.
pub fn err(text: &str) {
    use std::io::Write;
    let mut stderr = std::io::stderr().lock();
    let _ = stderr.write_all(cfmt::text_mode(text).as_bytes());
    let _ = stderr.flush();
}

/// Ends the program for `result`: prints the message of a failure on stderr and returns its code.
pub fn finish(result: Result<i32, Exit>) -> std::process::ExitCode {
    let code = match result {
        Ok(code) => code,
        Err(exit) => {
            err(&format!("{}\n", exit.message));
            exit.code
        }
    };
    std::process::ExitCode::from(u8::try_from(code).unwrap_or(1))
}

/// The three vertices of the full screen triangle: position, colour, texture coordinate.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

impl Vertex {
    /// Byte offsets of the three attributes, as `offsetof` gave them.
    pub const POSITION: u32 = 0;
    pub const COLOR: u32 = 12;
    pub const UV: u32 = 28;
    pub const SIZE: u32 = 36;

    /// The vertices as bytes, in memory order.
    pub fn bytes(vertices: &[Vertex]) -> Vec<u8> {
        vertices
            .iter()
            .flat_map(|v| v.position.iter().chain(&v.color).chain(&v.uv))
            .flat_map(|f| f.to_ne_bytes())
            .collect()
    }
}

/// Bench texture texel: `r = 40i + 10`, `g = 60j + 20`, `b = 15(i + 4j) + 7k`, `a = 255`,
/// with `i`, `j` the texel position in a 4 x 4 layer and `k` the layer (or depth slice).
pub fn pattern_texel(t: u32) -> [u8; 4] {
    let (i, j, k) = (t % 4, (t / 4) % 4, t / 16);
    [
        (40 * i + 10) as u8,
        (60 * j + 20) as u8,
        (15 * (i + 4 * j) + 7 * k) as u8,
        255,
    ]
}

/// `d3d12core.dll` in the folder of the given `d3d12.dll` (the whole name is replaced when it
/// has no folder), as `vkd3d_test` loads it first so that Windows does not take its own.
pub fn d3d12core_beside(dll: &str) -> String {
    match dll.rfind(['\\', '/']) {
        Some(slash) => format!("{}d3d12core.dll", &dll[..=slash]),
        None => "d3d12core.dll".to_owned(),
    }
}

/// Options of `vkd3d_test`, read only at `argv[5]` as the C++ did: `--cbv <register> <file>`
/// (a root constant buffer) needs 8 arguments, `--srv <register>` (a texture in a descriptor
/// table) needs 7. Anything else is ignored. Registers are `atoi` values, negative meaning absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vkd3dOptions {
    pub cbv_register: i32,
    pub cbv_file: Option<std::ffi::OsString>,
    pub srv_register: i32,
}

impl Vkd3dOptions {
    pub fn parse(args: &[std::ffi::OsString]) -> Self {
        let mut options = Self {
            cbv_register: -1,
            cbv_file: None,
            srv_register: -1,
        };
        let option = args
            .get(5)
            .map(|a| a.to_string_lossy().into_owned())
            .unwrap_or_default();
        if args.len() >= 8 && option == "--cbv" {
            options.cbv_register = cfmt::atoi(&args[6].to_string_lossy());
            options.cbv_file = Some(args[7].clone());
        }
        if args.len() >= 7 && option == "--srv" {
            options.srv_register = cfmt::atoi(&args[6].to_string_lossy());
        }
        options
    }
}

/// A constant buffer padded with zeros to a multiple of 256 bytes, as Direct3D 12 requires.
pub fn pad256(mut data: Vec<u8>) -> Vec<u8> {
    data.resize(data.len().div_ceil(256) * 256, 0);
    data
}

/// Native endian 32 bit floats of a readback buffer.
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

/// `texels` texels of [`pattern_texel`], RGBA8.
pub fn pattern(texels: u32) -> Vec<u8> {
    (0..texels).flat_map(pattern_texel).collect()
}
