//! `xapu.dll`: the XAPU library of the console, in Rust (`docs/CONSOLE_LIBS.md`, section 3).
//!
//! Hardware Opus decoding (through the `opus-decoder` crate of the workspace), decoding followed by pitch converting sample rate
//! conversion, and PCM sample rate conversion with a pitch ramp. `abi` holds the constants and the binary layouts, `converter`
//! the sample rate converter, `engine` the connections, streams and command queues, `exports` the six exports.
//! Written from the specification only. The console filter coefficients are unknown, the converter is a windowed sinc.

pub mod abi;
pub mod converter;
pub mod engine;
pub mod exports;

pub use exports::EXPORT_NAMES;
