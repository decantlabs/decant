//! `chad.dll`: the XDSP library of the console, in Rust (issue #139, `docs/CONSOLE_LIBS.md` section 4).
//!
//! Partitioned convolution (reverb) and forward and inverse FFT streams. `fft` holds the transforms in the packed real format of
//! the specification, `engine` the clients, contexts and the DSP work, `abi` the constants and structures and `exports` the seven
//! exports. Written from the specification only. Work completes synchronously inside the submit call.

pub mod abi;
pub mod engine;
pub mod exports;
pub mod fft;

pub use exports::EXPORT_NAMES;
