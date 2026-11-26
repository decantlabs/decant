//! `AcpHal.dll`: the audio control processor library of the console, in Rust (`docs/CONSOLE_LIBS.md`, section 2).
//!
//! It holds the APU heap allocator (`heap`), the SHAPE context arrays with their service thread and engine state (`shape`), the
//! `IAcpHal` object with its command and message rings (`hal`), the ABI constants and structures (`abi`), the 10 exports
//! (`exports`), the XMA engine that decodes the enabled XMA contexts into their output rings (`xma`), and the monitor that plays
//! what it decodes (`monitor`). The SRC, EQ, filter and DMA contexts are not run.
#![allow(non_snake_case)]

pub mod abi;
pub mod exports;
pub mod hal;
pub mod heap;
pub mod monitor;
pub mod priority;
pub mod shape;
pub mod stats;
pub mod xma;

pub use exports::EXPORT_NAMES;
