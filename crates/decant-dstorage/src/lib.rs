//! `dstorage_xs.dll` and `xcrdapi.dll`: the DirectStorage layer of Decant. A port of the former
//! `src/gdk/dstorage`, with the same exports (`DStorageGetFactory`, ordinal 1), behaviour and log
//! text.
//!
//! On Xbox a title loads its assets through DirectStorage, with console interfaces
//! (`IDStorageFactoryX`, `IDStorageQueueX`, ...) taken from `DStorageGetFactory`. This DLL
//! provides that entry point on PC, over PC DirectStorage when `dstorage.dll` is beside the
//! application, or a CPU fallback otherwise. The interface IDs and the method order of the console
//! are not public: they are read from a spec file (`spec`, `console`, format in
//! `docs/DSTORAGE_ABI.md`), and the queue engine (`engine`) never sees the console layout.
//!
//! External pieces, none of them needed to build:
//! - PC DirectStorage (`dstorage.dll`) is loaded at run time (`pc`).
//! - GDeflate decoding uses libdeflate with the NVIDIA additions, compiled and linked only when
//!   `DECANT_GDEFLATE_ROOT` is set at build time (`build.rs`, `libdeflate`).
//! - BCPack is decoded in process by the `bcpack` crate (`codecs`).

pub mod spec;

#[cfg(windows)]
mod codecs;
#[cfg(windows)]
mod console;
#[cfg(windows)]
mod engine;
#[cfg(windows)]
mod exports;
#[cfg(windows)]
#[cfg_attr(not(decant_gdeflate), allow(dead_code))]
mod fanout;
#[cfg(windows)]
pub mod libdeflate;
#[cfg(windows)]
mod log;
#[cfg(windows)]
mod pages;
#[cfg(windows)]
mod pc;
#[cfg(windows)]
mod pool;
#[cfg(windows)]
mod spans;
#[cfg(windows)]
pub mod decant;

#[cfg(windows)]
pub use exports::DStorageGetFactory;
