//! Console texture and surface layout: the GFX10 (RDNA2) tiling modes and tables, the surface
//! layout calculation, and the `xg` module that computes resource layouts the way the console
//! `xg_xs.dll` does. No archive or asset format is read here.

pub mod surface;
pub mod tiling;
pub mod xg;
