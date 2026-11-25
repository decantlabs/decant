//! What `gpu_diag` reads about this PC (adapters, driver versions, Windows, CPU, memory) and the
//! allowlisted record it prints, as a library: the Decant Launcher shows the same adapter facts
//! in its checks.

pub mod probe;
pub mod record;
pub mod sanitize;
