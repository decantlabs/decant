//! The Linux launch driver of the GDK host layer and the sandbox runners, the ports of
//! `src/gdk/launch_prep.sh`, `sandbox/run_airgapped.sh` and `sandbox/test_isolation.sh`.
//!
//! The binaries keep the scripts' commands, environment variables, messages and exit codes.
//! They still drive the same external programs (curl, tar, minisign, make, Proton, Wine and
//! docker) with the same arguments. The archive hash is computed here (SHA-256) instead of by
//! `sha256sum`. `DECANT_DRY_RUN=1` (or `--dry-run` first) prints each external command and file
//! change instead of making it.

pub mod ctx;
pub mod prep;
pub mod sandbox;

#[cfg(test)]
mod tests;
