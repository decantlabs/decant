//! `decant_xaudio2.dll`: a console title facing XAudio2 front that hands every call to the PC engine
//! (`XAudio2_9.dll`). A port of the former `src/decant/xaudio2/decant_xa2.cpp`, with the same
//! exports, counters and log lines.
//!
//! Why it can be thin: the Windows XAudio2 2.9 engine interface (`IXAudio2`) is the one the
//! console build exposes, and the public redistributable documentation says XAudio 2.9 is meant
//! to behave the same on both. The layer therefore wraps the engine (not the voices) and spends
//! its code where a console title can differ from the PC engine: the creation flags and processor
//! mask, interfaces queried by IID, formats (XMA and XMA2 exist only on Xbox, see `xma2_hook`)
//! and mastering voice flags. Anything the PC engine does not know is logged, never silently
//! dropped.
//!
//! Source: only the public Windows SDK header `xaudio2.h` (through the `windows` crate) and public
//! documentation. No console SDK file, no game file. The rules are plain functions here. The COM
//! wrapper and the exports are in `engine` and `exports`, the only modules with `unsafe` code.

#[cfg(windows)]
mod engine;
#[cfg(windows)]
mod exports;
pub mod stats;
pub mod xma2_hook;

pub use stats::DecantXa2Stats;

/// Flag bits the PC engine defines for `XAudio2Create`: `XAUDIO2_DEBUG_ENGINE` and
/// `XAUDIO2_STOP_ENGINE_WHEN_IDLE`.
pub const CREATE_KNOWN: u32 = 0x0001 | 0x2000;
/// `CreateMasteringVoice`: only `XAUDIO2_NO_VIRTUAL_AUDIO_CLIENT` is documented for the PC engine.
pub const MASTERING_KNOWN: u32 = 0x10000;
/// `XAUDIO2_USE_DEFAULT_PROCESSOR`.
pub const USE_DEFAULT_PROCESSOR: u32 = 0;
/// `XAUDIO2_E_INVALID_CALL`.
pub const XAUDIO2_E_INVALID_CALL: i32 = 0x8896_0001_u32 as i32;

/// XMA (0x0165) and XMA2 (0x0166).
pub fn is_xma(tag: u16) -> bool {
    tag == 0x0165 || tag == 0x0166
}

/// Format tags the PC engine mixes: PCM, IEEE float, ADPCM, the three xWMA tags, and
/// `WAVE_FORMAT_EXTENSIBLE`, which is judged by the engine.
pub fn is_pc_format(tag: u16) -> bool {
    matches!(tag, 0x0001 | 0x0003 | 0x0002 | 0xFFFE | 0x0161 | 0x0162 | 0x0163)
}

/// Formats the compact source voice creation diagnostic.
pub fn source_format_line(tag: u16, channels: u16, rate: u32, bits: u16, block: u16) -> String {
    format!("CreateSourceVoice: format tag 0x{tag:04X}, channels {channels}, sample rate {rate} Hz, bits {bits}, block size {block}")
}

/// A GUID written as the C layer logged it: `{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`, upper case.
pub fn guid_text(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> String {
    let d = data4;
    format!(
        "{{{data1:08X}-{data2:04X}-{data3:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

/// Cuts `text` to at most `max` bytes, on a character boundary.
fn truncated(text: &str, max: usize) -> &str {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The line the layer logs for `message`: prefixed with `[decant_xa2] `, with the limits of the C
/// buffers (511 bytes of message, 559 bytes in all).
pub fn log_line(message: &str) -> String {
    let line = format!("[decant_xa2] {}", truncated(message, 511));
    truncated(&line, 559).to_owned()
}

/// Logs one line: to the debugger with `DECANT_LOG`, and to the session log.
pub fn log(message: &str) {
    decant_log::trace(&log_line(message));
}

/// Whether `DECANT_XA2_TRACE` asks for every forwarded call (read once, as the C layer did).
pub fn trace_all() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("DECANT_XA2_TRACE").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert!(is_xma(0x0166) && is_xma(0x0165) && !is_xma(0x0001));
        for tag in [1, 2, 3, 0xFFFE, 0x161, 0x162, 0x163] {
            assert!(is_pc_format(tag), "{tag:#x}");
        }
        assert!(!is_pc_format(0x0166) && !is_pc_format(0x0055));
    }

    #[test]
    fn source_format_log_line_has_all_format_fields() {
        assert_eq!(source_format_line(0x0166, 2, 48000, 16, 4), "CreateSourceVoice: format tag 0x0166, channels 2, sample rate 48000 Hz, bits 16, block size 4");
    }

    #[test]
    fn guids_are_written_as_the_c_layer_wrote_them() {
        let text = guid_text(0x5ca1ab1e, 0x0001, 0x4000, [0x80, 0x00, 0x53, 0x49, 0x58, 0x58, 0x41, 0x32]);
        assert_eq!(text, "{5CA1AB1E-0001-4000-8000-534958584132}");
    }

    #[test]
    fn lines_keep_the_c_limits() {
        assert_eq!(log_line("-> StartEngine"), "[decant_xa2] -> StartEngine");
        assert_eq!(log_line(&"a".repeat(600)).len(), 13 + 511);
        assert_eq!(log_line(&"é".repeat(300)).len(), 13 + 510);
    }

    #[test]
    fn known_flags() {
        assert_eq!(0x4000 & !CREATE_KNOWN, 0x4000);
        assert_eq!(0x40000 & !MASTERING_KNOWN, 0x40000);
        assert_eq!(XAUDIO2_E_INVALID_CALL as u32, 0x8896_0001);
    }
}
