//! Offline Accessibility table: `api_id = {3E241977-6237-41E9-8559-102C2D9983F1}`, interface
//! `{D722B373-8C4D-4692-9E51-D6AD9B37AA7D}`. Titles ask for it while their boot graph completes
//! (`decant-specs/hle-gap.md` row 11). The answers are plain offline defaults:
//!
//! * Closed captions are off until the title enables them with `XClosedCaptionSetEnabled`. The
//!   style is white text on black, default edge and font style, scale 1.
//! * High contrast is off.
//! * Speech to text accepts every call and sends it nowhere, as there is no party chat.
//! * Speech synthesis has no installed voice. Synthesizers and streams are fixed handles, and every
//!   stream (from text or from SSML) is a WAVE file without samples (44 bytes, 16 kHz, mono,
//!   16 bit), so narration is silent and the title carries on.
//!
//! Slot order follows the thunks of the public PC GDK `xgameruntime.lib` for this interface (slots
//! 3 to 21). Slots 22 and 23 log a named "not implemented" (`local_title`). Every call is logged
//! with a bound (`calllog`). Signatures come from the public GDK `XAccessibility.h` and
//! `XSpeechSynthesizer.h`.
#![allow(unsafe_code)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::calllog::{Counters, logged_shim};
use crate::local_title::{E_NOT_SUFFICIENT_BUFFER, LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, put, query_interface, slot, token};
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, S_OK};

/// The table's `api_id`.
pub(crate) const XACCESSIBILITY_TABLE: Guid = Guid { data1: 0x3E241977, data2: 0x6237, data3: 0x41E9, data4: [0x85, 0x59, 0x10, 0x2C, 0x2D, 0x99, 0x83, 0xF1] };
/// The interface the title asks for.
pub(crate) const XACCESSIBILITY_IID: Guid = Guid { data1: 0xD722B373, data2: 0x8C4D, data3: 0x4692, data4: [0x9E, 0x51, 0xD6, 0xAD, 0x9B, 0x37, 0xAA, 0x7D] };

const NAME: &str = "Accessibility";
const SLOTS: usize = 24;
static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);

/// The object for the pair this table serves, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&XACCESSIBILITY_TABLE) && riid == Some(&XACCESSIBILITY_IID)).then(|| (OBJECT.as_out(), NAME))
}

struct Accessibility;

impl LocalTable for Accessibility {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[XACCESSIBILITY_IID];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        SLOT_NAMES.iter().find(|(number, _)| *number == slot).map(|&(_, name)| name)
    }
    fn object() -> *mut c_void {
        OBJECT.as_out()
    }
}
/// GDK function names by slot, only for the slots that are identified.
const SLOT_NAMES: [(usize, &str); 22] = [
    (0, "QueryInterface"),
    (1, "AddRef"),
    (2, "Release"),
    (3, "XClosedCaptionGetProperties"),
    (4, "XClosedCaptionSetEnabled"),
    (5, "XHighContrastGetMode"),
    (6, "XSpeechToTextSetPositionHint"),
    (7, "XSpeechToTextSendString"),
    (8, "XSpeechSynthesizerEnumerateInstalledVoices"),
    (9, "XSpeechSynthesizerCreate"),
    (10, "XSpeechSynthesizerCloseHandle"),
    (11, "XSpeechSynthesizerSetDefaultVoice"),
    (12, "XSpeechSynthesizerSetCustomVoice"),
    (13, "XSpeechSynthesizerCreateStreamFromText"),
    (14, "XSpeechSynthesizerCloseStreamHandle"),
    (15, "XSpeechSynthesizerGetStreamDataSize"),
    (16, "XSpeechSynthesizerGetStreamData"),
    (17, "XSpeechToTextBeginHypothesisString"),
    (18, "XSpeechToTextUpdateHypothesisString"),
    (19, "XSpeechToTextFinalizeHypothesisString"),
    (20, "XSpeechToTextCancelHypothesisString"),
    (21, "XSpeechSynthesizerCreateStreamFromSsml"),
];

/// `XColor`: A, R, G, B.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Color {
    pub(crate) a: u8,
    pub(crate) r: u8,
    pub(crate) g: u8,
    pub(crate) b: u8,
}

/// `XClosedCaptionProperties`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClosedCaptionProperties {
    pub(crate) background_color: Color,
    pub(crate) font_color: Color,
    pub(crate) window_color: Color,
    pub(crate) font_edge_attribute: u32,
    pub(crate) font_style: u32,
    pub(crate) font_scale: f32,
    pub(crate) enabled: bool,
}

/// `XHighContrastMode::Off`.
pub(crate) const HIGH_CONTRAST_OFF: u32 = 0;

/// A WAVE file of 16 kHz mono 16 bit PCM with no samples: RIFF size 36, a format chunk of 16
/// bytes (PCM, 1 channel, 16000 Hz, 32000 bytes/s, block 2, 16 bits), an empty data chunk.
pub(crate) const SILENCE: [u8; 44] = *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";

static CAPTIONS: AtomicBool = AtomicBool::new(false);
static SYNTHESIZER: u8 = 0;
static STREAM: u8 = 0;

type This = *const c_void;
type Handle = *mut c_void;

logged_shim!(NAME, COUNTS, This, 3 as "XClosedCaptionGetProperties", captions_get_entry => read_caption_style, (out: *mut ClosedCaptionProperties) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XClosedCaptionSetEnabled", captions_set_entry => switch_captions, (enabled: bool) -> i32);
logged_shim!(NAME, COUNTS, This, 5 as "XHighContrastGetMode", high_contrast_entry => read_contrast_mode, (out: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 6 as "XSpeechToTextSetPositionHint", stt_position_entry => acknowledge_id, (position: u32) -> i32);
logged_shim!(NAME, COUNTS, This, 7 as "XSpeechToTextSendString", stt_send_entry => drop_speech_text, (speaker: *const c_char, content: *const c_char, kind: u32) -> i32);
logged_shim!(NAME, COUNTS, This, 8 as "XSpeechSynthesizerEnumerateInstalledVoices", voices_entry => list_no_voices, (context: *mut c_void, callback: usize) -> i32);
logged_shim!(NAME, COUNTS, This, 9 as "XSpeechSynthesizerCreate", create_entry => new_synthesizer, (out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 10 as "XSpeechSynthesizerCloseHandle", close_entry => acknowledge_handle, (handle: Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 11 as "XSpeechSynthesizerSetDefaultVoice", default_voice_entry => acknowledge_handle, (handle: Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 12 as "XSpeechSynthesizerSetCustomVoice", custom_voice_entry => acknowledge_custom_voice, (handle: Handle, voice: *const c_char) -> i32);
logged_shim!(NAME, COUNTS, This, 13 as "XSpeechSynthesizerCreateStreamFromText", stream_text_entry => new_silent_stream, (handle: Handle, text: *const c_char, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 14 as "XSpeechSynthesizerCloseStreamHandle", close_stream_entry => acknowledge_handle, (handle: Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 15 as "XSpeechSynthesizerGetStreamDataSize", stream_size_entry => silent_stream_size, (handle: Handle, out: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 16 as "XSpeechSynthesizerGetStreamData", stream_data_entry => copy_silent_stream, (handle: Handle, size: usize, buffer: *mut u8, used: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 17 as "XSpeechToTextBeginHypothesisString", stt_begin_entry => begin_hypothesis, (speaker: *const c_char, content: *const c_char, kind: u32, out: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 18 as "XSpeechToTextUpdateHypothesisString", stt_update_entry => update_hypothesis, (id: u32, content: *const c_char) -> i32);
logged_shim!(NAME, COUNTS, This, 19 as "XSpeechToTextFinalizeHypothesisString", stt_finalize_entry => acknowledge_id, (id: u32) -> i32);
logged_shim!(NAME, COUNTS, This, 20 as "XSpeechToTextCancelHypothesisString", stt_cancel_entry => acknowledge_id, (id: u32) -> i32);
logged_shim!(NAME, COUNTS, This, 21 as "XSpeechSynthesizerCreateStreamFromSsml", stream_ssml_entry => new_silent_stream, (handle: Handle, ssml: *const c_char, out: *mut Handle) -> i32);

unsafe fn read_caption_style(out: *mut ClosedCaptionProperties) -> i32 {
    put(
        out,
        ClosedCaptionProperties {
            background_color: Color { a: 0xFF, r: 0, g: 0, b: 0 },
            font_color: Color { a: 0xFF, r: 0xFF, g: 0xFF, b: 0xFF },
            window_color: Color { a: 0, r: 0, g: 0, b: 0 },
            font_edge_attribute: 0,
            font_style: 0,
            font_scale: 1.0,
            enabled: CAPTIONS.load(Ordering::Acquire),
        },
    )
}

unsafe fn switch_captions(enabled: bool) -> i32 {
    CAPTIONS.store(enabled, Ordering::Release);
    S_OK
}

unsafe fn read_contrast_mode(out: *mut u32) -> i32 {
    put(out, HIGH_CONTRAST_OFF)
}

unsafe fn acknowledge_id(_value: u32) -> i32 {
    S_OK
}

unsafe fn acknowledge_handle(_handle: Handle) -> i32 {
    S_OK
}

unsafe fn drop_speech_text(_speaker: *const c_char, _content: *const c_char, _kind: u32) -> i32 {
    S_OK
}

unsafe fn begin_hypothesis(_speaker: *const c_char, _content: *const c_char, _kind: u32, out: *mut u32) -> i32 {
    put(out, token() as u32)
}

unsafe fn update_hypothesis(_id: u32, _content: *const c_char) -> i32 {
    S_OK
}

/// No voice is installed: the callback never runs.
unsafe fn list_no_voices(_context: *mut c_void, _callback: usize) -> i32 {
    S_OK
}

unsafe fn new_synthesizer(out: *mut Handle) -> i32 {
    put(out, (&raw const SYNTHESIZER).cast_mut().cast())
}

unsafe fn acknowledge_custom_voice(_handle: Handle, _voice: *const c_char) -> i32 {
    S_OK
}

unsafe fn new_silent_stream(_handle: Handle, _text: *const c_char, out: *mut Handle) -> i32 {
    put(out, (&raw const STREAM).cast_mut().cast())
}

unsafe fn silent_stream_size(_handle: Handle, out: *mut usize) -> i32 {
    put(out, SILENCE.len())
}

unsafe fn copy_silent_stream(_handle: Handle, size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    if buffer.is_null() {
        return E_INVALIDARG;
    }
    if size < SILENCE.len() {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe { buffer.copy_from_nonoverlapping(SILENCE.as_ptr(), SILENCE.len()) };
    if !used.is_null() {
        put(used, SILENCE.len());
    }
    S_OK
}

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<Accessibility>),
    slot!(no_count),
    slot!(no_count),
    slot!(captions_get_entry),
    slot!(captions_set_entry),
    slot!(high_contrast_entry),
    slot!(stt_position_entry),
    slot!(stt_send_entry),
    slot!(voices_entry),
    slot!(create_entry),
    slot!(close_entry),
    slot!(default_voice_entry),
    slot!(custom_voice_entry),
    slot!(stream_text_entry),
    slot!(close_stream_entry),
    slot!(stream_size_entry),
    slot!(stream_data_entry),
    slot!(stt_begin_entry),
    slot!(stt_update_entry),
    slot!(stt_finalize_entry),
    slot!(stt_cancel_entry),
    slot!(stream_ssml_entry),
    slot!(not_implemented::<Accessibility, 22>),
    slot!(not_implemented::<Accessibility, 23>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::E_NOTIMPL;
    use crate::exports::query_without_runtime;
    use crate::local_title::method;

    fn table() -> *mut c_void {
        let mut out = std::ptr::null_mut();
        assert_eq!(query_without_runtime(Some(&XACCESSIBILITY_TABLE), Some(&XACCESSIBILITY_IID), &mut out), (S_OK, Some(NAME)));
        out
    }

    #[test]
    fn captions_and_contrast_have_sane_defaults() {
        let t = table();
        type Get = unsafe extern "system" fn(This, *mut ClosedCaptionProperties) -> i32;
        type Set = unsafe extern "system" fn(This, bool) -> i32;
        type Mode = unsafe extern "system" fn(This, *mut u32) -> i32;
        assert_eq!(size_of::<ClosedCaptionProperties>(), 28);
        let mut p = std::mem::MaybeUninit::<ClosedCaptionProperties>::uninit();
        let mut mode = 7;
        // SAFETY: the slots get their GDK arguments.
        unsafe {
            assert_eq!(method::<Get>(t, 3)(t, p.as_mut_ptr()), S_OK);
            let p = p.assume_init();
            assert_eq!((p.font_color, p.font_scale), (Color { a: 0xFF, r: 0xFF, g: 0xFF, b: 0xFF }, 1.0));
            assert_eq!(method::<Set>(t, 4)(t, false), S_OK);
            assert_eq!(method::<Mode>(t, 5)(t, &mut mode), S_OK);
            assert_eq!(mode, HIGH_CONTRAST_OFF);
            assert_eq!(method::<Mode>(t, 5)(t, std::ptr::null_mut()), E_INVALIDARG);
        }
    }

    #[test]
    fn speech_is_an_empty_wave_file() {
        let t = table();
        type Create = unsafe extern "system" fn(This, *mut Handle) -> i32;
        type Stream = unsafe extern "system" fn(This, Handle, *const c_char, *mut Handle) -> i32;
        type Size = unsafe extern "system" fn(This, Handle, *mut usize) -> i32;
        type Data = unsafe extern "system" fn(This, Handle, usize, *mut u8, *mut usize) -> i32;
        type Close = unsafe extern "system" fn(This, Handle) -> i32;
        type Unknown = unsafe extern "system" fn(This) -> i32;
        let (mut synth, mut stream, mut size, mut used) = (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0);
        // SAFETY: the slots get their GDK arguments.
        unsafe {
            assert_eq!(method::<Create>(t, 9)(t, &mut synth), S_OK);
            for slot in [13, 21] {
                assert_eq!(method::<Stream>(t, slot)(t, synth, c"Hello".as_ptr(), &mut stream), S_OK);
                assert_eq!(method::<Size>(t, 15)(t, stream, &mut size), S_OK);
                let mut wav = vec![0u8; size];
                assert_eq!(method::<Data>(t, 16)(t, stream, size, wav.as_mut_ptr(), &mut used), S_OK);
                assert_eq!((&wav[..4], &wav[8..16], &wav[36..40], used), (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..], 44));
                assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
                assert_eq!(method::<Data>(t, 16)(t, stream, 10, wav.as_mut_ptr(), &mut used), E_NOT_SUFFICIENT_BUFFER);
                assert_eq!(method::<Close>(t, 14)(t, stream), S_OK);
            }
            assert_eq!(method::<Close>(t, 10)(t, synth), S_OK);
            assert_eq!(method::<Unknown>(t, 22)(t), E_NOTIMPL);
            assert_eq!(method::<Unknown>(t, 23)(t), E_NOTIMPL);
        }
    }
}
