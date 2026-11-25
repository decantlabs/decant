# Sound and input for a console GDK title on PC

Status: the layers are built and tested synthetically. Sources are the public Windows SDK header `xaudio2.h`, public Microsoft documentation and the public GDK header `GameInput.h` (October 2026 GDK, read from a local copy that is never committed). No console SDK file is used. Where Microsoft documents something only behind the Xbox developer agreement, this page says so and the layer does not implement it. The console audio libraries (AcpHal, XAPU, chad) are specified in [CONSOLE_LIBS.md](CONSOLE_LIBS.md).

## 1. XAudio2

### 1.1 What the public sources say

| # | Difference | Source | Layer |
|---|---|---|---|
| 1 | XMA is "only supported on Xbox One", XAudio 2.9 on PC mixes PCM, 32 bit float, ADPCM and xWMA | [XAudio 2.9 redistributable guide](https://learn.microsoft.com/en-us/windows/win32/xaudio2/xaudio2-redistributable) | `CreateSourceVoice` with format tag 0x0165 or 0x0166 is rejected with `XAUDIO2_E_INVALID_CALL` and logged, decoder hook `xma2_hook.rs` of `crates/decant-xaudio2` (1.4) |
| 2 | "XAudio 2.9 will choose a different CPU core on Xbox One than on PC", the processor argument of `XAudio2Create` and `IXAudio2Extension::GetProcessor` exist for this | same page, `xaudio2.h` (`XAUDIO2_USE_DEFAULT_PROCESSOR`, `IXAudio2Extension`) | a nondefault processor mask is logged and passed to the PC engine, `IXAudio2Extension` is answered by the PC engine |
| 3 | Console builds include `XAUDIO2.H`, and "some Xbox One-specific APIs are excluded from XAUDIO2REDIST.H". The page does not list them | same page | an interface the PC engine does not answer is returned as `E_NOINTERFACE` and its IID is logged, so the first console only interface a title asks for shows up |
| 4 | PC only: `XAUDIO2_NO_VIRTUAL_AUDIO_CLIENT`, `AUDIO_STREAM_CATEGORY`, `XAUDIO2_E_DEVICE_INVALIDATED`, `IXAudio2EngineCallback::OnCriticalError` | same page, `xaudio2.h` | flags other than the documented ones are logged, and if the PC engine refuses them the call is repeated without them |
| 5 | The 2.9 redistributable does not implement `CreateHrtfApo` (Windows 10 only) | same page | not exported by the layer |
| 6 | "On most PCs, XAudio 2.9 processes audio in 10 millisecond chunks" (`IXAudio2Extension::GetProcessingQuantum`) | same page | the PC value is returned, the console value is not public |

**Not covered, because the pages are not public.** The Xbox One audio architecture and SHAPE overview pages of the GDK documentation answer "Access restricted" (Xbox developer agreement) when fetched without membership. The sharing of hardware audio processor contexts with XAudio2 is therefore not implemented and not described here. One observation, not a documented fact: the PC `XAudio2_9.dll` on the development machine exports a function named `XAudio2CreateWithSharedContexts`, and no public documentation of it was found.

### 1.2 What the layer does

The crate `crates/decant-xaudio2` builds `decant_xaudio2.dll`. It exports `XAudio2Create` (ordinal 1), `CreateAudioReverb`, `CreateAudioVolumeMeter` and `XAudio2CreateWithVersionInfo`, with the ordinals of the PC `XAudio2_9.dll` because console modules import by ordinal (`build.rs` of the crate sets them), and the two counter functions `DecantXa2GetStats` and `DecantXa2ResetStats`. It loads the PC engine `XAudio2_9.dll` (or `XAudio2_9Redist.dll`). The engine it returns wraps the PC `IXAudio2`: every method is forwarded, and the cases of 1.1 are handled and counted (`DecantXa2GetStats`). Voices are not wrapped, they are the PC engine's own.

Set `DECANT_LOG=1` (OutputDebugString) or `DECANT_LOG_PATH` (file) to see what is not handled, `DECANT_XA2_TRACE=1` also logs every forwarded call. Under Wine and Proton the engine below is the builtin XAudio2 of Wine (FAudio), that path is untested.

### 1.3 Observed on the PC engine (checked by `xaudio2test`)

* `CreateMasteringVoice` returned `0x800401F0` (`CO_E_NOTINITIALIZED`) on a thread without COM. Whether a console title calls `CoInitializeEx` is not public, so the layer calls it (multithreaded) once per thread that reaches it, and logs the first time.
* The PC engine answers `XAUDIO2_E_INVALID_CALL` (`0x88960001`) for flag bits it does not define, in both `XAudio2Create` and `CreateMasteringVoice`. A title that passes console only bits would fail, so the layer retries without them.
* `XAUDIO2_VOICE_STATE::SamplesPlayed` is reset to zero when an end of stream buffer finishes, so the test keeps the largest value it saw.

### 1.4 XMA2

The layer contains no decoder. `xma2_hook.rs` defines the C ABI of an optional decoder DLL (`DecantXma2GetDecoder`, named by `DECANT_XMA2_DECODER`). The crate `decant-xma2` builds one such DLL, `decant_xma2.dll`, over FFmpeg: it loads a user supplied LGPL shared build of FFmpeg at run time (`DECANT_FFMPEG_DIR`, else its own folder) and ships nothing of it. The pure Rust decoder `decant-xma` serves the XMA contexts of AcpHal. The hook is built, but nothing yet feeds decoded PCM into a source voice, because that needs a wrapper around `SubmitSourceBuffer` and is only worth writing once a title is seen submitting XMA2 buffers. The layer logs every rejected XMA buffer format, which answers whether a given title needs XMA2 at all.

### 1.5 Checked and assumed

Checked: the PC side facts of 1.1 (read in the cited pages and header), the synthetic tests (section 3) and the observations of 1.3. Assumed, not verifiable without a title: that the `IXAudio2` vtable of a console title equals the PC 2.9 one (the redistributable guide says 2.9 is meant to match, but the console header is not public), that it never calls the Xbox only APIs listed in the restricted pages, and the console processing quantum.

### 1.6 XWMA, XMA2, SHAPE and XAPU

* **xWMA** needs nothing from us: the same format works through the PC `XAudio2_9.dll` (public XAudio2 documentation), so a title using it should play as is. Not tested here, because `xaudio2test` invents no xWMA data.
* **XMA2** keeps the status of 1.4. New: the container is WAV like with a seek table that may be byte swapped, so a decoder fed from the layer must read the seek table in the byte order of the console, and the source buffers of a title may be aligned on the boundary of the console allocator. Both are assumed, not seen on a title.
* **SHAPE and XAPU**: only a title that calls `IACPHAL` or the `XApu*` functions needs interception. These are separate DLLs and functions from `XAudio2_9.dll`, and the layer does not export them. They are implemented by `decant-acphal` and `decant-xapu` from [CONSOLE_LIBS.md](CONSOLE_LIBS.md).

## 2. GameInput

The public GDK documentation says the **same GameInput API, headers and library apply to PC and Xbox** ([GameInput FAQ](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/common/input/overviews/input-faq?view=gdk-2604)). On Xbox the runtime is part of the game OS image, on PC it is the redistributable or the inbox `GameInput.dll`. Documented differences:

* **Dynamic Latency Input** exists on console only, and the API is unimplemented on PC. The public `GameInput.h` of the October 2026 GDK contains no symbol for it (checked with a text search), so no PC title can call it. A console title that does would need a stub, which is not written because there is nothing to forward to.
* Background input (`EnableBackgroundInput`, API version 4) is PC only. On console, input arrives only while the title has focus.
* API versions: the October 2025 console runtime supports versions up to 3 and the October 2026 one up to 4. A title built for a later version than the PC runtime offers needs the redistributable beside the executable.

The crate `decant-gameinput` builds `GameInput.dll`, a shim with one always connected gamepad driven by a control file, and `decant-pad`, the tool that writes to that file. See [its README](../crates/decant-gameinput/README.md). A keyboard to gamepad mapping, which a title built for controllers would need if the player has none, is a separate feature. `gameinputtest` (in the crate `decant-xaudio2`) checks on a PC that `GameInput.dll` exports `GameInputCreate`, that it returns an object and that the object answers `IUnknown`. It also requests the current gamepad and keyboard readings through the `IGameInput` vtable and accepts a reading or `GAMEINPUT_E_READING_NOT_FOUND`.

## 3. Tests

`build.cmd` builds the crate with `cargo build --release --locked` (statically linked C runtime) and copies `decant_xaudio2.dll`, `xaudio2test.exe` and `gameinputtest.exe` into `build`. `cargo test -p decant-xaudio2` runs both programs against the DLL cargo built.

```
build\xaudio2test.exe build\decant_xaudio2.dll
build\gameinputtest.exe
```

`xaudio2test` creates an engine with a console style flag and processor, checks the export ordinals, queries `IUnknown`, `IXAudio2Extension` and a made up IID, submits an XMA2 format (refused), then creates a mastering voice and a float32 mono 48 kHz source voice and plays a half second 440 Hz sine, checking the sample counter of the engine (24,000 of 24,000 on a machine with an audio endpoint). Without an audio endpoint (the hosted CI runner) it prints SKIP for the sine and passes, and `DECANT_XA2_REQUIRE_DEVICE=1` turns the skip into a failure. Counters are checked through `DecantXa2GetStats`.
