# decant-xapu

`xapu.dll`, the XAPU library of the console, in Rust (specification in `docs/CONSOLE_LIBS.md`, section 3).

It drives Opus decoding, decoding followed by pitch converting sample rate conversion, and PCM sample rate conversion with a pitch ramp, through connections with streams and command queues.

## Exports

All six are exported by name, with the prototypes of the specification: `XApuConnect`, `XApuDisconnect`, `XApuEnqueueCommand`, `XApuDequeueResult`, `XApuGetResultCount` and `XApuGetUtilization`. The specification gives no ordinals, so there is no `build.rs` and no `.def` file. `build.cmd` builds the DLL with the static C runtime like the other shims.

## Layout of the code

* `abi`: result codes, constants, `XAPU_CONNECT_PARAMS`, `XAPU_CONNECTION_REGION`, and the 0x40 byte command and 0x50 byte result images with accessors at the offsets of the specification.
* `converter`: the pitch converting sample rate converter. A windowed sinc of 10 taps with a group delay of 5 frames, so at pitch 1.0 the output is the input delayed by exactly 5 frames. The pitch moves toward the target by the ramp per output frame, the ramp is clamped to 2 to the power of minus 18 up to 8 and the pitch to 0.125 up to 8. The cutoff drops to 1 / pitch above pitch 1.
* `engine`: connections, streams, the per stream queue depth, sequence numbers, the result queue, the result event (an auto reset event, created with option 8) and the four command types.
* `exports`: the six exports.
* Opus decoding uses the `opus-decoder` crate of the workspace, at 48 kHz, mono or stereo.
* Logging goes through `decant-log` (`DECANT_LOG`, `DECANT_LOG_PATH`). Connections and every failing call or result are logged.

## How commands run

Commands run synchronously inside `XApuEnqueueCommand`, in the order they arrive, and their result is committed at once. Results therefore come out in sequence order and the event is signalled per command. The hardware runs streams in parallel, which a caller cannot tell apart from this.

A start command activates its stream at once, also when its result fails (a bad channel count or pitch). Data commands on such a stream fail their result with `E_INVALIDARG`. Stop deactivates the stream when its result is dequeued.

## Open points

* The coefficients of the console filter are unknown. The windowed sinc is the one the specification allows, so output matches in level, delay and pitch behaviour but not sample for sample.
* The specification names the layout of the PCM input only through the output, which is floats. The convert PCM input is read as 32 bit floats, interleaved or, with connection option 2, one channel after the other. The frame count of a data command is the number of input frames, and the input length must hold them.
* The fields "first frame" and "frame count" of a data command do not matter for decoding: a packet always writes all its frames.
* A flush feeds 5 frames of silence to the converter so the last input frames come out, writes everything that is pending and starts a new segment. The state bits are ours where the specification lists them without rules: output ready when frames were written, more output available when at least a block (or, on a flush, anything) is still pending, need more input otherwise, saturation when a written sample exceeds full scale (nothing is clipped).
* The shared memory region is allocated, zeroed and handed back, nothing uses it.
* `XAPU_E_INVALID_COMMAND` and `XAPU_E_PROCESSING_ERROR` are defined and not returned: an unknown command type gives `E_INVALIDARG` as the specification says.

## Tests

`cargo test -p decant-xapu` runs the converter unit tests (delay, flush, seam free splitting of the input, level and low frequency preservation at several pitches, ramps and clamps), the connection rules (every range, the region, the event, the shared stream budget), the command rules (every error code, sequence numbers, queue depth, the processing buffer check, result ordering and layout, stop and reset), the PCM path (whole blocks, flush, capacity, planar output, saturation), the Opus paths on synthetic CELT packets (decode, bad packets, decoder reset, decode then convert) and the load of the built DLL.
