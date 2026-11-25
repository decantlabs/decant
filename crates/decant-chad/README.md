# decant-chad

`chad.dll`, the XDSP library of the console, in Rust (specification in `docs/CONSOLE_LIBS.md`, section 4).

It exposes partitioned convolution (reverb) and forward and inverse FFT streams.

## Exports

All seven are exported by name, with the prototypes of the specification: `XDspConnect`, `XDspConnectWithMaximumStreamLimit`, `XDspDisconnect`, `XDspActivate`, `XDspSubmitCommand`, `XDspSubmitCommandWithEnvelope` and `XDspDeactivate`. The specification gives no ordinals, so there is no `build.rs` and no `.def` file. `build.cmd` builds the DLL with the static C runtime like the other shims.

## Layout of the code

* `abi`: result codes, limits, the activation parameters (0x1C bytes, read at the offsets of the specification) and the 0x20 byte command.
* `fft`: a radix 2 complex transform in double precision and the packed real format of the specification (DC and Nyquist in the first pair, then the pairs of bins 1 up to N/2 minus 1). The forward transform is unscaled, the inverse divides by N.
* `engine`: clients (limits over all clients, memory ranges), contexts (activation rules, the impulse response partitions), the status array and the work: uniformly partitioned overlap save convolution with the envelope of the specification, FFT and inverse FFT with gains, block multiplier and polar format. Values that are not finite become 0.
* `exports`: the seven exports.
* Logging goes through `decant-log` (`DECANT_LOG`, `DECANT_LOG_PATH`). Connections and every failing call are logged.

Work completes synchronously inside the submit call, so the status array shows completed equal to submitted as soon as the call returns.

## Open points

The specification marks the polar order and the envelope storage as inferred, and leaves these readings open:

* The impulse response length is read as the length of each response. With flag 8 the second response follows the first and the same length is read again.
* "Partitions below 9" is read literally: an impulse response of fewer than 9 partitions gives `XDSP_E_IR_SIZE`.
* The input of a stereo convolution is interleaved unless flag 2 is set, the output is always the channels one after the other, as the specification says.
* The input gain and the output gain are applied to the FFT and the inverse FFT as well, the gain of a polar spectrum scales the magnitude only. The block multiplier is read for the forward FFT only.
* The impulse response is read from the caller's pointer without a range check, the specification gives no memory for it.
* Nothing sets the sticky error in status word 3 (work is synchronous and cannot fail), the check is there for callers that write it. `XDSP_E_NO_SLOT` is returned when the submitted count is 8 or more ahead of the completed count, which synchronous work reaches only when the caller edits the status words. `XDSP_E_BUSY` is defined and not used. The limit of 0x100 contexts cannot be reached because the activations of all clients are limited to 0x80. The stream limit and the priority of a client are stored and logged, the specification gives them no effect.

## Tests

`cargo test -p decant-chad` runs the transform tests (against the definition of the DFT, round trips, known signals, circular convolution), the connection and activation rules (every error code and order, limits over clients, contexts and the size of the impulse response), the DSP work (FFT, inverse FFT and polar format against the DFT, gains, the multiplier, non finite values, submit validation, sequence numbers, the sticky error and the slot limit, convolution against direct convolution block after block for mono, stereo, interleaved and one or two impulse responses, metadata partitions, the envelope groups and the right channel gains) and the load of the built DLL.
