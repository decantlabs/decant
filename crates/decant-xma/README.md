# decant-xma

A pure Rust decoder for XMA2 audio, the format used by Xbox titles. It takes a buffer of 2048 byte packets of one stream, walks the frames by bit position and decodes them to interleaved 16 bit PCM. Streams of one or two channels are supported, with frames of 512 samples per channel.

The crate has three parts. The `stream` module finds frames inside packets, `Decoder` turns frames into samples, and the `synth` module builds small valid streams for tests.

Licensed under the MIT license.
