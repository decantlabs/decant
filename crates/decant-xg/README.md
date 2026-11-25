# decant-xg

`xg_xs.dll`, the texture and resource layout library of the console, in Rust (specification in `docs/CONSOLE_LIBS.md`, section 1).

Given a texture description it says which GFX10 swizzle mode to use, how large the resource is, where each mip level and array slice lives, and the byte offset of any element. A title uses it to allocate texture memory and to place or tile data itself.

## Layout of the code

* `decant_texture::xg` holds the computation: swizzle mode choice, the mode and format tables, the address equations (the addrlib tables of `decant_texture::tiling` for the modes without exclusive or, the appendices of the specification for the others), the 1D, 2D and 3D layouts, the CMASK plane, the `XG_RESOURCE_LAYOUT` record, the conversion from a `D3D12_RESOURCE_DESC` and the descriptor functions.
* This crate wraps it in the console ABI: `exports` (the 24 exports with the prototypes of the specification), `abi` (the description structures), `state` (initialised flag, hardware version, exclusive locking) and `computer` (the texture computer object with its table of 12 methods).
* Logging goes through `decant-log` (`DECANT_LOG`, `DECANT_LOG_PATH`). Failures, hardware versions and calls of the stand ins are logged.

The export names are the ones of the specification. It gives no ordinals, so there is no `build.rs` and no `.def` file: rustc exports the `no_mangle` functions by name on the MSVC and the gnu targets. `build.cmd` builds the DLL with the static C runtime like the other shims.

## Open points

These stay as logging stand ins that return `E_NOTIMPL`, as the specification says:

* `XGEncodeDCC`, `XGDecodeDCC` and `XGCreateBVHComputer`.
* `XGCreateTextureComputerFromDescriptor`: the descriptor is read and validated, but the GFX10 image format table (element size per `IMG_FORMAT`) is not in the specification. Placeholder levels for descriptors whose mip count differs from the allocated count are not modelled either.
* The computer slots 7 (`GetTexelCoordinate`) and 11 (`GetTextureViewDescriptor`). Slot 10 is `GetResourceTiling` (`decant_texture::xg::resource_tiling`).
* Depth plus stencil formats (two planes) and the other formats the specification lists as valid but not modelled.
* Thick 3D layouts of mode 26 with 8 and 16 byte elements: the address equation data has no row for them.
* `XGSuggestSwizzleMode(desc, flags, layout)` writes `0x100 | mode` for the description with `Layout` 0. The prototype comes from the logged arguments, no console vector checks it.
* The 19041 build rules are not implemented, only the 22000 behaviour.

## Tests

`cargo test -p decant-xg -p decant-texture` runs every test vector of the library through the exports and the computer table (`tests/vectors.rs`), the load of the built DLL (`tests/dll.rs`) and the unit tests of `decant_texture::xg`, among them the comparison of the 2D layouts with `decant_texture::surface::compute` and the check that every address equation is one to one inside its block.
