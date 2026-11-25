# Review checks

The workflow `.github/workflows/review.yml` runs on pull requests and on manual dispatch. It has three jobs. It uploads nothing except the Wine artifact below and deploys nothing. Dedicated hardware checks stay manual, see [GPU_TESTING.md](GPU_TESTING.md).

## Review

Runs on a Windows hosted runner with the toolchain pinned in `rust-toolchain.toml` (limit 12 minutes).

* Repository check: `cargo run --locked -p decant-repo-check --bin check_repository`. It reviews every tracked path and its text and fails with `Review required:` and the paths to look at.
* On a pull request, the description must link its issue with `Closes #n` or `Part of #n`.
* `build.cmd` builds the DLLs and test programs (cargo, no Visual Studio developer prompt needed). `FFX_API_ROOT` can point to a preinstalled FidelityFX SDK, which is passed to `build.cmd` as its argument.
* The tests that need no GPU and no title file then run: `loadtest` and `xmem_abitest` against `xmem.dll`, `xaudio2test`, `gameinputtest`, `d3d12xs_test`, `dstorage_abi_test spec a` and `dstoragetest`. GPU execution is skipped on the hosted runner.

`gdk_smoketest` needs the companion DLLs of a staged title and is never run in CI.

## Rust workspace

Runs on a Windows hosted runner (limit 45 minutes): `cargo test --workspace --locked`, then `cargo clippy --workspace --all-targets --locked -- -D warnings`. This is the one job that downloads the locked dependencies. Its tests need no GPU and no title file.

## Wine

Runs on `ubuntu-latest` (limit 30 minutes). It installs the pinned toolchain with the `x86_64-pc-windows-gnu` target, `gcc-mingw-w64-x86-64` as the linker and `wine64`, and builds the GDK DLLs with `cargo build --release --locked`: `xmem.dll`, `decant_xaudio2.dll`, `decant_xma2.dll`, `xgameruntime.dll`, `dstorage_xs.dll`, `xcrdapi.dll`, `decant_ffx.dll`, `decant_bvh.dll` and `d3d12_xs.dll`, with their test programs, `gdk_smoketest.exe` and `gdk_config_reader.exe`. The C runtime is linked statically (`-C target-feature=+crt-static`), so no mingw DLL has to sit next to them. The export ordinals that MSVC gets from `/EXPORT:name,@N` reach the GNU linker as a module definition file that each `build.rs` writes from the same table.

The programs that need no GPU and no title file then run under Wine in a headless prefix with `WINEDEBUG=-all`: the seven of the Review job, `xgameruntime_test offline` and `xma2test`. Parts that need hardware skip themselves: no audio endpoint for `xaudio2test`, no `GameInput.dll` for `gameinputtest`, no D3D12 12_0 device for `d3d12xs_test`. `xma2test` is given a file that is not RIFF XMA2, so the job expects its clean refusal (`not a RIFF file`, exit code 1), which shows that the DLL loads and hands out its decoder table. The decoding itself is tested by the cargo tests. Every program runs even when an earlier one failed, and the step fails at the end with the number of failures.

Every built DLL and test program is uploaded as the workflow artifact `gdk-dlls-x86_64-pc-windows-gnu`, for manual tests under Wine on Linux. `make -C src/gdk all tests` builds the same set locally and copies it to `build/gdk`.
