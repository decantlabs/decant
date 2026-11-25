# Cross-build every GDK DLL and Wine-side test program with cargo for the target x86_64-pc-windows-gnu
# (rustup target add x86_64-pc-windows-gnu, and gcc-mingw-w64-x86-64 as the linker). build.cmd builds the
# same crates for MSVC on Windows. The CI job "wine" of .github/workflows/review.yml does what
# `make -C src/gdk all tests` does, then runs the tests under Wine.
#
#   make -C src/gdk              -> every DLL in build/gdk/
#   make -C src/gdk tests        -> every test program in build/gdk/
#   make -C src/gdk all tests    -> both, from one cargo build
#
# The DLLs: xmem.dll decant_xaudio2.dll decant_xma2.dll xgameruntime.dll dstorage_xs.dll xcrdapi.dll
# decant_ffx.dll decant_bvh.dll d3d12_xs.dll. The programs: loadtest.exe xmem_abitest.exe
# xaudio2test.exe gameinputtest.exe xma2test.exe xgameruntime_test.exe dstorage_abi_test.exe
# dstoragetest.exe d3d12xs_test.exe ffxtest.exe decant_bvh_test.exe gdk_smoketest.exe gdk_config_reader.exe.
#
# Run the tests under Wine, with Windows paths (the CI job lists the arguments of each):
#   wine64 build/gdk/loadtest.exe     Z:\\path\\to\\build\\gdk\\xmem.dll
#   wine64 build/gdk/xmem_abitest.exe Z:\\path\\to\\build\\gdk\\xmem.dll
CARGO     ?= cargo
ROOT      := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
OUT       ?= $(ROOT)/build/gdk
RS        := $(ROOT)
RUST_TARGET ?= x86_64-pc-windows-gnu
RELEASE   := $(RS)/target/$(RUST_TARGET)/release
# Static libgcc and winpthreads: no mingw runtime DLL has to sit next to the DLLs.
export RUSTFLAGS += -C target-feature=+crt-static

CRATES    := decant-xmem decant-xaudio2 decant-xma2 decant-xgameruntime decant-dstorage \
             decant-dstorage-xcrdapi decant-ffx decant-bvh decant-d3d12x decant-smoketest decant-gdk-config
DLLS      := xmem.dll decant_xaudio2.dll decant_xma2.dll xgameruntime.dll dstorage_xs.dll xcrdapi.dll \
             decant_ffx.dll decant_bvh.dll d3d12_xs.dll
TESTS     := loadtest.exe xmem_abitest.exe xaudio2test.exe gameinputtest.exe xma2test.exe \
             xgameruntime_test.exe dstorage_abi_test.exe dstoragetest.exe d3d12xs_test.exe \
             ffxtest.exe decant_bvh_test.exe gdk_smoketest.exe gdk_config_reader.exe

all: cargo-build
	mkdir -p $(OUT)
	cd $(RELEASE) && cp -f $(DLLS) $(OUT)/

tests: cargo-build
	mkdir -p $(OUT)
	cd $(RELEASE) && cp -f $(TESTS) $(OUT)/

# One cargo build makes every DLL and test program, cargo decides what is out of date. cargo runs inside
# the workspace so that its rust-toolchain.toml applies.
cargo-build:
	cd $(RS) && $(CARGO) build --release --locked --target $(RUST_TARGET) $(addprefix -p ,$(CRATES))

clean:
	$(RM) -r $(OUT)

.PHONY: all tests cargo-build clean
