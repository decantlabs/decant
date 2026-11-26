@echo off
setlocal
where cargo >nul 2>nul || (echo Install the Rust toolchain of rust-toolchain.toml. & exit /b 2)
set "R=%~dp0."
if not defined DECANT_BUILD set "DECANT_BUILD=%~dp0build"
for %%I in ("%DECANT_BUILD%") do set "DECANT_BUILD=%%~fI"
if not exist "%DECANT_BUILD%" mkdir "%DECANT_BUILD%" || exit /b 1
rem Every component is a Rust crate. Cargo finds the MSVC linker by itself, so no developer prompt is needed.
rem The static C runtime is the former /MT. cargo runs inside the workspace so that its rust-toolchain.toml applies.
rem The test of xgameruntime links the public GDK xgameruntime.lib when DECANT_GDK_ROOT is set (read by the crate build script).
rem PC DirectStorage is loaded at run time. With DECANT_GDEFLATE_ROOT set (the GDeflate folder of microsoft/DirectStorage) the GDeflate decoder is linked and the licences are copied.
if not defined DECANT_GDK_ROOT echo xgameruntime_test: online mode skipped, set DECANT_GDK_ROOT to the public GDK edition folder to build it.
set "RUSTFLAGS=%RUSTFLAGS% -C target-feature=+crt-static"
pushd "%R%" || exit /b 1
cargo build --release --locked --target-dir "%DECANT_BUILD%\cargo" -p decant-ffx -p decant-bvh -p decant-gdk-config -p decant-xmem -p decant-d3d12x -p decant-xaudio2 -p decant-xgameruntime -p decant-xma2 -p decant-dstorage -p decant-dstorage-xcrdapi -p decant-xg -p decant-acphal -p decant-xapu -p decant-chad -p decant-gameinput || (popd & exit /b 1)
popd
for %%F in (decant_ffx.dll decant_bvh.dll ffxtest.exe decant_bvh_test.exe gdk_config_reader.exe xmem.dll loadtest.exe xmem_abitest.exe d3d12_xs.dll d3d12xs_test.exe decant_xaudio2.dll xaudio2test.exe gameinputtest.exe xgameruntime.dll xgameruntime_test.exe decant_xma2.dll xma2test.exe dstorage_xs.dll xcrdapi.dll dstoragetest.exe dstorage_abi_test.exe xg_xs.dll AcpHal.dll xapu.dll chad.dll GameInput.dll decant-pad.exe) do copy /y "%DECANT_BUILD%\cargo\release\%%F" "%DECANT_BUILD%\%%F" >nul || exit /b 1
for %%F in (decant_ffx decant_bvh xmem d3d12_xs decant_xaudio2 xgameruntime decant_xma2 dstorage_xs xcrdapi xg_xs AcpHal xapu chad GameInput) do if exist "%DECANT_BUILD%\cargo\release\%%F.pdb" copy /y "%DECANT_BUILD%\cargo\release\%%F.pdb" "%DECANT_BUILD%\%%F.pdb" >nul || exit /b 1
rem XboxBvhBuilder.dll is decant_bvh.dll under the package name (it exports the mangled XboxBvhBuilder members too), so the shim staging replaces the console builder.
copy /y "%DECANT_BUILD%\cargo\release\decant_bvh.dll" "%DECANT_BUILD%\XboxBvhBuilder.dll" >nul || exit /b 1
if exist "%DECANT_BUILD%\cargo\release\decant_bvh.pdb" copy /y "%DECANT_BUILD%\cargo\release\decant_bvh.pdb" "%DECANT_BUILD%\XboxBvhBuilder.pdb" >nul || exit /b 1
if defined DECANT_GDEFLATE_ROOT if exist "%DECANT_GDEFLATE_ROOT%\3rdparty\libdeflate\COPYING" (
  copy /y "%DECANT_GDEFLATE_ROOT%\3rdparty\libdeflate\COPYING" "%DECANT_BUILD%\decant-storage-libdeflate-license.txt" >nul || exit /b 1
  copy /y "%DECANT_GDEFLATE_ROOT%\GDeflate\LICENSE" "%DECANT_BUILD%\decant-storage-apache-license.txt" >nul || exit /b 1
)
exit /b 0
