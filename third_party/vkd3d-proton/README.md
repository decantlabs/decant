# vkd3d-proton bridge hook

`vkd3d-decant-shader.patch` is a change to [vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton) at commit
`cae8353ed7959c448ac976342052de1ec2b81715`. The patch is under the LGPL 2.1 (`LGPL-2.1`), like vkd3d-proton. The licence text is in the `COPYING` file of the vkd3d-proton repository.

The patch adds 57 lines to `libs/vkd3d-shader/dxil.c`. It does nothing unless the environment variable
`VKD3D_DECANT_SHADER_DIR` is set. When it is set, for each shader it writes the shader container and the binding layout
vkd3d-proton chose, then runs `VKD3D_DECANT_SHADER_COMMAND`, which is expected to place a translated module in
`VKD3D_SHADER_OVERRIDE`. See [docs/SHADERS.md](../../docs/SHADERS.md).

To apply it: `cargo run -p decant-repo-check --bin third_party -- setup --vkd3d <vkd3d-proton checkout>` from the repository root, or from the checkout root,
`git apply vkd3d-decant-shader.patch`.

## Building the Windows DLLs

`build-windows.ps1` builds the x64 `d3d12.dll` and `d3d12core.dll` with the patch applied. Run it from a normal
PowerShell on Windows 11 (git, the `py` launcher and the Vulkan SDK for `glslangValidator` must be installed):

```
powershell -ExecutionPolicy Bypass -File third_party\vkd3d-proton\build-windows.ps1 [-Work <dir>]
```

Keep `-Work` short (for example `C:\vkb`). The script clones with `core.longpaths`, but the build itself still has
paths deep inside the dxil-spirv submodules, and a long work folder pushes them over the Windows path limit.

It works outside the repository (default `%USERPROFILE%\repos\vkd3d-build`): it downloads the WinLibs mingw-w64
toolchain (GCC 16.2.0, mingw-w64 14.0.0 UCRT, release `16.2.0posix-23.1.3-14.0.0-ucrt-r3`, zip SHA-256
`3c69305081e4a647144339a936244fcf7d3d9e558319e1ad7be2cd2886d4b6a1`, checked by the script), installs meson 1.12.1 and
ninja 1.13.2 in a venv, clones vkd3d-proton at the commit above with its submodules, applies the patch and runs
`meson setup --buildtype release --strip` and `ninja`. WinLibs provides `widl`, which the build needs. The DLLs land
in `<Work>\out`. They are never committed. The MSVC route of the upstream README also needs `widl`, so mingw is used.

The DLLs are not bit for bit reproducible (the PE header carries a build time), so their hashes differ per build.
The build of 23 Nov 2026 gave:

```
c2965763b26f459bf936dd57298966680d180bcafbde7aeca270f232269c8e4f  d3d12.dll
2e5c19ad3bb4181c505695a010a0634a6cc601435d8596d936aefd236ba886e6  d3d12core.dll
```

`d3d12core.dll` must sit in the same folder as `d3d12.dll` and be loaded first (or sit next to the executable).
Smoke test: `D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_0, IID_ID3D12Device)` returns `S_OK` on an RTX 3090.
The resulting binaries are LGPL 2.1.
