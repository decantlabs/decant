# Architecture

Decant runs an Xbox Series (GDK, Scarlett) title on PC. The title is an x86-64 PE executable, so its code runs natively. Decant sits between that code and the PC and supplies what the console had.

```
title executable (x86-64 PE, GDK)
 ├─ d3d12_xs.dll ........... D3D12.X layer over a PC D3D12 device     (decant-d3d12x)
 │    └─ shaders ........... console GFX10 code to HLSL and DXIL      (decant-shader, decant-dxc, decant-dxil)
 ├─ xmem.dll ............... console allocator                         (decant-xmem)
 ├─ dstorage_xs.dll ........ DirectStorage console ABI, BCPack         (decant-dstorage, decant-bcpack)
 ├─ xgameruntime.dll ....... interposer over the PC game runtime       (decant-xgameruntime)
 ├─ GameInput.dll .......... one scripted gamepad                      (decant-gameinput)
 ├─ audio .................. XAudio2, AcpHal, XAPU, chad, XMA2         (decant-xaudio2, decant-acphal, decant-xapu, decant-chad, decant-xma2)
 ├─ xg_xs.dll .............. texture and resource layout               (decant-xg, decant-texture)
 └─ ray tracing and upscaling: BVH builds, FidelityFX forwarding      (decant-bvh, decant-ffx)
```

## GPU

Console shader binaries are RDNA2 (GFX10) machine code. `decant-shader` decodes them, lifts them to an intermediate representation, structures the control flow and prints HLSL for a whole pipeline at once, with the real root signature in hand. `decant-dxc` compiles the HLSL to DXIL, which runs on native Direct3D 12 and on vkd3d-proton. Details are in [docs/SHADER_TRANSLATOR.md](docs/SHADER_TRANSLATOR.md) and [docs/SHADERS.md](docs/SHADERS.md).

`d3d12_xs.dll` answers the console's Direct3D 12 entry points. It creates a PC device and returns it inside proxies that follow the console's method order, which is kept as a data table. It detiles console textures with a compute pass, creates pipelines on first bind, and emulates the frame events a console title waits for. See [docs/D3D12X_LAYER.md](docs/D3D12X_LAYER.md) for the state, [docs/D3D12X_DIFFERENCES.md](docs/D3D12X_DIFFERENCES.md) for what public sources say about the differences, and [docs/DSTORAGE_ABI.md](docs/DSTORAGE_ABI.md) for DirectStorage.

## System APIs

Each console library the title imports and Windows does not provide has its own crate, with the same export names and ordinals. [docs/CONSOLE_LIBS.md](docs/CONSOLE_LIBS.md) specifies the XG, AcpHal, XAPU and chad libraries and the physical page functions of XMem. [docs/AUDIO_INPUT.md](docs/AUDIO_INPUT.md) covers audio and input. [docs/DEBUG_TOOLS.md](docs/DEBUG_TOOLS.md) lists the switches that show what a title does.

The layers log through `decant-log`. Calls that a layer could not really answer are counted in one shared block that `decant-launch` reads.

## Launch

On Windows, `decant-launch` stages the title into a writable overlay folder with our DLLs in front, then runs it under the debugging API. It sees every thread from its first instruction and every exception before the title does, and it logs every DLL, debug line and exception.

On Linux, `launch_prep` (crate `decant-linux`) prepares a Wine and Proton stack. The GDK API itself comes from the Ferestre runtime, and our DLLs cover the console only parts. Launching runs inside an air gapped container by default. See [docs/LAUNCH_ARCHITECTURE.md](docs/LAUNCH_ARCHITECTURE.md) and [src/gdk/README.md](src/gdk/README.md).

`decant-audit-imports` compares the imports of a title with what the PC and our DLLs provide, and `decant-stub` generates stand in DLLs for what nothing provides.

## Testing

Tests need no GPU and no console files unless a crate says otherwise. Hardware tests follow [docs/GPU_TESTING.md](docs/GPU_TESTING.md). CI is described in [docs/CI.md](docs/CI.md).
