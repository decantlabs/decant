# Hardware test protocol

Hardware tests run on a local build of the commit under test, using the same commit for every comparison. The runner is `gpu_diag` of the `decant-gpu-diag` crate:

```bat
gpu_diag [--build <folder>] [--backend <FidelityFX backend DLL>] [--adapter <n>] [--commit <text>] [--json]
```

## What it does

* Lists the DXGI adapters with their D3D12 ray tracing tier (`D3D12_OPTIONS5`) and tells the default adapter, the selected adapter and software adapters apart.
* Runs `loadtest.exe`, `decant_bvh_test.exe` and `ffxtest.exe` from the build folder one after another, each with a 45 second deadline on its process tree. A stalled test is stopped and recorded as a timeout. If DXR or a compatible FidelityFX backend is unavailable, the test is recorded as unavailable. `--backend` names the FidelityFX backend DLL as local input only.
* Prints an allowlisted record: component, version, public hardware and driver, capability, elapsed milliseconds, result (pass, fail, timeout, unavailable) and a sanitized error category. It never records serial numbers, user names, paths, environment, command lines or raw output.
* Saves nothing and sends nothing. The record is shown for review, and sharing it is up to the person who ran it.

## What to submit

Fill in the GPU form with the GPU model, public driver and Windows versions, CPU, RAM, commit, test name, elapsed time, result, sanitized errors and reproduction steps. Record build, prebuild size and GPU reported compacted size separately. Do not compare throughput from one triangle. Raw logs stay local.

The local baseline is an RTX 4080. A Ryzen 7 9800X3D, RTX 5090 and 64 GB configuration is a proposed volunteer configuration.
