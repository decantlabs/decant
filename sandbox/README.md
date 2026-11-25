# Air gapped sandbox

A Docker container for running untrusted Windows executables (package DLLs, a title executable, test programs) under Wine with no network and a read only view of the title package. Use it for anything that executes code from a package, the native tools (the repository root) only read files and do not need it. `launch_prep launch` (crate `decant-linux`) runs inside this container by default.

| Property | How |
|---|---|
| No network | `network_mode: none`: only `lo` exists, no DNS, no route |
| Pinned base | the Dockerfile pins `ubuntu:24.04` by digest, refresh deliberately |
| Package read only | `$DECANT_PACKAGE_DIR` mounted at `/data/game:ro` |
| Shims read only | `build/gdk` (from `make -C src/gdk all tests`) mounted at `/data/stubs:ro` |
| Writable state contained | `sandbox/prefix` (Wine prefix) and `sandbox/logs` only, both ignored by git |
| Least privilege | nonroot user with the host uid/gid, `cap_drop: ALL` plus `SYS_PTRACE` for Wine debugging, `no-new-privileges`, no `privileged` |
| Bounded | 16 GiB memory, 8 CPUs, no restart |

The image (Ubuntu 24.04, distribution Wine, Mesa Vulkan, Xvfb) needs the network once, at `docker compose build`. Nothing is downloaded at run time.

```sh
make -C src/gdk all tests
export DECANT_PACKAGE_DIR=/path/to/package
run_airgapped                  # build, start, refuse to continue unless isolated
test_isolation                 # 12 checks: network, filesystem, privileges
docker exec -it decant-sandbox bash
  wine64 /data/stubs/xmem_abitest.exe 'Z:\data\stubs\xmem.dll'
docker compose -f sandbox/docker-compose.yml down
```

`run_airgapped` and `test_isolation` are binaries of the `decant-linux` crate (`cargo build --release -p decant-linux` in the repository root). `run_airgapped` aborts and stops the container if any interface other than `lo` exists or any outbound TCP or HTTP probe succeeds. `test_isolation` checks network, filesystem and privileges and exits nonzero on any failure.

Limits: Docker's isolation is not a VM boundary. The GPU is not passed through, so this runs CPU side and loader tests, not rendering. For rendering, follow docs/GPU_TESTING.md on dedicated hardware without network access.
