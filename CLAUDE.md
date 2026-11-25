# Instructions for AI agents and contributors

Decant runs Xbox Series (GDK) titles natively on PC. Read [ARCHITECTURE.md](ARCHITECTURE.md) first. This file holds the working rules.

## Layout

* Every component is a Rust crate in `crates/decant-*`. The list is in [crates/README.md](crates/README.md).
* `src/gdk/` has the Linux build of the console layers and the launch stack. `sandbox/` has the air gapped container and the Windows Sandbox loop. `third_party/` holds code under other licences.
* The toolchain is pinned in `rust-toolchain.toml` (edition 2024). Workspace lints deny `unsafe_code`. Crates that need it, such as the DLLs and the GPU benches, allow it in their own manifest.

## Build and test

* `setup.ps1 -Check` reports the tools a machine needs.
* `build.cmd` builds the DLLs and test programs into `build\`.
* `cargo test --workspace --locked`
* `cargo clippy --workspace --all-targets --locked -- -D warnings`
* `cargo run --locked -p decant-repo-check --bin check_repository`

Run all of them on the rebased tree before every push. Run `build.cmd` and the native test programs too when you touch a DLL. Report the commands and results in the pull request, and say what you did not test. Never claim a result you did not run. If you are stuck after three attempts on one idea, stop and report.

## Git

* Work in your own git worktree or branch. Do not switch the branch of a shared clone.
* Keep one topic per commit. Pull and rebase on `main` before each push. Never force push to a branch someone else works on.
* Commit messages and pull requests name their issue (`Part of #n` or `Closes #n`).

## Code style

* Write Rust for the pinned toolchain. Keep functions small and name things after what they do.
* ABI facts that come from measurement are data tables with a comment that says what was measured and on what. Do not guess a console ABI. Log the unknown and return a clear error such as `E_NOTIMPL`.
* A layer that cannot really answer a call counts it through `decant-log` so a run shows what the title asked for.
* Tests use synthetic inputs made for the test. They need no GPU and no console files unless the crate says so. A test that needs hardware skips itself with a message.
* Delete the code a change replaces in the same change.

## Code from other projects

* Our own code, or MIT, Apache or BSD code with its copyright line kept and a one or two line header `// <what> from <owner/repo> at <revision>, file <path>. Copyright <holder>, <licence>.`
* Code under another licence goes in its own folder under `third_party/`, with a README that names the licence and an entry in `third_party/manifest.json`. `check_repository` fails otherwise.
* Facts and ideas from projects without a licence may be used, their code may not.
* Look at what other projects already solved before you design a fix, then write the code yourself.

## Safety

* Never run a title executable on the host. Run untrusted code in `sandbox/`, which has no network.
* Console files, keys and logs that hold paths or data from a user's machine stay on that machine. Quote only the line that shows where a run stopped.

## Documentation

* Plain, factual English. State what was measured and where the measurement stops.
* No semicolons and no dashes as punctuation. Rewrite the sentence instead. Code, identifiers, file names and commands stay as they are.
* Bullets use `*`.
* Update the document that describes a component in the same change that alters it.
