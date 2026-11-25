# Instructions for AI agents

Read [CLAUDE.md](CLAUDE.md) first. It holds the rules for every agent. The points below must never be missed.

* Work in your own git worktree. Never switch the branch of a shared clone.
* Before each push, on the rebased tree: `build.cmd` and the native test programs when you touch a DLL, then in the repository root `cargo test --workspace --locked`, `cargo clippy --workspace --all-targets --locked -- -D warnings` and `cargo run --locked -p decant-repo-check --bin check_repository`.
* Never run a title executable on the host. Untrusted code runs inside `sandbox/`.
* Never claim a result you did not run. Report the commands, the results and what you did not test.
* Stuck after three attempts on one idea: stop and report.
* Check what other projects already solved before you design a fix, and respect their licences (see "Code from other projects" in CLAUDE.md).
* Docs are plain English with no semicolons and no dashes as punctuation. Bullets use `*`.
