# Contributing

1. Choose an open issue, or open one for what you want to change. Say what you take so others can coordinate.
2. Create a topic branch and keep each change focused.
3. Run the checks before you push:
   * `cargo test --workspace --locked`
   * `cargo clippy --workspace --all-targets --locked -- -D warnings`
   * `cargo run --locked -p decant-repo-check --bin check_repository`
   * `build.cmd` and the native tests when you touch a DLL
4. Open a pull request against `main` and link the issue. Give the commands you ran, the tool versions, the hardware, the results and anything you did not test.
5. Respond to review and rerun the affected checks.

## Licence

Decant is MIT licensed. By contributing you agree that your contribution is licensed under the same terms (inbound equals outbound). There is no separate contributor agreement.

## Code from other projects

* Our own code, or MIT, Apache or BSD code with its copyright line kept and a one or two line header `// <what> from <owner/repo> at <revision>, file <path>. Copyright <holder>, <licence>.`
* Code under another licence goes in its own folder under `third_party/` with a README that names the licence and an entry in `third_party/manifest.json`.
* Facts and ideas from projects without a licence may be used, their code may not.

## What not to submit

Do not submit console executables, keys, memory dumps, secrets or personal identifiers. Test inputs are synthetic and made for the test. Follow [SECURITY.md](SECURITY.md) for vulnerabilities.
