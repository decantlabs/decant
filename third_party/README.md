# Third party code under other licences

Everything in this directory is under its own licence, **not** the project licence. It is kept here, apart from
the rest of the repository, so that the licence of each piece is obvious and nothing under the project licence is
mixed with it.

| Directory | Licence | Upstream | What it is |
|---|---|---|---|
| [`vkd3d-proton/`](vkd3d-proton/README.md) | LGPL 2.1 | vkd3d-proton at `cae8353e` | The 57 line bridge hook that lets vkd3d-proton use translated shaders |

`manifest.json` lists every file, its licence, the upstream repository and the exact upstream commit.

## Putting the pieces in place

`third_party` (the `decant-repo-check` crate) places each piece where it is needed. It never downloads
anything: you clone the upstream projects yourself, and the program checks the commit and applies the patch. Run it
from the repository root:

```
cargo run -p decant-repo-check --bin third_party -- list
cargo run -p decant-repo-check --bin third_party -- check
cargo run -p decant-repo-check --bin third_party -- setup --vkd3d <vkd3d-proton checkout> [--dry-run]
```

`setup` stops if a checkout is not at the expected commit, and does nothing for a patch that is already applied.

## Rules

* Anything not under the project licence goes in its own subdirectory here, with its licence named in its README
  and an entry in `manifest.json`. `check` (run by the tests of the Rust workspace) fails otherwise.
* These pieces are separate programs or patches to separate programs. Nothing under the project licence links
  with them, the project's tools only run them as processes and read their output files.
