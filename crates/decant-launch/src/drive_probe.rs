//! Test helper for `tests/game_drive.rs`, not part of the launcher: reads the file named by its
//! first argument (a path such as `H:/hello.txt`) and checks the answer of `Path::exists` and of
//! `std::fs::read`. Exit 0 when the content is `hello`, 5 when the file cannot be opened, 6 when the
//! content is something else, 7 when `exists` says no.

fn main() {
    let path = std::env::args().nth(1).unwrap_or_default();
    if !std::path::Path::new(&path).exists() {
        std::process::exit(7);
    }
    std::process::exit(match std::fs::read(&path) {
        Ok(bytes) if bytes == b"hello" => 0,
        Ok(_) => 6,
        Err(_) => 5,
    });
}
