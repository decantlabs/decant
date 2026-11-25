//! A stand in for `xbox_decode_test_xbox.exe` with the same command line
//! (`<pack> <report> <shader name>`). It decodes nothing: it prints the invented listing of
//! `synthetic::listing`, chosen from a hash of the name, with the instruction forms the scripts
//! look for. Placed where the scripts expect the decoder, it lets the Python scripts and the Rust
//! tools be compared on the same listings when the GPL decoder is not built.
//!
//! Usage: `cargo run -p decant-shader-analysis --example stand_in_decoder -- <pack> <report> <name>`

#[allow(dead_code)]
#[path = "../src/synthetic.rs"]
mod synthetic;

fn main() {
    print!("{}", synthetic::listing(&std::env::args().nth(3).unwrap_or_default()));
}
