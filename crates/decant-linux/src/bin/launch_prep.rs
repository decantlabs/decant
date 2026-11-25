//! `launch_prep`, the port of `src/gdk/launch_prep.sh`. See `decant_linux::prep`.

use decant_linux::ctx::{Ctx, System};

fn main() {
    let mut ctx = Ctx::from_process(Box::new(System));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = decant_linux::prep::main(&mut ctx, &args);
    std::process::exit(code);
}
