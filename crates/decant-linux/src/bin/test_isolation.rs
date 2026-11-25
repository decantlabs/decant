//! `test_isolation`, the port of `sandbox/test_isolation.sh`: twelve checks of network, file
//! system and privilege isolation. The sandbox folder is SANDBOX_DIR, else `sandbox/` of the
//! checkout.

use decant_linux::ctx::{Ctx, Exit, System};

fn main() {
    let mut ctx = Ctx::from_process(Box::new(System));
    if std::env::args().nth(1).as_deref() == Some("--dry-run") {
        ctx.dry_run = true;
    }
    let dir = decant_linux::sandbox::default_dir(&ctx);
    let code = match decant_linux::sandbox::test_isolation(&mut ctx, &dir) {
        Ok(()) => 0,
        Err(Exit(code)) => code,
    };
    std::process::exit(code);
}
