//! Checks the tracked paths and their text before review, as `scripts/check_repository.py` did:
//! `Reviewed <n> textual paths` and exit code 0, or `Review required: <paths>` on the error
//! stream and exit code 1.
//!
//! Usage: `check_repository [--root <repository folder>]`

use std::io::Write;
use std::process::ExitCode;

use decant_repo_check::rules::{Outcome, check, tracked};
use decant_repo_check::{NL, root};

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let result = root(&mut args).and_then(|root| {
        let names = tracked(&root)?;
        check(&root, &names).map_err(|stop| stop.0)
    });
    match result {
        Ok(Outcome::Reviewed(n)) => {
            print!("Reviewed {n} textual paths{NL}");
            ExitCode::SUCCESS
        }
        Ok(Outcome::Review(paths)) => {
            let _ = write!(std::io::stderr(), "Review required: {}{NL}", paths.join(", "));
            ExitCode::FAILURE
        }
        Err(stop) => {
            let _ = write!(std::io::stderr(), "{}{NL}", stop.replace('\n', NL));
            ExitCode::FAILURE
        }
    }
}
