//! Lists, checks and puts in place the third party pieces kept in `third_party/` (see
//! `third_party/README.md`), as `scripts/third_party.py` did.
//!
//! ```text
//! third_party list
//! third_party check
//! third_party setup [--vkd3d DIR] [--dry-run]
//! ```
//!
//! `setup` never downloads anything. Each DIR is a git checkout you cloned yourself: it is checked
//! to be at the commit named in `third_party/manifest.json`, then the patch is applied. A patch
//! that is already applied is left alone. `--root <folder>` names the repository when the current
//! folder is not inside it.

use std::io::Write;
use std::process::ExitCode;

use decant_repo_check::third::{check, list, load_manifest, setup_one};
use decant_repo_check::{NL, root};

const USAGE: &str = "usage: third_party [-h] {list,check,setup} ...";
const SETUP_USAGE: &str = "usage: third_party setup [-h] [--vkd3d VKD3D] [--dry-run]";

/// An argument error as `argparse` reports it: usage, then the message, exit code 2.
fn usage_error(usage: &str, prog: &str, message: &str) -> ExitCode {
    let _ = write!(std::io::stderr(), "{usage}{NL}{prog}: error: {message}{NL}");
    ExitCode::from(2)
}

fn help() -> ExitCode {
    print!(
        "{USAGE}{NL}{NL}Lists, checks and puts in place the third party pieces kept in third_party/.{NL}{NL}positional arguments:{NL}  {{list,check,setup}}{NL}{NL}options:{NL}  -h, --help          show this help message and exit{NL}"
    );
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let root = match root(&mut args) {
        Ok(r) => r,
        Err(e) => {
            let _ = write!(std::io::stderr(), "{e}{NL}");
            return ExitCode::FAILURE;
        }
    };
    let third = root.join("third_party");
    let Some(command) = args.first().cloned() else {
        return usage_error(USAGE, "third_party", "the following arguments are required: command");
    };
    let (mut vkd3d, mut dry_run) = (None, false);
    let mut rest = args[1..].iter();
    match command.as_str() {
        "-h" | "--help" => return help(),
        "list" | "check" => {
            if let Some(extra) = rest.next() {
                return usage_error(USAGE, "third_party", &format!("unrecognized arguments: {extra}"));
            }
        }
        "setup" => {
            while let Some(arg) = rest.next() {
                let (key, inline) = match arg.split_once('=') {
                    Some((k, v)) => (k, Some(v.to_owned())),
                    None => (arg.as_str(), None),
                };
                match key {
                    "--vkd3d" => {
                        let Some(value) = inline.or_else(|| rest.next().cloned()) else {
                            return usage_error(SETUP_USAGE, "third_party setup", &format!("argument {key}: expected one argument"));
                        };
                        vkd3d = Some(value);
                    }
                    "--dry-run" => dry_run = true,
                    "-h" | "--help" => {
                        print!("{SETUP_USAGE}{NL}");
                        return ExitCode::SUCCESS;
                    }
                    _ => return usage_error(USAGE, "third_party", &format!("unrecognized arguments: {arg}")),
                }
            }
        }
        other => {
            return usage_error(USAGE, "third_party", &format!("argument command: invalid choice: '{other}' (choose from 'list', 'check', 'setup')"));
        }
    }
    let items = match load_manifest(&third) {
        Ok(i) => i,
        Err(e) => {
            let _ = write!(std::io::stderr(), "{e}{NL}");
            return ExitCode::FAILURE;
        }
    };
    match command.as_str() {
        "list" => {
            for line in list(&items) {
                print!("{line}{NL}");
            }
            ExitCode::SUCCESS
        }
        "check" => match check(&third) {
            Ok(problems) => {
                for p in &problems {
                    print!("{p}{NL}");
                }
                if problems.is_empty() {
                    print!("third_party: OK{NL}");
                    ExitCode::SUCCESS
                } else {
                    print!("third_party: {} problem(s){NL}", problems.len());
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                let _ = write!(std::io::stderr(), "{e}{NL}");
                ExitCode::FAILURE
            }
        },
        _ => {
            let targets = [("vkd3d", vkd3d.filter(|v| !v.is_empty()))];
            if targets.iter().all(|(_, v)| v.is_none()) {
                return usage_error(USAGE, "third_party", "give the checkout: --vkd3d");
            }
            let mut errors = Vec::new();
            for (name, item) in &items {
                let target = item.get("target").and_then(|t| t.as_str()).unwrap_or_default();
                let Some(checkout) = targets.iter().find(|(t, _)| *t == target).and_then(|(_, v)| v.clone()) else { continue };
                if let Some(error) = setup_one(&third, name, item, &checkout, dry_run, &mut |line| print!("{line}{NL}")) {
                    errors.push(error);
                }
            }
            for e in &errors {
                let _ = write!(std::io::stderr(), "{e}{NL}");
            }
            if errors.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
    }
}
