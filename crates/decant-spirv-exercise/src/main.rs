//! Runs the validator exercise with the local SPIRV-Tools and prints the tool version and one row
//! per module. Exits with 1 when an outcome is not the expected one, 2 without SPIRV-Tools.
//!
//! Usage: `spirv_exercise` (SPIRV-Tools from `SPIRV_TOOLS`, `VULKAN_SDK` or the `PATH`)

use std::process::ExitCode;

use decant_spirv_exercise::{Outcome, TARGET_ENV, Tools, check, variants};

fn main() -> ExitCode {
    let Some(tools) = Tools::find() else {
        eprintln!("SPIRV-Tools not found: set SPIRV_TOOLS to the folder with spirv-as and spirv-val, or install the Vulkan SDK");
        return ExitCode::from(2);
    };
    println!("{}, target environment {TARGET_ENV}\n", tools.version());
    println!("| Module | Change | Expected | spirv-val exit | First message |");
    println!("|---|---|---|---|---|");
    let mut unexpected = 0;
    for v in variants() {
        let expected_valid = v.name == "minimal";
        let (exit, message, valid) = match check(&tools, v.name, &v.text) {
            Ok(o @ Outcome::Validated { .. }) => {
                let valid = o.valid();
                let Outcome::Validated { exit, message, .. } = o else { unreachable!() };
                (exit.map_or("none".to_owned(), |c| c.to_string()), message, valid)
            }
            Ok(Outcome::NotAssembled { exit }) => (format!("not assembled ({})", exit.map_or("none".to_owned(), |c| c.to_string())), String::new(), false),
            Err(e) => (format!("error: {e}"), String::new(), false),
        };
        unexpected += usize::from(valid != expected_valid);
        let expected = if expected_valid { "valid" } else { "invalid" };
        println!("| {} | {} | {expected} | {exit} | {} |", v.name, v.change, message.replace('|', "\\|"));
    }
    if unexpected == 0 { ExitCode::SUCCESS } else { ExitCode::from(1) }
}
