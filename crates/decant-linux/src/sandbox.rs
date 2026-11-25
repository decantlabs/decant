//! The sandbox runners, ports of `sandbox/run_airgapped.sh` (build, start, refuse to continue
//! unless isolated) and `sandbox/test_isolation.sh` (12 checks: network, file system,
//! privileges). Same docker commands, messages and exit codes. The compose file and the
//! Dockerfile stay as they are.

use std::io::Write as _;

use crate::ctx::{Ctx, Exit, Res, Sink};

/// The container name of `sandbox/docker-compose.yml`, as both scripts wrote it.
pub const CONTAINER: &str = "decant-sandbox";

/// The message of `${DECANT_PACKAGE_DIR:?...}`.
const NEED_GAME_DIR: &str = "Set DECANT_PACKAGE_DIR to the (read-only) game package directory";

/// `$(cd dir && pwd)`: the folder as an absolute path.
pub fn absolute(dir: &str) -> String {
    std::path::absolute(dir).map(|p| p.to_string_lossy().replace('\\', "/")).unwrap_or_else(|_| dir.to_owned())
}

/// The sandbox folder of a standalone run: SANDBOX_DIR, else `sandbox/` of the checkout.
pub fn default_dir(ctx: &Ctx) -> String {
    let root = ctx.var_or("DECANT_ROOT", &crate::prep::default_root());
    absolute(&ctx.var_or("SANDBOX_DIR", &format!("{root}/sandbox")))
}

/// `: "${DECANT_PACKAGE_DIR:?...}"` (line `line` of the script) then
/// `export DECANT_UID=${DECANT_UID:-$(id -u)} DECANT_GID=${DECANT_GID:-$(id -g)}`: both
/// values are read before either is exported.
fn environment(ctx: &mut Ctx, dir: &str, line: u32) -> Res {
    if ctx.var("DECANT_PACKAGE_DIR").is_none_or(|v| v.is_empty()) {
        let _ = ctx.out.flush();
        let _ = writeln!(ctx.err, "{}: line {line}: DECANT_PACKAGE_DIR: {NEED_GAME_DIR}", ctx.prog);
        return Err(Exit(1));
    }
    let mut values = Vec::new();
    for (name, flag) in [("DECANT_UID", "-u"), ("DECANT_GID", "-g")] {
        let value = match ctx.var(name).filter(|v| !v.is_empty()) {
            Some(value) => value,
            None => {
                let cmd = ctx.command("id").cwd(dir).arg(flag).stdout(Sink::Capture);
                ctx.exec(&cmd).stdout.trim_end_matches('\n').to_owned()
            }
        };
        values.push((name, value));
    }
    for (name, value) in values {
        ctx.export(name, &value);
    }
    Ok(())
}

fn docker(ctx: &Ctx, dir: &str, args: &[&str]) -> crate::ctx::Cmd {
    ctx.command("docker").cwd(dir).args(args.iter().copied())
}

/// `docker compose down` after a failed check, then exit 1 (or the code of `down` itself when it
/// fails, as `set -e` would have it).
fn abort(ctx: &mut Ctx, dir: &str, message: &str) -> Exit {
    ctx.echo(message);
    let cmd = docker(ctx, dir, &["compose", "down"]);
    let code = ctx.exec(&cmd).code;
    Exit(if code == 0 { 1 } else { code })
}

/// The output of `docker exec decant-sandbox ls /sys/class/net`, as `$(...)` keeps it.
fn interfaces(ctx: &mut Ctx, dir: &str, stderr: Sink) -> (i32, String) {
    let cmd = docker(ctx, dir, &["exec", CONTAINER, "ls", "/sys/class/net"]).stdout(Sink::Capture).stderr(stderr);
    let out = ctx.exec(&cmd);
    (out.code, out.stdout.trim_end_matches('\n').to_owned())
}

/// `run_airgapped.sh`, with `dir` the sandbox folder (absolute).
pub fn run_airgapped(ctx: &mut Ctx, dir: &str) -> Res {
    for line in [
        "=== DECANT AIR-GAPPED SANDBOX ===",
        "Network: NONE (network_mode: none)",
        "Game data: READ-ONLY mount",
        "No internet, no LAN, no DNS, no nothing.",
        "",
    ] {
        ctx.echo(line);
    }
    environment(ctx, dir, 15)?;
    // The writable mounts are made as the invoking user (docker would make them as root).
    for sub in ["prefix", "logs", "../build/gdk"] {
        ctx.mkdir_p(&format!("{dir}/{sub}"))?;
    }

    ctx.echo("[1/4] Building sandbox container...");
    let cmd = docker(ctx, dir, &["compose", "build", "--quiet"]).stderr(Sink::Stdout);
    ctx.exec_ok(&cmd)?;

    ctx.echo("[2/4] Starting sandbox (network: none)...");
    let cmd = docker(ctx, dir, &["compose", "up", "-d"]).stderr(Sink::Stdout);
    ctx.exec_ok(&cmd)?;

    ctx.echo("[3/4] Verifying network isolation...");
    // These probes MUST fail: if one succeeds, something is wrong.
    if interfaces(ctx, dir, Sink::Inherit).1 != "lo" && !ctx.dry_run {
        return Err(abort(ctx, dir, "FATAL: container has a network interface besides lo! Aborting."));
    }
    let tcp = docker(ctx, dir, &["exec", CONTAINER, "timeout", "3", "bash", "-c", "echo > /dev/tcp/1.1.1.1/53"]);
    if ctx.exec(&tcp.stderr(Sink::Null)).code == 0 && !ctx.dry_run {
        return Err(abort(ctx, dir, "FATAL: Network is NOT isolated! Aborting."));
    }
    let http = docker(ctx, dir, &["exec", CONTAINER, "curl", "-s", "--max-time", "2", "https://google.com"]);
    if ctx.exec(&http.stderr(Sink::Null)).code == 0 && !ctx.dry_run {
        return Err(abort(ctx, dir, "FATAL: Network is NOT isolated! Aborting."));
    }
    for line in [
        "  ✓ Only the loopback interface exists",
        "  ✓ No outbound TCP (1.1.1.1:53 unreachable)",
        "  ✓ No HTTP connectivity (curl fails)",
        "  ✓ Network isolation VERIFIED",
        "[4/4] Sandbox ready.",
        "",
        "Enter the sandbox:",
        "  docker exec -it decant-sandbox bash",
        "",
        "Build the shims first if /data/stubs is empty: make -C src/gdk all tests",
        "",
        "Inside the sandbox:",
        "  /data/game/     - game files (read-only)",
        "  /data/stubs/    - DLL shims and Wine tests (read-only, build/gdk)",
        "  /data/logs/     - output logs",
        "  ~/.wine/        - Wine prefix",
        "",
    ] {
        ctx.echo(line);
    }
    ctx.echo(&format!("Stop: docker compose -f {dir}/docker-compose.yml down"));
    Ok(())
}

/// One check of the isolation suite: commands joined with `&&`, then whether it should pass.
pub struct Probe {
    pub name: &'static str,
    /// true: `check` (must succeed). false: `check_fail` (must be blocked).
    pub must_pass: bool,
    pub steps: &'static [Step],
}

/// One command of a probe, after `docker exec decant-sandbox`.
pub enum Step {
    Run(&'static [&'static str]),
    /// `[ "$(command)" = lo ]`
    OutputIsLo(&'static [&'static str]),
}

/// The twelve checks of `test_isolation.sh`, in its order, grouped under their headings.
pub const SUITE: [(&str, &[Probe]); 3] = [
    (
        "--- Network isolation ---",
        &[
            Probe {
                name: "Only the loopback interface exists",
                must_pass: true,
                steps: &[Step::OutputIsLo(&["ls", "/sys/class/net"])],
            },
            Probe {
                name: "No TCP to 1.1.1.1:53",
                must_pass: false,
                steps: &[Step::Run(&["timeout", "3", "bash", "-c", "echo > /dev/tcp/1.1.1.1/53"])],
            },
            Probe {
                name: "No TCP to docker0 gateway 172.17.0.1:22",
                must_pass: false,
                steps: &[Step::Run(&["timeout", "3", "bash", "-c", "echo > /dev/tcp/172.17.0.1/22"])],
            },
            Probe {
                name: "No DNS resolution",
                must_pass: false,
                steps: &[Step::Run(&["getent", "hosts", "google.com"])],
            },
            Probe {
                name: "No HTTP outbound",
                must_pass: false,
                steps: &[Step::Run(&["curl", "-s", "--max-time", "2", "http://google.com"])],
            },
            Probe {
                name: "No HTTPS outbound",
                must_pass: false,
                steps: &[Step::Run(&["curl", "-s", "--max-time", "2", "https://google.com"])],
            },
        ],
    ),
    (
        "--- Filesystem isolation ---",
        &[
            Probe {
                name: "Cannot write to game data",
                must_pass: false,
                steps: &[Step::Run(&["touch", "/data/game/test"])],
            },
            Probe {
                name: "Cannot write to stubs",
                must_pass: false,
                steps: &[Step::Run(&["touch", "/data/stubs/test"])],
            },
            Probe {
                name: "Can write to logs",
                must_pass: true,
                steps: &[Step::Run(&["touch", "/data/logs/test"]), Step::Run(&["rm", "/data/logs/test"])],
            },
            Probe {
                name: "Can write to Wine prefix",
                must_pass: true,
                steps: &[
                    Step::Run(&["touch", "/home/runner/.wine/test"]),
                    Step::Run(&["rm", "/home/runner/.wine/test"]),
                ],
            },
        ],
    ),
    (
        "--- Privilege isolation ---",
        &[
            Probe {
                name: "Not running as root",
                must_pass: false,
                steps: &[Step::Run(&["sh", "-c", "test $(id -u) -eq 0"])],
            },
            Probe {
                name: "no-new-privileges is set",
                must_pass: true,
                steps: &[Step::Run(&["grep", "-q", "NoNewPrivs:[[:space:]]*1", "/proc/1/status"])],
            },
        ],
    ),
];

/// Runs the commands of a probe with stderr hidden (`eval "$2" 2>/dev/null`): true when all
/// succeed.
fn probe_succeeds(ctx: &mut Ctx, dir: &str, probe: &Probe) -> bool {
    for step in probe.steps {
        let ok = match step {
            Step::Run(args) => {
                let cmd = docker(ctx, dir, &["exec", CONTAINER]).args(args.iter().copied()).stderr(Sink::Null);
                ctx.exec(&cmd).code == 0
            }
            Step::OutputIsLo(args) => {
                let cmd = docker(ctx, dir, &["exec", CONTAINER]).args(args.iter().copied());
                let out = ctx.exec(&cmd.stdout(Sink::Capture).stderr(Sink::Null));
                out.stdout.trim_end_matches('\n') == "lo"
            }
        };
        if !ok {
            return false;
        }
    }
    true
}

/// `test_isolation.sh`, with `dir` the sandbox folder (absolute).
pub fn test_isolation(ctx: &mut Ctx, dir: &str) -> Res {
    ctx.echo("=== ISOLATION TEST SUITE ===");
    // Recreate the sandbox so the mounts are always fresh.
    let down = docker(ctx, dir, &["compose", "down"]).stdout(Sink::Null).stderr(Sink::Null);
    ctx.exec(&down);
    environment(ctx, dir, 41)?;
    for sub in ["prefix", "logs"] {
        ctx.mkdir_p(&format!("{dir}/{sub}"))?;
    }
    let up = docker(ctx, dir, &["compose", "up", "-d"]).stderr(Sink::Null);
    ctx.exec_ok(&up)?;

    let (mut pass, mut fail) = (0, 0);
    for (heading, probes) in SUITE {
        ctx.echo("");
        ctx.echo(heading);
        for probe in probes {
            let succeeded = probe_succeeds(ctx, dir, probe);
            let line = match (probe.must_pass, succeeded) {
                (true, true) => format!("  ✓ {}", probe.name),
                (true, false) => format!("  ✗ {}", probe.name),
                (false, false) => format!("  ✓ {} (correctly blocked)", probe.name),
                (false, true) => format!("  ✗ SECURITY FAILURE: {}", probe.name),
            };
            if probe.must_pass == succeeded {
                pass += 1;
            } else {
                fail += 1;
            }
            ctx.echo(&line);
        }
    }
    ctx.echo("");
    ctx.echo("--- Results ---");
    ctx.echo(&format!("Passed: {pass} / {}", pass + fail));
    ctx.echo(&format!("Failed: {fail} / {}", pass + fail));
    if fail == 0 {
        ctx.echo("ALL TESTS PASSED - sandbox is properly isolated");
        Ok(())
    } else {
        ctx.echo("SECURITY ISSUES DETECTED");
        Err(Exit(1))
    }
}
