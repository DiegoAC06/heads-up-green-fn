"""Run the pre-merge, real HTTP-judge baseline checks from Windows PowerShell.

Save as tools/validate_http.py in your repository. Run from repo root:
    python -u tools/validate_http.py --deals 20 --seeds 11    # smoke check
    python -u tools/validate_http.py --deals 1000 --seeds 1,7,42 --include-m1

Runs pytest, launches both bots on ports 5108 and 5109 for each match,
invokes the actual judge, saves logs, and stops the servers. Does not
commit, push, change engine files, or submit anything.
"""

import argparse
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ("baseline_random", "baseline_always_half", "baseline_always_pot", "baseline_calling_station")
AGENT_PORT = 5108
OPP_PORT = 5109


def port_is_used(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server(script, port, log_path, m2="1"):
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["POKERBOT_M2"] = m2
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    finally:
        log.close()
    return process


def wait_for_server(proc, port, log_path, startup_seconds=15):
    url = f"http://127.0.0.1:{port}/"
    start = time.monotonic()
    while time.monotonic() - start < startup_seconds:
        if proc.poll() is not None:
            break
        try:
            with urlopen(url, timeout=0.4) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.15)
    detail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
    raise RuntimeError(f"Server {url} did not become ready within {startup_seconds}s.\n{detail}")


def stop_server(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def check_judge_report(output, exit_code, target_seconds):
    verdict = re.search(r"RESULT:\s*([A-Z0-9_]+)", output)
    reports = re.findall(r"agent1 half[12]:\s*thinking\s*([0-9.]+)s[^\n]*?errors\s+(\d+)", output)
    reasons = []
    if exit_code != 0:
        reasons.append(f"judge exit code {exit_code}")
    if not verdict:
        reasons.append("missing match verdict")
    if len(reports) != 2:
        reasons.append("missing per-half agent1 error/time statistics")
    if "forfeit" in output.lower() or "timed out" in output.lower():
        reasons.append("report contains forfeit/timeout text; inspect full log")
    max_time = max((float(t) for t, _ in reports), default=float("nan"))
    if any(int(e) for _, e in reports):
        reasons.append("nonzero judge errors")
    if reports and max_time >= 8:
        reasons.append("8-second official time limit reached")
    decision = verdict.group(1) if verdict else "UNKNOWN"
    if decision not in ("AGENT1_WIN", "CHOP"):
        reasons.append(f"unexpected verdict: {decision}")
    warning = bool(reports and max_time > target_seconds)
    return decision, max_time, warning, reasons


def main():
    parser = argparse.ArgumentParser(description="Run final HTTP matches against the four baseline bots")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seeds", default="1,7,42")
    parser.add_argument("--target-time", type=float, default=4.0)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--include-m1", action="store_true",
                        help="Also test M2 vs the current code with detection disabled")
    args = parser.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if args.deals < 1 or not seeds:
        parser.error("Give a positive --deals value and at least one seed")
    if port_is_used(AGENT_PORT) or port_is_used(OPP_PORT):
        parser.error("Ports 5108 or 5109 already have listeners; stop them before running")
    if not (ROOT / "judge_engine.py").is_file():
        parser.error("Run from the full tournament repo; judge_engine.py is missing")

    log_dir = ROOT / "tools" / "http_validation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_tests:
        print("\nRunning full pytest suite...", flush=True)
        env = os.environ.copy()
        env["POKERBOT_M2"] = "1"
        check = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=env)
        if check.returncode:
            raise SystemExit("Unit tests failed. Stopping before HTTP matches.")

    failed = 0
    total = 0
    opponent_names = BASELINES + (("m1_no_detection",) if args.include_m1 else ())
    for baseline in opponent_names:
        is_m1 = baseline == "m1_no_detection"
        opponent_script = (ROOT / "agent.py" if is_m1
                           else ROOT / "baseline-bots" / (baseline + ".py"))
        if not opponent_script.is_file():
            raise FileNotFoundError(opponent_script)
        for seed in seeds:
            total += 1
            safe_seed = re.sub(r"[^a-zA-Z0-9_-]", "_", seed)
            prefix = f"{baseline}_seed_{safe_seed}"
            agent_log = log_dir / (prefix + "_agent.log")
            opponent_log = log_dir / (prefix + "_opponent.log")
            report_log = log_dir / (prefix + "_judge.txt")
            agent = opponent = None
            print(f"\n[{total}] {baseline}, seed {seed}, {args.deals} paired deals", flush=True)
            try:
                agent = start_server(ROOT / "agent.py", AGENT_PORT, agent_log)
                opponent = start_server(opponent_script, OPP_PORT, opponent_log,
                                        m2="0" if is_m1 else "1")
                wait_for_server(agent, AGENT_PORT, agent_log)
                wait_for_server(opponent, OPP_PORT, opponent_log)
                command = [
                    sys.executable, "-u", str(ROOT / "judge_engine.py"),
                    "--agent1", f"http://127.0.0.1:{AGENT_PORT}",
                    "--agent2", f"http://127.0.0.1:{OPP_PORT}",
                    "--deals", str(args.deals), "--seed", seed,
                ]
                run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
                content = run.stdout + "\n" + run.stderr
                report_log.write_text(content, encoding="utf-8")
                verdict, seconds, warning, reasons = check_judge_report(
                    content, run.returncode, args.target_time
                )
                state = "PASS" if not reasons else "CHECK"
                if reasons:
                    failed += 1
                print(f"  {state}: {verdict}; max agent1 half time {seconds:.3f}s "
                      f"{'(over preferred target)' if warning else ''}", flush=True)
                for reason in reasons:
                    print(f"  - {reason}", flush=True)
                print(f"  Full report: {report_log}", flush=True)
            except (Exception, subprocess.TimeoutExpired) as exc:
                failed += 1
                print(f"  FAILED: {exc}", flush=True)
            finally:
                stop_server(opponent)
                stop_server(agent)
                time.sleep(0.25)

    print(f"\nCompleted {total} HTTP matches; {failed} require attention.", flush=True)
    print(f"Reports: {log_dir}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
