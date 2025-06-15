#!/usr/bin/env python3
import argparse
import numpy as np
import subprocess
import os
import shutil
import re
import time
import random
import uuid
import errno
from pathlib import Path
import concurrent.futures

# --- Configuration: adjust these paths as needed ---
BASE_PARAM_FILE = Path("/home/justinfearis/concept/param/Bachelor/Uncertain/Uncertain.param")
OUTPUT_DIR      = Path("/home/justinfearis/concept/output/Usikkertest2")
CONCEPT_BIN     = Path("/home/justinfearis/concept/concept")

# --- Functions to prepare and run one simulation ---

def make_paramfile(idx: int, seed_amp: int, seed_phase: int):
    # Build unique tag and run directory
    tag        = f"id{idx:04d}_amp{seed_amp}_phase{seed_phase}"
    run_dir    = OUTPUT_DIR / tag
    param_path = OUTPUT_DIR / f"params_{tag}.param"

    # Copy base .param and read lines
    if param_path.exists():
        param_path.unlink()
    shutil.copy(BASE_PARAM_FILE, param_path)
    original = param_path.read_text().splitlines()

    # Remove any old overrides: tag, path.*, random_seeds
    lines = []
    skip_random = False
    for l in original:
        if re.match(r"^\s*tag\s*", l):
            continue
        if re.match(r"^\s*path\.output_dir", l) or re.match(r"^\s*path\.job_dir", l):
            continue
        if not skip_random and re.match(r"^\s*random_seeds\s*=" , l):
            skip_random = True
            continue
        if skip_random:
            if re.match(r"^\s*}\s*", l):
                skip_random = False
            continue
        lines.append(l)

    # Construct override block
    override = [
        f'_tag               = "{tag}"',
        f'path.output_dir   = "{run_dir}"',
        f'path.job_dir      = "{run_dir}/job"',
        "",
        "random_seeds = {",
        "    'general'              : 0,",
        f"    'primordial amplitudes': {seed_amp},",
        f"    'primordial phases'    : {seed_phase},",
        "}",
        ""
    ]

    # Write new parameter file
    param_path.write_text("\n".join(override + lines))
    return param_path, run_dir


def run_concept(param_path: Path, run_dir: Path):
    # Prepare run and job directories
    run_dir.mkdir(parents=True, exist_ok=True)
    job_dir = run_dir / f"job_{uuid.uuid4().hex[:6]}"
    job_dir.mkdir()
    bin_dir = run_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    local_bin = bin_dir / "concept"
    shutil.copy(CONCEPT_BIN, local_bin)

    # Single-thread environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    fake_home = run_dir / "fake_HOME"
    fake_home.mkdir()
    env["HOME"] = str(fake_home)

    # Execute CONCEPT with retry logic
    cmd = [str(local_bin), "-p", str(param_path), "-n", "1", "--local"]
    logpath = run_dir / f"{param_path.stem}.log"
    for attempt in range(1, 6):
        with open(logpath, "a") as logfile:
            if attempt > 1:
                logfile.write(f"\n--- RETRY {attempt} ---\n")
            try:
                subprocess.check_call(cmd, cwd=str(job_dir), env=env,
                                      stdout=logfile, stderr=subprocess.STDOUT)
                break
            except (subprocess.CalledProcessError, OSError):
                time.sleep(random.uniform(0.01, 0.05))
    shutil.rmtree(fake_home, ignore_errors=True)


def _run_one(task):
    idx, amp, phase = task
    cfg, run_dir = make_paramfile(idx, amp, phase)
    run_concept(cfg, run_dir)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Run CONCEPT with random seeds")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Ensemble mode
    p_ens = sub.add_parser("ensemble", help="Run an ensemble of random-seed simulations")
    p_ens.add_argument("--total-samples", type=int, required=True,
                       help="Number of random realizations to run")
    p_ens.add_argument("--parallel", type=int, default=1,
                       help="Number of single-core jobs in parallel")
    p_ens.add_argument("--generate-only", action="store_true", help="Only write out the .param files, then exit")

    # Single run mode
    p_sin = sub.add_parser("single", help="Run a single simulation with given seeds")
    p_sin.add_argument("--seed-amp",   type=int, required=True)
    p_sin.add_argument("--seed-phase", type=int, required=True)
    p_sin.add_argument("--index",      type=int, default=1,
                       help="Index for naming this run")

    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "ensemble":
        # Generate random seeds
        amps   = np.random.randint(0, 2**31-1, size=args.total_samples)
        phases = np.random.randint(0, 2**31-1, size=args.total_samples)
        tasks  = [(i, a, p) for i, (a, p) in enumerate(zip(amps, phases), start=1)]
        if args.generate_only:
            for idx, amp, phase in tasks:
                param_path, _ = make_paramfile(idx, amp, phase)
                print("[GEN] wrote", param_path.name)
            return
        if args.parallel > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as exe:
                for idx in exe.map(_run_one, tasks):
                    print(f"Completed run {idx}/{args.total_samples}")
        else:
            for task in tasks:
                print(f"Starting run {task[0]}/{args.total_samples}")
                _run_one(task)

    else:
        cfg, run_dir = make_paramfile(args.index,
                                      args.seed_amp, args.seed_phase)
        run_concept(cfg, run_dir)
        print(f"Completed single run idx={args.index}")

if __name__ == "__main__":
    main()
