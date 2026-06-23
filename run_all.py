"""
run_all.py — runs full FTL-IGM driving pipeline

Usage:
    python run_all.py

Steps:
    1. collect_data.py   — collect real HighwayEnv trajectories
    2. train.py          — train conditional model
    3. inverse_learning.py — invert concept + all evaluations
"""

import subprocess, sys, os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

def run(script, args=""):
    cmd = f"{sys.executable} {os.path.join(SCRIPTS, script)} {args}"
    print(f"\n{'='*60}\n{cmd}\n{'='*60}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"FAILED: {script}")
        sys.exit(1)

if __name__ == "__main__":
    run("collect_data.py", "--n_train 150 --n_fewshot 5")
    run("train.py",        "--epochs 80")
    run("inverse_learning.py", "--steps 1000 --n_closed_loop 10")
    print("\n\nDone. Check ../Results/ for all plots.")
