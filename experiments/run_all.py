"""Run every experiment in order, skipping any whose results already exist.

Resumable: each experiment writes results/expN_*.json and is skipped if that file is
present, so an interrupted run can simply be restarted.  Order is cheapest-first so that
partial runs still produce a usable paper.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# cheapest-and-most-essential first, so a partial run still yields a usable paper
STEPS = [
    ("exp0_critical_values.py", "exp0_critical_values.json"),
    ("exp3_lrv.py",             "exp3_lrv.json"),
    ("exp4_gapping.py",         "exp4_gapping.json"),
    ("exp7_cost.py",            "exp7_cost.json"),
    ("exp2_coverage_law.py",    "exp2_coverage_law.json"),
    ("exp1_main.py",            "exp1_main.json"),
    ("exp6_real.py",            "exp6_real.json"),
    ("exp5_ablations.py",       "exp5_ablations.json"),
    ("exp9_wellcond.py",        "exp9_wellcond.json"),
    ("exp8_lrv_rate.py",        "exp8_lrv_rate.json"),
]


def main():
    only = sys.argv[1:] or None
    for script, out in STEPS:
        if only and not any(o in script for o in only):
            continue
        if os.path.exists(os.path.join(RES, out)):
            print(f"[skip] {script} ({out} exists)", flush=True)
            continue
        log = os.path.join(RES, script.replace(".py", ".log"))
        print(f"[run ] {script} -> {out}", flush=True)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, script)],
                                 stdout=fh, stderr=subprocess.STDOUT)
        ok = os.path.exists(os.path.join(RES, out))
        print(f"[{'done' if ok else 'FAIL'}] {script} rc={rc} "
              f"({time.time()-t0:.0f}s, log: {os.path.basename(log)})", flush=True)
    print("[all] finished", flush=True)


if __name__ == "__main__":
    main()
