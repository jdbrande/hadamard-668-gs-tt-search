"""Acceptance-diagnostics tests. Run: python3 tests_acceptance.py"""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import json
import os
import shutil
import subprocess
import sys

import numpy as np

PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args, capture_output=True,
                          text=True, timeout=400)


# ---- 1. cycle history + cap-pressure telemetry ------------------------------
wd = "acc_work"
shutil.rmtree(wd, ignore_errors=True)
subprocess.run([PY, "worker.py", "--route", "gs", "--n", "13",
                "--worker-id", "0", "--workdir", wd, "--pool-cap", "300",
                "--batch", "20000", "--max-pairs", "500000"],
               capture_output=True, text=True, timeout=300)
hist = os.path.join(wd, "worker_000", "cycles.jsonl")
check("per-cycle history (cycles.jsonl) is written", os.path.exists(hist))
rec = json.loads(open(hist).read().splitlines()[-1])
check("history rows carry generated/accepted/dup/psd/pools/pool_cap",
      {"generated", "accepted", "dup_rejected", "build_psd_rej",
       "pools", "pool_cap", "secs", "cycle"} <= set(rec))
p = json.load(open(os.path.join(wd, "worker_000", "progress.json")))
check("progress.json reports pool_cap, cap_pressure, bins_at_cap",
      {"pool_cap", "cap_pressure", "bins_at_cap"} <= set(p)
      and p["pool_cap"] == 300)

# ---- 2. acceptance command: real run + verdicts ------------------------------
r = ctl(["acceptance", "--workdir", wd, "--cycles", "50"])
check("acceptance command summarizes acc/min, gen/min, dup%, psd%, @cap",
      "acc/min" in r.stdout and "w" not in r.stderr
      and "route gs" in r.stdout)

# synthetic saturated worker -> classifier must say pool-cap saturated
sat = os.path.join(wd, "worker_007")
os.makedirs(sat, exist_ok=True)
json.dump({"route": "gs"}, open(os.path.join(sat, "meta.json"), "w"))
rows = []
for c in range(60):
    rows.append(json.dumps({
        "t": 0, "cycle": 100 + c, "secs": 21.0, "status": "running",
        "pools": {"1": 20000, "3": 20000, "5": 20000},
        "pool_cap": 20000, "generated": 0, "accepted": 0,
        "dup_rejected": 0, "build_psd_rej": 0, "pairs_hashed": 0,
        "probes": 0, "probe_psd_rej": 0, "psd_rejected": 0,
        "collisions": 0, "buckets": 0}))
open(os.path.join(sat, "cycles.jsonl"), "w").write("\n".join(rows) + "\n")
r = ctl(["acceptance", "--workdir", wd])
check("classifier flags (f) pool-cap saturation for the production "
      "signature (21s cycles, 0 gen, pools pinned at cap)",
      "POOL-CAP SATURATED" in r.stdout or "generation gated" in r.stdout)

# duplicate-saturation signature -> (c)
dup = os.path.join(wd, "worker_008")
os.makedirs(dup, exist_ok=True)
json.dump({"route": "gs"}, open(os.path.join(dup, "meta.json"), "w"))
rows = [json.dumps({"t": 0, "cycle": c, "secs": 30.0, "status": "running",
                    "pools": {"1": 50}, "pool_cap": 20000,
                    "generated": 100000, "accepted": 1,
                    "dup_rejected": 5000, "build_psd_rej": 98000,
                    "pairs_hashed": 0, "probes": 0, "probe_psd_rej": 0,
                    "psd_rejected": 98000, "collisions": 0, "buckets": 0})
        for c in range(10)]
open(os.path.join(dup, "cycles.jsonl"), "w").write("\n".join(rows) + "\n")
r = ctl(["acceptance", "--workdir", wd])
check("classifier flags (c) duplicate saturation on dup-heavy history",
      "duplicate saturation" in r.stdout)

# ---- 3. experiment mode is isolated ------------------------------------------
sentinel = os.path.join(wd, "SENTINEL")
open(sentinel, "w").write("main work untouched\n")
before = sorted(os.listdir(wd))
shutil.rmtree("experiments", ignore_errors=True)
r = ctl(["experiment", "--workdir", wd, "--route", "gs", "--n", "13",
         "--seeds", "2", "--minutes", "0.4", "--pool-cap", "200",
         "--batch", "20000"])
check("experiment mode runs and reports per-seed acceptance",
      "acc/min" in r.stdout and "experiment dir" in r.stdout)
check("experiment never touches the main workdir",
      sorted(os.listdir(wd)) == before
      and open(sentinel).read().startswith("main work"))
exp_dirs = os.listdir("experiments")
check("experiment artifacts kept in isolated experiments/ dir",
      len(exp_dirs) == 1
      and os.path.exists(os.path.join("experiments", exp_dirs[0],
                                      "worker_000", "progress.json")))
shutil.rmtree("experiments", ignore_errors=True)
shutil.rmtree(wd, ignore_errors=True)
for f in ("hadamard_52.csv",):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} ACCEPTANCE TESTS PASSED")
