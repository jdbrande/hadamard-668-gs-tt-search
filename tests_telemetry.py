"""Telemetry tests. Run: python3 tests_telemetry.py"""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

import core
import engine
import worker as wk

rng = np.random.default_rng(31)
PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


# ---- 1. counters increment at the exact creation/accept/reject points ------
route = wk.GSRoute(13)
c = defaultdict(int)
batch = route.generate(3, 20000, rng, counters=c)
check("generate() counts raw generated candidates (pre-filter)",
      c["generated"] > 0 and c["generated"] >= len(batch))
check("generated == build_psd_rejects + returned (exact accounting)",
      c["generated"] == c["build_psd_rej"] + len(batch)
      and c["build_psd_rej"] > 0)

pool = engine.Pool(13, periodic=True)
c2 = defaultdict(int)
pool.add(batch[:50], counters=c2)
first_accepted = c2["accepted"]
check("Pool.add counts accepted candidates",
      first_accepted == len(pool.seqs) and first_accepted > 0)
c3 = defaultdict(int)
pool.add(batch[:50], counters=c3)   # exact resubmission: all dups
check("Pool.add counts duplicate/equivalent rejects",
      c3["accepted"] == 0 and c3["dup_rejected"] == 50)
c4 = defaultdict(int)
pool.add(-batch[:50], counters=c4)  # negation-equivalents: also dups
check("negation-equivalents are counted as dup_rejected, not accepted",
      c4["accepted"] == 0 and c4["dup_rejected"] == 50)

# ---- 2. write_json_atomic leaves no temp files ------------------------------
os.makedirs("tel_tmp", exist_ok=True)
for i in range(50):
    wk.write_json_atomic("tel_tmp/p.json", {"i": i})
check("write_json_atomic: last write wins, no temp litter",
      json.load(open("tel_tmp/p.json"))["i"] == 49
      and not [f for f in os.listdir("tel_tmp") if ".tmp" in f])
shutil.rmtree("tel_tmp")

# ---- 3. worker writes progress.json with the agreed schema ------------------
wd = "tel_work"
shutil.rmtree(wd, ignore_errors=True)
cmd = [PY, "worker.py", "--route", "gs", "--n", "13", "--worker-id", "0",
       "--workdir", wd, "--pool-cap", "300", "--batch", "20000",
       "--max-pairs", "500000"]
subprocess.run(cmd, capture_output=True, text=True, timeout=300)
ppath = os.path.join(wd, "worker_000", "progress.json")
check("progress.json is written", os.path.exists(ppath))
p = json.load(open(ppath))
need_keys = {"opt_version", "worker_id", "route", "n", "pid", "started",
             "updated", "cycle", "cycle_seconds", "totals", "latest_cycle",
             "pools", "status", "partial_telemetry", "telemetry_started_at"}
check("progress.json carries the agreed schema", need_keys <= set(p))
need_tot = {"generated", "accepted", "dup_rejected", "pairs_hashed",
            "buckets", "probes", "build_psd_rej", "probe_psd_rej",
            "psd_rejected", "collisions"}
check("totals carry all agreed counters", need_tot <= set(p["totals"]))
check("generated/accepted/dup counters are live (nonzero after a run)",
      p["totals"]["generated"] > 0 and p["totals"]["accepted"] > 0)
check("fresh workdir -> telemetry is complete, not partial",
      p["partial_telemetry"] is False)
check("solved run reports status=solved", p["status"] == "solved")
check("latest_cycle reports rates",
      "rate_pairs_per_sec" in p["latest_cycle"]
      and "rate_probes_per_sec" in p["latest_cycle"])

# ---- 4. resume does not reset cumulative totals -----------------------------
t1 = dict(p["totals"])
os.remove(os.path.join(wd, "STOP"))
os.remove(os.path.join(wd, "SOLUTION.json"))
subprocess.run(cmd, capture_output=True, text=True, timeout=300)
p2 = json.load(open(ppath))
check("resume carries cumulative totals forward (never resets)",
      all(p2["totals"][k] >= t1[k] for k in t1)
      and p2["totals"]["pairs_hashed"] > 0
      and p2["cycle"] >= p["cycle"])
check("resume with existing progress.json keeps partial flag unchanged",
      p2["partial_telemetry"] is False)

# ---- 5. pools that predate telemetry -> partial_telemetry=true --------------
os.remove(ppath)                      # simulate a pre-telemetry run
os.remove(os.path.join(wd, "STOP"))
os.remove(os.path.join(wd, "SOLUTION.json"))
subprocess.run(cmd, capture_output=True, text=True, timeout=300)
p3 = json.load(open(ppath))
check("pre-existing pools without progress.json -> partial_telemetry=true "
      "with telemetry_started_at",
      p3["partial_telemetry"] is True and "telemetry_started_at" in p3)

# ---- 6. progress writes never corrupt NPZ checkpoints ------------------------
data = np.load(os.path.join(wd, "worker_000", "pools.npz"))
check("pools.npz loads cleanly after telemetry-enabled runs",
      len(data.files) > 0)
mig = json.load(open(os.path.join(wd, "worker_000", "migration.json")))
check("caches/migration intact after telemetry-enabled runs",
      mig["opt_version"] == wk.OPT_VERSION
      and os.path.exists(os.path.join(wd, "worker_000", "caches.npz")))

# ---- 7. dashboard: prefers progress.json, falls back to logs, stale ----------
os.environ["H668_WORKDIR"] = wd
os.environ["H668_WORKERS"] = "2"
sys.path.insert(0, ".")
dash = importlib.import_module("kid_dashboard_v3")
importlib.reload(dash)

d0 = dash.progress(0)
check("dashboard prefers progress.json and labels the source",
      d0["source"] == "progress.json"
      and d0["totals"]["generated"] > 0)

logdir = os.path.join(wd, "worker_001")
os.makedirs(logdir, exist_ok=True)
oldline = ("[w1 12:00:00] cycle 41 (2780.3s) pools={'1': 11411} "
           "pairs=2,842,251,410 buckets=2,842,251,410 "
           "probes=6,368,789,023 collchk=0 psdskip=27,438,963,780 "
           "baseline_keys=0 incr_keys=30")
with open(os.path.join(logdir, "log.txt"), "w") as fh:
    fh.write(oldline + "\n")
d1 = dash.progress(1)
check("dashboard falls back to old log format when progress.json absent",
      d1["source"] == "log fallback"
      and d1["totals"]["pairs_hashed"] == 2_842_251_410
      and d1["totals"]["probes"] == 6_368_789_023
      and d1["totals"]["probe_psd_rej"] == 27_438_963_780
      and d1["cycle"] == 41)
check("log fallback computes rates and marks telemetry partial",
      d1["latest_cycle"]["rate_pairs_per_sec"] > 0
      and d1["partial_telemetry"] is True)

old_time = time.time() - 3600
os.utime(os.path.join(logdir, "log.txt"), (old_time, old_time))
d1b = dash.progress(1)
check("stale detection: source silent for >10 min is flagged",
      d1b["stale"] is True)
check("fresh progress.json is not stale", dash.progress(0)["stale"] is False)

html_page = dash.page()
check("dashboard page renders with mixed telemetry sources",
      "progress.json" in html_page and "log fallback" in html_page
      and "Cumulative totals" in html_page)

shutil.rmtree(wd, ignore_errors=True)
if os.path.exists("hadamard_52.csv"):
    os.remove("hadamard_52.csv")

print(f"\nALL {PASS} TELEMETRY TESTS PASSED")
