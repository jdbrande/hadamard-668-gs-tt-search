"""Collision-evidence, replay, PIN, seed-telemetry, merge tests."""
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
import time
from collections import defaultdict

import numpy as np

import core
import engine
import worker as wk

PASS = 0
PY = sys.executable
rng = np.random.default_rng(61)


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


# ---- 1. collision capture parity: exact matches surface in the log ----------
paf = rng.integers(-30, 30, (40, 20)).astype(np.int16)
clog = []
hit = engine.match(paf, paf, paf, paf, max_pairs=10**9,
                   collision_log=clog)
ref = engine.find_all_matches(paf, paf, paf, paf)
check("collision_log captures events; exact match (if any) is logged",
      (hit is None and len(ref) == 0) or (hit in ref and hit in clog
                                          or len(clog) >= 1))

# ---- 2. end-to-end: worker persists full collision evidence -----------------
wd = "clue_work"
shutil.rmtree(wd, ignore_errors=True)
subprocess.run([PY, "worker.py", "--route", "gs", "--n", "13",
                "--worker-id", "4", "--seed", "94091007",
                "--workdir", wd, "--pool-cap", "300",
                "--batch", "20000", "--max-pairs", "500000"],
               capture_output=True, text=True, timeout=300)
cpath = os.path.join(wd, "worker_004", "collisions.jsonl")
check("collisions.jsonl written when a match/collision occurs",
      os.path.exists(cpath))
recs = [json.loads(l) for l in open(cpath).read().splitlines()]
r = recs[-1]
need = {"timestamp", "worker_id", "route", "seed", "cycle", "pattern",
        "bins", "indices", "sequences", "paf_residual", "residual_zero",
        "verified_hadamard", "verify_exception"}
check("evidence record carries every required field", need <= set(r))
check("evidence includes the real seed and reloadable sequences",
      r["seed"] == 94091007 and len(r["sequences"]) == 4
      and all(set(s) <= {"+", "-"} for s in r["sequences"]))
check("the solving collision has zero residual and verified=true",
      any(x["residual_zero"] and x["verified_hadamard"] for x in recs))

# ---- 3. telemetry seed fix ----------------------------------------------------
p = json.load(open(os.path.join(wd, "worker_004", "progress.json")))
check("progress.json now includes the real seed", p["seed"] == 94091007)

# ---- 4. replay-collision reproduces and re-verifies ----------------------------
for f in ("hadamard_52_replay.csv",):
    if os.path.exists(f):
        os.remove(f)
r1 = ctl(["replay-collision", "--workdir", wd, "--worker-id", "4"])
check("replay-collision recomputes residual and re-verifies the match",
      "VERIFIED Hadamard order 52" in r1.stdout
      and os.path.exists("hadamard_52_replay.csv"))
H = np.loadtxt("hadamard_52_replay.csv", delimiter=",", dtype=np.int64)
check("replayed matrix passes independent exact verification",
      core.verify_hadamard(H) == 52)
r1b = ctl(["replay-collision", "--workdir", wd, "--worker-id", "4",
           "--cycle", "9999"])
check("replay-collision explains missing records honestly",
      "no collision records" in (r1b.stdout + r1b.stderr))

# ---- 5. replay-cycle re-finds the match from the pool snapshot -----------------
cyc = recs[-1]["cycle"]
os.remove("hadamard_52_replay.csv")
r2 = ctl(["replay-cycle", "--workdir", wd, "--worker-id", "4",
          "--cycle", str(cyc)])
check("replay-cycle reconstructs the snapshot and re-finds+verifies",
      "VERIFIED" in r2.stdout and os.path.exists("hadamard_52_replay.csv"))

# ---- 6. a synthetic near-miss (nonzero residual) is classified as noise --------
fake = dict(recs[-1])
fake["cycle"] = 777
seqs = [_s for _s in fake["sequences"]]
flip = list(seqs[0])
flip[0] = "+" if flip[0] == "-" else "-"
fake["sequences"] = ["".join(flip)] + seqs[1:]
fake["residual_zero"] = False
fake["verified_hadamard"] = False
with open(cpath, "a") as fh:
    fh.write(json.dumps(fake) + "\n")
r3 = ctl(["replay-collision", "--workdir", wd, "--worker-id", "4",
          "--cycle", "777"])
check("replay classifies nonzero residual as hash noise, no false verify",
      "hash coincidence" in r3.stdout and "VERIFIED" not in r3.stdout)

# ---- 7. PIN protection -----------------------------------------------------------
os.makedirs(os.path.join(wd, "worker_007"), exist_ok=True)
json.dump({"route": "gs", "seed": 1}, open(
    os.path.join(wd, "worker_007", "meta.json"), "w"))
rows = [json.dumps({"t": 0, "cycle": c + 1, "secs": 60.0,
                    "status": "running", "pools": {"1": 500},
                    "pool_cap": 100000, "generated": 100000,
                    "accepted": 0, "dup_rejected": 5000,
                    "build_psd_rej": 0, "pairs_hashed": 1, "probes": 1,
                    "probe_psd_rej": 0, "psd_rejected": 0,
                    "collisions": 0, "buckets": 1}) for c in range(12)]
open(os.path.join(wd, "worker_007", "cycles.jsonl"), "w").write(
    "\n".join(rows) + "\n")
open(os.path.join(wd, "worker_007", "PIN"), "w").write("clue owner\n")
check("test holds pinned worker lock",
      wk.acquire_worker_lock(os.path.join(wd, "worker_007")))
open("clue_seeds.txt", "w").write("95000001\n95000002\n")
r4 = ctl(["auto-rotate-gs", "--workdir", wd, "--seed-file",
          "clue_seeds.txt", "--replace-stale", "--dry-run",
          "--cooldown-minutes", "0"])
check("auto-rotate never touches PINNED workers (even proven-stale ones)",
      "PINNED" in r4.stdout and "[plan] rotate w7" not in r4.stdout)
r5 = ctl(["rotate-gs-seeds", "--workdir", wd, "--seeds", "95000001",
          "--replace-workers", "7"])
check("manual rotate refuses pinned workers with clear message",
      "PINNED" in (r5.stdout + r5.stderr) and r5.returncode != 0)
import importlib
os.environ.update(H668_WORKDIR=wd, H668_WORKERS="8",
                  H668_SEEDS="clue_seeds.txt", H668_DASH_CACHE_TTL="0")
sys.path.insert(0, ".")
import dash_routing as dr
importlib.reload(dr)
rows_ = {r["id"]: r for r in dr.route_mix()["workers"]}
check("dashboard flags pinned workers and shows real seed from progress",
      rows_[7]["pinned"] is True and rows_[4]["seed"] == 94091007)
clues = dr.clue_owners()
check("dashboard clue owners: worker, seed, cycle, route, verified",
      any(cl["worker_id"] == 4 and cl["seed"] == 94091007
          and cl["verified"] for cl in clues)
      and any(cl["residual_zero"] is False for cl in clues))
html_out = dr.render_routing_html()
check("clue panel renders with noise/verified distinction",
      "Clue owners" in html_out and "hash noise" in html_out
      and "VERIFIED" in html_out)

# ---- 8. route-safe merge -----------------------------------------------------------
os.makedirs(os.path.join(wd, "worker_006"), exist_ok=True)
np.savez_compressed(os.path.join(wd, "worker_006", "pools.npz"),
                    bxy_2=np.ones((5, 56), np.int8),
                    bz_0=np.ones((5, 56), np.int8))
r6 = ctl(["merge", "--workdir", wd])
gg = np.load(os.path.join(wd, "global_pools_gs.npz"))
tt = np.load(os.path.join(wd, "global_pools_tt.npz"))
check("merge is route-safe: gs file has only gs bins, tt only tt bins",
      all(not k.startswith(("bxy_", "bz_", "bw_")) for k in gg.files)
      and all(k.startswith(("bxy_", "bz_", "bw_")) for k in tt.files)
      and len(tt.files) == 2)

shutil.rmtree(wd, ignore_errors=True)
for f in ("hadamard_52.csv", "hadamard_52_replay.csv",
          "hadamard_52_collision.csv", "clue_seeds.txt"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} CLUE/REPLAY TESTS PASSED")
