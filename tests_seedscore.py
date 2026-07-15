"""Seed leaderboard tests. Run: python3 tests_seedscore.py"""
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

import worker as wk

PASS = 0
PY = sys.executable
WD = "score_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"FAIL  {name}")
        sys.exit(1)
    PASS += 1
    print(f"ok    {name}")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", WD],
                          capture_output=True, text=True, timeout=300)


def mk_worker(wid, seed, route="gs", apm=150, dup=10, colls=(),
              pinned=False, cycles=20, stale=False):
    d = os.path.join(WD, f"worker_{wid:03d}")
    os.makedirs(d, exist_ok=True)
    json.dump({"worker_id": wid, "route": route, "seed": seed,
               "started": time.time() - 7200},
              open(os.path.join(d, "meta.json"), "w"))
    acc = apm  # per 60s cycle
    rows = [json.dumps({"t": 0, "cycle": c + 1, "secs": 60.0,
                        "status": "running", "pools": {"1": 500, "3": 480},
                        "pool_cap": 1000, "generated": 100000,
                        "accepted": acc, "dup_rejected": dup,
                        "build_psd_rej": 98000, "pairs_hashed": 1000,
                        "probes": 1000, "probe_psd_rej": 50000,
                        "psd_rejected": 50000,
                        "collisions": len(colls or ()),
                        "buckets": 1000}) for c in range(cycles)]
    open(os.path.join(d, "cycles.jsonl"), "w").write("\n".join(rows) + "\n")
    if colls is not None:
        with open(os.path.join(d, "collisions.jsonl"), "w") as fh:
            for kind, hours in colls:
                fh.write(json.dumps({
                    "timestamp": time.time() - 7200 + hours * 3600,
                    "worker_id": wid, "route": route, "seed": seed,
                    "cycle": 5, "pattern": [1, 1, 1, 1],
                    "bins": ["1", "1", "1", "1"],
                    "indices": [0, 0, 0, 0], "sequences": ["+", "+",
                                                           "+", "+"],
                    "paf_residual": [0] if kind != "noisy" else [4],
                    "residual_zero": kind in ("res0", "verified"),
                    "verified_hadamard": kind == "verified",
                    "verify_exception": None}) + "\n")
    wk.write_json_atomic(os.path.join(d, "progress.json"),
                         {"route": route, "seed": seed,
                          "updated": time.time() - (7200 if stale else 5),
                          "cycle": cycles, "status": "running",
                          "bins_at_cap": 0,
                          "totals": {"accepted": acc * cycles},
                          "telemetry_started_at": time.time() - 7200,
                          "pools": {"1": 500, "3": 480},
                          "pool_cap": 1000, "latest_cycle": {}})
    if pinned:
        open(os.path.join(d, "PIN"), "w").write("clue owner\n")
    return d


shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
# seed 111: MANY noisy collisions, high apm
mk_worker(0, 111, apm=250, colls=[("noisy", 1)] * 60)
# seed 222: ONE residual_zero, lower apm, pinned clue owner
mk_worker(1, 222, apm=120, colls=[("res0", 2)], pinned=True)
# seed 333: verified hadamard
mk_worker(2, 333, apm=100, colls=[("verified", 3)])
# seed 444: no collisions.jsonl at all
mk_worker(3, 444, apm=180, colls=None)
# corrupt line in seed 111's file
with open(os.path.join(WD, "worker_000", "collisions.jsonl"), "a") as fh:
    fh.write("{not json!!\n")

r = ctl(["seed-leaderboard"])
out = r.stdout
check("command runs and writes work/seed_leaderboard.json",
      os.path.exists(os.path.join(WD, "seed_leaderboard.json")))
lb = json.load(open(os.path.join(WD, "seed_leaderboard.json")))
rank = [s["seed"] for s in lb["seeds"]]

# ---- 1. residual_zero outranks many hash-noise collisions ----
check("seed with ONE residual_zero outranks seed with 60 noisy "
      "collisions and higher acc/min",
      rank.index(222) < rank.index(111))
check("verified_hadamard outranks residual_zero",
      rank.index(333) < rank.index(222))
s111 = next(s for s in lb["seeds"] if s["seed"] == 111)
check("noisy collisions capped: 60 noisy contribute <= +5 pts",
      s111["noisy"] == 60 and s111["score"] < 1000)

# ---- 2. pinned clue worker clearly marked ----
s222 = next(s for s in lb["seeds"] if s["seed"] == 222)
check("pinned clue owner marked in JSON and table",
      s222["pinned"] is True and "PINNED(clue owner)" in out)

# ---- 3. missing collisions.jsonl handled ----
s444 = next(s for s in lb["seeds"] if s["seed"] == 444)
check("missing collisions.jsonl handled: seed still ranked with zero "
      "evidence", s444["noisy"] == 0 and s444["verified"] == 0
      and 444 in rank)

# ---- 4. corrupt lines skipped with warning ----
check("corrupt collision line skipped with a printed warning",
      "WARNING" in out and "corrupt" in out
      and any("corrupt" in w for w in lb["warnings"]))

# ---- 5. JSON schema stable ----
need_top = {"schema_version", "generated", "weights", "warnings",
            "seeds", "workers"}
need_seed = {"seed", "routes", "worker_ids", "pinned", "stale",
             "minutes", "accepted_per_min", "dup_pct", "build_pass_pct",
             "screen_rej_pct", "median_cycle_secs", "bins_at_cap",
             "pool_balance", "verified", "residual_zero", "noisy",
             "first_evidence_hours", "score"}
check("JSON schema stable: top-level and per-seed keys",
      need_top <= set(lb) and lb["schema_version"] == 1
      and all(need_seed <= set(s) for s in lb["seeds"]))
r2 = ctl(["seed-leaderboard"])
lb2 = json.load(open(os.path.join(WD, "seed_leaderboard.json")))
check("re-run produces identical schema and identical ranking",
      set(lb2) == set(lb)
      and [s["seed"] for s in lb2["seeds"]] == rank)

# ---- extras: evidence timing bonus, per-worker summaries, retired dirs ----
check("per-worker summaries present with dirs",
      len(lb["workers"]) == 4
      and all("dir" in w and "score" in w for w in lb["workers"]))
os.makedirs(os.path.join(WD, "retired_gs_1"), exist_ok=True)
shutil.move(os.path.join(WD, "worker_002"),
            os.path.join(WD, "retired_gs_1", "worker_002"))
r3 = ctl(["seed-leaderboard"])
lb3 = json.load(open(os.path.join(WD, "seed_leaderboard.json")))
check("retired worker dirs still contribute their seed's evidence",
      any(s["seed"] == 333 and s["verified"] == 1 for s in lb3["seeds"]))

# ---- dashboard card ----
os.environ.update(H668_WORKDIR=WD, H668_WORKERS="4",
                  H668_DASH_CACHE_TTL="0", H668_SEEDS="nonexistent.txt")
sys.path.insert(0, ".")
import dash_routing as dr
importlib.reload(dr)
lbc = dr.seed_leaderboard()
check("dashboard card: top seeds with counts and near-zero-weight note",
      lbc is not None and lbc["verified"] == 1
      and lbc["residual_zero"] == 1 and lbc["noisy"] == 60)
html_out = dr.render_routing_html()
check("panel renders Seed Leaderboard card with pinned marker",
      "Seed Leaderboard" in html_out and "📌" in html_out
      and "near-zero weight" in html_out)
payload = dr.routing_payload()
check("routing.json payload carries seed_leaderboard (read-only)",
      payload["seed_leaderboard"]["noisy"] == 60
      and payload["read_only"] is True)

shutil.rmtree(WD, ignore_errors=True)
print(f"\nALL {PASS} SEED-SCORE TESTS PASSED")
