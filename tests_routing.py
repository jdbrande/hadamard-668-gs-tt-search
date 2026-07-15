"""Routing/heartbeat/rotation tests. Run: python3 tests_routing.py"""
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

rng = np.random.default_rng(51)
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
                          text=True, timeout=600)


# ---- 1. heartbeat fires mid-match; interrupt aborts without losing keys ----
route = wk.GSRoute(13)
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    for _ in range(40):
        if len(pools[s].seqs) >= 64:
            break
        pools[s].add(route.generate(s, 20000, rng),
                     need=64 - len(pools[s].seqs))
beats = []
st = wk.MatchState("rt_state.json")
res = wk.incremental_match(
    route, pools, st, 500, 2, defaultdict(int), caches={},
    heartbeat=lambda phase, extra=None: beats.append((phase, extra)),
    should_stop=lambda: False)
check("heartbeat fires during matching with phase + counters + ETA",
      len(beats) > 0 and beats[0][0].startswith("matching key")
      and {"pairs_hashed", "probes", "eta_key_seconds",
           "key_pairs_planned"} <= set(beats[-1][1]))

# interrupt mid-run on MATCHLESS pools: completed keys persist, the
# interrupted key is NOT marked done, and completing later checks
# everything (no potentially valid region skipped)
def any_match(pls):
    for pat in route.patterns:
        for (b1, b2), (b3, b4) in route.splits(pat):
            pf = lambda b: core.paf_half_batch(pls[b].seqs)
            if engine.find_all_matches(pf(b1), pf(b2), pf(b3), pf(b4)):
                return True
    return False

tiny = {b: engine.Pool(13, True) for b in route.bins}
k = 3
while True:
    for b in route.bins:
        tiny[b].seqs = pools[b].seqs[:k].copy()
        tiny[b].keys = set()
    if not any_match(tiny) or k == 1:
        break
    k -= 1
if any_match(tiny):
    print("note: could not build matchless pools; skipping interrupt test")
else:
    if os.path.exists("rt_state.json"):
        os.remove("rt_state.json")
    st1 = wk.MatchState("rt_state.json")
    calls = {"n": 0}
    r_int = wk.incremental_match(
        route, tiny, st1, 500, 1, defaultdict(int), caches={},
        heartbeat=None,
        should_stop=lambda: calls.__setitem__("n", calls["n"] + 1)
        or calls["n"] > 15)
    completed = len(st1.marks)
    st1b = wk.MatchState("rt_state.json")
    st1b.load(tiny, wk.bin_key)
    check("interrupt: aborts cleanly, watermarks persisted only for "
          "completed keys",
          r_int is None and st1b.marks == st1.marks
          and completed < sum(len(route.splits(p))
                              for p in route.patterns))
    r_fin = wk.incremental_match(route, tiny, st1b, 500, 1,
                                 defaultdict(int), caches={},
                                 should_stop=lambda: False)
    st2 = wk.MatchState("rt_state.json")
    st2.load(tiny, wk.bin_key)
    n_keys = sum(len(route.splits(p)) for p in route.patterns)
    check(f"finishing after interrupt covers all {n_keys} keys "
          "(no potentially valid region skipped)",
          r_fin is None and len(st2.marks) == n_keys)
    if os.path.exists("rt_state.json"):
        os.remove("rt_state.json")

# ---- 2. live worker: mid-cycle progress.json heartbeats + STOP_WORKER ------
wd = "rt_work"
shutil.rmtree(wd, ignore_errors=True)
env = {**os.environ}
proc = subprocess.Popen(
    [PY, "worker.py", "--route", "gs", "--n", "41", "--worker-id", "0",
     "--workdir", wd, "--pool-cap", "4000", "--batch", "50000",
     "--max-pairs", "300000", "--chunk", "50",
     "--heartbeat-seconds", "0.2"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
ppath = os.path.join(wd, "worker_000", "progress.json")
seen_matching, updates = False, set()
deadline = time.time() + 120
while time.time() < deadline:
    if os.path.exists(ppath):
        try:
            p = json.load(open(ppath))
            updates.add(p["updated"])
            if p.get("status") == "matching":
                seen_matching = True
                break
        except (ValueError, OSError):
            pass
    time.sleep(0.5)
check("live worker updates progress.json mid-cycle (status=matching)",
      seen_matching and len(updates) >= 2)
open(os.path.join(wd, "worker_000", "STOP_WORKER"), "w").write("test\n")
try:
    proc.wait(timeout=180)
    stopped = True
except subprocess.TimeoutExpired:
    proc.kill()
    stopped = False
check("per-worker STOP_WORKER interrupts a worker mid-cycle within minutes",
      stopped)
p = json.load(open(ppath))
check("stopped worker records status=stopped", p["status"] == "stopped")

# ---- 3. acceptance verdict: fresh TT mid-cycle is not 'generation gated' ---
tt = os.path.join(wd, "worker_003")
os.makedirs(tt, exist_ok=True)
json.dump({"route": "tt"}, open(os.path.join(tt, "meta.json"), "w"))
wk.write_json_atomic(os.path.join(tt, "progress.json"), {
    "route": "tt", "updated": time.time(), "cycle": 0, "status": "matching",
    "telemetry_started_at": time.time() - 3600, "totals": {"accepted": 0},
    "pools": {str(i): 50 for i in range(19)}, "pool_cap": 100000,
    "latest_cycle": {"phase": "matching key 3/12 block 1/4",
                     "eta_key_seconds": 7200}})
r = ctl(["acceptance", "--workdir", wd])
check("acceptance labels mid-cycle TT as long-cycle-in-progress with ETA, "
      "not generation gated",
      "mid-cycle" in r.stdout and "generation gated" not in
      [l for l in r.stdout.splitlines() if " 3 " in l or "tt" in l][0]
      if any(" 3 " in l for l in r.stdout.splitlines()) else
      "mid-cycle" in r.stdout)
# stale-silent worker -> UNPROVEN BUSY
wk.write_json_atomic(os.path.join(tt, "progress.json"), {
    "route": "tt", "updated": time.time() - 7 * 3600, "cycle": 0,
    "status": "running", "telemetry_started_at": time.time() - 9 * 3600,
    "totals": {"accepted": 0}, "pools": {str(i): 50 for i in range(19)},
    "pool_cap": 100000, "latest_cycle": {}})
r = ctl(["acceptance", "--workdir", wd])
check("acceptance flags silent-busy workers as UNPROVEN BUSY",
      "UNPROVEN BUSY" in r.stdout)

# ---- 4. recommend-routes ------------------------------------------------------
gsd = os.path.join(wd, "worker_000")
p = json.load(open(os.path.join(gsd, "progress.json")))
p["totals"]["accepted"] = 50000
p["telemetry_started_at"] = time.time() - 3600
p["updated"] = time.time()
wk.write_json_atomic(os.path.join(gsd, "progress.json"), p)
r = ctl(["recommend-routes", "--workdir", wd, "--stale-hours", "3"])
check("recommend-routes summarizes per-route productivity",
      "acc/min" in r.stdout and "gs" in r.stdout and "tt" in r.stdout)
check("recommend-routes recommends shifting stale-zero TT slots to GS "
      "with exact rotate command",
      "RECOMMENDATION" in r.stdout and "rotate-gs-seeds" in r.stdout)

# ---- 5. rotate-gs-seeds: safe replacement -------------------------------------
sentinel = json.load(open(os.path.join(tt, "progress.json")))
r = ctl(["rotate-gs-seeds", "--workdir", wd, "--seeds", "91000000",
         "--replace-workers", "3", "--gs-n", "13", "--pool-cap", "200",
         "--wait-minutes", "1"])
check("rotate: merges first, retires old dir, launches fresh GS seed",
      "merging pools first" in r.stdout and "retired old dir" in r.stdout
      and "seed=91000000" in r.stdout)
retired = os.listdir(os.path.join(wd, "retired"))
check("rotate: old worker state preserved under retired/, never deleted",
      len(retired) == 1 and json.load(open(os.path.join(
          wd, "retired", retired[0], "progress.json")))["route"] == "tt")
time.sleep(3)
newp = os.path.join(wd, "worker_003", "progress.json")
deadline = time.time() + 90
ok_new = False
while time.time() < deadline:
    try:
        np_ = json.load(open(newp))
        if np_["route"] == "gs":
            ok_new = True
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
check("rotate: replacement worker runs as GS in a fresh dir", ok_new)
open(os.path.join(wd, "worker_003", "STOP_WORKER"), "w").write("cleanup\n")
time.sleep(2)
r = ctl(["stop", "--workdir", wd])
time.sleep(3)
shutil.rmtree(wd, ignore_errors=True)
for f in ("hadamard_52.csv", "hadamard_116.csv"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} ROUTING TESTS PASSED")
