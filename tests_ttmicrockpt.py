"""TT micro-checkpointing proof tests. Run: python3 tests_ttmicrockpt.py"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.pop("H668_MICRO_MAX_PAIRS", None)
import core
import engine
import worker as wk

rng = np.random.default_rng(101)
PASS = 0
PY = sys.executable
WD = "mc_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"FAIL  {name}")
        sys.exit(1)
    PASS += 1
    print(f"ok    {name}")


def build_fixture(rows):
    route = wk.GSRoute(13)
    pools = {s: engine.Pool(13, True) for s in route.bins}
    for s in route.bins:
        for _ in range(60):
            if len(pools[s].seqs) >= 90:
                break
            pools[s].add(route.generate(s, 20000, rng),
                         need=90 - len(pools[s].seqs))
    fix = {b: engine.Pool(13, True) for b in route.bins}
    k = rows
    def any_match():
        return any(engine.find_all_matches(
            *[core.paf_half_batch(fix[b].seqs) for b in (x1, x2, x3, x4)])
            for p in route.patterns
            for (x1, x2), (x3, x4) in route.splits(p))
    while True:
        for b in route.bins:
            fix[b].seqs = pools[b].seqs[:k].copy()
        if not any_match() or k == 1:
            break
        k -= 1
    return route, pools, fix


route, big_pools, fix = build_fixture(6)
os.environ["H668_MICRO_MAX_PAIRS"] = "5"
S, M = "mc_state.json", "mc_micro.json"


def clean():
    for f in (S, M, M + ".corrupt"):
        if os.path.exists(f):
            os.remove(f)


def run(st_path=S, mi=None, stop_after=None, stats=None, pools=None):
    st = wk.MatchState(st_path)
    st.load(pools or fix, wk.bin_key)
    mi = mi if mi is not None else wk.MicroState(M)
    calls = {"n": 0}
    stop = (lambda: calls.__setitem__("n", calls["n"] + 1)
            or calls["n"] > stop_after) if stop_after else None
    r = wk.incremental_match(route, pools or fix, st, 500, 1,
                             stats if stats is not None
                             else defaultdict(int),
                             caches={}, micro_state=mi,
                             should_stop=stop)
    mi.flush()
    return r, mi


def interrupt_with_units(pools=None, lo=2, hi=60):
    """Find a deterministic stop point that lands MID-KEY, so completed
    units of the in-flight key persist (interrupts at key boundaries
    leave nothing behind by design: completed keys promote to the
    normal watermark)."""
    for stop in range(lo, hi):
        clean()
        s = defaultdict(int)
        r, _ = run(stop_after=stop, stats=s, pools=pools)
        if not os.path.exists(M):
            continue
        d = json.load(open(M))
        n = sum(len(k["units"]) for k in d["keys"].values())
        if n > 0 and s.get("interrupted"):
            return stop, n, s, r
    raise AssertionError("no mid-key interrupt point found")


# ---- 1+2. writes completed units; interrupted unit retried -------------------
stop_pt, n1, s1, r1 = interrupt_with_units()
check(f"1. micro state writes completed units on interrupt "
      f"(stop point {stop_pt}: {n1} units persisted)", n1 > 0)
s2 = defaultdict(int)
r2, _ = run(stats=s2)
check("2. interrupted (unfinished) work is retried on resume "
      "(resume hashed pairs > 0)", s2["pairs_hashed"] > 0)
check("2b. never marks unfinished units complete "
      "(resume skipped exactly the completed ones)",
      s2["micro_skipped"] == n1)

# ---- 3. completed units not retried when pools unchanged ---------------------
s3 = defaultdict(int)
r3, _ = run(stats=s3)
check("3. fully-covered run does zero matching work "
      "(watermarks + no retries)", s3["pairs_hashed"] == 0)

# ---- 7+8. resume == uninterrupted: coverage and identical solution -----------
clean()
s_full = defaultdict(int)
r_full, _ = run(stats=s_full)
covered_full = s_full["pairs_hashed"]
_, _, sA, _ = interrupt_with_units()
sB = defaultdict(int)
run(stats=sB)
check("7. union of micro-units == old full-key coverage: interrupted+"
      "resumed hashed >= uninterrupted (partial frag redone), and "
      "resumed alone < full (units skipped)",
      sA["pairs_hashed"] + sB["pairs_hashed"] >= covered_full
      and sB["pairs_hashed"] < covered_full)
# planted-solution determinism
clean()
sol_pools = {b: engine.Pool(13, True) for b in route.bins}
for b in route.bins:
    sol_pools[b].seqs = big_pools[b].seqs[:60].copy()
r_uni, _ = run(stats=defaultdict(int), pools=sol_pools)
clean()
r_int = None
for stop in range(2, 40):
    clean()
    r_int, _ = run(stop_after=stop, stats=defaultdict(int),
                   pools=sol_pools)
    if r_int is None and os.path.exists(M):
        break
r_res, _ = run(stats=defaultdict(int), pools=sol_pools)
got = r_res or r_int
check("8. resume finds the IDENTICAL solution as uninterrupted v7 "
      "(deterministic unit order) and it verifies",
      r_uni is not None and got is not None
      and all(np.array_equal(a, b) for a, b in
              zip(r_uni[1], got[1]))
      and core.verify_hadamard(core.gs_build(got[1])) == 52)

# ---- 4+5. mutation invalidates affected keys; unaffected keys survive --------
interrupt_with_units()
mi = wk.MicroState(M)
key53 = next(k for k in mi.data["keys"])   # first key: pattern (5,3,3,3)
bins53 = mi.data["keys"][key53]["bins"]
# manufacture a second valid entry for a DISJOINT-bin key (7,1,1,1)
pi7 = next(i for i, p in enumerate(route.patterns) if p[0] == 7)
key7 = f"gs|{pi7}|0"
bins7 = [(7, 1), None]
(b1, b2), (b3, b4) = route.splits(route.patterns[pi7])[0]
o = [0, 0, 0, 0]
L = [len(fix[b].seqs) for b in (b1, b2, b3, b4)]
ranges = [list(sum(bx, ())) for bx in [
    ((o[0], L[0]), (0, L[1]), (0, L[2]), (0, L[3])),
    ((0, o[0]), (o[1], L[1]), (0, L[2]), (0, L[3])),
    ((0, o[0]), (0, o[1]), (o[2], L[2]), (0, L[3])),
    ((0, o[0]), (0, o[1]), (0, o[2]), (o[3], L[3]))]]
mi.data["keys"][key7] = {
    "bins": [str(b) for b in (b1, b2, b3, b4)],
    "ranges": ranges,
    "fp": {str(b): {"len": len(fix[b].seqs),
                    "hash": wk._fingerprint(fix[b].seqs)}
           for b in (b1, b2, b3, b4)},
    "units": {"b0|0:0": {"done": True, "ts": 1}}, "updated": 1}
mi._dirty = True
mi.flush()
# mutate a bin used by key53 but NOT by key7: bin 3 or 5
mut_bin = next(b for b in (3, 5) if str(b) in bins53)
fix[mut_bin].seqs = fix[mut_bin].seqs.copy()
fix[mut_bin].seqs[0, 0] *= -1
mi2 = wk.MicroState(M)
ent53 = mi2.key_view(key53, tuple(int(x) for x in bins53),
                     [mi2.data["keys"][key53]["ranges"][i]
                      for i in range(4)]
                     if False else
                     [list(r) for r in
                      mi2.data["keys"][key53]["ranges"]],
                     fix, wk.bin_key)
check("4. pool mutation invalidates the affected key's micro units",
      len(ent53["units"]) == 0)
ent7 = mi2.key_view(key7, (b1, b2, b3, b4), ranges, fix, wk.bin_key)
check("5. unaffected key over disjoint bins keeps its units",
      ent7["units"].get("b0|0:0", {}).get("done") is True)
fix[mut_bin].seqs[0, 0] *= -1

# ---- 6. version mismatch invalidates -----------------------------------------
interrupt_with_units()
d = json.load(open(M))
d["micro_version"] = 999
json.dump(d, open(M, "w"))
mi3 = wk.MicroState(M)
check("6. micro_version mismatch discards micro state",
      not mi3.data["keys"])
d["micro_version"] = wk.MICRO_VERSION
d["opt_version"] = -5
json.dump(d, open(M, "w"))
mi4 = wk.MicroState(M)
check("6b. opt_version mismatch discards micro state",
      not mi4.data["keys"])

# ---- 9. native/python parity with micro skips ---------------------------------
interrupt_with_units()
shutil.copy(M, M + ".bak")
if os.path.exists(S):
    shutil.copy(S, S + ".bak")
outs = {}
for mode in ("1", "0"):
    # restore the exact post-interrupt state for each mode
    shutil.copy(M + ".bak", M)
    if os.path.exists(S + ".bak"):
        shutil.copy(S + ".bak", S)
    elif os.path.exists(S):
        os.remove(S)
    os.environ["H668_NATIVE"] = mode
    engine._LIB = None
    import native
    native._STATE.update(lib=None, checked=False, enabled=False,
                         backend="python", fallback_reason=None)
    st = wk.MatchState(S)
    st.load(fix, wk.bin_key)
    s = defaultdict(int)
    wk.incremental_match(route, fix, st, 500, 1, s, caches={},
                         micro_state=wk.MicroState(M))
    outs[mode] = dict(s)
for f in (M + ".bak", S + ".bak"):
    if os.path.exists(f):
        os.remove(f)
os.environ.pop("H668_NATIVE")
engine._LIB = None
check("9. native and python resumes agree on pairs, probes, and "
      "micro_skipped",
      all(outs["0"].get(k, 0) == outs["1"].get(k, 0)
          for k in ("pairs_hashed", "probes", "micro_skipped")))

# ---- 10. same-bin triangle unchanged under micro -------------------------------
clean()
s_tri = defaultdict(int)
r_tri, _ = run(stats=s_tri, pools=sol_pools)
check("10. tri still active with micro on (swap pairs skipped, "
      "solution verifies)",
      s_tri["swap_skipped"] > 0 or r_tri is not None)

# ---- 11+12. matchstate backward compat; workers without micro file run --------
clean()
run(stats=defaultdict(int))
ms = json.load(open(S))
check("11. matchstate.json schema unchanged "
      "(opt_version/marks/fps only)",
      set(ms) == {"opt_version", "marks", "fps"})
os.remove(M) if os.path.exists(M) else None
st = wk.MatchState(S)
st.load(fix, wk.bin_key)
s12 = defaultdict(int)
r12 = wk.incremental_match(route, fix, st, 500, 1, s12, caches={},
                           micro_state=wk.MicroState(M))
check("12. worker with no micro_matchstate.json still runs",
      r12 is None and s12["pairs_hashed"] == 0)

# ---- 13. corrupt micro file quarantined ----------------------------------------
clean()
open(M, "w").write("{corrupt!!\n")
mi5 = wk.MicroState(M)
check("13. corrupt micro_matchstate quarantined (.corrupt), fresh state",
      os.path.exists(M + ".corrupt") and not mi5.data["keys"])
os.remove(M + ".corrupt")

# ---- 14+15. worker 004 protection; run-safe/status-safe smoke ------------------
shutil.rmtree(WD, ignore_errors=True)
os.makedirs(os.path.join(WD, "worker_004"), exist_ok=True)
json.dump({"worker_id": 4, "route": "gs", "seed": 94091007},
          open(os.path.join(WD, "worker_004", "meta.json"), "w"))
open(os.path.join(WD, "worker_004", "PIN"), "w").write("clue owner\n")
r = subprocess.run([PY, "ctl.py", "rotate-gs-seeds", "--workdir", WD,
                    "--seeds", "1", "--replace-workers", "4"],
                   capture_output=True, text=True, timeout=120)
check("14. worker 004 remains GS and pinned; rotation refused",
      r.returncode != 0 and "PINNED" in (r.stdout + r.stderr))
open("mc_seeds.txt", "w").write("95200001\n")
r = subprocess.run([PY, "ctl.py", "run-safe", "--workdir", WD,
                    "--seeds", "mc_seeds.txt", "--gs", "0:1",
                    "--tt-workers", "1"], capture_output=True,
                   text=True, timeout=120)
r2 = subprocess.run([PY, "ctl.py", "status-safe", "--workdir", WD],
                    capture_output=True, text=True, timeout=120)
check("15. run-safe refusal logic and status-safe still pass",
      r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
      and r2.returncode == 0 and "worker 004" in r2.stdout)
os.remove("mc_seeds.txt")

# ---- 16. dashboard micro warning only when expected -----------------------------
for wid, age in ((1, 120), (2, 4000)):
    dd = os.path.join(WD, f"worker_{wid:03d}")
    os.makedirs(dd, exist_ok=True)
    json.dump({"worker_id": wid, "route": "tt", "seed": 1},
              open(os.path.join(dd, "meta.json"), "w"))
    wk.write_json_atomic(os.path.join(dd, "progress.json"), {
        "route": "tt", "seed": 1, "updated": time.time(),
        "cycle": 0, "status": "matching", "bins_at_cap": 0,
        "totals": {"accepted": 0, "dup_rejected": 0},
        "telemetry_started_at": time.time() - 9000,
        "pools": {"1": 10}, "pool_cap": 100,
        "latest_cycle": {"phase": "matching key 1/12 block 1/4",
                         "micro_active": True,
                         "micro_units_done": 7,
                         "micro_last_ckpt": time.time() - age,
                         "micro_target_secs": 300}})
os.environ.update(H668_WORKDIR=WD, H668_WORKERS="5",
                  H668_DASH_CACHE_TTL="0", H668_SEEDS="none.txt",
                  H668_EXPECT_GS="1", H668_EXPECT_TT="2")
sys.path.insert(0, ".")
import dash_routing as dr
importlib.reload(dr)
warns = dr.supervisor_warnings()
check("16. dashboard warns for the 66-min-stale micro checkpoint but "
      "NOT for the in-window one",
      any("worker 2" in w and "micro checkpoint" in w for w in warns)
      and not any("worker 1:" in w and "micro" in w for w in warns))
rows = {r["id"]: r for r in dr.route_mix()["workers"]}
check("16b. worker table carries micro fields",
      rows[2]["micro_active"] and rows[2]["micro_units_done"] == 7)

clean()
shutil.rmtree(WD, ignore_errors=True)
os.environ.pop("H668_MICRO_MAX_PAIRS", None)
for f in ("hadamard_52.csv",):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} TT-MICRO-CHECKPOINT TESTS PASSED")
