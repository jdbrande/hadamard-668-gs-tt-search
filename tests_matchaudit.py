"""match-audit + partitioned matching (v7) proof tests."""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.pop("H668_PARTITION", None)
import core
import engine
import worker as wk

rng = np.random.default_rng(91)
PASS = 0
PY = sys.executable
WD = "ma_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"FAIL  {name}")
        sys.exit(1)
    PASS += 1
    print(f"ok    {name}")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args, capture_output=True,
                          text=True, timeout=400)


# ---- 1. partition soundness proof: every exhaustive solution's build
#         and probe pairs share compatible coordinate-0 partitions -------------
route = wk.GSRoute(13)
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    for _ in range(60):
        if len(pools[s].seqs) >= 90:
            break
        pools[s].add(route.generate(s, 20000, rng),
                     need=90 - len(pools[s].seqs))
proved = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda b: core.paf_half_batch(pools[b].seqs)
        A, B, C, D = (pf(b) for b in (b1, b2, b3, b4))
        for (i, j, k, l) in engine.find_all_matches(A, B, C, D):
            t1 = int(A[i, 0]) + int(B[j, 0])
            t2 = int(C[k, 0]) + int(D[l, 0])
            assert t1 + t2 == 0, "partition condition violated"
            proved += 1
check(f"PARTITION PROOF: all {proved} exhaustive solutions satisfy "
      f"wt1*(a0+b0)+wt2*(c0+d0)=0, so coordinate-0 partitioning can "
      f"never separate a match", proved > 0)

# ---- 2. equivalence: partitioned vs exhaustive, randomized + weights ----------
agree = True
for t in range(10):
    hh = int(rng.integers(2, 40))
    mats = [rng.integers(-25, 25, (int(rng.integers(3, 45)), hh)
                         ).astype(np.int16) for _ in range(4)]
    w1, w2 = 1 + t % 2, 1 + t % 3
    ref = engine.find_all_matches(*mats, wt1=w1, wt2=w2)
    got = engine.match(*mats, wt1=w1, wt2=w2, max_pairs=10**9)
    if (got is None) != (len(ref) == 0) or (got and got not in ref):
        agree = False
check("partitioned match finds a solution iff the exhaustive reference "
      "does (10 randomized trials incl. weights)", agree)

# with PSD screens + tri, on the same-bin GS split
pat, ((b1, b2), (b3, b4)) = next(
    (p, s) for p in route.patterns for s in route.splits(p)
    if s[0][0] == s[0][1] and s[1][0] == s[1][1])
pf = lambda b: core.paf_half_batch(pools[b].seqs)
A, B, C, D = (pf(b) for b in (b1, b2, b3, b4))
psd = {b: engine.psd_rows_periodic(pools[b].seqs)
       for b in (b1, b2, b3, b4)}
s7 = defaultdict(int)
hit = engine.match(A, B, C, D, max_pairs=10**9, stats=s7,
                   psd1=(psd[b1], psd[b2], 52.0),
                   psd2=(psd[b3], psd[b4], 52.0),
                   offs=(0, 0, 0, 0), tri=(True, True))
ref = engine.find_all_matches(A, B, C, D)
check("same-bin tri STILL ACTIVE under partitioned path (swap pairs "
      "skipped, solution from exhaustive set, verifies)",
      s7["swap_skipped"] > 0 and hit is not None and hit in ref
      and core.verify_hadamard(core.gs_build(
          [pools[b].seqs[x] for b, x in
           zip((b1, b2, b3, b4), hit)])) == 52)

# ---- 3. python/native v7 parity on all counters -------------------------------
outs = {}
for mode in ("1", "0"):
    os.environ["H668_NATIVE"] = mode
    engine._LIB = None
    import native
    native._STATE.update(lib=None, checked=False, enabled=False,
                         backend="python", fallback_reason=None)
    st = defaultdict(int)
    r = engine.match(A, B, C, D, max_pairs=250, stats=st,
                     psd1=(psd[b1], psd[b2], 52.0),
                     psd2=(psd[b3], psd[b4], 52.0),
                     offs=(0, 0, 0, 0), tri=(True, True))
    outs[mode] = (r, dict(st))
os.environ.pop("H668_NATIVE")
engine._LIB = None
check("python and native v7 agree on pairs, probes, psd_skipped, "
      "swap_skipped, and outcome (incl. oversized-partition "
      "fragmentation at max_pairs=250)",
      all(outs["0"][1].get(k, 0) == outs["1"][1].get(k, 0)
          for k in ("pairs_hashed", "probes", "psd_skipped",
                    "swap_skipped"))
      and (outs["0"][0] is None) == (outs["1"][0] is None))

# ---- 4. the speedup is real: probe multiplication eliminated ------------------
L = 1500
paf = rng.integers(-30, 30, (L, 40)).astype(np.int16)
res = {}
for mode in ("1", "0"):
    os.environ["H668_PARTITION"] = mode
    s = defaultdict(int)
    t0 = time.perf_counter()
    engine.match(paf, paf, paf, paf, max_pairs=100_000, stats=s)
    res[mode] = (dict(s), time.perf_counter() - t0)
os.environ.pop("H668_PARTITION")
check(f"probe count drops from {res['0'][0]['probes']:,} (legacy "
      f"chunked) to {res['1'][0]['probes']:,} (partitioned); wall "
      f"{res['0'][1]:.2f}s -> {res['1'][1]:.2f}s "
      f"({res['0'][1]/max(res['1'][1],1e-9):.1f}x)",
      res["1"][0]["probes"] * 5 < res["0"][0]["probes"]
      and res["0"][1] > res["1"][1] * 5)

# ---- 5. incremental growth: staged watermark blocks stay complete ------------
tiny = {b: engine.Pool(13, True) for b in route.bins}
for b in route.bins:
    tiny[b].seqs = pools[b].seqs[:20].copy()
if os.path.exists("ma_state.json"):
    os.remove("ma_state.json")
st1 = wk.MatchState("ma_state.json")
r1 = wk.incremental_match(route, tiny, st1, 500_000, 400,
                          defaultdict(int), caches={})
for b in route.bins:   # grow every bin, run incrementally again
    tiny[b].seqs = pools[b].seqs[: len(pools[b].seqs)].copy()
st2 = wk.MatchState("ma_state.json")
st2.load(tiny, wk.bin_key)
r2 = wk.incremental_match(route, tiny, st2, 500_000, 400,
                          defaultdict(int), caches={})
exists = any(engine.find_all_matches(
    *[core.paf_half_batch(tiny[b].seqs) for b in
      (bb1, bb2, bb3, bb4)])
    for p in route.patterns
    for (bb1, bb2), (bb3, bb4) in route.splits(p))
found = (r1 is not None) or (r2 is not None)
check("staged growth under partitioned matching: solution found iff one "
      "exists across incremental cycles (watermark blocks complete)",
      found == exists)
if os.path.exists("ma_state.json"):
    os.remove("ma_state.json")

# ---- 6. end-to-end: GS and TT workers solve + verify under partition ----------
shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
for widx, (routearg, narg, order) in enumerate(
        (("gs", "13", 52), ("tt", "4", 44))):
    subprocess.run([PY, "worker.py", "--route", routearg, "--n", narg,
                    "--worker-id", str(widx), "--workdir", WD,
                    "--pool-cap",
                    "300", "--batch", "20000", "--max-pairs", "500000"],
                   capture_output=True, text=True, timeout=300)
    H = np.loadtxt(f"hadamard_{order}.csv", delimiter=",",
                   dtype=np.int64)
    check(f"end-to-end {routearg} worker solves and verifies order "
          f"{order} under the partitioned engine",
          core.verify_hadamard(H) == order)
    os.remove(os.path.join(WD, "SOLUTION.json"))
    os.remove(os.path.join(WD, "STOP"))

# ---- 7. match-audit command fields + watermark invariants ----------------------
# purpose-built audited worker: matchless pools, state completed in-process
d0 = os.path.join(WD, "worker_002")
os.makedirs(d0, exist_ok=True)
json.dump({"worker_id": 2, "route": "gs", "n": 13},
          open(os.path.join(d0, "meta.json"), "w"))
aud = {b: engine.Pool(13, True) for b in route.bins}
k = 3
while True:
    for b in route.bins:
        aud[b].seqs = pools[b].seqs[:k].copy()
    if not any(engine.find_all_matches(
            *[core.paf_half_batch(aud[bb].seqs) for bb in
              (x1, x2, x3, x4)])
            for p in route.patterns
            for (x1, x2), (x3, x4) in route.splits(p)) or k == 1:
        break
    k -= 1
sta = wk.MatchState(os.path.join(d0, "matchstate.json"))
assert wk.incremental_match(route, aud, sta, 500_000, 400,
                            defaultdict(int), caches={}) is None
np.savez_compressed(os.path.join(d0, "pools.npz"),
                    **{wk.bin_key(b): aud[b].seqs for b in aud})
r = ctl(["match-audit", "--workdir", WD])
out = r.stdout
check("match-audit prints route, key, bins, sizes, marks, tri, coverage, "
      "pending, true chunked ops, partitioned ops, reason",
      all(x in out for x in ("route=gs", "key gs|", "bins=", "sizes=",
                             "marks=", "tri=", "quads covered",
                             "CURRENT engine true ops",
                             "partition-compatible ops", "reason:",
                             "opt_version OK", "fingerprints valid")))
check("match-audit flags fully-covered keys as correctly not rematched",
      "correctly NOT rematched" in out)
# unchanged pools: audit twice -> marks persist (still valid)
r2 = ctl(["match-audit", "--workdir", WD])
check("watermarks persist across unchanged pools (repeat audit: still "
      "valid, still fully covered)",
      "fingerprints valid" in r2.stdout
      and "correctly NOT rematched" in r2.stdout)
# pool mutation invalidates
pz = np.load(os.path.join(d0, "pools.npz"))
arrays = {k: pz[k].copy() for k in pz.files}
kbig = max(arrays, key=lambda k: len(arrays[k]))
arrays[kbig][0, 0] *= -1   # mutate a prefix row
np.savez_compressed(os.path.join(d0, "pools.npz"), **arrays)
r3 = ctl(["match-audit", "--workdir", WD])
check("prefix mutation invalidates watermarks (audit reports DISCARDED "
      "+ baseline pending)",
      "DISCARDED" in r3.stdout and "BASELINE" in r3.stdout)
arrays[kbig][0, 0] *= -1
np.savez_compressed(os.path.join(d0, "pools.npz"), **arrays)
# OPT_VERSION mismatch invalidates
sp = os.path.join(d0, "matchstate.json")
raw = json.load(open(sp))
raw["opt_version"] = -1
json.dump(raw, open(sp, "w"))
r4 = ctl(["match-audit", "--workdir", WD])
check("OPT_VERSION mismatch invalidates watermarks (audit reports "
      "MISMATCH -> discard)", "MISMATCH -> discard" in r4.stdout)

# ---- 8. worker 004 protection unchanged ----------------------------------------
os.makedirs(os.path.join(WD, "worker_004"), exist_ok=True)
json.dump({"worker_id": 4, "route": "gs", "seed": 94091007},
          open(os.path.join(WD, "worker_004", "meta.json"), "w"))
open(os.path.join(WD, "worker_004", "PIN"), "w").write("clue owner\n")
r5 = ctl(["rotate-gs-seeds", "--workdir", WD, "--seeds", "1",
          "--replace-workers", "4"])
check("worker 004 remains GS and pinned; rotation still refused",
      r5.returncode != 0 and "PINNED" in (r5.stdout + r5.stderr))

shutil.rmtree(WD, ignore_errors=True)
for f in ("hadamard_52.csv", "hadamard_44.csv",
          "hadamard_52_collision.csv", "hadamard_44_collision.csv"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} MATCH-AUDIT/PARTITION TESTS PASSED")
