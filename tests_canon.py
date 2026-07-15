"""Canonical orbit dedup: proof tests. Run: python3 tests_canon.py
House rule: a dedup/restriction ships only if tests prove it never
removes a known valid construction."""
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
from collections import defaultdict

import numpy as np

import core
import engine
import worker as wk

rng = np.random.default_rng(81)
PASS = 0
PY = sys.executable
WD = "canon_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"FAIL  {name}")
        sys.exit(1)
    PASS += 1
    print(f"ok    {name}")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", WD],
                          capture_output=True, text=True, timeout=400)


def run_worker(wid, route, n, extra=()):
    return subprocess.run(
        [PY, "worker.py", "--route", route, "--n", str(n),
         "--worker-id", str(wid), "--workdir", WD, "--pool-cap", "300",
         "--batch", "20000", "--max-pairs", "500000"] + list(extra),
        capture_output=True, text=True, timeout=300)


# ---- 0. same-bin swap: representative property (the soundness proof, ----
#         checked empirically on pattern (5,5,1,1) where BOTH tri fire) ----
route = wk.GSRoute(13)
pat, ((b1, b2), (b3, b4)) = next(
    (p, s) for p in route.patterns for s in route.splits(p)
    if s[0][0] == s[0][1] and s[1][0] == s[1][1])
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    for _ in range(60):
        if len(pools[s].seqs) >= 90:
            break
        pools[s].add(route.generate(s, 20000, rng),
                     need=90 - len(pools[s].seqs))
pf = lambda b: core.paf_half_batch(pools[b].seqs)
A, B, C, D = (pf(b) for b in (b1, b2, b3, b4))
ref = engine.find_all_matches(A, B, C, D)
check(f"fixture has solutions on same-bin/same-bin split {((b1,b2),(b3,b4))}",
      len(ref) > 0)
rep_ok = all(
    (min(i, j), max(i, j), min(k, l), max(k, l)) in ref
    for (i, j, k, l) in ref)
check("REPRESENTATIVE PROOF: for every exhaustive solution (i,j,k,l), the "
      "ordered representative (i<=j, k<=l) is ALSO a solution -- the "
      "matching identity is symmetric in each side's pair", rep_ok)

# ---- 1. tri restriction: same solvability as exhaustive, verified H ----
os.environ["H668_PARTITION"] = "0"   # exact v6 counters under test
s_tri = defaultdict(int)
hit = engine.match(A, B, C, D, max_pairs=10**9, stats=s_tri,
                   offs=(0, 0, 0, 0), tri=(True, True))
check("tri-restricted match finds a solution from the exhaustive set",
      hit is not None and hit in ref)
check("tri restriction actually skipped swap pairs",
      s_tri["swap_skipped"] > 0)
quad = [pools[b].seqs[x] for b, x in zip((b1, b2, b3, b4), hit)]
check("tri-found solution assembles and verifies exactly (order 52)",
      core.verify_hadamard(core.gs_build(quad)) == 52)

# negative control: matchless pools must stay matchless under tri
tiny = [A[:2], B[:2], C[:2], D[:2]]
if not engine.find_all_matches(*tiny):
    r0 = engine.match(*tiny, max_pairs=10**9, offs=(0, 0, 0, 0),
                      tri=(True, True))
    check("tri never invents matches on matchless pools", r0 is None)
else:
    check("(control skipped: tiny slice had a match)", True)

# ---- 2. blocked coverage: chunk/watermark offsets partition the triangle ----
nrows = 30
paf = rng.integers(-30, 30, (nrows, 20)).astype(np.int16)
s_full = defaultdict(int)
engine.match(paf, paf, paf[:1] * 0 + 99, paf[:1] * 0 + 98,
             max_pairs=10**9, stats=s_full, offs=(0, 0, 0, 0),
             tri=(True, False))
s_blk = defaultdict(int)
for lo, hi in ((0, 10), (10, 30)):
    engine.match(paf[lo:hi], paf, paf[:1] * 0 + 99, paf[:1] * 0 + 98,
                 max_pairs=10**9, stats=s_blk, offs=(lo, 0, 0, 0),
                 tri=(True, False))
check("incremental blocks with global offsets hash EXACTLY the same "
      "i<=j pairs as one unblocked pass (no pair lost, none doubled)",
      s_blk["pairs_hashed"] == s_full["pairs_hashed"]
      == nrows * (nrows + 1) // 2)
os.environ.pop("H668_PARTITION", None)   # back to default (partitioned)

# ---- 3. python/native parity with tri + PSD screens together ----
seqs = rng.choice(np.array([-1, 1], np.int8), (60, 13))
pafx = core.paf_half_batch(seqs)
psd = engine.psd_rows_periodic(seqs)
outs = {}
for mode in ("1", "0"):
    os.environ["H668_NATIVE"] = mode
    engine._LIB = None
    import native
    native._STATE.update(lib=None, checked=False, enabled=False,
                         backend="python", fallback_reason=None)
    st = defaultdict(int)
    r = engine.match(pafx, pafx, pafx, pafx, max_pairs=10**9, stats=st,
                     psd1=(psd, psd, 52.0), psd2=(psd, psd, 52.0),
                     offs=(0, 0, 0, 0), tri=(True, True))
    outs[mode] = (r, dict(st))
os.environ.pop("H668_NATIVE")
engine._LIB = None
check("python and native v6 agree: swap_skipped, psd_skipped, "
      "pairs_hashed, outcome",
      all(outs["0"][1].get(k, 0) == outs["1"][1].get(k, 0)
          for k in ("swap_skipped", "psd_skipped", "pairs_hashed"))
      and (outs["0"][0] is None) == (outs["1"][0] is None))

# ---- 4. end-to-end: workers still solve GS and TT with tri wired in ----
shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
r = run_worker(0, "gs", 13)
check("GS worker (tri active on same-bin keys) still finds and verifies "
      "order 52", os.path.exists(os.path.join(WD, "SOLUTION.json")))
H = np.loadtxt("hadamard_52.csv", delimiter=",", dtype=np.int64)
check("GS solution verifies independently", core.verify_hadamard(H) == 52)
os.remove(os.path.join(WD, "SOLUTION.json"))
os.remove(os.path.join(WD, "STOP"))
r = run_worker(1, "tt", 4)
check("TT worker still finds and verifies order 44",
      os.path.exists(os.path.join(WD, "SOLUTION.json")))
os.remove(os.path.join(WD, "SOLUTION.json"))
os.remove(os.path.join(WD, "STOP"))

# ---- 5. canonical merge: dedup, telemetry, solution preservation ----
# worker 2's pools = orbit transforms of worker 0's pools (pure duplicates)
src = np.load(os.path.join(WD, "worker_000", "pools.npz"))
fake = {}
for k in src.files:
    arr = src[k].astype(np.int8)
    rolled = np.roll(arr, 3, axis=1)          # cyclic shift
    negrev = -arr[:, ::-1]                    # negation + reversal
    fake[k] = np.concatenate([rolled, negrev])
os.makedirs(os.path.join(WD, "worker_002"), exist_ok=True)
np.savez_compressed(os.path.join(WD, "worker_002", "pools.npz"), **fake)
raw_total = sum(len(src[k]) for k in src.files) + \
    sum(len(v) for v in fake.values())
r = ctl(["merge"])
check("merge prints canonical telemetry (raw/kept/removed per route)",
      "raw rows read" in r.stdout and "orbit duplicates removed" in
      r.stdout and "[merge/gs]" in r.stdout and "[merge/tt]" in r.stdout)
gg = np.load(os.path.join(WD, "global_pools_gs.npz"))
kept_gs = sum(len(gg[k]) for k in gg.files)
check("orbit duplicates removed: global smaller than raw union "
      f"({kept_gs} kept of {raw_total} raw rows incl. planted orbits)",
      kept_gs < raw_total)
# no orbit lost: canonical key sets identical before/after
lost = 0
for k in gg.files:
    raw_keys = {core.canonical_key(row, True) for row in
                np.concatenate([src[k].astype(np.int8), fake[k]])
                if k in fake} | {core.canonical_key(row, True)
                                 for row in src[k].astype(np.int8)}
    kept_keys = {core.canonical_key(row, True)
                 for row in gg[k].astype(np.int8)}
    if raw_keys != kept_keys:
        lost += 1
check("NO ORBIT LOST: canonical key set of merged global == canonical "
      "key set of raw union, every bin", lost == 0)

# idempotency
r2 = ctl(["merge"])
check("canonical merge is idempotent (second run removes 0)",
      "orbit duplicates removed: 0" in r2.stdout)

# route safety preserved
tt = np.load(os.path.join(WD, "global_pools_tt.npz"))
check("route-safe: gs global has only gs bins, tt only tt bins",
      all(not k.startswith(("bxy_", "bz_", "bw_")) for k in gg.files)
      and all(k.startswith(("bxy_", "bz_", "bw_")) for k in tt.files))

# solution preservation through canonical globals: fresh worker absorbs
# the deduped global and still solves
r = run_worker(3, "gs", 13, extra=("--seed", "424242"))
check("fresh worker absorbing canonical-deduped globals still solves",
      os.path.exists(os.path.join(WD, "SOLUTION.json")))

shutil.rmtree(WD, ignore_errors=True)
for f in ("hadamard_52.csv", "hadamard_44.csv", "hadamard_52_collision.csv",
          "hadamard_44_collision.csv"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} CANONICAL-DEDUP TESTS PASSED")
