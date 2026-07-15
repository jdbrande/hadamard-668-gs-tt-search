"""Migration tests. Run: python3 tests_migration.py

Contract: existing pools are reused (never start from zero when pools
exist), corrupt files are quarantined, dedup and caches are rebuilt under a
version stamp, and NO checked-pair history is trusted unless recorded by the
current optimization version -- proven by an adversarial legacy matchstate
that falsely claims a solution-containing region was already checked.
"""
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

import core
import engine
import worker as wk

rng = np.random.default_rng(23)
PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def build_solvable_gs_pools(n=13, per_bin=80):
    """Deduped pools guaranteed to contain at least one GS solution."""
    route = wk.GSRoute(n)
    while True:
        pools = {s: engine.Pool(n, True) for s in route.bins}
        for s in route.bins:
            for _ in range(40):
                if len(pools[s].seqs) >= per_bin:
                    break
                pools[s].add(route.generate(s, 20000, rng),
                             need=per_bin - len(pools[s].seqs))
        for pat in route.patterns:
            for (b1, b2), (b3, b4) in route.splits(pat):
                pf = lambda b: core.paf_half_batch(pools[b].seqs)
                if engine.find_all_matches(pf(b1), pf(b2), pf(b3), pf(b4)):
                    return route, pools
        # extremely unlikely to loop at n=13, but stay deterministic-safe


route, pools = build_solvable_gs_pools()

# ---- 1. legacy pools are REUSED, with duplicates/equivalents deduped -------
wd = "mig_work"
shutil.rmtree(wd, ignore_errors=True)
mydir = os.path.join(wd, "worker_000")
os.makedirs(mydir)
legacy = {}
for b, p in pools.items():
    dup = np.concatenate([p.seqs, p.seqs[:10], -p.seqs[:10],
                          np.roll(p.seqs[:10], 3, axis=1)])
    legacy[wk.bin_key(b)] = dup
np.savez_compressed(os.path.join(mydir, "pools.npz"), **legacy)

# ---- 2. adversarial legacy matchstate: claims EVERYTHING already checked ---
huge = {f"gs|{pi}|{si}": {str(b): 10_000_000 for b in route.bins}
        for pi in range(len(route.patterns)) for si in range(3)}
json.dump({"opt_version": 1, "marks": huge, "fps": {}},
          open(os.path.join(mydir, "matchstate.json"), "w"))

cmd = [PY, "worker.py", "--route", "gs", "--n", "13", "--worker-id", "0",
       "--workdir", wd, "--pool-cap", "500", "--batch", "20000",
       "--max-pairs", "500000"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
out = r.stdout + r.stderr

check("legacy pools are reused (resume message, not fresh start)",
      "resumed pools" in out)
check("legacy checked-pair claims are DISCARDED (version mismatch logged)",
      "discarding watermarks" in out)
check("migration index written with note that history was not imported",
      json.load(open(os.path.join(mydir, "migration.json")))
      ["checked_pair_history_imported"] is False)
# Baseline evidence: the ONLY way to find a solution here is to rescan the
# region the adversarial matchstate claimed was already checked. A solved
# run therefore proves the baseline pass executed. (If it solves later than
# cycle 1, the stats line also shows baseline_keys > 0.)
check("solution hidden in the falsely-'checked' region was still found "
      "(proves the baseline full pass ran)",
      os.path.exists(os.path.join(wd, "SOLUTION.json")))
H = np.loadtxt("hadamard_52.csv", delimiter=",", dtype=np.int64)
check("migrated-pool solution verifies exactly", core.verify_hadamard(H) == 52)

mig = json.load(open(os.path.join(mydir, "migration.json")))
raw_total = sum(mig["raw_rows_loaded"].values())
kept_total = sum(mig["stored_after_dedup"].values())
check("duplicates/equivalents were deduped during migration "
      f"({raw_total} raw -> {kept_total} kept)", kept_total < raw_total)
check("optimization-versioned index carries the current version stamp",
      mig["opt_version"] == wk.OPT_VERSION)
check("versioned PAF/PSD cache files were created",
      os.path.exists(os.path.join(mydir, "caches.npz"))
      and json.load(open(os.path.join(mydir, "cacheinfo.json")))
      ["opt_version"] == wk.OPT_VERSION)
shutil.rmtree(wd, ignore_errors=True)

# ---- 3. ctl.py migrate: offline, idempotent, quarantines corrupt files -----
wd = "mig_ctl"
shutil.rmtree(wd, ignore_errors=True)
good = os.path.join(wd, "worker_000")
bad = os.path.join(wd, "worker_001")
os.makedirs(good), os.makedirs(bad)
np.savez_compressed(os.path.join(good, "pools.npz"), **legacy)
with open(os.path.join(bad, "pools.npz"), "wb") as fh:
    fh.write(b"corrupt bytes, not a zip")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", wd],
                          capture_output=True, text=True, timeout=300)


r1 = ctl(["migrate"])
check("ctl migrate: healthy legacy pool migrated with dedup + caches",
      "migrated" in r1.stdout
      and os.path.exists(os.path.join(good, "migration.json"))
      and os.path.exists(os.path.join(good, "caches.npz")))
check("ctl migrate: corrupt pool quarantined, not fatal",
      "quarantined" in r1.stdout
      and os.path.isdir(os.path.join(wd, "bad_npz_backup")))
r2 = ctl(["migrate"])
check("ctl migrate is idempotent (second run: up to date)",
      "up to date" in r2.stdout and "already current" in r2.stdout)

# ---- 4. cache version stamp: stale caches invalidated and rebuilt ----------
info_path = os.path.join(good, "cacheinfo.json")
info = json.load(open(info_path))
info["opt_version"] = wk.OPT_VERSION - 1
json.dump(info, open(info_path, "w"))
gpools = {b: engine.Pool(route.lengths[b], route.periodic)
          for b in route.bins}
data = np.load(os.path.join(good, "pools.npz"))
for b in route.bins:
    if wk.bin_key(b) in data:
        gpools[b].add(data[wk.bin_key(b)].astype(np.int8))
msgs = []
caches = wk.load_caches(good, route, gpools, wd, log=msgs.append)
check("stale cache version is invalidated on load",
      caches == {} and any("invalidating" in m for m in msgs))
caches = wk.ensure_caches(route, gpools, caches)
wk.save_caches(good, route, gpools, caches)
check("rebuilt cache carries the current version stamp",
      json.load(open(info_path))["opt_version"] == wk.OPT_VERSION)
msgs = []
caches2 = wk.load_caches(good, route, gpools, wd, log=msgs.append)
check("valid caches are reused on next load",
      len(caches2) > 0 and any("reused" in m for m in msgs))
shutil.rmtree(wd, ignore_errors=True)

# ---- 5. baseline once, then strictly incremental ---------------------------
from collections import defaultdict
state = wk.MatchState(os.path.join(".", "mig_state.json"))
small = {b: engine.Pool(route.lengths[b], route.periodic)
         for b in route.bins}
for b in route.bins:                       # tiny pools with no solution
    while True:
        small[b].seqs = np.empty((0, route.lengths[b]), np.int8)
        small[b].keys = set()
        small[b].add(pools[b].seqs[:3])
        break
# shrink until no matches exist in the tiny pools
def any_match(pls):
    for pat in route.patterns:
        for (b1, b2), (b3, b4) in route.splits(pat):
            pf = lambda b: core.paf_half_batch(pls[b].seqs)
            if engine.find_all_matches(pf(b1), pf(b2), pf(b3), pf(b4)):
                return True
    return False
k = 3
while any_match(small) and k > 0:
    k -= 1
    for b in route.bins:
        small[b].seqs = small[b].seqs[:k]
if any_match(small):
    print("note: could not build matchless tiny pools; skipping 5a/5b")
else:
    caches = {}
    s1 = defaultdict(int)
    assert wk.incremental_match(route, small, state, 500000, 4000, s1,
                                caches=caches) is None
    check("baseline pass hashes pairs on first run", s1["pairs_hashed"] > 0)
    s2 = defaultdict(int)
    assert wk.incremental_match(route, small, state, 500000, 4000, s2,
                                caches=caches) is None
    check("second pass with unchanged pools does ZERO pair work "
          "(watermarks cover everything)",
          s2["pairs_hashed"] == 0 and s2["probes"] == 0)
    # grow pools with the full solvable set: solution must now be found,
    # via incremental blocks only
    for b in route.bins:
        small[b].add(pools[b].seqs.copy())
    s3 = defaultdict(int)
    found = wk.incremental_match(route, small, state, 500000, 4000, s3,
                                 caches=caches)
    check("growth is scanned incrementally and the new solution is found",
          found is not None and s3["incremental_keys"] > 0)
    pat, quad = found
    check("incrementally-found solution assembles and verifies exactly",
          core.verify_hadamard(core.gs_build(quad)) == 52)
os.remove("mig_state.json") if os.path.exists("mig_state.json") else None
if os.path.exists("hadamard_52.csv"):
    os.remove("hadamard_52.csv")

print(f"\nALL {PASS} MIGRATION TESTS PASSED")
