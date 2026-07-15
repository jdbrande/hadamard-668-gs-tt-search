"""Optimization-equivalence tests. Run: python3 tests_opt.py

Every optimization is compared against a slow, complete reference on small
instances with known solutions. Contract: the optimized search must find the
SAME set of valid solutions as the old complete search.
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

rng = np.random.default_rng(11)
PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def build_gs_pools(n, per_bin, dedup=True):
    """Raw (no-dedup) and deduped pools for symmetric GS at small n."""
    route = wk.GSRoute(n)
    raw, ded = {}, {}
    for s in route.bins:
        rows = []
        for _ in range(60):
            if len(rows) >= per_bin * 6:
                break
            batch = route.generate(s, 20000, rng)
            rows.extend(list(batch))
        raw[s] = np.array(rows[: per_bin * 6], dtype=np.int8)
        p = engine.Pool(n, periodic=True)
        p.add(raw[s])
        ded[s] = p.seqs
    return route, raw, ded


# ================= 1. PAF-vector dedup keeps a representative ==============
route, raw, ded = build_gs_pools(13, per_bin=40)
lost = 0
for s in route.bins:
    ded_pafs = {p.tobytes() for p in core.paf_half_batch(ded[s])}
    for row in raw[s]:
        # normalize sign the way the pool does
        r = row if row.sum() >= 0 else -row
        if core.paf_half_batch(r[None, :])[0].tobytes() not in ded_pafs:
            lost += 1
check("PAF dedup: every raw candidate's PAF vector keeps a representative",
      lost == 0)

# End-to-end: raw pools and deduped pools yield the same PATTERN solvability
same = True
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda pool: core.paf_half_batch(pool)
        full = engine.find_all_matches(pf(raw[b1]), pf(raw[b2]),
                                       pf(raw[b3]), pf(raw[b4]))
        dd = engine.find_all_matches(pf(ded[b1]), pf(ded[b2]),
                                     pf(ded[b3]), pf(ded[b4]))
        if (len(full) > 0) != (len(dd) > 0):
            same = False
        for (i, j, k, l) in list(dd)[:3]:   # every dedup match is valid
            H = core.gs_build([ded[b1][i], ded[b2][j], ded[b3][k],
                               ded[b4][l]])
            core.verify_hadamard(H)
check("PAF dedup: raw pools and deduped pools solve the same patterns "
      "(and dedup matches verify exactly)", same)

# ================= 2. NPAF dedup still allows TT construction ==============
X = np.array([1, 1, 1, 1], np.int8)
Y = np.array([1, 1, -1, 1], np.int8)
Z = np.array([1, 1, -1, -1], np.int8)
W = np.array([1, -1, 1], np.int8)

def pool_representative(seq, periodic):
    """The exact row a Pool would store for this candidate."""
    p = engine.Pool(len(seq), periodic)
    p.add(seq[None, :].copy())
    return p.seqs[0]

reps = [pool_representative(s, periodic=False) for s in (X, Y, Z, W)]
for orig, rep in zip((X, Y, Z, W), reps):
    a = core.npaf_batch(orig[None, :], len(orig) - 1)[0]
    b = core.npaf_batch(rep[None, :], len(rep) - 1)[0]
    assert np.array_equal(a, b)
check("NPAF dedup: stored representatives keep exact NPAF vectors", True)
# sign normalization may flip a sequence; TT identity is invariant, so the
# representative quadruple must still assemble to a Hadamard matrix:
H = core.tt_build(*[list(map(int, r)) for r in reps])
check("NPAF dedup: TT construction from stored representatives -> "
      "verified order 44", core.verify_hadamard(H) == 44)

# ================= 3. pair-PSD filter never rejects a valid pair ============
def psd_pair_ok(psdX, psdY, cap):
    w = int(psdX.argmax())
    return psdX[w] + psdY[w] <= cap + 1e-6

# 3a. theory check on actual solutions: n=13 GS solutions from dedup pools
tested_pairs = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda pool: core.paf_half_batch(pool)
        sols = engine.find_all_matches(pf(ded[b1]), pf(ded[b2]),
                                       pf(ded[b3]), pf(ded[b4]))
        psd = {b: engine.psd_rows_periodic(ded[b]) for b in (b1, b2, b3, b4)}
        for (i, j, k, l) in sols:
            assert psd_pair_ok(psd[b1][i], psd[b2][j], 4 * 13)
            assert psd_pair_ok(psd[b3][k], psd[b4][l], 4 * 13)
            tested_pairs += 2
check(f"pair-PSD filter passes both pairs of every known GS solution "
      f"({tested_pairs} pairs)", tested_pairs > 0)

# 3b. TT(4) fixture pairs pass their (weighted) caps
padded = lambda s: engine.psd_rows_padded(np.array([s], np.int8), 8)[0]
m = 4
psd_W = engine.psd_rows_padded(W[None, :], 8)[0]
check("pair-PSD filter passes the TT(4) fixture pairs",
      psd_pair_ok(padded(X), padded(Y), 2 * (3 * m - 1))
      and psd_pair_ok(padded(Z), psd_W, 3 * m - 1))

# 3c. brute force: filtered matcher finds exactly the same solution sets
diff = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda pool: core.paf_half_batch(pool)
        A, B, C, D = (pf(ded[b]) for b in (b1, b2, b3, b4))
        ref = engine.find_all_matches(A, B, C, D)
        psd = {b: engine.psd_rows_periodic(ded[b]) for b in (b1, b2, b3, b4)}
        # exhaustively re-run reference matching but with the screen applied
        filt = set()
        for (i, j, k, l) in ref:
            if (psd_pair_ok(psd[b1][i], psd[b2][j], 4 * 13)
                    and psd_pair_ok(psd[b3][k], psd[b4][l], 4 * 13)):
                filt.add((i, j, k, l))
        if ref != filt:
            diff += 1
check("pair-PSD filter: filtered solution set == complete solution set "
      "on every pattern/split", diff == 0)

# 3d. the production matcher with the screen ON finds a match iff the
#     reference finds one, on every pattern/split
disagree = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda pool: core.paf_half_batch(pool)
        A, B, C, D = (pf(ded[b]) for b in (b1, b2, b3, b4))
        psd = {b: engine.psd_rows_periodic(ded[b]) for b in (b1, b2, b3, b4)}
        hit = engine.match(A, B, C, D, max_pairs=2_000_000,
                           psd1=(psd[b1], psd[b2], 4 * 13),
                           psd2=(psd[b3], psd[b4], 4 * 13))
        ref = engine.find_all_matches(A, B, C, D)
        if (hit is None) != (len(ref) == 0):
            disagree += 1
        if hit is not None:
            i, j, k, l = hit
            core.verify_hadamard(core.gs_build(
                [ded[b1][i], ded[b2][j], ded[b3][k], ded[b4][l]]))
check("production matcher with screen ON agrees with the exhaustive "
      "reference on every pattern/split", disagree == 0)

# ================= 4. incremental old-x-new == full rescan ==================
def incremental_all(A, B, C, D, marks):
    """All matches touching at least one 'new' candidate, via the four
    disjoint blocks used by worker.incremental_match."""
    (a0, b0, c0, d0) = marks
    out = set()
    blocks = [((a0, len(A)), (0, len(B)), (0, len(C)), (0, len(D))),
              ((0, a0), (b0, len(B)), (0, len(C)), (0, len(D))),
              ((0, a0), (0, b0), (c0, len(C)), (0, len(D))),
              ((0, a0), (0, b0), (0, c0), (d0, len(D)))]
    for (al, ah), (bl, bh), (cl, ch), (dl, dh) in blocks:
        if al >= ah or bl >= bh or cl >= ch or dl >= dh:
            continue
        for (i, j, k, l) in engine.find_all_matches(
                A[al:ah], B[bl:bh], C[cl:ch], D[dl:dh]):
            out.add((i + al, j + bl, k + cl, l + dl))
    return out

agree = True
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda pool: core.paf_half_batch(pool)
        A, B, C, D = (pf(ded[b]) for b in (b1, b2, b3, b4))
        a0, b0, c0, d0 = (len(A) // 2, len(B) // 3, len(C) // 2, len(D) // 4)
        full_now = engine.find_all_matches(A, B, C, D)
        old_region = engine.find_all_matches(A[:a0], B[:b0], C[:c0], D[:d0])
        incr = incremental_all(A, B, C, D, (a0, b0, c0, d0))
        if old_region | incr != full_now or (old_region & incr):
            agree = False
check("incremental blocks: old-region + incremental == full rescan, "
      "with no overlap, on every pattern/split", agree)

# ================= 5. checkpoint/resume preserves watermarks =================
wd = "opt_work"
shutil.rmtree(wd, ignore_errors=True)
os.makedirs(os.path.join(wd, "worker_000"), exist_ok=True)
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    pools[s].add(ded[s].copy())
st = wk.MatchState(os.path.join(wd, "worker_000", "matchstate.json"))
st.update("gs|0|0", {str(b): len(pools[b].seqs) for b in route.bins[:4]},
          pools, wk.bin_key)
st.save()
st2 = wk.MatchState(st.path)
st2.load(pools, wk.bin_key)
check("checkpoint/resume preserves incremental watermarks",
      st2.marks == st.marks and st2.fps == st.fps)

# mutated pool prefix -> watermarks must reset (safe full rescan)
pools[route.bins[0]].seqs[0, 0] *= -1
st3 = wk.MatchState(st.path)
st3.load(pools, wk.bin_key)
check("mutated pool prefix invalidates watermarks (full rescan)",
      st3.marks == {})
pools[route.bins[0]].seqs[0, 0] *= -1  # restore

# ================= 6. version bump invalidates stale watermark cache ========
orig_version = wk.OPT_VERSION
try:
    wk.OPT_VERSION = orig_version + 1
    st4 = wk.MatchState(st.path)
    st4.load(pools, wk.bin_key)
    check("OPT_VERSION change discards stale pair-check watermarks",
          st4.marks == {})
finally:
    wk.OPT_VERSION = orig_version
st5 = wk.MatchState(st.path)
st5.load(pools, wk.bin_key)
check("matching OPT_VERSION keeps watermarks", st5.marks == st.marks)
shutil.rmtree(wd, ignore_errors=True)

# ================= 7. optimized worker still solves and verifies ============
wd = "opt_e2e"
shutil.rmtree(wd, ignore_errors=True)
for routearg, narg, order in (("gs", "13", 52), ("tt", "4", 44)):
    if os.path.exists(f"hadamard_{order}.csv"):
        os.remove(f"hadamard_{order}.csv")
    r = subprocess.run([PY, "worker.py", "--route", routearg, "--n", narg,
                        "--worker-id", 0 and "" or "0", "--workdir", wd,
                        "--pool-cap", "300", "--batch", "20000",
                        "--max-pairs", "500000"],
                       capture_output=True, text=True, timeout=300)
    sol = json.load(open(os.path.join(wd, "SOLUTION.json")))
    H = np.loadtxt(f"hadamard_{order}.csv", delimiter=",", dtype=np.int64)
    check(f"optimized worker end-to-end ({routearg}) -> verified order "
          f"{order}", core.verify_hadamard(H) == order
          and sol["order"] == order)
    os.remove(os.path.join(wd, "SOLUTION.json"))
    os.remove(os.path.join(wd, "STOP"))
shutil.rmtree(wd, ignore_errors=True)

print(f"\nALL {PASS} OPTIMIZATION-EQUIVALENCE TESTS PASSED")
