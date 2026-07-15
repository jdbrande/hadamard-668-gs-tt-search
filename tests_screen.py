"""Full-frequency PSD screen: proof tests. Run: python3 tests_screen.py
House rule: a filter ships only if tests prove it never rejects a known
valid construction."""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import os
import sys
from collections import defaultdict

import numpy as np

import core
import engine
import worker as wk

rng = np.random.default_rng(71)
PASS = 0


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


route = wk.GSRoute(13)
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    for _ in range(40):
        if len(pools[s].seqs) >= 80:
            break
        pools[s].add(route.generate(s, 20000, rng),
                     need=80 - len(pools[s].seqs))

# ---- 1. PROOF CHECK: every pair of every known solution satisfies the cap
#         at EVERY frequency (stronger than what the screen tests) ----------
tested = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda b: core.paf_half_batch(pools[b].seqs)
        sols = engine.find_all_matches(pf(b1), pf(b2), pf(b3), pf(b4))
        psd = {b: engine.psd_rows_periodic(pools[b].seqs)
               for b in (b1, b2, b3, b4)}
        for (i, j, k, l) in sols:
            assert np.all(psd[b1][i] + psd[b2][j] <= 4 * 13 + 1e-6)
            assert np.all(psd[b3][k] + psd[b4][l] <= 4 * 13 + 1e-6)
            tested += 2
check(f"theorem: all {tested} known-solution pairs satisfy the PSD cap "
      f"at every frequency (so no top-K subset can reject them)",
      tested > 0)

# ---- 2. screened solution set == complete solution set ---------------------
diff = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda b: core.paf_half_batch(pools[b].seqs)
        A, B, C, D = (pf(b) for b in (b1, b2, b3, b4))
        ref = engine.find_all_matches(A, B, C, D)
        psd = {b: engine.psd_rows_periodic(pools[b].seqs)
               for b in (b1, b2, b3, b4)}
        wA, wB = engine.top_freqs(psd[b1]), engine.top_freqs(psd[b2])
        wC, wD = engine.top_freqs(psd[b3]), engine.top_freqs(psd[b4])
        ok1 = engine._screen(psd[b1], wA, psd[b2], wB, 4 * 13)
        ok2 = engine._screen(psd[b3], wC, psd[b4], wD, 4 * 13)
        filt = {(i, j, k, l) for (i, j, k, l) in ref
                if ok1[i, j] and ok2[k, l]}
        if filt != ref:
            diff += 1
check("K=4 two-sided screen: filtered solution set == complete set on "
      "every pattern/split", diff == 0)

# ---- 3. TT(4) fixture passes at every frequency ------------------------------
X = np.array([1, 1, 1, 1], np.int8)
Y = np.array([1, 1, -1, 1], np.int8)
Z = np.array([1, 1, -1, -1], np.int8)
W = np.array([1, -1, 1], np.int8)
pp = lambda s: engine.psd_rows_padded(np.asarray(s, np.int8)[None, :], 8)[0]
m = 4
check("TT(4) fixture: X+Y and Z+W satisfy their weighted caps at every "
      "frequency",
      np.all(pp(X) + pp(Y) <= 2 * (3 * m - 1) + 1e-6)
      and np.all(pp(Z) + pp(W) <= (3 * m - 1) + 1e-6))

# ---- 4. end-to-end with the new screen: production matcher still agrees
#         with the exhaustive reference everywhere ----------------------------
disagree = 0
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        pf = lambda b: core.paf_half_batch(pools[b].seqs)
        A, B, C, D = (pf(b) for b in (b1, b2, b3, b4))
        psd = {b: engine.psd_rows_periodic(pools[b].seqs)
               for b in (b1, b2, b3, b4)}
        hit = engine.match(A, B, C, D, max_pairs=4_000_000,
                           psd1=(psd[b1], psd[b2], 4 * 13),
                           psd2=(psd[b3], psd[b4], 4 * 13))
        ref = engine.find_all_matches(A, B, C, D)
        if (hit is None) != (len(ref) == 0):
            disagree += 1
        if hit is not None:
            core.verify_hadamard(core.gs_build(
                [pools[b].seqs[x] for b, x in
                 zip((b1, b2, b3, b4), hit)]))
check("production matcher with K-screen agrees with exhaustive reference "
      "and matches verify exactly", disagree == 0)

# ---- 5. monotone strengthening: K=4 skips >= K=1 skips, matches unchanged ---
seqs = rng.choice(np.array([-1, 1], np.int8), (150, 167))
paf = core.paf_half_batch(seqs)
psd = engine.psd_rows_periodic(seqs)
skips = {}
for k in (1, 4):
    os.environ["H668_SCREEN_K"] = str(k)
    import importlib
    importlib.reload(engine)
    s = defaultdict(int)
    r = engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s,
                     psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
    skips[k] = (s["psd_skipped"], r)
os.environ.pop("H668_SCREEN_K")
importlib.reload(engine)
check("K=4 rejects at least as much as K=1 with identical match outcome",
      skips[4][0] >= skips[1][0] and (skips[4][1] is None)
      == (skips[1][1] is None))

# ---- 6. python/native parity of the new screen --------------------------------
s_nat, s_py = defaultdict(int), defaultdict(int)
r_nat = engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s_nat,
                     psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
lib = engine._LIB
engine._LIB = None
os.environ["H668_NATIVE"] = "0"
r_py = engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s_py,
                    psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
os.environ.pop("H668_NATIVE")
engine._LIB = lib
check("python and native v5 screens: identical skips, pairs, and outcome",
      s_nat["psd_skipped"] == s_py["psd_skipped"]
      and s_nat["pairs_hashed"] == s_py["pairs_hashed"]
      and (r_nat is None) == (r_py is None))

print(f"\nALL {PASS} SCREEN PROOF TESTS PASSED")
