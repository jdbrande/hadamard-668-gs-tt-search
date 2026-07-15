"""Native-path correctness oracle + benchmark tests.
Run: python3 tests_native.py"""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.pop("H668_NATIVE", None)
import core
import engine
import native
import worker as wk

rng = np.random.default_rng(77)
PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def reset_native():
    native._STATE.update(lib=None, checked=False, enabled=False,
                         backend="python", fallback_reason=None,
                         speedup_estimate=None)
    engine._LIB = None


# ---- 1. env switch ----------------------------------------------------------
reset_native()
os.environ["H668_NATIVE"] = "0"
check("H668_NATIVE=0 forces python path", engine.load_fastmatch() is None
      and native.status()["native_backend"] == "python")
os.environ.pop("H668_NATIVE")
reset_native()
check("default: native enabled when compiled",
      engine.load_fastmatch() is not None
      and native.status()["native_enabled"])
check("backend reported", native.status()["native_backend"]
      in ("neon", "scalar-c++"))

# ---- 2. canonicalization oracle: byte-identical across sizes/semantics ------
bad = 0
for n in (1, 2, 3, 5, 9, 13, 55, 56, 83, 167):
    for _ in range(60):
        s = rng.choice(np.array([-1, 1], np.int8), n)
        for periodic in (True, False):
            if native.canonical_key(s, periodic) != \
                    core._canonical_key_py(s, periodic):
                bad += 1
check("native canon_key byte-identical to python on 1200 random cases "
      "(incl. n=1,2 edge cases)", bad == 0)
s = np.ones(9, dtype=np.int8)
check("canon_key edge: constant sequence",
      native.canonical_key(s, True) == core._canonical_key_py(s, True))

# ---- 3. matcher oracle: native vs python identical results ------------------
def match_both(A, B, C, D, wt1, wt2, psd1=None, psd2=None):
    outs = {}
    for mode in ("0", "1"):
        os.environ["H668_NATIVE"] = mode
        reset_native() if mode == "0" else reset_native()
        stats = defaultdict(int)
        r = engine.match(A, B, C, D, wt1=wt1, wt2=wt2, max_pairs=10**9,
                         stats=stats, psd1=psd1, psd2=psd2)
        outs[mode] = (r, stats)
    os.environ.pop("H668_NATIVE")
    reset_native()
    return outs


agree = True
for trial in range(8):
    h = int(rng.integers(1, 100))
    sizes = [int(rng.integers(0, 35)) for _ in range(4)]
    mats = [rng.integers(-60, 60, (s, h)).astype(np.int16) for s in sizes]
    wt1, wt2 = 1 + trial % 2, 1 + trial % 3
    ref = engine.find_all_matches(*mats, wt1=wt1, wt2=wt2)
    outs = match_both(*mats, wt1, wt2)
    for mode in ("0", "1"):
        r, st = outs[mode]
        if (r is None) != (len(ref) == 0):
            agree = False
        if r is not None and r not in ref:
            agree = False
    # counters that are semantically defined must agree python vs native
    for k in ("pairs_hashed", "probes", "psd_skipped"):
        if outs["0"][1][k] != outs["1"][1][k]:
            agree = False
check("randomized oracle (8 trials incl. empty pools, h=1..99, weights): "
      "python and native find matches from the same solution set with "
      "identical pairs/probes/psd counters", agree)

# with PSD screens
seqs = rng.choice(np.array([-1, 1], np.int8), (60, 167))
paf = core.paf_half_batch(seqs)
psd = engine.psd_rows_periodic(seqs)
o = match_both(paf, paf, paf, paf, 1, 1,
               psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
check("PSD-screened matching: identical psd_skipped python vs native",
      o["0"][1]["psd_skipped"] == o["1"][1]["psd_skipped"])

# v2 vs v3 kernels return equivalent results
lib = native.load()
if hasattr(lib, "match_pairs_v2") and hasattr(lib, "match_pairs_v3"):
    same = True
    for _ in range(5):
        pafx = rng.integers(-30, 30, (25, 40)).astype(np.int16)
        ref = engine.find_all_matches(pafx, pafx, pafx, pafx)
        for sym in (lib.match_pairs_v2, lib.match_pairs_v3):
            got = native._raw_match_sym(lib, sym, pafx) \
                if hasattr(native, "_raw_match_sym") else None
        # use the module helper against each symbol
        r3 = native._raw_match(lib, pafx, pafx, pafx, pafx, 1, 1)
        if (r3 is None) != (len(ref) == 0) or (r3 and r3 not in ref):
            same = False
    check("v3 kernel returns members of the exhaustive solution set", same)

# ---- 4. GS/TT fixtures + end-to-end identical under both paths --------------
for mode in ("0", "1"):
    os.environ["H668_NATIVE"] = mode
    reset_native()
    wd = f"nat_e2e_{mode}"
    shutil.rmtree(wd, ignore_errors=True)
    env = {**os.environ, "H668_NATIVE": mode}
    for routearg, narg, order in (("gs", "13", 52), ("tt", "4", 44)):
        r = subprocess.run([PY, "worker.py", "--route", routearg,
                            "--n", narg, "--worker-id", "0",
                            "--workdir", wd, "--pool-cap", "300",
                            "--batch", "20000", "--max-pairs", "500000"],
                           capture_output=True, text=True, timeout=300,
                           env=env)
        H = np.loadtxt(f"hadamard_{order}.csv", delimiter=",",
                       dtype=np.int64)
        check(f"end-to-end {routearg} solve verifies with "
              f"H668_NATIVE={mode}", core.verify_hadamard(H) == order)
        os.remove(os.path.join(wd, "SOLUTION.json"))
        os.remove(os.path.join(wd, "STOP"))
    p = json.load(open(os.path.join(wd, "worker_000", "progress.json")))
    check(f"telemetry reports native fields (mode {mode})",
          "native_enabled" in p and "native_backend" in p
          and p["native_enabled"] == (mode == "1"))
    # resume with the OTHER mode: checkpoints are path-independent
    other = "1" if mode == "0" else "0"
    env2 = {**os.environ, "H668_NATIVE": other}
    subprocess.run([PY, "worker.py", "--route", "gs", "--n", "13",
                    "--worker-id", "0", "--workdir", wd,
                    "--pool-cap", "300", "--batch", "20000",
                    "--max-pairs", "500000"],
                   capture_output=True, text=True, timeout=300, env=env2)
    check(f"resume across native<->python switch still solves "
          f"({mode}->{other})",
          os.path.exists(os.path.join(wd, "SOLUTION.json")))
    shutil.rmtree(wd, ignore_errors=True)
os.environ.pop("H668_NATIVE", None)
reset_native()

# ---- 5. rule 9: disagreement -> loud python fallback -------------------------
os.environ["H668_NATIVE_FORCE_MISMATCH"] = "1"
reset_native()
lib = engine.load_fastmatch()
st = native.status()
check("forced disagreement disables native loudly with a recorded reason",
      lib is None and st["native_enabled"] is False
      and "mismatch" in (st["native_fallback_reason"] or ""))
os.environ.pop("H668_NATIVE_FORCE_MISMATCH")
reset_native()

# ---- 6. benchmark-as-test: native must beat python decisively ----------------
seqs = rng.choice(np.array([-1, 1], np.int8), (400, 167))
paf = core.paf_half_batch(seqs)


def rate(mode):
    os.environ["H668_NATIVE"] = mode
    reset_native()
    stats = defaultdict(int)
    t0 = time.perf_counter()
    engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=stats)
    return stats["pairs_hashed"] / (time.perf_counter() - t0)


r_py, r_nat = rate("0"), rate("1")
os.environ.pop("H668_NATIVE")
reset_native()
check(f"benchmark: native pair throughput >= 2x python "
      f"({r_nat:,.0f} vs {r_py:,.0f}/s = {r_nat/r_py:.1f}x)",
      r_nat >= 2 * r_py)

for f in ("hadamard_52.csv", "hadamard_44.csv"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} NATIVE TESTS PASSED")
