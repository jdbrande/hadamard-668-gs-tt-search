#!/usr/bin/env python3
"""Task A: profile the true hot loops on deterministic fixtures.
Usage: python3 bench_hotloops.py [--pool work/worker_000/pools.npz]
Prints a ranked table: hotspot, python time, native time, calls/sec,
% of profiled runtime, recommended action."""
import argparse
import os
import time
from collections import defaultdict

import numpy as np

import core
import engine
import worker as wk

rng = np.random.default_rng(42)


def timeit(fn, *a, reps=1, **kw):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a, **kw)
    return (time.perf_counter() - t0) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None,
                    help="optional realistic pools.npz from a worker dir")
    ap.add_argument("--rows", type=int, default=700)
    a = ap.parse_args()

    n, h = 167, 83
    if a.pool and os.path.exists(a.pool):
        data = np.load(a.pool)
        key = max(data.files, key=lambda k: len(data[k]))
        seqs = data[key].astype(np.int8)[: a.rows]
        n = seqs.shape[1]
        h = (n - 1) // 2
        print(f"fixture: {len(seqs)} rows from {a.pool} (length {n})")
    else:
        seqs = rng.choice(np.array([-1, 1], np.int8), (a.rows, n))
        print(f"fixture: {a.rows} synthetic rows (length {n})")

    paf = core.paf_half_batch(seqs)
    psd = engine.psd_rows_periodic(seqs)
    N = len(paf)
    pair_count = N * N
    results = []

    # 1+2. pair hashing + bucket building (they are one pass in this design)
    def py_hash():
        table = defaultdict(list)
        for i in range(N):
            rows = paf[i][None, :].astype(np.int32) + paf
            for j in range(N):
                table[rows[j].astype(np.int16).tobytes()].append((i, j))
        return table

    t_py, table = timeit(py_hash)
    results.append(("pair hashing + bucket build", t_py, None, pair_count,
                    "port (dominant matcher cost)"))

    # 3. probe loop (python)
    def py_probe():
        hits = 0
        for k in range(N):
            rows = -(paf[k][None, :].astype(np.int32) + paf)
            keys = rows.astype(np.int16)
            for l in range(N):
                if keys[l].tobytes() in table:
                    hits += 1
        return hits

    t_pr, _ = timeit(py_probe)
    results.append(("probe loop", t_pr, None, pair_count,
                    "port (dominant matcher cost)"))

    # 4. PSD skip logic (vectorized python screen decision)
    w = psd.argmax(axis=1).astype(np.int32)

    def py_psd():
        vals = psd[np.arange(N), w]
        return (vals[:, None] + psd[:, w.astype(np.intp)].T
                <= 4 * n + 1e-6).sum()

    t_psd, _ = timeit(py_psd, reps=5)
    results.append(("PSD pair screen", t_psd, None, pair_count,
                    "keep in native pair loop (1 lookup/pair)"))

    # 5. collision checking (probe against a table with guaranteed hits)
    some = list(table.keys())[:200]
    def py_coll():
        c = 0
        for kbytes in some * 50:
            for (i, j) in table.get(kbytes, ()):
                c += 1
                break
        return c
    t_c, _ = timeit(py_coll, reps=3)
    results.append(("collision verification", t_c, None, len(some) * 50,
                    "leave (rare in production: collchk=0)"))

    # 6. Pool.add duplicate/equivalence checking (canonical_key dominated)
    sample = seqs[:120]
    def py_canon():
        return [core._canonical_key_py(r, True) if
                hasattr(core, "_canonical_key_py") else
                core.canonical_key(r, True) for r in sample]
    t_k, _ = timeit(py_canon)
    results.append(("Pool.add dedup (canonical_key)", t_k, None, len(sample),
                    "port (pure-python O(n^2) per row)"))

    # 7. candidate generation
    route_gs, route_tt = wk.GSRoute(167), wk.TTRoute(56)
    t_g, _ = timeit(route_gs.generate, 3, 20000, rng)
    results.append(("candidate generation (gs)", t_g, None, 20000,
                    "leave (numpy FFT-bound)"))
    t_t, _ = timeit(route_tt.generate, ("xy", 2), 20000, rng)
    results.append(("candidate generation (tt)", t_t, None, 20000,
                    "leave (numpy FFT-bound)"))

    # native comparisons where a native path exists
    import native
    if native.load() is not None:
        stats = defaultdict(int)
        t_nm, _ = timeit(engine.match, paf, paf, paf, paf,
                         max_pairs=10**9, stats=stats)
        results[0] = (results[0][0], results[0][1], t_nm * 0.5,
                      results[0][3], results[0][4])
        results[1] = (results[1][0], results[1][1], t_nm * 0.5,
                      results[1][3], results[1][4])
        if hasattr(native, "canonical_key"):
            t_nk, _ = timeit(lambda: [native.canonical_key(r, True)
                                      for r in sample])
            results[5] = (results[5][0], results[5][1], t_nk,
                          results[5][3], results[5][4])

    total = sum(r[1] for r in results)
    results.sort(key=lambda r: -r[1])
    print(f"\n{'hotspot':<34} {'python':>9} {'native':>9} "
          f"{'calls/sec':>12} {'%runtime':>9}  action")
    for name, tp, tn, calls, action in results:
        nat = f"{tn*1000:8.1f}ms" if tn else "        -"
        print(f"{name:<34} {tp*1000:7.1f}ms {nat} "
              f"{calls/tp:>12,.0f} {100*tp/total:>8.1f}%  {action}")
    print("\n(native matcher time split 50/50 across hash+probe rows; "
          "the C++ pass fuses them)")


if __name__ == "__main__":
    main()
