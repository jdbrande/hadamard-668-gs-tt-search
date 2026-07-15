#!/usr/bin/env python3
"""Task F benchmark: python vs native on deterministic fixtures.
Usage: python3 bench_native.py [--rows 400]"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np


def run(rows):
    import core
    import engine
    import worker as wk
    rng = np.random.default_rng(99)
    n, h = 167, 83
    seqs = rng.choice(np.array([-1, 1], np.int8), (rows, n))
    paf = core.paf_half_batch(seqs)
    psd = engine.psd_rows_periodic(seqs)

    # raw throughput (no PSD screen): comparable to dashboard pair ops/sec
    stats0 = defaultdict(int)
    t0 = time.perf_counter()
    engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=stats0)
    raw_dt = time.perf_counter() - t0
    raw_pairs = stats0["pairs_hashed"] / raw_dt
    raw_probes = stats0["probes"] / raw_dt

    # screened run (production configuration)
    stats = defaultdict(int)
    t0 = time.perf_counter()
    engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=stats,
                 psd1=(psd, psd, 4 * n), psd2=(psd, psd, 4 * n))
    dt = time.perf_counter() - t0
    pairs = raw_pairs
    probes = raw_probes
    # end-to-end mini cycle: incremental_match on a solvable toy instance
    route = wk.GSRoute(13)
    pools = {s: engine.Pool(13, True) for s in route.bins}
    g = np.random.default_rng(5)
    for s in route.bins:
        for _ in range(20):
            if len(pools[s].seqs) >= 64:
                break
            pools[s].add(route.generate(s, 20000, g),
                         need=64 - len(pools[s].seqs))
    st = wk.MatchState("bench_state.json")
    t1 = time.perf_counter()
    wk.incremental_match(route, pools, st, 2_000_000, 4000,
                         defaultdict(int), caches={})
    cyc = time.perf_counter() - t1
    if os.path.exists("bench_state.json"):
        os.remove("bench_state.json")
    return {"pair_ops_sec": pairs, "probe_ops_sec": probes,
            "psd_rej_sec": stats["psd_skipped"] / dt,
            "match_time": raw_dt, "cycle_time": cyc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    a = ap.parse_args()

    os.environ["H668_NATIVE"] = "0"
    py = run(a.rows)
    os.environ["H668_NATIVE"] = "1"
    import engine
    engine._LIB = None            # drop cached python decision
    import native
    native._STATE.update(lib=None, checked=False)
    nat = run(a.rows)
    import native as nv
    st = nv.status()

    print(f"\nfixture: {a.rows} candidates/side, h=83, PSD screen ON")
    print(f"{'metric':<24} {'python':>14} {'native':>14} {'speedup':>9}")
    for key, label in (("pair_ops_sec", "pair ops/sec"),
                       ("probe_ops_sec", "probe ops/sec"),
                       ("psd_rej_sec", "PSD rejects/sec"),
                       ("match_time", "match wall time (s)"),
                       ("cycle_time", "e2e mini-cycle (s)")):
        p, q = py[key], nat[key]
        ratio = (p / q) if "time" in key else (q / max(p, 1e-9))
        if "time" in key:
            print(f"{label:<24} {p:>14.3f} {q:>14.3f} {ratio:>8.1f}x")
        else:
            print(f"{label:<24} {p:>14,.0f} {q:>14,.0f} {ratio:>8.1f}x")
    print(f"\nbackend: {st['native_backend']}   "
          f"enabled: {st['native_enabled']}")
    if not st["native_enabled"]:
        print(f"fallback reason: {st['native_fallback_reason']}")


if __name__ == "__main__":
    main()
