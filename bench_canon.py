#!/usr/bin/env python3
"""Canonical-dedup benchmark: merge cost/size and same-bin pair reduction."""
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np


def main():
    import core
    import engine
    import worker as wk
    rng = np.random.default_rng(9)
    WD = "bench_canon_work"
    shutil.rmtree(WD, ignore_errors=True)

    # --- merge: 2 workers x 3 gs bins x 8k rows, 50% planted orbit dups ---
    # (random +-1 rows: canonical-dedup cost is independent of the sieve)
    route = wk.GSRoute(167)
    bins3 = sorted(route.bins)[:3]
    base = {}
    for b in bins3:
        base[wk.bin_key(b)] = rng.choice(
            np.array([-1, 1], np.int8), (8000, 167))
    for widx in (0, 1):
        d = os.path.join(WD, f"worker_{widx:03d}")
        os.makedirs(d, exist_ok=True)
        pools = {}
        for k, arr in base.items():
            half = arr[widx * 4000:(widx + 1) * 4000]
            dup = -np.roll(arr[:2000] if widx else arr[6000:],
                           widx + 1, axis=1)[:, ::-1]
            pools[k] = np.concatenate([half, dup])
        np.savez_compressed(os.path.join(d, "pools.npz"), **pools)
    raw = sum(4000 + 2000 for _ in base) * 2
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "ctl.py", "merge",
                        "--workdir", WD], capture_output=True, text=True)
    dt = time.perf_counter() - t0
    gg = np.load(os.path.join(WD, "global_pools_gs.npz"))
    kept = sum(len(gg[k]) for k in gg.files)
    size = os.path.getsize(os.path.join(WD, "global_pools_gs.npz"))
    print(f"merge (canonical): {raw:,} raw rows -> {kept:,} kept "
          f"({raw - kept:,} orbit dups removed) in {dt:.2f}s "
          f"({raw / dt:,.0f} rows/s)")
    print(f"global_pools_gs.npz: {size / 1024:.0f} KB "
          f"(raw would be ~{size * raw / max(kept, 1) / 1024:.0f} KB)")
    shutil.rmtree(WD, ignore_errors=True)

    # --- same-bin pair reduction ---
    paf = rng.integers(-40, 40, (500, 83)).astype(np.int16)
    for tri in (False, True):
        s = defaultdict(int)
        t0 = time.perf_counter()
        engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s,
                     offs=(0, 0, 0, 0), tri=(tri, tri))
        dtm = time.perf_counter() - t0
        print(f"same-bin key, tri={tri!s:5}: pairs hashed "
              f"{s['pairs_hashed']:>9,}  probes {s['probes']:>9,}  "
              f"swap skipped {s['swap_skipped']:>9,}  wall {dtm:.3f}s")


if __name__ == "__main__":
    main()
