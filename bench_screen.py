#!/usr/bin/env python3
"""K-sweep benchmark for the PSD pair screen on SIEVE-PASSING candidates
(the realistic population -- raw random sequences overstate rejection)."""
import os
import time
from collections import defaultdict

import numpy as np


def main():
    import core
    import worker as wk
    rng = np.random.default_rng(3)
    route = wk.GSRoute(167)
    seqs = []
    while len(seqs) < 500:
        batch = route.generate(3, 200000, rng)
        seqs.extend(list(batch))
    seqs = np.array(seqs[:500], np.int8)
    print(f"fixture: {len(seqs)} sieve-passing symmetric candidates "
          f"(n=167, |sum|=3)")
    print(f"{'K':>3} {'rejected%':>10} {'pairs hashed':>13} "
          f"{'wall s':>8} {'eff pair ops/sec':>17}")
    base = None
    for k in (0, 1, 2, 4, 8, 16):
        os.environ["H668_SCREEN_K"] = str(max(k, 1))
        import importlib
        import engine
        importlib.reload(engine)
        psd = engine.psd_rows_periodic(seqs)
        paf = core.paf_half_batch(seqs)
        s = defaultdict(int)
        t0 = time.perf_counter()
        if k == 0:
            engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s)
        else:
            engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s,
                         psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
        dt = time.perf_counter() - t0
        attempted = s["pairs_hashed"] + s["psd_skipped"] // 2
        rej = 100 * (s["psd_skipped"] / 2) / max(attempted, 1)
        eff = attempted / dt
        if base is None:
            base = dt
        print(f"{k:>3} {rej:>9.1f}% {s['pairs_hashed']:>13,} "
              f"{dt:>8.3f} {eff:>17,.0f}  ({base/dt:.2f}x vs K=0)")
    os.environ.pop("H668_SCREEN_K", None)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""K-sweep benchmark for the PSD pair screen on SIEVE-PASSING candidates
(the realistic population -- raw random sequences overstate rejection)."""
import os
import time
from collections import defaultdict

import numpy as np


def main():
    import core
    import worker as wk
    rng = np.random.default_rng(3)
    route = wk.GSRoute(167)
    seqs = []
    while len(seqs) < 500:
        batch = route.generate(3, 200000, rng)
        seqs.extend(list(batch))
    seqs = np.array(seqs[:500], np.int8)
    print(f"fixture: {len(seqs)} sieve-passing symmetric candidates "
          f"(n=167, |sum|=3)")
    print(f"{'K':>3} {'rejected%':>10} {'pairs hashed':>13} "
          f"{'wall s':>8} {'eff pair ops/sec':>17}")
    base = None
    for k in (0, 1, 2, 4, 8, 16):
        os.environ["H668_SCREEN_K"] = str(max(k, 1))
        import importlib
        import engine
        importlib.reload(engine)
        psd = engine.psd_rows_periodic(seqs)
        paf = core.paf_half_batch(seqs)
        s = defaultdict(int)
        t0 = time.perf_counter()
        if k == 0:
            engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s)
        else:
            engine.match(paf, paf, paf, paf, max_pairs=10**12, stats=s,
                         psd1=(psd, psd, 668.0), psd2=(psd, psd, 668.0))
        dt = time.perf_counter() - t0
        attempted = s["pairs_hashed"] + s["psd_skipped"] // 2
        rej = 100 * (s["psd_skipped"] / 2) / max(attempted, 1)
        eff = attempted / dt
        if base is None:
            base = dt
        print(f"{k:>3} {rej:>9.1f}% {s['pairs_hashed']:>13,} "
              f"{dt:>8.3f} {eff:>17,.0f}  ({base/dt:.2f}x vs K=0)")
    os.environ.pop("H668_SCREEN_K", None)


if __name__ == "__main__":
    main()
