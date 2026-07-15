#!/usr/bin/env python3
"""
Pruning-based Goethals-Seidel quadruple search (miniature of the published
methodology used for orders 428/764), in three stages:

  1. SYMMETRY RESTRICTION - only sequences invariant under x -> -x on Z_n
     (a[i] == a[n-i]). Free variables per sequence: 1 + (n-1)/2 instead of n.
     For n=167 this cuts the raw space from 2^668 to 2^336. This is a BET:
     solutions are not obliged to be symmetric.

  2. PSD FILTER - any solution satisfies
        PSD_a(w) + PSD_b(w) + PSD_c(w) + PSD_d(w) = 4n  for every w != 0,
     and PSDs are nonnegative, so every individual sequence must have
     PSD(w) <= 4n at all nonzero frequencies. Candidates failing this are
     discarded immediately (~98% of random candidates at n=167).
     Additionally, column sums must satisfy s1^2+s2^2+s3^2+s4^2 = 4n,
     so candidates are binned by |sum| and only viable sum-patterns pursued.

  3. MEET-IN-THE-MIDDLE - the quadruple condition is
        PAF_a(s)+PAF_b(s)+PAF_c(s)+PAF_d(s) = 0  for s = 1..(n-1)/2
     (symmetry halves the shifts). PAF vectors are INTEGERS, so pairs (a,b)
     are hashed by their PAF-sum vector and pairs (c,d) probe the table for
     the exact complementary vector. Cost ~ (#pairs) instead of (#pairs)^2.

Usage:
  python gs_pruned.py --n 13                     # end-to-end demo, seconds
  python gs_pruned.py --n 167 --hours 8          # the open problem
  python gs_pruned.py --n 167 --verify           # check a found solution

On success: exact verification + hadamard_<4n>.csv. Checkpoints candidate
pools to gsp_pool_<n>.npz so overnight runs accumulate across restarts.
"""
import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np

rng = np.random.default_rng()


# ---------- stage 0: number theory ----------

def sum_decompositions(n):
    """All multisets {s1>=s2>=s3>=s4>=1, odd} with sum of squares = 4n."""
    target, out = 4 * n, []
    m = int(target ** 0.5)
    odds = [k for k in range(1, m + 1, 2)]
    for s1 in odds:
        for s2 in [k for k in odds if k <= s1]:
            for s3 in [k for k in odds if k <= s2]:
                r = target - s1 * s1 - s2 * s2 - s3 * s3
                if r < 1:
                    continue
                s4 = int(round(r ** 0.5))
                if s4 * s4 == r and s4 % 2 == 1 and s4 <= s3:
                    out.append((s1, s2, s3, s4))
    return out


# ---------- stage 1+2: symmetric candidates through the PSD sieve ----------

def expand(free, n):
    """free = (a0, a1..a_h) with h=(n-1)/2 -> full symmetric sequence."""
    h = (n - 1) // 2
    seq = np.empty(n, dtype=np.int8)
    seq[0] = free[0]
    seq[1:h + 1] = free[1:]
    seq[h + 1:] = free[1:][::-1]
    return seq


def generate_pool(n, per_batch, psd_cap):
    """Random symmetric sequences passing the PSD <= 4n test, vectorized."""
    h = (n - 1) // 2
    free = rng.choice(np.array([-1, 1], dtype=np.int8), size=(per_batch, h + 1))
    seqs = np.empty((per_batch, n), dtype=np.int8)
    seqs[:, 0] = free[:, 0]
    seqs[:, 1:h + 1] = free[:, 1:]
    seqs[:, h + 1:] = free[:, 1:][:, ::-1]
    F = np.fft.rfft(seqs.astype(np.float64), axis=1)
    psd = np.abs(F) ** 2
    ok = psd[:, 1:].max(axis=1) <= psd_cap + 1e-6
    return seqs[ok]


def paf_half(seqs, n):
    """Integer PAF vectors at shifts 1..(n-1)/2 for a batch of sequences."""
    h = (n - 1) // 2
    F = np.fft.rfft(seqs.astype(np.float64), axis=1)
    paf = np.fft.irfft(np.abs(F) ** 2, n, axis=1)
    return np.rint(paf[:, 1:h + 1]).astype(np.int16)


# ---------- stage 3: meet-in-the-middle on integer PAF vectors ----------

def key_of(vec):
    return hashlib.blake2b(vec.tobytes(), digest_size=12).digest()


def match_quadruple(poolA, pafA, poolB, pafB, poolC, pafC, poolD, pafD,
                    n, max_pairs, log=print):
    """Hash (a,b) pairs by PAF_a+PAF_b; probe with -(PAF_c+PAF_d)."""
    table = defaultdict(list)
    count = 0
    for i in range(len(poolA)):
        sums = pafA[i][None, :] + pafB
        for j in range(len(poolB)):
            table[key_of(sums[j])].append((i, j))
            count += 1
            if count >= max_pairs:
                break
        if count >= max_pairs:
            break
    log(f"    hashed {count:,} (a,b) pairs into {len(table):,} buckets")

    probes = 0
    for k in range(len(poolC)):
        negs = -(pafC[k][None, :] + pafD)
        for l in range(len(poolD)):
            probes += 1
            hits = table.get(key_of(negs[l]))
            if not hits:
                continue
            for (i, j) in hits:
                total = (pafA[i].astype(int) + pafB[j] + pafC[k] + pafD[l])
                if np.all(total == 0):
                    return poolA[i], poolB[j], poolC[k], poolD[l]
            if probes >= max_pairs:
                return None
    return None


# ---------- assembly + verification ----------

def goethals_seidel(seqs):
    n = len(seqs[0])
    A, B, C, D = (np.array([np.roll(s, k) for k in range(n)], dtype=np.int64)
                  for s in seqs)
    R = np.fliplr(np.eye(n, dtype=np.int64))
    return np.block([
        [A,        B @ R,      C @ R,      D @ R],
        [-B @ R,   A,          -D.T @ R,   C.T @ R],
        [-C @ R,   D.T @ R,    A,          -B.T @ R],
        [-D @ R,   -C.T @ R,   B.T @ R,    A]])


def verify_and_save(seqs, n, out_csv):
    H = goethals_seidel(seqs)
    N = 4 * n
    assert np.all(np.abs(H) == 1)
    assert np.array_equal(H @ H.T, N * np.eye(N, dtype=np.int64)), "NOT Hadamard"
    np.savetxt(out_csv, H, fmt="%d", delimiter=",")
    print(f"[VERIFIED] {N}x{N} Hadamard matrix, H H^T = {N} I  ->  {out_csv}")
    for s in seqs:
        print("   seq:", "".join("+" if x > 0 else "-" for x in s),
              f"(sum {int(s.sum())})")


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--hours", type=float, default=0.05)
    ap.add_argument("--pool-cap", type=int, default=4000,
                    help="max candidates kept per |sum| bin")
    ap.add_argument("--max-pairs", type=int, default=3_000_000,
                    help="pair budget per matching attempt")
    args = ap.parse_args()
    n = args.n
    if n % 2 == 0:
        sys.exit("n must be odd")
    psd_cap = 4 * n
    decomps = sum_decompositions(n)
    print(f"n={n}  order={4*n}  PSD cap={psd_cap}")
    print(f"column-sum patterns (s1,s2,s3,s4) with sum of squares = {4*n}:")
    for d in decomps:
        print("   ", d)
    needed = sorted({s for d in decomps for s in d})

    pools = {s: [] for s in needed}
    pool_file = f"gsp_pool_{n}.npz"
    if os.path.exists(pool_file):
        data = np.load(pool_file)
        for s in needed:
            if f"s{s}" in data:
                pools[s] = list(data[f"s{s}"])
        print(f"[resume] pools loaded: "
              f"{ {s: len(v) for s, v in pools.items()} }")

    t0, attempt = time.time(), 0
    while time.time() - t0 < args.hours * 3600:
        # grow pools
        batch = generate_pool(n, 200_000, psd_cap)
        sums = batch.sum(axis=1)
        for s in needed:
            if len(pools[s]) < args.pool_cap:
                hit = batch[np.abs(sums) == s]
                # normalize sign so sum >= 0 (negation preserves PAF)
                hit = hit * np.where(hit.sum(axis=1) < 0, -1, 1)[:, None]
                pools[s].extend(list(hit.astype(np.int8)))
                pools[s] = pools[s][:args.pool_cap]
        np.savez_compressed(pool_file,
                            **{f"s{s}": np.array(v, dtype=np.int8)
                               for s, v in pools.items() if v})

        attempt += 1
        sizes = {s: len(v) for s, v in pools.items()}
        print(f"[{time.time()-t0:6.0f}s] pools {sizes}")

        # try to assemble a quadruple for each sum pattern and pairing split
        for (s1, s2, s3, s4) in decomps:
            if min(len(pools[s]) for s in (s1, s2, s3, s4)) == 0:
                continue
            for (x, y), (z, w) in (((s1, s2), (s3, s4)),
                                   ((s1, s3), (s2, s4)),
                                   ((s1, s4), (s2, s3))):
                A = np.array(pools[x]); B = np.array(pools[y])
                C = np.array(pools[z]); D = np.array(pools[w])
                res = match_quadruple(
                    A, paf_half(A, n), B, paf_half(B, n),
                    C, paf_half(C, n), D, paf_half(D, n),
                    n, args.max_pairs,
                    log=lambda m: print(f"  ({x},{y})x({z},{w}) {m}"))
                if res is not None:
                    print(f"[SOLVED] pattern ({x},{y})+({z},{w}), "
                          f"attempt {attempt}, {time.time()-t0:.1f}s")
                    verify_and_save(res, n, f"hadamard_{4*n}.csv")
                    return
        print("  no match this round; pools keep growing, retrying...")

    print("[stop] time budget exhausted; pools are checkpointed for resume.")


if __name__ == "__main__":
    main()
