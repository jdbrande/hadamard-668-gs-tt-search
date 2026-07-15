"""Pools and meet-in-the-middle matching, with optional compiled backend.

v3 additions (each covered by tests_opt.py equivalence tests):
  * match() is COMPLETE over its inputs: chunk sizes shrink to respect
    max_pairs instead of silently dropping rows (required for watermarks).
  * optional pair-PSD screen: a pair (a,b) is skipped iff
      PSD_a(w*) + PSD_b(w*) > cap  at w* = argmax_w PSD_a(w).
    Soundness: any exact PAF/NPAF match has sum-of-PSDs equal to the budget
    at EVERY frequency, so both of its pairs pass every per-frequency screen.
    The screen can only discard pairs that cannot be part of any solution.
  * find_all_matches(): slow exhaustive reference matcher used by tests to
    prove the fast paths return the same solution sets.
"""
import ctypes
import hashlib
import os
from collections import defaultdict

import numpy as np

import core

_LIB = None   # kept for tests that monkeypatch engine._LIB


def load_fastmatch():
    """Checked native library, honoring H668_NATIVE (see native.py).
    Tests may set engine._LIB = None to force the python path temporarily."""
    global _LIB
    if _LIB is not None:
        return _LIB
    import native
    _LIB = native.load()
    return _LIB


class Pool:
    def __init__(self, length, periodic):
        self.length = length
        self.periodic = periodic
        self.seqs = np.empty((0, length), dtype=np.int8)
        self.keys = set()

    def add(self, batch, need=None, counters=None):
        """Sign-normalize, dedup by canonical key, append. Returns #new.
        Dedup removes only candidates whose canonical orbit (hence exact
        PAF/NPAF vector) already has a stored representative -- proven
        loss-free by tests_opt.py.
        counters (optional dict) is incremented in place with telemetry:
        "accepted" (appended) and "dup_rejected" (canonical key already
        present). Rows skipped by the need-cap are counted as neither."""
        if len(batch) == 0:
            return 0
        if need is not None:
            batch = batch[: max(0, need) * 2 + 8]
        sums = batch.sum(axis=1)
        batch = batch * np.where(sums < 0, -1, 1)[:, None].astype(np.int8)
        fresh = []
        for row in batch:
            if need is not None and len(fresh) >= need:
                break
            k = hashlib.blake2b(
                core.canonical_key(row, self.periodic), digest_size=16
            ).digest()
            if k not in self.keys:
                self.keys.add(k)
                fresh.append(row)
            elif counters is not None:
                counters["dup_rejected"] = counters.get("dup_rejected", 0) + 1
        if counters is not None:
            counters["accepted"] = counters.get("accepted", 0) + len(fresh)
        if fresh:
            self.seqs = np.concatenate([self.seqs, np.array(fresh, np.int8)])
        return len(fresh)

    def cap(self, limit):
        if len(self.seqs) > limit:
            self.seqs = self.seqs[:limit]

    def state(self):
        return self.seqs


# ---------------------------------------------------------------- PSD rows

def psd_rows_periodic(seqs):
    """Nonzero-frequency PSD rows, float32. Row i pairs with seqs[i]."""
    F = np.fft.rfft(seqs.astype(np.float64), axis=1)
    return (np.abs(F[:, 1:]) ** 2).astype(np.float32)


def psd_rows_padded(seqs, pad):
    """Padded PSD rows (nonperiodic route), DC excluded, float32."""
    F = np.fft.rfft(seqs.astype(np.float64), pad, axis=1)
    return (np.abs(F[:, 1:]) ** 2).astype(np.float32)


SCREEN_K = int(os.environ.get("H668_SCREEN_K", "4"))


def top_freqs(psd, k=None):
    """Per-row indices of the k largest-PSD frequencies (int32, (n,k))."""
    k = k or SCREEN_K
    k = max(1, min(k, psd.shape[1]))
    idx = np.argpartition(-psd, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(psd, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1).astype(np.int32)


def _screen(psdX, wX, psdY, wY, cap):
    """Boolean (nx, ny): True where the pair SURVIVES. Checks each side's
    top-K frequencies; sound because a valid pair satisfies the cap at
    EVERY frequency (see tests_screen.py)."""
    if psdX is None or cap is None or cap <= 0:
        return None
    ok = np.ones((len(psdX), len(psdY)), dtype=bool)
    for t in range(wX.shape[1]):
        w = wX[:, t]
        vals = psdX[np.arange(len(psdX)), w]
        ok &= vals[:, None] + psdY[:, w.astype(np.intp)].T <= cap + 1e-6
    if wY is not None:
        for t in range(wY.shape[1]):
            w = wY[:, t]
            vals = psdY[np.arange(len(psdY)), w]
            ok &= (psdX[:, w.astype(np.intp)]
                   + vals[None, :] <= cap + 1e-6)
    return ok


def _numpy_match(pafA, pafB, pafC, pafD, wt1, wt2, stats,
                 psd1=None, psd2=None, collision_log=None, lo=0,
                 offs=(0, 0, 0, 0), tri=(False, False)):
    a_off, b_off, c_off, d_off = offs
    a_off += lo
    psdA, psdB, cap1 = psd1 if psd1 else (None, None, None)
    psdC, psdD, cap2 = psd2 if psd2 else (None, None, None)
    wA = top_freqs(psdA) if psdA is not None else None
    wB = top_freqs(psdB) if psdB is not None else None
    wC = top_freqs(psdC) if psdC is not None else None
    wD = top_freqs(psdD) if psdD is not None else None
    ok1 = _screen(psdA, wA, psdB, wB, cap1) if psdA is not None else None
    ok2 = _screen(psdC, wC, psdD, wD, cap2) if psdC is not None else None

    table = defaultdict(list)
    for i in range(len(pafA)):
        rows = wt1 * (pafA[i][None, :].astype(np.int32) + pafB)
        for j in range(len(pafB)):
            if tri[0] and (a_off + i) > (b_off + j):
                stats["swap_skipped"] += 1
                continue
            if ok1 is not None and not ok1[i, j]:
                stats["psd_skipped"] += 1
                continue
            table[rows[j].astype(np.int16).tobytes()].append((i, j))
            stats["pairs_hashed"] += 1
    stats["buckets"] += len(table)
    for k in range(len(pafC)):
        rows = -wt2 * (pafC[k][None, :].astype(np.int32) + pafD)
        keys = rows.astype(np.int16)
        for l in range(len(pafD)):
            if tri[1] and (c_off + k) > (d_off + l):
                stats["swap_skipped"] += 1
                continue
            if ok2 is not None and not ok2[k, l]:
                stats["psd_skipped"] += 1
                continue
            stats["probes"] += 1
            hits = table.get(keys[l].tobytes())
            if hits:
                stats["collisions"] += len(hits)
                i, j = hits[0]
                if collision_log is not None:
                    collision_log.append((i + lo, j, k, l))
                return (i, j, k, l)
    return None


def match(pafA, pafB, pafC, pafD, wt1=1, wt2=1,
          max_pairs=20_000_000, chunk=4000, stats=None,
          psd1=None, psd2=None, on_progress=None, collision_log=None,
          offs=(0, 0, 0, 0), tri=(False, False), micro=None):
    """Find (i,j,k,l) with wt1*(pafA[i]+pafB[j]) + wt2*(pafC[k]+pafD[l]) = 0.

    COMPLETE over the given arrays: chunking never drops rows. Optional
    psd1=(psdA,psdB,cap1) / psd2=(psdC,psdD,cap2) enable the sound pair-PSD
    screen. Returns global indices or None.

    on_progress(rows_done, rows_total) is called after each chunk (every
    few seconds of work). If it returns False, matching ABORTS cleanly:
    stats["interrupted"] is set and None is returned. Aborting is safe --
    callers must not advance watermarks for interrupted regions."""
    if stats is None:
        stats = defaultdict(int)
    h = pafA.shape[1]
    lib = load_fastmatch()
    if os.environ.get("H668_PARTITION", "1") != "0" \
            and min(len(pafA), len(pafB), len(pafC), len(pafD)) > 0:
        return _match_partitioned(
            pafA, pafB, pafC, pafD, wt1, wt2, max_pairs, stats,
            psd1, psd2, on_progress, collision_log, offs, tri, lib,
            micro=micro)
    eff = max(1, min(chunk, max_pairs // max(len(pafB), 1)))
    psdA_all = psd1[0] if psd1 else None
    for lo in range(0, len(pafA), eff):
        if on_progress is not None and \
                on_progress(lo, len(pafA)) is False:
            stats["interrupted"] += 1
            return None
        A = np.ascontiguousarray(pafA[lo:lo + eff])
        pA = (psdA_all[lo:lo + eff], psd1[1], psd1[2]) if psd1 else None
        if lib is not None:
            r = _cc_match(lib, A, pafB, pafC, pafD, h, wt1, wt2,
                          pA, psd2, stats, collision_log, lo,
                          offs=offs, tri=tri)
        else:
            r = _numpy_match(A, pafB, pafC, pafD, wt1, wt2, stats,
                             psd1=pA, psd2=psd2,
                             collision_log=collision_log, lo=lo,
                             offs=offs, tri=tri)
        if r is not None:
            return (r[0] + lo, r[1], r[2], r[3])
    if on_progress is not None:
        on_progress(len(pafA), len(pafA))
    return None


def _cc_match(lib, A, pafB, pafC, pafD, h, wt1, wt2, psd1, psd2, stats,
              collision_log=None, lo=0, offs=(0, 0, 0, 0),
              tri=(False, False)):
    out = (ctypes.c_longlong * 4)()
    st = (ctypes.c_longlong * 5)()
    use_v6 = hasattr(lib, "match_pairs_v6")
    use_v5 = hasattr(lib, "match_pairs_v5")
    use_v4 = hasattr(lib, "match_pairs_v4")
    coll_buf = (ctypes.c_longlong * (64 * 4))()
    coll_n = ctypes.c_longlong(0)

    def P16(x):
        return np.ascontiguousarray(x, dtype=np.int16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16))

    def PF(x):
        return (np.ascontiguousarray(x, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)) if x is not None else None)

    def PI(x):
        return (np.ascontiguousarray(x, dtype=np.int32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)) if x is not None else None)

    if psd1:
        psdA, psdB, cap1 = psd1
        wA, wB = top_freqs(psdA), top_freqs(psdB)
        F, K = psdA.shape[1], wA.shape[1]
    else:
        psdA = psdB = wA = wB = None
        cap1, F, K = 0.0, 0, 0
    if psd2:
        psdC, psdD, cap2 = psd2
        wC, wD = top_freqs(psdC), top_freqs(psdD)
        F, K = psdC.shape[1], wC.shape[1]
    else:
        psdC = psdD = wC = wD = None
        cap2 = 0.0
    # keep references alive through the call
    holders = [np.ascontiguousarray(x, dtype=np.float32)
               for x in (psdA, psdB, psdC, psdD) if x is not None]
    if use_v6:
        st6 = (ctypes.c_longlong * 7)()
        found = lib.match_pairs_v6(
            P16(A), len(A), P16(pafB), len(pafB),
            P16(pafC), len(pafC), P16(pafD), len(pafD),
            h, wt1, wt2,
            PF(psdA), PI(wA), PF(psdB), PI(wB),
            PF(psdC), PI(wC), PF(psdD), PI(wD),
            F, K, cap1, cap2,
            offs[0] + lo, offs[1], offs[2], offs[3],
            1 if tri[0] else 0, 1 if tri[1] else 0,
            out, st6, coll_buf, 64, ctypes.byref(coll_n))
        for x in range(5):
            st[x] = st6[x]
        stats["swap_skipped"] += st6[5] + st6[6]
        if collision_log is not None:
            for x in range(coll_n.value):
                collision_log.append(
                    (coll_buf[4 * x] + lo, coll_buf[4 * x + 1],
                     coll_buf[4 * x + 2], coll_buf[4 * x + 3]))
    elif use_v5:
        found = lib.match_pairs_v5(
            P16(A), len(A), P16(pafB), len(pafB),
            P16(pafC), len(pafC), P16(pafD), len(pafD),
            h, wt1, wt2,
            PF(psdA), PI(wA), PF(psdB), PI(wB),
            PF(psdC), PI(wC), PF(psdD), PI(wD),
            F, K, cap1, cap2, out, st, coll_buf, 64,
            ctypes.byref(coll_n))
        if collision_log is not None:
            for x in range(coll_n.value):
                collision_log.append(
                    (coll_buf[4 * x] + lo, coll_buf[4 * x + 1],
                     coll_buf[4 * x + 2], coll_buf[4 * x + 3]))
    elif use_v4:
        found = lib.match_pairs_v4(
            P16(A), len(A), P16(pafB), len(pafB),
            P16(pafC), len(pafC), P16(pafD), len(pafD),
            h, wt1, wt2,
            PF(psdA), PI(wA), PF(psdB),
            PF(psdC), PI(wC), PF(psdD),
            F, cap1, cap2, out, st, coll_buf, 64, ctypes.byref(coll_n))
        if collision_log is not None:
            for x in range(coll_n.value):
                collision_log.append(
                    (coll_buf[4 * x] + lo, coll_buf[4 * x + 1],
                     coll_buf[4 * x + 2], coll_buf[4 * x + 3]))
    else:
        entry = getattr(lib, "match_pairs_v3", None) or lib.match_pairs_v2
        found = entry(
            P16(A), len(A), P16(pafB), len(pafB),
            P16(pafC), len(pafC), P16(pafD), len(pafD),
            h, wt1, wt2,
            PF(psdA), PI(wA), PF(psdB),
            PF(psdC), PI(wC), PF(psdD),
            F, cap1, cap2, out, st)
    del holders
    stats["pairs_hashed"] += st[0]
    stats["buckets"] += st[1]
    stats["probes"] += st[2]
    stats["collisions"] += st[3]
    stats["psd_skipped"] += st[4]
    if found:
        return (out[0], out[1], out[2], out[3])
    return None


def find_all_matches(pafA, pafB, pafC, pafD, wt1=1, wt2=1):
    """Exhaustive reference matcher (slow, tests only): the complete set of
    (i,j,k,l) with wt1*(A_i+B_j) + wt2*(C_k+D_l) == 0."""
    table = defaultdict(list)
    for i in range(len(pafA)):
        for j in range(len(pafB)):
            v = wt1 * (pafA[i].astype(np.int32) + pafB[j])
            table[v.astype(np.int16).tobytes()].append((i, j))
    out = set()
    for k in range(len(pafC)):
        for l in range(len(pafD)):
            v = -wt2 * (pafC[k].astype(np.int32) + pafD[l])
            for (i, j) in table.get(v.astype(np.int16).tobytes(), ()):
                out.add((i, j, k, l))
    return out


def _groups_by_coord(paf):
    """value -> int64 index array, grouped on PAF coordinate 0."""
    v = paf[:, 0].astype(np.int64)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    cuts = np.flatnonzero(np.diff(sv)) + 1
    out = {}
    for chunk_idx in np.split(order, cuts):
        out[int(v[chunk_idx[0]])] = chunk_idx.astype(np.int64)
    return out


def _match_partitioned(pafA, pafB, pafC, pafD, wt1, wt2, max_pairs,
                       stats, psd1, psd2, on_progress, collision_log,
                       offs, tri, lib, micro=None):
    """Partition the pair space by the exact necessary condition
    wt1*(a0+b0) + wt2*(c0+d0) = 0 (coordinate 0 of the combined PAF sum
    must vanish, because the FULL vector must vanish for a match). Every
    possibly-matching quadruple has its build pair and probe pair in the
    same partition, so per-partition matching with one shared table is
    complete -- and the chunk x full-reprobe multiplication of the legacy
    path disappears. Oversized partitions are fragmented on the build
    side only (reprobe confined to that partition). See tests_matchaudit.
    """
    h = pafA.shape[1]
    gA, gB = _groups_by_coord(pafA), _groups_by_coord(pafB)
    gC, gD = _groups_by_coord(pafC), _groups_by_coord(pafD)
    psdA, psdB, cap1 = psd1 if psd1 else (None, None, 0.0)
    psdC, psdD, cap2 = psd2 if psd2 else (None, None, 0.0)
    if psd1:
        wA, wB = top_freqs(psdA), top_freqs(psdB)
        F, K = psdA.shape[1], wA.shape[1]
    else:
        wA = wB = None
        F, K = (psd2[0].shape[1], top_freqs(psd2[0]).shape[1]) \
            if psd2 else (0, 0)
    if psd2:
        wC, wD = top_freqs(psdC), top_freqs(psdD)
        F, K = psdC.shape[1], wC.shape[1]
    else:
        wC = wD = None
    parts = []
    for t1 in sorted({a + b for a in gA for b in gB}):
        num = -wt1 * t1
        if num % wt2:
            continue
        t2 = num // wt2
        bsegs = [(gA[a], gB[t1 - a]) for a in gA if (t1 - a) in gB]
        psegs = [(gC[c], gD[t2 - c]) for c in gC if (t2 - c) in gD]
        if bsegs and psegs:
            parts.append((t1, (bsegs, psegs)))
    done_rows = 0
    total_rows = sum(len(x) for _, p in parts for x, _ in p[0]) or 1
    # micro-checkpointing: unit = (coord-0 partition value, fragment
    # index). Fragmentation is DETERMINISTIC given (groups, frag_pairs),
    # so a unit label identifies exactly one build-subset x partition-
    # probe product. Completed units are skipped only when the micro
    # state's fingerprints/version/ranges validate (see worker.MicroState).
    frag_pairs = max_pairs
    if micro is not None and micro.frag_pairs:
        frag_pairs = min(max_pairs, micro.frag_pairs)
    for t1_label, (bsegs, psegs) in parts:
        if on_progress is not None and \
                on_progress(done_rows, total_rows) is False:
            stats["interrupted"] += 1
            if micro is not None:
                micro.flush()
            return None
        pairs = sum(len(x) * len(y) for x, y in bsegs)
        frags = [bsegs]
        if pairs > frag_pairs:
            frags = []
            cur, cur_pairs = [], 0
            for x, y in bsegs:
                step = max(1, frag_pairs // max(len(y), 1))
                for lo in range(0, len(x), step):
                    seg = (x[lo:lo + step], y)
                    segp = len(seg[0]) * len(y)
                    if cur_pairs + segp > frag_pairs and cur:
                        frags.append(cur)
                        cur, cur_pairs = [], 0
                    cur.append(seg)
                    cur_pairs += segp
            if cur:
                frags.append(cur)
        for fi, fsegs in enumerate(frags):
            unit = f"{t1_label}:{fi}"
            if micro is not None and micro.done(unit):
                stats["micro_skipped"] += 1
                continue
            r = _run_partition(lib, pafA, pafB, pafC, pafD, h, wt1, wt2,
                               psdA, wA, psdB, wB, psdC, wC, psdD, wD,
                               F, K, cap1, cap2, fsegs, psegs, offs,
                               tri, stats, collision_log)
            if r is not None:
                if micro is not None:
                    micro.flush()
                return r
            if micro is not None:
                micro.mark(unit)
        done_rows += sum(len(x) for x, _ in bsegs)
    if micro is not None:
        micro.flush()
    if on_progress is not None:
        on_progress(total_rows, total_rows)
    return None


def _run_partition(lib, pafA, pafB, pafC, pafD, h, wt1, wt2,
                   psdA, wA, psdB, wB, psdC, wC, psdD, wD, F, K,
                   cap1, cap2, bsegs, psegs, offs, tri, stats,
                   collision_log):
    if lib is not None and hasattr(lib, "match_pairs_v7"):
        return _cc_partition(lib, pafA, pafB, pafC, pafD, h, wt1, wt2,
                             psdA, wA, psdB, wB, psdC, wC, psdD, wD,
                             F, K, cap1, cap2, bsegs, psegs, offs, tri,
                             stats, collision_log)
    # python reference: dict table over build segments, probe segments
    table = {}
    a_off, b_off, c_off, d_off = offs
    for xi, yj in bsegs:
        for i in xi:
            row = wt1 * (pafA[i].astype(np.int32) + pafB[yj])
            for pos, j in enumerate(yj):
                if tri[0] and (a_off + i) > (b_off + j):
                    stats["swap_skipped"] += 1
                    continue
                if psdA is not None and _pair_reject(
                        psdA, psdB, wA, wB, i, j, cap1):
                    stats["psd_skipped"] += 1
                    continue
                table.setdefault(
                    row[pos].astype(np.int16).tobytes(), []).append(
                    (int(i), int(j)))
                stats["pairs_hashed"] += 1
    stats["buckets"] += stats["pairs_hashed"] * 0 + len(table)
    for xk, yl in psegs:
        for k in xk:
            row = -wt2 * (pafC[k].astype(np.int32) + pafD[yl])
            keys = row.astype(np.int16)
            for pos, l in enumerate(yl):
                if tri[1] and (c_off + k) > (d_off + l):
                    stats["swap_skipped"] += 1
                    continue
                if psdC is not None and _pair_reject(
                        psdC, psdD, wC, wD, k, l, cap2):
                    stats["psd_skipped"] += 1
                    continue
                stats["probes"] += 1
                hits = table.get(keys[pos].tobytes())
                if hits:
                    stats["collisions"] += len(hits)
                    i, j = hits[0]
                    if collision_log is not None:
                        collision_log.append((i, j, int(k), int(l)))
                    return (i, j, int(k), int(l))
    return None


def _pair_reject(psdX, psdY, wX, wY, i, j, cap):
    for t in range(wX.shape[1]):
        w = int(wX[i, t])
        if psdX[i, w] + psdY[j, w] > cap + 1e-6:
            return True
    for t in range(wY.shape[1]):
        w = int(wY[j, t])
        if psdX[i, w] + psdY[j, w] > cap + 1e-6:
            return True
    return False


def _cc_partition(lib, pafA, pafB, pafC, pafD, h, wt1, wt2,
                  psdA, wA, psdB, wB, psdC, wC, psdD, wD, F, K,
                  cap1, cap2, bsegs, psegs, offs, tri, stats,
                  collision_log):
    def P16(x):
        return np.ascontiguousarray(x, np.int16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16))

    def PF(x):
        return (np.ascontiguousarray(x, np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)) if x is not None
            else ctypes.POINTER(ctypes.c_float)())

    def PI(x):
        return (np.ascontiguousarray(x, np.int32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)) if x is not None
            else ctypes.POINTER(ctypes.c_int32)())

    def PL(x):
        return np.ascontiguousarray(x, np.int64).ctypes.data_as(
            ctypes.POINTER(ctypes.c_longlong))

    def flat(segs, side):
        idx = np.concatenate([s[side] for s in segs]) if segs else \
            np.empty(0, np.int64)
        starts = np.zeros(len(segs) + 1, np.int64)
        for i, s in enumerate(segs):
            starts[i + 1] = starts[i] + len(s[side])
        return idx.astype(np.int64), starts

    biA, bsA = flat(bsegs, 0)
    biB, bsB = flat(bsegs, 1)
    piC, psC = flat(psegs, 0)
    piD, psD = flat(psegs, 1)
    table_pairs = sum(len(x) * len(y) for x, y in bsegs)
    out = (ctypes.c_longlong * 4)()
    st = (ctypes.c_longlong * 7)()
    coll_buf = (ctypes.c_longlong * (64 * 4))()
    coll_n = ctypes.c_longlong(0)
    found = lib.match_pairs_v7(
        P16(pafA), P16(pafB), P16(pafC), P16(pafD), h, wt1, wt2,
        PF(psdA), PI(wA), PF(psdB), PI(wB),
        PF(psdC), PI(wC), PF(psdD), PI(wD),
        F, K, cap1, cap2,
        PL(biA), PL(biB), PL(bsA), PL(bsB), len(bsegs),
        PL(piC), PL(piD), PL(psC), PL(psD), len(psegs),
        offs[0], offs[1], offs[2], offs[3],
        1 if tri[0] else 0, 1 if tri[1] else 0, max(table_pairs, 1),
        out, st, coll_buf, 64, ctypes.byref(coll_n))
    for key, x in zip(("pairs_hashed", "buckets", "probes",
                       "collisions", "psd_skipped"), range(5)):
        stats[key] += st[x]
    stats["swap_skipped"] += st[5] + st[6]
    if collision_log is not None:
        for x in range(coll_n.value):
            collision_log.append((coll_buf[4 * x], coll_buf[4 * x + 1],
                                  coll_buf[4 * x + 2],
                                  coll_buf[4 * x + 3]))
    if found:
        return tuple(int(v) for v in out)
    return None
