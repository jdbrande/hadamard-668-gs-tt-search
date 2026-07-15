"""One search worker. Deterministic seed, private directory, restart-safe.

Hardening (v2):
  * flock on work/worker_XXX/lock -- a second process with the same worker id
    and workdir exits immediately with code 2 and a clear message.
  * checkpoint writes go to <path>.tmp.<pid>.<rand>.npz then os.replace().
  * any unreadable .npz is quarantined to work/bad_npz_backup/ and the worker
    starts that pool fresh instead of crashing.
  * meta.json (wid, route, n, pid) is written for ctl.py status.

Protocol (unchanged):
  work/STOP                     -> all workers exit at next cycle boundary
  work/SOLUTION.json            -> written atomically by the winner
  hadamard_<N>.csv              -> written ONLY after core.verify_hadamard
  work/worker_XX/pools.npz      -> checkpointed candidate pools
  work/global_pools_<route>.npz -> merged pools, absorbed when newer
"""
import argparse
import fcntl
import json
import os
import secrets
import shutil
import time
import zipfile
from collections import defaultdict

import numpy as np

import core
import engine

_LOCK_HANDLE = None  # held open for process lifetime

# Bump whenever matching/dedup/filter semantics or cache formats change:
# stale incremental watermarks AND persisted PAF/PSD caches are then
# discarded; the next cycle does a baseline full pass over existing pools.
OPT_VERSION = 3


def _fingerprint(arr):
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(arr).tobytes(),
                           digest_size=12).hexdigest()


MICRO_VERSION = 1


class MicroState:
    """Sub-key micro-checkpoints for v7 partitioned matching.

    Coverage claim of a completed unit "b{block}|{t1}:{frag}": the exact
    build-fragment x partition-probe product of coord-0 partition t1,
    fragment frag, inside watermark block `block` of `key`, computed over
    the pools identified by the recorded fingerprints and ranges. Units
    are skipped ONLY while every guard holds; any doubt discards (safe:
    re-checking pairs is always sound; skipping unproven ranges never is).

    Guards (any mismatch discards the affected key's units):
      * micro_version + opt_version (global)
      * frag_pairs used for deterministic fragmentation (global)
      * per key: bins, the exact block ranges, and EXACT per-bin
        (length, prefix-hash) fingerprints -- growth discards too, since
        partition composition depends on every row in range.
    Normal matchstate.json is untouched: when a key completes, its
    watermark is written exactly as before and the key's micro entries
    are cleared (promotion = the superset watermark takes over).
    """

    def __init__(self, path, target_secs=None):
        self.path = path
        self.target_secs = target_secs if target_secs is not None else \
            float(os.environ.get("H668_MICRO_TARGET_SECS", "300"))
        self.flush_secs = max(15.0, min(60.0, self.target_secs / 5))
        self.frag_pairs = int(os.environ.get(
            "H668_MICRO_MAX_PAIRS",
            str(int(self.target_secs
                    * float(os.environ.get("H668_MICRO_RATE",
                                           "12e6"))))))
        self.data = {"micro_version": MICRO_VERSION,
                     "opt_version": OPT_VERSION,
                     "frag_pairs": self.frag_pairs, "keys": {}}
        self._dirty = False
        self._last_flush = time.time()
        self.units_done_session = 0
        self.last_ckpt = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            d = json.load(open(self.path))
        except (OSError, ValueError):
            try:
                os.replace(self.path, self.path + ".corrupt")
            except OSError:
                pass
            print("[micro] unreadable micro_matchstate; quarantined, "
                  "starting fresh (safe: only re-checks)")
            return
        if d.get("micro_version") != MICRO_VERSION or \
                d.get("opt_version") != OPT_VERSION or \
                d.get("frag_pairs") != self.frag_pairs:
            print(f"[micro] version/params mismatch "
                  f"(micro {d.get('micro_version')}/{MICRO_VERSION}, "
                  f"opt {d.get('opt_version')}/{OPT_VERSION}, frag "
                  f"{d.get('frag_pairs')}/{self.frag_pairs}); "
                  f"discarding micro state (full re-check of open keys)")
            return
        self.data["keys"] = d.get("keys", {})

    def key_view(self, key, bins, ranges, pools, bin_key_fn):
        """Validate-or-reset this key's entry against the CURRENT pools
        and block ranges; return a per-block view factory."""
        want_bins = [str(b) for b in bins]
        want_fp = {}
        for b in bins:
            bk = bin_key_fn(b)
            want_fp[str(b)] = {"len": len(pools[b].seqs),
                               "hash": _fingerprint(pools[b].seqs)}
        ent = self.data["keys"].get(key)
        reason = None
        if ent is not None:
            if ent.get("bins") != want_bins:
                reason = "bins changed"
            elif ent.get("ranges") != [list(r) for r in ranges]:
                reason = "block ranges changed (growth/watermark moved)"
            else:
                for b, fp in ent.get("fp", {}).items():
                    got = want_fp.get(b)
                    if got is None or got["len"] != fp["len"] or \
                            got["hash"] != fp["hash"]:
                        reason = f"fingerprint mismatch on bin {b}"
                        break
        if ent is None or reason is not None:
            if ent is not None:
                print(f"[micro] {key}: discarding micro units "
                      f"({reason}); safe re-check")
            ent = {"bins": want_bins,
                   "ranges": [list(r) for r in ranges],
                   "fp": want_fp, "units": {}, "updated": time.time()}
            self.data["keys"][key] = ent
            self._dirty = True
        return ent

    def block_view(self, key, block_idx):
        ms = self

        class _View:
            frag_pairs = self.frag_pairs

            def done(_s, unit):
                return bool(ms.data["keys"][key]["units"].get(
                    f"b{block_idx}|{unit}", {}).get("done"))

            def mark(_s, unit):
                ms.data["keys"][key]["units"][f"b{block_idx}|{unit}"] = \
                    {"done": True, "ts": round(time.time(), 1)}
                ms.data["keys"][key]["updated"] = time.time()
                ms.units_done_session += 1
                ms._dirty = True
                if time.time() - ms._last_flush >= ms.flush_secs:
                    ms.flush()

            def flush(_s):
                ms.flush()
        return _View()

    def clear_key(self, key):
        if key in self.data["keys"]:
            del self.data["keys"][key]
            self._dirty = True
            self.flush()

    def flush(self):
        if not self._dirty:
            return
        write_json_atomic(self.path, self.data)
        self._dirty = False
        self._last_flush = time.time()
        self.last_ckpt = self._last_flush


class MatchState:
    """Incremental-matching watermarks, persisted per worker.

    marks[key][bin] = how many candidates of that bin were already fully
    cross-checked for that (pattern, split). Validity is guarded two ways:
      * opt_version mismatch  -> discard everything (full rescan)
      * pool fingerprint mismatch (pool prefix changed since the marks were
        recorded, e.g. manual edits or old-format pools) -> discard everything
    Discarding watermarks is always safe: it only re-checks pairs."""

    def __init__(self, path):
        self.path = path
        self.marks = {}
        self.fps = {}

    def load(self, pools, bin_key_fn, log=print):
        if not os.path.exists(self.path):
            return
        try:
            data = json.load(open(self.path))
        except (OSError, ValueError):
            log("[matchstate] unreadable; full rescan")
            return
        if data.get("opt_version") != OPT_VERSION:
            log(f"[matchstate] opt_version {data.get('opt_version')} != "
                f"{OPT_VERSION}; discarding watermarks (full rescan)")
            return
        for bk, fp in data.get("fps", {}).items():
            pool = next((p for b, p in pools.items()
                         if bin_key_fn(b) == bk), None)
            if pool is None or len(pool.seqs) < fp["len"] or \
                    _fingerprint(pool.seqs[:fp["len"]]) != fp["hash"]:
                log(f"[matchstate] fingerprint mismatch on {bk}; "
                    f"discarding watermarks (full rescan)")
                return
        self.marks = data.get("marks", {})
        self.fps = data.get("fps", {})

    def get(self, key, bins):
        return {b: int(self.marks.get(key, {}).get(b, 0)) for b in bins}

    def update(self, key, counts, pools, bin_key_fn):
        self.marks[key] = {b: int(c) for b, c in counts.items()}
        for b, p in pools.items():
            bk = bin_key_fn(b)
            self.fps[bk] = {"len": int(len(p.seqs)),
                            "hash": _fingerprint(p.seqs)}

    def save(self):
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"opt_version": OPT_VERSION, "marks": self.marks,
                       "fps": self.fps}, fh)
        os.replace(tmp, self.path)


def ensure_caches(route, pools, caches):
    """Keep per-bin PAF/PSD rows aligned with (append-only) pools.
    caches[b] = [rows_covered, paf_array, psd_array]."""
    for b, p in pools.items():
        n = len(p.seqs)
        if b not in caches:
            if n:
                caches[b] = [n, route.paf(p.seqs), route.psd(p.seqs)]
            continue
        m, paf, psd = caches[b]
        if m > n:            # pool shrank/changed: rebuild from scratch
            caches[b] = [n, route.paf(p.seqs), route.psd(p.seqs)]
        elif m < n:          # extend for new rows only
            new = p.seqs[m:]
            caches[b] = [n, np.concatenate([paf, route.paf(new)]),
                         np.concatenate([psd, route.psd(new)])]
    return caches


def save_caches(mydir, route, pools, caches):
    arrays, info = {}, {"opt_version": OPT_VERSION, "bins": {}}
    for b, (m, paf, psd) in caches.items():
        bk = bin_key(b)
        arrays[f"paf_{bk}"] = paf
        arrays[f"psd_{bk}"] = psd
        info["bins"][bk] = {"rows": int(m),
                            "hash": _fingerprint(pools[b].seqs[:m])}
    save_npz(os.path.join(mydir, "caches.npz"), arrays)
    tmp = os.path.join(mydir, f"cacheinfo.json.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(info, fh)
    os.replace(tmp, os.path.join(mydir, "cacheinfo.json"))


def load_caches(mydir, route, pools, workdir, log=print):
    """Load persisted PAF/PSD caches if (and only if) the version stamp and
    per-bin pool-prefix fingerprints check out. Anything stale is ignored
    and rebuilt -- rebuilding is always safe."""
    caches = {}
    info_path = os.path.join(mydir, "cacheinfo.json")
    if not os.path.exists(info_path):
        return caches
    try:
        info = json.load(open(info_path))
    except (OSError, ValueError):
        log("[caches] unreadable cacheinfo; rebuilding caches")
        return caches
    if info.get("opt_version") != OPT_VERSION:
        log(f"[caches] cache version {info.get('opt_version')} != "
            f"{OPT_VERSION}; invalidating and rebuilding")
        return caches
    data = safe_load_npz(os.path.join(mydir, "caches.npz"), workdir)
    if data is None:
        return caches
    for b, p in pools.items():
        bk = bin_key(b)
        meta = info.get("bins", {}).get(bk)
        if (meta is None or f"paf_{bk}" not in data
                or meta["rows"] > len(p.seqs)
                or _fingerprint(p.seqs[:meta["rows"]]) != meta["hash"]):
            continue
        caches[b] = [meta["rows"],
                     data[f"paf_{bk}"].astype(np.int16),
                     data[f"psd_{bk}"].astype(np.float32)]
    if caches:
        log(f"[caches] reused persisted PAF/PSD caches for "
            f"{len(caches)} bin(s)")
    return caches


def write_migration_index(mydir, route, pools, raw_counts, log=print):
    """Versioned migration record. Existing pools are REUSED as candidates;
    checked-pair history is never imported from it (watermarks live in
    matchstate.json and are guarded separately by version + fingerprints)."""
    path = os.path.join(mydir, "migration.json")
    old = None
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except (OSError, ValueError):
            old = None
    if old is not None and old.get("opt_version") == OPT_VERSION:
        return old  # already migrated under this version
    info = {"opt_version": OPT_VERSION, "route": route.name,
            "migrated_at": time.time(),
            "raw_rows_loaded": {str(k): int(v)
                                for k, v in raw_counts.items()},
            "stored_after_dedup": {str(b): int(len(p.seqs))
                                   for b, p in pools.items()},
            "checked_pair_history_imported": False,
            "note": ("pools reused as candidates; baseline full pass will "
                     "run because no trusted checked-pair record exists "
                     "for this opt_version")}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(info, fh, indent=1)
    os.replace(tmp, path)
    total_raw = sum(raw_counts.values())
    total_kept = sum(len(p.seqs) for p in pools.values())
    log(f"[migrate] reused {total_raw} raw candidate rows -> {total_kept} "
        f"after dedup; index written (opt_version {OPT_VERSION}); "
        f"checked-pair history NOT imported -> baseline pass")
    return info


def incremental_match(route, pools, state, max_pairs, chunk, stats,
                      caches=None, heartbeat=None, should_stop=None,
                      collision_sink=None, micro_state=None):
    """Check exactly the quadruples not covered by the watermarks: with old
    region a0 x b0 x c0 x d0 already checked, the four disjoint blocks
      A[a0:] x B x C x D
      A[:a0] x B[b0:] x C x D
      A[:a0] x B[:b0] x C[c0:] x D
      A[:a0] x B[:b0] x C[:c0] x D[d0:]
    exactly cover the complement (proven equivalent to a full rescan by
    tests_opt.py). Returns (pattern, quad) or None; advances watermarks."""
    caches = ensure_caches(route, pools, caches if caches is not None
                           else {})

    def paf_of(b):
        return caches[b][1]

    def psd_of(b):
        return caches[b][2]

    n_keys = sum(len(route.splits(p)) for p in route.patterns)
    key_i = 0
    t_start = time.time()
    pairs_done_prior = 0
    for pi, pat in enumerate(route.patterns):
        for si, ((b1, b2), (b3, b4)) in enumerate(route.splits(pat)):
            key_i += 1
            bins = (b1, b2, b3, b4)
            if min(len(pools[b].seqs) for b in bins) == 0:
                continue
            key = f"{route.name}|{pi}|{si}"
            old = state.get(key, [str(b) for b in bins])
            o = [old[str(b)] for b in bins]
            L = [len(pools[b].seqs) for b in bins]
            stats["baseline_keys" if max(o) == 0 else
                  "incremental_keys"] += 1
            pafs = [paf_of(b) for b in bins]
            psds = [psd_of(b) for b in bins]
            cap1, cap2 = route.pair_caps
            blocks = [
                ((o[0], L[0]), (0, L[1]), (0, L[2]), (0, L[3])),
                ((0, o[0]), (o[1], L[1]), (0, L[2]), (0, L[3])),
                ((0, o[0]), (0, o[1]), (o[2], L[2]), (0, L[3])),
                ((0, o[0]), (0, o[1]), (0, o[2]), (o[3], L[3])),
            ]
            planned = sum((ah - al) * (bh - bl)
                          for (al, ah), (bl, bh), _, _ in blocks
                          if ah > al and bh > bl)
            key_micro = None
            if micro_state is not None and \
                    os.environ.get("H668_MICRO", "1") != "0":
                key_micro = micro_state.key_view(
                    key, bins, [sum(blocks[x], ()) for x in range(4)],
                    pools, bin_key)
            for bi, ((a_lo, a_hi), (b_lo, b_hi), (c_lo, c_hi),
                     (d_lo, d_hi)) in enumerate(blocks):
                if a_lo >= a_hi or b_lo >= b_hi or c_lo >= c_hi \
                        or d_lo >= d_hi:
                    continue

                def on_progress(rows, total, _b=(b_lo, b_hi),
                                _bi=bi):
                    if should_stop is not None and should_stop():
                        return False
                    if heartbeat is not None:
                        done = (stats["pairs_hashed"]
                                + stats["psd_skipped"] // 2)
                        rate = max(done / max(
                            time.time() - t_start, 1e-9), 1.0)
                        remaining = max(planned - (done
                                                   - pairs_done_prior), 0)
                        heartbeat(
                            phase=f"matching key {key_i}/{n_keys} "
                                  f"block {_bi + 1}/4",
                            extra={
                                "pattern": [int(x) for x in pat],
                                "bins": [str(b) for b in bins],
                                "key_pairs_planned": int(planned),
                                "pairs_hashed": int(
                                    stats["pairs_hashed"]),
                                "probes": int(stats["probes"]),
                                "probe_psd_rej": int(
                                    stats["psd_skipped"]),
                                "psd_rejected": int(
                                    stats["psd_skipped"]),
                                "generated": int(stats["generated"]),
                                "accepted": int(stats["accepted"]),
                                "dup_rejected": int(
                                    stats["dup_rejected"]),
                                "eta_key_seconds": int(
                                    remaining / rate),
                                "micro_active": micro_state is not None,
                                "micro_units_done": int(getattr(
                                    micro_state, "units_done_session",
                                    0)) if micro_state else 0,
                                "micro_skipped": int(
                                    stats.get("micro_skipped", 0)),
                                "micro_last_ckpt": getattr(
                                    micro_state, "last_ckpt", None)
                                if micro_state else None,
                                "micro_target_secs": getattr(
                                    micro_state, "target_secs", None)
                                if micro_state else None,
                            })
                    return True

                clog = [] if collision_sink is not None else None
                mview = (micro_state.block_view(key, bi)
                         if key_micro is not None else None)
                hit = engine.match(
                    pafs[0][a_lo:a_hi], pafs[1][b_lo:b_hi],
                    pafs[2][c_lo:c_hi], pafs[3][d_lo:d_hi],
                    wt1=route.wt[0], wt2=route.wt[1],
                    max_pairs=max_pairs, chunk=chunk, stats=stats,
                    psd1=(psds[0][a_lo:a_hi], psds[1][b_lo:b_hi], cap1),
                    psd2=(psds[2][c_lo:c_hi], psds[3][d_lo:d_hi], cap2),
                    on_progress=on_progress, collision_log=clog,
                    offs=(a_lo, b_lo, c_lo, d_lo),
                    tri=(b1 == b2, b3 == b4), micro=mview)
                if clog:
                    for (ci, cj, ck, cl) in clog:
                        gi = (a_lo + ci, b_lo + cj, c_lo + ck, d_lo + cl)
                        quad = [pools[b1].seqs[gi[0]],
                                pools[b2].seqs[gi[1]],
                                pools[b3].seqs[gi[2]],
                                pools[b4].seqs[gi[3]]]
                        collision_sink(pat, [str(b) for b in bins],
                                       gi, quad, route)
                if stats.get("interrupted"):
                    # do NOT advance this key's watermark; completed keys
                    # were already persisted below. Resume redoes only
                    # the interrupted key.
                    state.save()
                    return None
                if hit is not None:
                    i, j, k, l = hit
                    quad = [pools[b1].seqs[a_lo + i],
                            pools[b2].seqs[b_lo + j],
                            pools[b3].seqs[c_lo + k],
                            pools[b4].seqs[d_lo + l]]
                    return pat, quad
            pairs_done_prior = (stats["pairs_hashed"]
                                + stats["psd_skipped"] // 2)
            state.update(key, {str(b): n for b, n in zip(bins, L)},
                         pools, bin_key)
            state.save()   # persist per completed key: multi-hour cycles
                           # now checkpoint incrementally
            if micro_state is not None:
                micro_state.clear_key(key)   # promoted: watermark now
                                             # covers a superset
    return None


def acquire_worker_lock(worker_dir):
    """Exclusive non-blocking flock. True if acquired. Lock file records PID."""
    global _LOCK_HANDLE
    os.makedirs(worker_dir, exist_ok=True)
    fh = open(os.path.join(worker_dir, "lock"), "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fh.seek(0)
    fh.truncate()
    fh.write(str(os.getpid()))
    fh.flush()
    _LOCK_HANDLE = fh  # flock releases automatically on process exit
    return True


def quarantine(path, workdir, reason):
    bad_dir = os.path.join(workdir, "bad_npz_backup")
    os.makedirs(bad_dir, exist_ok=True)
    dest = os.path.join(
        bad_dir, f"{os.path.basename(path)}.{int(time.time())}.{os.getpid()}")
    try:
        shutil.move(path, dest)
        print(f"[quarantine] {path} -> {dest} ({reason})")
    except OSError as e:
        print(f"[quarantine] could not move {path}: {e}")
    return dest


def safe_load_npz(path, workdir):
    """Load an .npz; on any corruption, quarantine and return None."""
    if not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, KeyError) as e:
        quarantine(path, workdir, f"unreadable: {type(e).__name__}: {e}")
        return None


def write_json_atomic(path, data):
    """Atomic JSON write: unique temp per process, fsync, rename."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # fsync not available on some filesystems; rename is atomic
    os.replace(tmp, path)


def save_npz(path, arrays):
    """Atomic checkpoint: unique temp name per process, then rename."""
    tmp = f"{path}.tmp.{os.getpid()}.{secrets.token_hex(4)}.npz"
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class GSRoute:
    """Symmetric Goethals-Seidel, length n. Weights (1,1)."""
    name = "gs"

    def __init__(self, n):
        self.n = n
        self.h = (n - 1) // 2
        self.patterns = core.gs_sum_patterns(n)
        self.bins = sorted({s for p in self.patterns for s in p})
        self.lengths = {s: n for s in self.bins}
        self.periodic = True
        self.cap = 4 * n
        self.wt = (1, 1)
        self.pair_caps = (4 * n, 4 * n)

    def generate(self, s, count, rng, counters=None):
        batch = core.symmetric_with_sum(count, self.n, s, rng)
        keep = core.psd_pass_periodic(batch, self.cap)
        if counters is not None:
            counters["generated"] = counters.get("generated", 0) + len(batch)
            counters["build_psd_rej"] = (counters.get("build_psd_rej", 0)
                                         + int(len(batch) - keep.sum()))
        return batch[keep]

    def paf(self, seqs):
        return core.paf_half_batch(seqs)

    def psd(self, seqs):
        return engine.psd_rows_periodic(seqs)

    def splits(self, pat):
        s1, s2, s3, s4 = pat
        return [((s1, s2), (s3, s4)), ((s1, s3), (s2, s4)),
                ((s1, s4), (s2, s3))]

    def assemble(self, quad):
        return core.gs_build(quad)


class TTRoute:
    """Turyn-type TT(m): X,Y length m (weight 1); Z length m, W length m-1
    (weight 2). Hadamard order 4(3m-1)."""
    name = "tt"

    def __init__(self, m):
        self.m = m
        self.h = m - 1
        self.patterns = core.tt_sum_patterns(m)
        self.periodic = False
        self.pad = 1
        while self.pad < 2 * m:
            self.pad *= 2
        self.wt = (1, 2)
        self.pair_caps = (2 * (3 * m - 1), 3 * m - 1)
        self.bins = sorted({("xy", p[0]) for p in self.patterns}
                           | {("xy", p[1]) for p in self.patterns}
                           | {("z", p[2]) for p in self.patterns}
                           | {("w", p[3]) for p in self.patterns})
        self.lengths = {b: (m - 1 if b[0] == "w" else m) for b in self.bins}

    def generate(self, b, count, rng, counters=None):
        role, s = b
        L = self.lengths[b]
        batch = core.random_with_sum(count, L, s, rng)
        cap = 2 * (3 * self.m - 1) if role == "xy" else (3 * self.m - 1)
        keep = core.psd_pass_padded(batch, cap, self.pad)
        if counters is not None:
            counters["generated"] = counters.get("generated", 0) + len(batch)
            counters["build_psd_rej"] = (counters.get("build_psd_rej", 0)
                                         + int(len(batch) - keep.sum()))
        return batch[keep]

    def paf(self, seqs):
        return core.npaf_batch(seqs, self.h)

    def psd(self, seqs):
        return engine.psd_rows_padded(seqs, self.pad)

    def splits(self, pat):
        sx, sy, sz, sw = pat
        return [((("xy", sx), ("xy", sy)), (("z", sz), ("w", sw)))]

    def assemble(self, quad):
        return core.tt_build(*quad)


class Telemetry:
    """Cumulative + per-cycle counters, persisted to progress.json.

    Resume rules (spec F): totals from an existing progress.json are carried
    forward, never reset. If no progress.json exists but pools do, history
    is unknown -> counters start at 0 with partial_telemetry=true and
    telemetry_started_at=now. Totals are never invented."""

    KEYS = ("generated", "accepted", "dup_rejected", "pairs_hashed",
            "buckets", "probes", "build_psd_rej", "probe_psd_rej",
            "psd_rejected", "collisions")

    def __init__(self, mydir, wid, route, pools_preexisted, pool_cap=0,
                 seed=None):
        self.seed = seed
        self.path = os.path.join(mydir, "progress.json")
        self.history_path = os.path.join(mydir, "cycles.jsonl")
        self.pool_cap = pool_cap
        self.wid = wid
        self.route = route
        self.totals = {k: 0 for k in self.KEYS}
        self.cycle = 0
        self.started = time.time()
        self.partial = False
        self.telemetry_started_at = time.time()
        self.recent_cycle_seconds = []
        prev = None
        if os.path.exists(self.path):
            try:
                prev = json.load(open(self.path))
            except (OSError, ValueError):
                prev = None
        if prev is not None:
            for k in self.KEYS:
                self.totals[k] = int(prev.get("totals", {}).get(k, 0))
            self.cycle = int(prev.get("cycle", 0))
            self.partial = bool(prev.get("partial_telemetry", False))
            self.telemetry_started_at = prev.get("telemetry_started_at",
                                                 self.telemetry_started_at)
            self.recent_cycle_seconds = list(
                prev.get("recent_cycle_seconds", []))[-12:]
        elif pools_preexisted:
            self.partial = True   # pools predate telemetry; history unknown

    def record_cycle(self, cycle_stats, secs, pools, status):
        self.cycle += 1
        cyc = {k: int(cycle_stats.get(k, 0)) for k in
               ("generated", "accepted", "dup_rejected", "pairs_hashed",
                "buckets", "probes", "build_psd_rej", "psd_skipped",
                "collisions")}
        cyc["probe_psd_rej"] = cyc.pop("psd_skipped")
        cyc["psd_rejected"] = cyc["build_psd_rej"] + cyc["probe_psd_rej"]
        for k in self.KEYS:
            self.totals[k] += cyc.get(k, 0)
        s = max(secs, 1e-9)
        cyc["rate_pairs_per_sec"] = int(cyc["pairs_hashed"] / s)
        cyc["rate_probes_per_sec"] = int(cyc["probes"] / s)
        cyc["rate_psd_rej_per_sec"] = int(cyc["psd_rejected"] / s)
        self.recent_cycle_seconds = (self.recent_cycle_seconds
                                     + [round(secs, 1)])[-12:]
        self.write(cyc, secs, pools, status)
        # append-only per-cycle history for `ctl.py acceptance`
        rec = {"t": round(time.time(), 1), "cycle": self.cycle,
               "secs": round(secs, 1), "status": status,
               "pools": {str(b): len(p.seqs) for b, p in pools.items()},
               "pool_cap": self.pool_cap, **cyc}
        try:
            with open(self.history_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def heartbeat(self, pools, phase, extra=None, min_interval=30.0):
        """Mid-cycle progress.json update (throttled). Adds phase telemetry
        so multi-hour cycles are observable: current phase, current key/bin,
        in-cycle counters, pair progress and ETA. Never touches totals."""
        now = time.time()
        if now - getattr(self, "_last_hb", 0) < min_interval:
            return
        self._last_hb = now
        latest = dict(extra or {})
        latest["phase"] = phase
        self.write(latest, now - getattr(self, "_cycle_t0", now), pools,
                   "matching" if phase.startswith("matching")
                   else "running")

    def write(self, latest, secs, pools, status):
        import native
        nstat = native.status(with_speedup=(self.cycle <= 1))
        write_json_atomic(self.path, {
            "opt_version": OPT_VERSION,
            **nstat,
            "worker_id": self.wid,
            "route": self.route.name,
            "n": getattr(self.route, "n", getattr(self.route, "m", None)),
            "pid": os.getpid(),
            "seed": self.seed,
            "started": self.started,
            "updated": time.time(),
            "cycle": self.cycle,
            "cycle_seconds": round(secs, 1),
            "totals": dict(self.totals),
            "latest_cycle": latest,
            "pools": {str(b): len(p.seqs) for b, p in pools.items()},
            "pool_cap": self.pool_cap,
            "cap_pressure": {str(b): round(len(p.seqs)
                                           / max(self.pool_cap, 1), 4)
                             for b, p in pools.items()},
            "bins_at_cap": sum(1 for p in pools.values()
                               if self.pool_cap
                               and len(p.seqs) >= self.pool_cap),
            "recent_cycle_seconds": self.recent_cycle_seconds,
            "partial_telemetry": self.partial,
            "telemetry_started_at": self.telemetry_started_at,
            "status": status,
        })


def bin_key(b):
    return f"b{b}" if not isinstance(b, tuple) else f"b{b[0]}_{b[1]}"


def run_worker(route, wid, base_seed, workdir, pool_cap, batch,
               max_pairs, chunk, cycle_seconds, heartbeat_seconds=30.0):
    mydir = os.path.join(workdir, f"worker_{wid:03d}")
    os.makedirs(workdir, exist_ok=True)

    if not acquire_worker_lock(mydir):
        print(f"[w{wid}] REFUSING TO START: another process already holds "
              f"{mydir}/lock (same workdir + worker id). "
              f"Inspect with: python3 ctl.py status --workdir {workdir}")
        return 2

    with open(os.path.join(mydir, "meta.json"), "w") as fh:
        json.dump({"worker_id": wid, "route": route.name,
                   "n": getattr(route, "n", getattr(route, "m", None)),
                   "pid": os.getpid(), "seed": base_seed,
                   "started": time.time()}, fh)

    rng = np.random.default_rng(base_seed + 1000 * wid)
    pool_path = os.path.join(mydir, "pools.npz")
    stop_path = os.path.join(workdir, "STOP")
    glob_path = os.path.join(workdir, f"global_pools_{route.name}.npz")
    glob_seen = 0.0

    pools = {b: engine.Pool(route.lengths[b], route.periodic)
             for b in route.bins}
    raw_counts = {}
    data = safe_load_npz(pool_path, workdir)
    if data is not None:
        for b in route.bins:
            if bin_key(b) in data:
                arr = data[bin_key(b)].astype(np.int8)
                raw_counts[bin_key(b)] = len(arr)
                pools[b].add(arr)
        print(f"[w{wid}] resumed pools:",
              {str(b): len(p.seqs) for b, p in pools.items()})
    else:
        print(f"[w{wid}] no usable checkpoint; starting fresh")

    write_migration_index(mydir, route, pools, raw_counts)
    caches = load_caches(mydir, route, pools, workdir)
    telemetry = Telemetry(mydir, wid, route,
                          pools_preexisted=data is not None,
                          pool_cap=pool_cap, seed=base_seed)
    telemetry.write({}, 0.0, pools, "running")

    state = MatchState(os.path.join(mydir, "matchstate.json"))
    state.load(pools, bin_key)

    log = open(os.path.join(mydir, "log.txt"), "a", buffering=1)

    def out(msg):
        line = f"[w{wid} {time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        log.write(line + "\n")

    coll_path = os.path.join(mydir, "collisions.jsonl")

    def record_collision(pat, bins_s, gi, quad, rt):
        import traceback
        residual = None
        verified = False
        err = None
        try:
            pafs = [rt.paf(np.asarray(q, np.int8)[None, :])[0].astype(int)
                    for q in quad]
            residual = (rt.wt[0] * (pafs[0] + pafs[1])
                        + rt.wt[1] * (pafs[2] + pafs[3])).tolist()
            if not any(residual):
                H = rt.assemble(quad)
                N = core.verify_hadamard(H)
                np.savetxt(os.path.abspath(f"hadamard_{N}_collision.csv"),
                           H, fmt="%d", delimiter=",")
                verified = True
        except Exception:
            err = traceback.format_exc()
        rec = {"timestamp": time.time(), "worker_id": wid,
               "route": rt.name, "seed": base_seed,
               "cycle": telemetry.cycle + 1,
               "pattern": [int(x) for x in pat], "bins": bins_s,
               "indices": [int(x) for x in gi],
               "sequences": ["".join("+" if v > 0 else "-" for v in q)
                             for q in quad],
               "paf_residual": residual,
               "residual_zero": bool(residual is not None
                                     and not any(residual)),
               "verified_hadamard": verified,
               "verify_exception": err}
        try:
            with open(coll_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        out(f"COLLISION EVIDENCE saved: bins {bins_s} idx {gi} "
            f"residual_zero={rec['residual_zero']} verified={verified}")

    micro = MicroState(os.path.join(mydir, "micro_matchstate.json"))

    my_stop = os.path.join(mydir, "STOP_WORKER")

    def should_stop():
        return os.path.exists(stop_path) or os.path.exists(my_stop)

    cycle = 0
    while not should_stop():
        cycle += 1
        t0 = time.time()
        telemetry._cycle_t0 = t0
        telemetry.heartbeat(pools, "generating", min_interval=0)

        stats = defaultdict(int)
        for b in route.bins:
            if len(pools[b].seqs) < pool_cap:
                pools[b].add(route.generate(b, batch, rng, counters=stats),
                             need=pool_cap - len(pools[b].seqs),
                             counters=stats)

        try:
            if (os.path.exists(glob_path)
                    and os.path.getmtime(glob_path) > glob_seen):
                glob_seen = os.path.getmtime(glob_path)
                gdata = safe_load_npz(glob_path, workdir)
                if gdata is not None:
                    for b in route.bins:
                        if bin_key(b) in gdata:
                            pools[b].add(
                                gdata[bin_key(b)].astype(np.int8),
                                need=pool_cap * 4 - len(pools[b].seqs))
                    out("absorbed merged global pools")
        except OSError:
            pass  # merge file replaced mid-check; retry next cycle

        save_npz(pool_path, {bin_key(b): p.seqs for b, p in pools.items()})

        found = incremental_match(
            route, pools, state, max_pairs, chunk, stats, caches=caches,
            micro_state=micro,
            heartbeat=lambda phase, extra=None: telemetry.heartbeat(
                pools, phase, extra, min_interval=heartbeat_seconds),
            should_stop=should_stop, collision_sink=record_collision)
        save_caches(mydir, route, pools, caches)
        if found is not None:
            pat, quad = found
            out(f"MATCH on pattern {pat}; assembling + verifying...")
            H = route.assemble(quad)
            N = core.verify_hadamard(H)  # raises unless exactly Hadamard
            csv = os.path.abspath(f"hadamard_{N}.csv")
            np.savetxt(csv, H, fmt="%d", delimiter=",")
            sol = {"order": int(N), "route": route.name, "csv": csv,
                   "pattern": [int(x) for x in pat], "worker": wid,
                   "sequences": [[int(v) for v in s] for s in quad]}
            tmp = os.path.join(
                workdir, f"SOLUTION.json.tmp.{os.getpid()}")
            with open(tmp, "w") as fh:
                json.dump(sol, fh, indent=1)
            os.replace(tmp, os.path.join(workdir, "SOLUTION.json"))
            with open(stop_path, "w") as fh:
                fh.write(f"solved by worker {wid}\n")
            out(f"VERIFIED Hadamard order {N} -> {csv}")
            for s in quad:
                out("  seq: " + "".join(
                    "+" if v > 0 else "-" for v in s))
            telemetry.record_cycle(stats, time.time() - t0, pools, "solved")
            return 0
        telemetry.record_cycle(stats, time.time() - t0, pools, "running")
        sizes = {str(b): len(p.seqs) for b, p in pools.items()}
        out(f"cycle {cycle} ({time.time()-t0:.1f}s) pools={sizes} "
            f"pairs={stats['pairs_hashed']:,} buckets={stats['buckets']:,} "
            f"probes={stats['probes']:,} collchk={stats['collisions']:,} "
            f"psdskip={stats['psd_skipped']:,} "
            f"baseline_keys={stats['baseline_keys']} "
            f"incr_keys={stats['incremental_keys']}")
        while (time.time() - t0 < cycle_seconds and not should_stop()):
            time.sleep(0.5)
    telemetry.write({}, 0.0, pools, "stopped")
    out("STOP observed (global or per-worker); exiting cleanly")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["gs", "tt"], required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="gs: sequence length (167); tt: m (56)")
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--pool-cap", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--max-pairs", type=int, default=20_000_000)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--cycle-seconds", type=float, default=0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0,
                    help="mid-cycle progress.json update interval")
    a = ap.parse_args()
    route = GSRoute(a.n or 167) if a.route == "gs" else TTRoute(a.n or 56)
    raise SystemExit(run_worker(route, a.worker_id, a.seed, a.workdir,
                                a.pool_cap, a.batch, a.max_pairs, a.chunk,
                                a.cycle_seconds, a.heartbeat_seconds))


if __name__ == "__main__":
    main()
