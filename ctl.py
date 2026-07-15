"""Orchestration CLI (hardened).

  python3 ctl.py launch         --workers 12   # skips already-alive IDs
  python3 ctl.py launch-missing --workers 12   # explicit gap-filling launch
  python3 ctl.py status          [--workdir work]
  python3 ctl.py monitor         [--workdir work] [--once]
  python3 ctl.py watch           [--workdir work]
  python3 ctl.py stop            [--workdir work]
  python3 ctl.py merge           [--workdir work]
  python3 ctl.py clean-bad       [--workdir work]
  python3 ctl.py kill-duplicates [--workdir work]
  python3 ctl.py verify hadamard_668.csv

Liveness is decided by two independent signals:
  1. flock probe on work/worker_NNN/lock (authoritative: a live worker holds it)
  2. `ps -axo pid=,args=` command-line scan (portable macOS/Linux, no pgrep)
"""
import argparse
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np

import core


# ------------------------------------------------------------ detection

def lock_holder(worker_dir):
    """(alive, pid_from_lockfile). Probe by trying to take the flock."""
    path = os.path.join(worker_dir, "lock")
    if not os.path.exists(path):
        return False, None
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            txt = fh.read().strip()
            return True, int(txt) if txt.isdigit() else None
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.seek(0)
        txt = fh.read().strip()
        return False, int(txt) if txt.isdigit() else None
    finally:
        fh.close()


def ps_workers(workdir):
    """{wid: [(pid, args), ...]} for processes whose command line mentions
    worker.py, this workdir (as given or absolute), and a --worker-id."""
    out = subprocess.run(["ps", "-axo", "pid=,args="],
                         capture_output=True, text=True).stdout
    targets = {workdir, os.path.abspath(workdir)}
    found = {}
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"(\d+)\s+(.*)", line)
        if not m or "worker.py" not in m.group(2):
            continue
        pid, args = int(m.group(1)), m.group(2)
        wd = re.search(r"--workdir\s+(\S+)", args)
        wi = re.search(r"--worker-id\s+(\d+)", args)
        if not wd or not wi:
            continue
        if wd.group(1) not in targets and \
                os.path.abspath(wd.group(1)) not in targets:
            continue
        found.setdefault(int(wi.group(1)), []).append((pid, args))
    return found


def worker_dirs(workdir):
    return sorted(glob.glob(os.path.join(workdir, "worker_[0-9]*")))


def alive_ids(workdir):
    """IDs considered alive: lock held OR a matching process in ps."""
    ids = set()
    for d in worker_dirs(workdir):
        wid = int(os.path.basename(d).split("_")[1])
        if lock_holder(d)[0]:
            ids.add(wid)
    ids.update(ps_workers(workdir).keys())
    return ids


def read_meta(worker_dir):
    try:
        return json.load(open(os.path.join(worker_dir, "meta.json")))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------ commands

def _spawn(wid, a):
    route = "tt" if wid < round(a.workers * a.tt_share) else "gs"
    n = a.tt_m if route == "tt" else a.gs_n
    cmd = [sys.executable, "worker.py", "--route", route, "--n", str(n),
           "--worker-id", str(wid), "--seed", str(a.seed),
           "--workdir", a.workdir, "--pool-cap", str(a.pool_cap),
           "--max-pairs", str(a.max_pairs)]
    logf = open(os.path.join(a.workdir, f"launch_w{wid:03d}.out"), "a")
    p = subprocess.Popen(cmd, stdout=logf, stderr=logf)
    print(f"launched worker {wid} route={route} n={n} pid={p.pid}")
    return p


def launch_missing(a):
    os.makedirs(a.workdir, exist_ok=True)
    stop = os.path.join(a.workdir, "STOP")
    if os.path.exists(os.path.join(a.workdir, "SOLUTION.json")):
        print("SOLUTION.json already present; not launching. "
              "See ctl.py watch.")
        return []
    if os.path.exists(stop):
        os.remove(stop)
        print("removed stale STOP file")
    alive = alive_ids(a.workdir)
    missing = [w for w in range(a.workers) if w not in alive]
    if alive:
        print(f"already alive: {sorted(alive)}")
    if not missing:
        print(f"all {a.workers} workers already running; nothing to launch")
        return []
    procs = [_spawn(w, a) for w in missing]
    print(f"launched {len(procs)} missing worker(s): {missing}")
    if a.wait:
        for p in procs:
            p.wait()
    return procs


def status(a):
    rows = []
    ps = ps_workers(a.workdir)
    for d in worker_dirs(a.workdir):
        wid = int(os.path.basename(d).split("_")[1])
        alive, lockpid = lock_holder(d)
        meta = read_meta(d)
        pids = [p for p, _ in ps.get(wid, [])]
        rows.append((wid, meta.get("route", "?"),
                     lockpid or (pids[0] if pids else "-"),
                     "ALIVE" if (alive or pids) else "dead",
                     "DUPLICATE!" if len(pids) > 1 else "",
                     os.path.join(d, "log.txt")))
    for wid in sorted(set(ps) - {r[0] for r in rows}):
        rows.append((wid, "?", ps[wid][0][0], "ALIVE (no dir yet)",
                     "DUPLICATE!" if len(ps[wid]) > 1 else "", "-"))
    if not rows:
        print("no workers found in", a.workdir)
        return
    print(f"{'id':>4} {'route':>6} {'pid':>8} {'state':<18} {'':<11} log")
    for r in sorted(rows):
        print(f"{r[0]:>4} {r[1]:>6} {str(r[2]):>8} {r[3]:<18} {r[4]:<11} {r[5]}")
    n_alive = len(alive_ids(a.workdir))
    print(f"unique alive worker ids: {n_alive}")


def kill_duplicates(a):
    ps = ps_workers(a.workdir)
    killed = 0
    for wid, procs in sorted(ps.items()):
        if len(procs) <= 1:
            continue
        d = os.path.join(a.workdir, f"worker_{wid:03d}")
        _, lockpid = lock_holder(d)
        pids = [p for p, _ in procs]
        keep = lockpid if lockpid in pids else min(pids)
        for pid in pids:
            if pid == keep:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"worker {wid}: SIGTERM duplicate pid {pid} "
                      f"(kept {keep})")
                killed += 1
            except OSError as e:
                print(f"worker {wid}: could not kill {pid}: {e}")
    if killed:
        time.sleep(2)
        for wid, procs in ps_workers(a.workdir).items():
            for pid, _ in procs[1:]:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"worker {wid}: SIGKILL straggler {pid}")
                except OSError:
                    pass
    print(f"killed {killed} duplicate process(es)"
          if killed else "no duplicates found")


def clean_bad(a):
    bad_dir = os.path.join(a.workdir, "bad_npz_backup")
    moved, removed = [], []
    for path in glob.glob(os.path.join(a.workdir, "**", "*.npz"),
                          recursive=True):
        if os.path.sep + "bad_npz_backup" + os.path.sep in path:
            continue
        base = os.path.basename(path)
        if ".tmp." in base:
            if time.time() - os.path.getmtime(path) > 300:
                os.remove(path)
                removed.append(path)
            continue
        try:
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("crc mismatch")
            np.load(path, allow_pickle=False).close()
        except Exception as e:
            os.makedirs(bad_dir, exist_ok=True)
            dest = os.path.join(bad_dir, f"{base}.{int(time.time())}")
            os.replace(path, dest)
            moved.append((path, dest, type(e).__name__))
    for src, dest, why in moved:
        print(f"quarantined {src} -> {dest} ({why})")
    for p in removed:
        print(f"removed stale temp {p}")
    if not moved and not removed:
        print("all .npz files healthy; no stale temps")


def monitor(a):
    while True:
        sol = os.path.join(a.workdir, "SOLUTION.json")
        if os.path.exists(sol):
            print("SOLUTION FOUND:", open(sol).read())
            return
        lines = []
        for lg in sorted(glob.glob(os.path.join(a.workdir, "worker_*",
                                                "log.txt"))):
            tail = open(lg).readlines()
            if tail:
                lines.append(tail[-1].rstrip())
        os.system("clear" if os.name != "nt" else "cls")
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "-", a.workdir,
              f"- unique alive: {len(alive_ids(a.workdir))}")
        print("\n".join(lines) if lines else "(no worker logs yet)")
        if a.once:
            return
        time.sleep(5)


def watch(a):
    sol = os.path.join(a.workdir, "SOLUTION.json")
    while not os.path.exists(sol):
        time.sleep(2)
    data = json.load(open(sol))
    print(f"VERIFIED SOLUTION: order {data['order']} via route "
          f"{data['route']}, worker {data['worker']}, "
          f"pattern {data['pattern']}")
    print("CSV:", data.get("csv", f"hadamard_{data['order']}.csv"))
    for s in data["sequences"]:
        print("  seq:", "".join("+" if v > 0 else "-" for v in s))


def stop(a):
    with open(os.path.join(a.workdir, "STOP"), "w") as fh:
        fh.write("manual stop\n")
    print("STOP file written; workers exit at next cycle boundary.")


def merge(a):
    merged = {}
    for pool in glob.glob(os.path.join(a.workdir, "worker_*", "pools.npz")):
        try:
            data = np.load(pool, allow_pickle=False)
        except Exception as e:
            print(f"skipping unreadable {pool} ({type(e).__name__}); "
                  f"run: python3 ctl.py clean-bad")
            continue
        for key in data.files:
            arr = data[key].astype(np.int8)
            merged[key] = (np.unique(np.concatenate([merged[key], arr]),
                                     axis=0)
                           if key in merged else np.unique(arr, axis=0))
    def route_of_key(k):
        return "tt" if k.startswith(("bxy_", "bz_", "bw_")) else "gs"

    def canon_dedup(arr, periodic):
        """Keep one representative per canonical orbit. Uses the SAME
        representative rule as the worker path (core.canonical_key over
        negation x rotations x reversal for periodic; negation x reversal
        for nonperiodic), so worker absorb sees identical keys."""
        if len(arr) == 0:
            return arr, 0
        seen = set()
        keep = []
        for row in arr:
            k = core.canonical_key(np.ascontiguousarray(row, np.int8),
                                   periodic)
            if k not in seen:
                seen.add(k)
                keep.append(row)
        removed = len(arr) - len(keep)
        return (np.array(keep, dtype=np.int8)
                if keep else arr[:0]), removed

    for route in ("gs", "tt"):
        keys = {k: v for k, v in merged.items()
                if route_of_key(k) == route}
        raw_rows = sum(len(v) for v in keys.values())
        removed_total = 0
        for k in list(keys):
            deduped, removed = canon_dedup(keys[k], periodic=(route == "gs"))
            keys[k] = deduped
            removed_total += removed
        kept_rows = sum(len(v) for v in keys.values())
        print(f"[merge/{route}] raw rows read: {raw_rows:,}  "
              f"canonical kept: {kept_rows:,}  "
              f"orbit duplicates removed: {removed_total:,}")
        out = os.path.join(a.workdir, f"global_pools_{route}.npz")
        tmp = f"{out}.tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, **keys)
        os.replace(tmp, out)
        print(f"global_pools_{route}: "
              f"{ {k: len(v) for k, v in keys.items()} or 'empty' }")


def migrate(a):
    """Offline migration of every worker directory: validate + quarantine
    pools, dedup, recompute PAF/PSD caches, write versioned migration index.
    Idempotent; safe to run while workers are stopped. Never imports
    checked-pair history -- first optimized run does a baseline full pass."""
    import worker as wk
    done, skipped = 0, 0
    for d in worker_dirs(a.workdir):
        pool_path = os.path.join(d, "pools.npz")
        if not os.path.exists(pool_path):
            continue
        mig = os.path.join(d, "migration.json")
        if os.path.exists(mig):
            try:
                if json.load(open(mig)).get("opt_version") == wk.OPT_VERSION:
                    print(f"{d}: already migrated (opt_version "
                          f"{wk.OPT_VERSION}); up to date")
                    skipped += 1
                    continue
            except (OSError, ValueError):
                pass
        data = wk.safe_load_npz(pool_path, a.workdir)
        if data is None:
            print(f"{d}: pools.npz was corrupt -> quarantined to "
                  f"bad_npz_backup; worker will start fresh")
            continue
        keys = list(data.files)
        is_tt = any(k.startswith(("bxy_", "bz_", "bw_")) for k in keys)
        if is_tt:
            m = max(data[k].shape[1] for k in keys)
            route = wk.TTRoute(m)
        else:
            n = data[keys[0]].shape[1]
            route = wk.GSRoute(n)
        pools = {b: __import__("engine").Pool(route.lengths[b],
                                              route.periodic)
                 for b in route.bins}
        raw_counts = {}
        for b in route.bins:
            bk = wk.bin_key(b)
            if bk in data.files:
                arr = data[bk].astype(np.int8)
                raw_counts[bk] = len(arr)
                pools[b].add(arr)
        wk.write_migration_index(d, route, pools, raw_counts)
        caches = wk.ensure_caches(route, pools, {})
        wk.save_caches(d, route, pools, caches)
        wk.save_npz(pool_path, {wk.bin_key(b): p.seqs
                                for b, p in pools.items()})
        total_raw = sum(raw_counts.values())
        total_kept = sum(len(p.seqs) for p in pools.values())
        print(f"{d}: route={route.name} migrated {total_raw} raw rows -> "
              f"{total_kept} deduped; caches rebuilt "
              f"(opt_version {wk.OPT_VERSION})")
        done += 1
    print(f"migration complete: {done} migrated, {skipped} already current")


def acceptance(a):
    """Summarize candidate acceptance over the last N cycles per worker and
    diagnose which bottleneck class applies:
      (a) seed saturation (b) generator exhaustion (c) duplicate saturation
      (d) PSD too restrictive (e) route imbalance (f) pool-cap saturation."""
    N = a.cycles
    print(f"{'id':>4} {'route':>6} {'cyc':>6} {'acc/min':>9} {'gen/min':>11} "
          f"{'dup%':>6} {'psd%':>6} {'@cap':>7} {'verdict'}")
    fleet = []
    for d in worker_dirs(a.workdir):
        wid = int(os.path.basename(d).split("_")[1])
        hist_path = os.path.join(d, "cycles.jsonl")
        recs = []
        if os.path.exists(hist_path):
            for line in open(hist_path).read().splitlines()[-N:]:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
        if not recs:
            try:
                p = json.load(open(os.path.join(d, "progress.json")))
                recs = [{**p.get("latest_cycle", {}),
                         "secs": p.get("cycle_seconds", 0),
                         "pools": p.get("pools", {}),
                         "pool_cap": p.get("pool_cap", 0),
                         "cycle": p.get("cycle", 0)}]
            except (OSError, ValueError):
                continue
        mins = max(sum(r.get("secs", 0) for r in recs) / 60.0, 1e-9)
        gen = sum(r.get("generated", 0) for r in recs)
        acc = sum(r.get("accepted", 0) for r in recs)
        dup = sum(r.get("dup_rejected", 0) for r in recs)
        psd = sum(r.get("build_psd_rej", 0) for r in recs)
        last = recs[-1]
        cap = int(last.get("pool_cap", 0))
        pools = last.get("pools", {})
        at_cap = sum(1 for v in pools.values() if cap and v >= cap)
        nbins = max(len(pools), 1)
        route = json.load(open(os.path.join(d, "meta.json"))).get(
            "route", "?") if os.path.exists(
            os.path.join(d, "meta.json")) else "?"
        examined = acc + dup
        dup_rate = 100 * dup / max(examined, 1)
        psd_rate = 100 * psd / max(gen, 1)
        status_j = {}
        try:
            status_j = json.load(open(os.path.join(d, "progress.json")))
        except (OSError, ValueError):
            pass
        hb_age = time.time() - status_j.get("updated", 0) \
            if status_j else float("inf")
        mid_cycle = status_j.get("status") == "matching" or \
            str(status_j.get("latest_cycle", {}).get("phase", "")
                ).startswith("matching")
        if cap and at_cap == nbins:
            verdict = "(f) POOL-CAP SATURATED: raise --pool-cap"
        elif gen == 0 and at_cap == 0 and last.get("cycle", 0) == 0:
            if mid_cycle and hb_age < 120:
                ph = status_j["latest_cycle"].get("phase", "matching")
                eta = status_j["latest_cycle"].get("eta_key_seconds")
                verdict = (f"mid-cycle ({ph}"
                           + (f", key ETA {eta/3600:.1f}h" if eta
                              else "") + ") -- long cycle in progress")
            elif hb_age > 3600:
                verdict = (f"UNPROVEN BUSY: no telemetry for "
                           f"{hb_age/3600:.1f}h; heartbeats absent -- "
                           f"restart to enable them")
            else:
                verdict = "no completed cycles yet (startup/long cycle)"
        elif gen == 0:
            verdict = "(f) generation gated (bins at/near cap)"
        elif dup_rate > 90:
            verdict = "(c) duplicate saturation: add seeds/workers"
        elif psd_rate > 99.5:
            verdict = "(d) PSD-bound: expected ~98%; investigate if higher"
        elif acc / mins < 1 and gen / mins > 1000:
            verdict = "(c/f) low acceptance: check need-cap + dup rate"
        else:
            verdict = "healthy"
        fleet.append((route, acc / mins))
        print(f"{wid:>4} {route:>6} {last.get('cycle', 0):>6} "
              f"{acc/mins:>9,.1f} {gen/mins:>11,.0f} {dup_rate:>5.1f}% "
              f"{psd_rate:>5.1f}% {at_cap:>3}/{nbins:<3} {verdict}")
    by_route = {}
    for r, apm in fleet:
        by_route.setdefault(r, []).append(apm)
    for r, v in sorted(by_route.items()):
        print(f"route {r}: {sum(v):,.1f} accepted/min across "
              f"{len(v)} workers")
    print("\nnotes: (a) seed saturation and (b) generator exhaustion are "
          "impossible at n=167 scale (2^84 space per bin); build-PSD "
          "rejecting ~98% is the designed sieve, not a fault.")


def experiment(a):
    """Safe seed experiment in an ISOLATED directory: never touches the
    main workdir. Runs short-lived workers on fresh seeds and reports
    accepted/min per seed so you can pick seeds/caps before committing."""
    import worker as wk_mod  # noqa: F401  (validates import early)
    ts = int(time.time())
    exp = os.path.join("experiments", f"exp_{ts}")
    os.makedirs(exp, exist_ok=True)
    assert os.path.abspath(exp) != os.path.abspath(a.workdir)
    print(f"experiment dir: {exp} (main workdir untouched)")
    procs = []
    for i, seed in enumerate(range(a.base_seed, a.base_seed + a.seeds)):
        cmd = [sys.executable, "worker.py", "--route", a.route,
               "--n", str(a.n), "--worker-id", str(i),
               "--seed", str(seed), "--workdir", exp,
               "--pool-cap", str(a.pool_cap), "--batch", str(a.batch),
               "--max-pairs", "2000000"]
        logf = open(os.path.join(exp, f"launch_w{i:03d}.out"), "a")
        procs.append((seed, subprocess.Popen(cmd, stdout=logf,
                                             stderr=logf)))
    print(f"running {a.seeds} seed(s) for {a.minutes} min...")
    deadline = time.time() + a.minutes * 60
    while time.time() < deadline and any(p.poll() is None
                                         for _, p in procs):
        time.sleep(2)
    with open(os.path.join(exp, "STOP"), "w") as fh:
        fh.write("experiment budget reached\n")
    for _, p in procs:
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    print(f"{'seed':>10} {'accepted':>10} {'generated':>12} "
          f"{'acc/min':>9} {'dup%':>6}")
    for i, (seed, _) in enumerate(procs):
        try:
            p = json.load(open(os.path.join(
                exp, f"worker_{i:03d}", "progress.json")))
        except (OSError, ValueError):
            print(f"{seed:>10} (no telemetry)")
            continue
        t = p["totals"]
        mins = max((p["updated"] - p["telemetry_started_at"]) / 60, 1e-9)
        ex = t["accepted"] + t["dup_rejected"]
        print(f"{seed:>10} {t['accepted']:>10,} {t['generated']:>12,} "
              f"{t['accepted']/mins:>9,.1f} "
              f"{100*t['dup_rejected']/max(ex,1):>5.1f}%")
    print(f"results kept in {exp}; delete when done (safe -- isolated).")


def recommend_routes(a):
    """Summarize per-route productivity and recommend worker allocation.
    Rules: high CPU + no telemetry update for --stale-hours => 'unproven
    busy'. If one route accepts candidates and the other has accepted
    nothing for hours, recommend shifting slots (with exact commands)."""
    ps = ps_workers(a.workdir)
    cpu = {}
    out = subprocess.run(["ps", "-axo", "pid=,%cpu="],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                cpu[int(parts[0])] = float(parts[1])
            except ValueError:
                pass
    routes = {}
    for d in worker_dirs(a.workdir):
        wid = int(os.path.basename(d).split("_")[1])
        try:
            p = json.load(open(os.path.join(d, "progress.json")))
        except (OSError, ValueError):
            continue
        r = p.get("route", "?")
        age_h = (time.time() - p.get("updated", 0)) / 3600
        mins = max((p.get("updated", 0)
                    - p.get("telemetry_started_at", 0)) / 60, 1e-9)
        apm = p.get("totals", {}).get("accepted", 0) / mins
        wcpu = max((cpu.get(pid, 0) for pid, _ in ps.get(wid, [])),
                   default=0.0)
        info = routes.setdefault(r, {"workers": [], "apm": 0.0,
                                     "unproven": [], "stale_h": 0.0})
        info["workers"].append(wid)
        info["apm"] += apm
        info["stale_h"] = max(info["stale_h"], age_h)
        if wcpu > 50 and age_h > a.stale_hours:
            info["unproven"].append(wid)
    print(f"{'route':>6} {'workers':>18} {'acc/min':>10} "
          f"{'max stale':>10} {'unproven busy'}")
    for r, i in sorted(routes.items()):
        print(f"{r:>6} {str(sorted(i['workers'])):>18} "
              f"{i['apm']:>10,.1f} {i['stale_h']:>9.1f}h "
              f"{sorted(i['unproven']) or '-'}")
    gs, tt = routes.get("gs", {}), routes.get("tt", {})
    print()
    for r, i in routes.items():
        if i["unproven"]:
            print(f"* route {r}: workers {sorted(i['unproven'])} are "
                  f"UNPROVEN BUSY (high CPU, telemetry silent > "
                  f"{a.stale_hours}h). If they predate the heartbeat "
                  f"patch, restart them to gain mid-cycle visibility "
                  f"before drawing conclusions.")
    if gs.get("apm", 0) > 0 and tt and tt.get("apm", 0) == 0 \
            and tt.get("stale_h", 0) > a.stale_hours:
        keep = sorted(tt.get("workers", []))[:2]
        move = [w for w in sorted(tt.get("workers", [])) if w not in keep]
        print(f"* RECOMMENDATION: GS is accepting "
              f"({gs['apm']:,.0f}/min); TT has accepted 0 for "
              f"{tt['stale_h']:.1f}h. Keep {len(keep)} TT worker(s) "
              f"{keep} for evidence-gathering with heartbeats, and "
              f"rotate {move} to GS with fresh seeds:")
        seeds = ",".join(str(90001000 + k) for k in range(len(move)))
        print(f"    python3 ctl.py rotate-gs-seeds --workdir "
              f"{a.workdir} --seeds {seeds} --replace-workers "
              f"{','.join(map(str, move))}")
    elif tt.get("apm", 0) > 0:
        print("* TT is accepting candidates; no rebalance needed.")
    else:
        print("* Insufficient evidence to recommend a shift yet; "
              "let heartbeats accumulate.")


def rotate_gs_seeds(a):
    """Safely replace selected workers with fresh-seed GS workers.
    Steps: merge pools (preserve their candidates in global), per-worker
    STOP_WORKER (interrupt-safe mid-cycle), wait for lock release, retire
    dir to work/retired/ (NEVER deleted), launch fresh GS worker with the
    provided seed. Old state fully preserved."""
    ids = [int(x) for x in a.replace_workers.split(",")]
    pinned = [w for w in ids if os.path.exists(
        os.path.join(a.workdir, f"worker_{w:03d}", "PIN"))]
    if pinned:
        sys.exit(f"refusing: workers {pinned} are PINNED "
                 f"(remove work/worker_NNN/PIN to allow rotation)")
    seeds = [int(x) for x in a.seeds.split(",")]
    if len(seeds) != len(ids):
        sys.exit(f"need one seed per worker: {len(ids)} workers, "
                 f"{len(seeds)} seeds")
    print("merging pools first so retiring workers' candidates are "
          "preserved in the global pools...")
    merge(a)
    for wid in ids:
        d = os.path.join(a.workdir, f"worker_{wid:03d}")
        if os.path.isdir(d):
            with open(os.path.join(d, "STOP_WORKER"), "w") as fh:
                fh.write("rotate-gs-seeds\n")
            print(f"worker {wid}: STOP_WORKER written; waiting for "
                  f"clean exit (interrupt-safe, may take ~1 min)...")
    deadline = time.time() + a.wait_minutes * 60
    pending = set(ids)
    while pending and time.time() < deadline:
        for wid in list(pending):
            d = os.path.join(a.workdir, f"worker_{wid:03d}")
            if not os.path.isdir(d) or not lock_holder(d)[0]:
                pending.discard(wid)
        time.sleep(2)
    if pending:
        print(f"WARNING: workers {sorted(pending)} did not exit within "
              f"{a.wait_minutes} min; NOT retiring them. Re-run once "
              f"they stop (heartbeat-enabled workers stop in minutes; "
              f"pre-heartbeat workers only check at cycle boundaries).")
        ids = [w for w in ids if w not in pending]
    retired = os.path.join(a.workdir, "retired")
    os.makedirs(retired, exist_ok=True)
    for wid, seed in zip(ids, seeds):
        d = os.path.join(a.workdir, f"worker_{wid:03d}")
        if os.path.isdir(d):
            dest = os.path.join(retired, f"worker_{wid:03d}.{int(time.time())}")
            os.replace(d, dest)
            print(f"worker {wid}: retired old dir -> {dest} (preserved)")
        cmd = [sys.executable, "worker.py", "--route", "gs",
               "--n", str(a.gs_n), "--worker-id", str(wid),
               "--seed", str(seed), "--workdir", a.workdir,
               "--pool-cap", str(a.pool_cap),
               "--max-pairs", str(a.max_pairs)]
        logf = open(os.path.join(a.workdir, f"launch_w{wid:03d}.out"), "a")
        p = subprocess.Popen(cmd, stdout=logf, stderr=logf)
        print(f"worker {wid}: fresh GS worker launched, seed={seed}, "
              f"pid={p.pid}")
    print("rotation complete; old state under work/retired/, global "
          "pools untouched except the pre-rotation merge.")


def _gs_worker_assessment(d, stale_cycles, dup_threshold,
                          min_apm):
    """Classify one worker for auto-rotation. Returns (decision, info).
    decision in: 'rotate', 'skip'. Rotation requires PROOF, never wall
    time: N completed cycles with ~zero acceptance, high dup%, no cap
    pressure, no promising signals -- and only for alive GS workers."""
    wid = int(os.path.basename(d).split("_")[1])
    meta, prog = {}, {}
    for name, dst in (("meta.json", "meta"), ("progress.json", "prog")):
        try:
            obj = json.load(open(os.path.join(d, name)))
            if dst == "meta":
                meta = obj
            else:
                prog = obj
        except (OSError, ValueError):
            pass
    route = prog.get("route") or meta.get("route")
    info = {"wid": wid, "route": route, "seed": meta.get("seed")}
    if os.path.exists(os.path.join(d, "PIN")):
        return "skip", {**info, "why": "PINNED (never auto-rotated)"}
    if route != "gs":
        return "skip", {**info, "why": "not a GS worker (never touched)"}
    alive = lock_holder(d)[0] or wid in ps_workers(
        os.path.dirname(d)).keys()
    if not alive:
        return "skip", {**info, "why": "not alive (use launch-missing)"}
    recs = []
    hist = os.path.join(d, "cycles.jsonl")
    if os.path.exists(hist):
        for line in open(hist).read().splitlines():
            try:
                r = json.loads(line)
                if r.get("secs", 0) > 0:
                    recs.append(r)
            except ValueError:
                pass
    recs = recs[-stale_cycles:]
    if len(recs) < stale_cycles:
        # startup or long cycle: activity alone never triggers rotation,
        # and absence of completed cycles is not proof of staleness
        phase = str(prog.get("latest_cycle", {}).get("phase", ""))
        return "skip", {**info, "why":
                        f"only {len(recs)}/{stale_cycles} completed "
                        f"cycles (mid-cycle/startup"
                        + (f": {phase}" if phase else "") + ")"}
    mins = sum(r["secs"] for r in recs) / 60.0
    acc = sum(r.get("accepted", 0) for r in recs)
    dup = sum(r.get("dup_rejected", 0) for r in recs)
    coll = sum(r.get("collisions", 0) for r in recs)
    at_cap = max(int(r.get("pool_cap", 0)
                     and sum(1 for v in r.get("pools", {}).values()
                             if v >= r["pool_cap"])) for r in recs)
    apm = acc / max(mins, 1e-9)
    examined = acc + dup
    dup_pct = 100.0 * dup / examined if examined else None
    info.update(apm=round(apm, 2), dup_pct=(round(dup_pct, 1)
                                            if dup_pct is not None
                                            else None),
                cycles=recs[-1].get("cycle"), at_cap=at_cap,
                collisions=coll)
    if apm >= min_apm:
        return "skip", {**info, "why": f"accepting ({apm:.1f}/min)"}
    if at_cap > 0:
        return "skip", {**info, "why":
                        f"{at_cap} bin(s) at cap: cap issue, not seed"}
    if dup_pct is None:
        return "skip", {**info, "why":
                        "no candidates examined in window; ambiguous"}
    if dup_pct < dup_threshold:
        return "skip", {**info, "why":
                        f"dup {dup_pct:.1f}% < threshold; not proven"}
    if coll > 0:
        return "skip", {**info, "why":
                        f"promising signals in window ({coll} "
                        f"collision checks); keeping"}
    return "rotate", {**info, "why":
                      f"PROVEN STALE: {apm:.2f} acc/min, "
                      f"dup {dup_pct:.1f}%, 0 bins at cap, no signals, "
                      f"over {len(recs)} completed cycles"}


def _used_seeds(workdir):
    used = set()
    log = os.path.join(workdir, "seed_rotations.jsonl")
    if os.path.exists(log):
        for line in open(log).read().splitlines():
            try:
                used.add(int(json.loads(line)["new_seed"]))
            except (ValueError, KeyError):
                pass
    for d in worker_dirs(workdir):
        try:
            s = json.load(open(os.path.join(d, "meta.json"))).get("seed")
            if s is not None:
                used.add(int(s))
        except (OSError, ValueError):
            pass
    return used


def _last_rotation_ts(workdir):
    log = os.path.join(workdir, "seed_rotations.jsonl")
    ts = 0.0
    if os.path.exists(log):
        for line in open(log).read().splitlines():
            try:
                ts = max(ts, float(json.loads(line)["timestamp"]))
            except (ValueError, KeyError):
                pass
    return ts


def auto_rotate_gs(a):
    """Automatic GS seed rotation, telemetry-proof-gated. Rotates only
    workers that pass _gs_worker_assessment (never TT, never mid-cycle
    activity alone, never wall time alone). Old dirs are retired to
    work/retired_gs_<ts>/, never deleted. Rotations are logged to
    work/seed_rotations.jsonl and rate-limited by --cooldown-minutes."""
    seeds = []
    for line in open(a.seed_file).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(int(line))

    def one_pass():
        used = _used_seeds(a.workdir)
        avail = [s for s in seeds if s not in used]
        since = time.time() - _last_rotation_ts(a.workdir)
        if since < a.cooldown_minutes * 60:
            print(f"[auto-rotate] cooldown: last rotation "
                  f"{since/60:.1f} min ago < {a.cooldown_minutes} min; "
                  f"no action")
            return 0
        plans, skips = [], []
        for d in worker_dirs(a.workdir):
            decision, info = _gs_worker_assessment(
                d, a.stale_cycles, a.dup_threshold,
                a.min_accepted_per_min)
            (plans if decision == "rotate" else skips).append(info)
        for s in skips:
            print(f"[skip] w{s['wid']} ({s.get('route')}): {s['why']}")
        plans = plans[: a.max_rotations]
        if not plans:
            print("[auto-rotate] no proven-stale GS workers")
            return 0
        if len(avail) < len(plans):
            print(f"[auto-rotate] only {len(avail)} unused seeds for "
                  f"{len(plans)} rotations; truncating")
            plans = plans[: len(avail)]
        for p, seed in zip(plans, avail):
            print(f"[plan] rotate w{p['wid']} (seed {p['seed']}) -> "
                  f"fresh seed {seed} :: {p['why']}")
        if a.dry_run:
            print(f"[dry-run] {len(plans)} rotation(s) planned; "
                  f"nothing changed")
            return 0
        if not a.replace_stale:
            print("pass --replace-stale to act (or --dry-run to plan)")
            return 0
        ts = int(time.time())
        retired_root = os.path.join(a.workdir, f"retired_gs_{ts}")
        print("[auto-rotate] merging pools first (preserve candidates)")
        merge(a)
        for p in plans:
            d = os.path.join(a.workdir, f"worker_{p['wid']:03d}")
            with open(os.path.join(d, "STOP_WORKER"), "w") as fh:
                fh.write("auto-rotate-gs\n")
        deadline = time.time() + a.wait_minutes * 60
        pend = {p["wid"] for p in plans}
        while pend and time.time() < deadline:
            for wid in list(pend):
                d = os.path.join(a.workdir, f"worker_{wid:03d}")
                if not lock_holder(d)[0]:
                    pend.discard(wid)
            time.sleep(1)
        done = 0
        for p, seed in zip(plans, avail):
            wid = p["wid"]
            if wid in pend:
                print(f"[warn] w{wid} did not exit in time; NOT rotated")
                continue
            d = os.path.join(a.workdir, f"worker_{wid:03d}")
            os.makedirs(retired_root, exist_ok=True)
            dest = os.path.join(retired_root, f"worker_{wid:03d}")
            os.replace(d, dest)
            cmd = [sys.executable, "worker.py", "--route", "gs",
                   "--n", str(a.gs_n), "--worker-id", str(wid),
                   "--seed", str(seed), "--workdir", a.workdir,
                   "--pool-cap", str(a.pool_cap),
                   "--max-pairs", str(a.max_pairs)]
            logf = open(os.path.join(a.workdir,
                                     f"launch_w{wid:03d}.out"), "a")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf)
            rec = {"timestamp": time.time(), "old_worker_id": wid,
                   "old_seed": p.get("seed"), "new_seed": seed,
                   "reason": p["why"], "old_accepted_per_min": p["apm"],
                   "old_dup_pct": p["dup_pct"], "old_cycle": p["cycles"],
                   "retired_to": dest, "new_pid": proc.pid}
            with open(os.path.join(a.workdir, "seed_rotations.jsonl"),
                      "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[rotated] w{wid}: retired -> {dest}; fresh GS "
                  f"seed {seed} pid {proc.pid}")
            done += 1
        print(f"[auto-rotate] {done} rotation(s) done; worker count "
              f"unchanged")
        return done

    if a.watch:
        pid_path = os.path.join(a.workdir, "auto_rotate.pid")
        with open(pid_path, "w") as fh:
            json.dump({"pid": os.getpid(),
                       "cooldown_minutes": a.cooldown_minutes,
                       "watch_interval_seconds": a.watch_interval_seconds,
                       "started": time.time()}, fh)
        try:
            tick = 0
            while True:
                tick += 1
                print(f"--- auto-rotate watch tick {tick} "
                      f"({time.strftime('%H:%M:%S')}) ---")
                one_pass()
                if a.max_ticks and tick >= a.max_ticks:
                    return
                time.sleep(a.watch_interval_seconds)
        finally:
            try:
                os.remove(pid_path)
            except OSError:
                pass
    else:
        one_pass()


def _load_collisions(workdir, wid):
    path = os.path.join(workdir, f"worker_{wid:03d}", "collisions.jsonl")
    out = []
    if os.path.exists(path):
        for line in open(path).read().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _seq_from_str(s):
    return np.array([1 if ch == "+" else -1 for ch in s], dtype=np.int8)


def replay_collision(a):
    """Reproduce and re-verify a saved collision, read-only w.r.t. work/.
    Recomputes the PAF residual from the stored sequences and, if it is
    exactly zero, assembles and verifies the Hadamard matrix."""
    import worker as wk
    recs = _load_collisions(a.workdir, a.worker_id)
    if a.cycle is not None:
        recs = [r for r in recs if r.get("cycle") == a.cycle]
    if not recs:
        sys.exit(f"no collision records for worker {a.worker_id}"
                 + (f" cycle {a.cycle}" if a.cycle is not None else "")
                 + " (records begin after the evidence patch; earlier "
                 "collisions were counters only and cannot be replayed)")
    for n, r in enumerate(recs):
        print(f"--- record {n}: worker {r['worker_id']} seed {r['seed']} "
              f"cycle {r['cycle']} route {r['route']} "
              f"pattern {r['pattern']} bins {r['bins']} "
              f"idx {r['indices']}")
        quad = [_seq_from_str(seq) for seq in r["sequences"]]
        route = wk.GSRoute(len(quad[0])) if r["route"] == "gs" \
            else wk.TTRoute(len(quad[0]))
        pafs = [route.paf(q[None, :])[0].astype(int) for q in quad]
        residual = (route.wt[0] * (pafs[0] + pafs[1])
                    + route.wt[1] * (pafs[2] + pafs[3]))
        nz = int(np.abs(residual).sum())
        verdict = ("ZERO -- exact match" if nz == 0
                   else "nonzero -- hash coincidence, mathematically empty")
        print(f"    recomputed PAF residual L1 = {nz} ({verdict})")
        if nz == 0:
            H = route.assemble(quad)
            N = core.verify_hadamard(H)
            csv = f"hadamard_{N}_replay.csv"
            np.savetxt(csv, H, fmt="%d", delimiter=",")
            print(f"    VERIFIED Hadamard order {N} -> {csv}")
        else:
            print("    stored verdict matches: "
                  f"residual_zero={r['residual_zero']} "
                  f"verified={r['verified_hadamard']}")


def replay_cycle(a):
    """Re-run matching over the pool snapshot a worker had at a given
    cycle (pools are append-only, so slicing to that cycle's recorded bin
    sizes reconstructs the exact candidate set). Read-only; reports every
    hash collision and any exact match. Can be expensive: bound with
    --max-pairs."""
    import worker as wk
    import engine
    d = os.path.join(a.workdir, f"worker_{a.worker_id:03d}")
    hist = [json.loads(l) for l in
            open(os.path.join(d, "cycles.jsonl")).read().splitlines()]
    rec = next((r for r in hist if r.get("cycle") == a.cycle), None)
    if rec is None:
        sys.exit(f"cycle {a.cycle} not found in cycles.jsonl")
    meta = json.load(open(os.path.join(d, "meta.json")))
    route = wk.GSRoute(meta.get("n") or 167) if meta["route"] == "gs" \
        else wk.TTRoute(meta.get("n") or 56)
    data = np.load(os.path.join(d, "pools.npz"), allow_pickle=False)
    pools = {}
    for b in route.bins:
        bk = wk.bin_key(b)
        want = int(rec["pools"].get(str(b), 0))
        arr = data[bk][:want] if bk in data.files else \
            np.empty((0, route.lengths[b]), np.int8)
        if len(arr) < want:
            print(f"warning: bin {b} has {len(arr)} rows now but cycle "
                  f"recorded {want}; snapshot incomplete")
        p = engine.Pool(route.lengths[b], route.periodic)
        p.seqs = arr.astype(np.int8)
        pools[b] = p
    print(f"replaying cycle {a.cycle} snapshot: "
          f"{ {str(b): len(p.seqs) for b, p in pools.items()} }")
    total_coll, found = 0, None
    for pi, pat in enumerate(route.patterns):
        for (b1, b2), (b3, b4) in route.splits(pat):
            P = [pools[b] for b in (b1, b2, b3, b4)]
            if min(len(p.seqs) for p in P) == 0:
                continue
            pafs = [route.paf(p.seqs) for p in P]
            clog = []
            hit = engine.match(pafs[0], pafs[1], pafs[2], pafs[3],
                               wt1=route.wt[0], wt2=route.wt[1],
                               max_pairs=a.max_pairs,
                               collision_log=clog)
            if clog:
                print(f"  pattern {pat}: {len(clog)} hash "
                      f"collision(s) at indices {clog[:8]}")
                total_coll += len(clog)
            if hit is not None:
                found = (pat, [P[x].seqs[hit[x]] for x in range(4)])
    if found is not None:
        pat, quad = found
        H = route.assemble(quad)
        N = core.verify_hadamard(H)
        csv = f"hadamard_{N}_replay.csv"
        np.savetxt(csv, H, fmt="%d", delimiter=",")
        print(f"EXACT MATCH re-found and VERIFIED: order {N} -> {csv}")
    else:
        print(f"replay complete: {total_coll} hash collision(s), "
              f"no exact match in this snapshot (consistent with the "
              f"original run)")


SEED_WEIGHTS = {
    # evidence hierarchy: math evidence dominates, raw noise cannot
    "verified_hadamard_each": 10000.0,
    "residual_zero_each": 1000.0,
    "noisy_collision_each": 0.5,
    "noisy_collision_cap": 5.0,
    "accepted_per_min_max_pts": 100.0,   # linear up to 300 acc/min
    "accepted_per_min_ref": 300.0,
    "dup_pct_penalty_per_pct": -1.0,
    "capped_bin_penalty": -25.0,
    "pool_balance_max_pts": 25.0,
    "build_pass_max_pts": 15.0,          # linear up to 3% sieve pass
    "build_pass_ref_pct": 3.0,
    "screen_rej_weight": 0.0,            # informational: reflects the
                                         # pool population, not the seed
    "cycle_secs_penalty_max": -10.0,     # linear up to 600s median
    "cycle_secs_ref": 600.0,
    "first_evidence_max_pts": 20.0,      # 20 - hours_to_first_evidence
}


def _collect_worker_stats(d, max_cycles=200):
    """Read one worker dir (live or retired), tolerating missing or
    corrupt files. Returns (stats dict, warnings list)."""
    warns = []
    meta, prog = {}, {}
    for name, dst in (("meta.json", meta), ("progress.json", prog)):
        try:
            dst.update(json.load(open(os.path.join(d, name))))
        except (OSError, ValueError):
            pass
    recs, corrupt = [], 0
    cpath = os.path.join(d, "cycles.jsonl")
    if os.path.exists(cpath):
        for line in open(cpath).read().splitlines()[-max_cycles:]:
            try:
                r = json.loads(line)
                if r.get("secs", 0) > 0:
                    recs.append(r)
            except ValueError:
                corrupt += 1
    if corrupt:
        warns.append(f"{cpath}: skipped {corrupt} corrupt line(s)")
    colls, ccorrupt = [], 0
    kpath = os.path.join(d, "collisions.jsonl")
    if os.path.exists(kpath):
        for line in open(kpath).read().splitlines():
            try:
                colls.append(json.loads(line))
            except ValueError:
                ccorrupt += 1
    if ccorrupt:
        warns.append(f"{kpath}: skipped {ccorrupt} corrupt line(s)")
    mins = sum(r["secs"] for r in recs) / 60.0
    gen = sum(r.get("generated", 0) for r in recs)
    acc = sum(r.get("accepted", 0) for r in recs)
    dup = sum(r.get("dup_rejected", 0) for r in recs)
    bpsd = sum(r.get("build_psd_rej", 0) for r in recs)
    pairs = sum(r.get("pairs_hashed", 0) + r.get("probes", 0)
                for r in recs)
    screened = sum(r.get("psd_rejected", 0) + r.get("probe_psd_rej", 0)
                   for r in recs)
    secs_list = sorted(r["secs"] for r in recs)
    started = prog.get("telemetry_started_at") or meta.get("started")
    verified = sum(1 for x in colls if x.get("verified_hadamard"))
    res0 = sum(1 for x in colls if x.get("residual_zero")
               and not x.get("verified_hadamard"))
    noisy = sum(1 for x in colls if not x.get("residual_zero"))
    evid_ts = [x["timestamp"] for x in colls
               if x.get("residual_zero") or x.get("verified_hadamard")]
    first_ev_h = ((min(evid_ts) - started) / 3600
                  if evid_ts and started else None)
    pools = prog.get("pools", {})
    cap = prog.get("pool_cap", 0)
    fills = [v / cap for v in pools.values()] if cap and pools else []
    if fills and sum(fills):
        mean = sum(fills) / len(fills)
        var = sum((f - mean) ** 2 for f in fills) / len(fills)
        balance = max(0.0, 1.0 - (var ** 0.5) / max(mean, 1e-9))
    else:
        balance = 0.0
    updated = prog.get("updated", 0)
    return {"dir": d,
            "worker_id": meta.get("worker_id",
                                  int(os.path.basename(d).split("_")[1])
                                  if "_" in os.path.basename(d) else -1),
            "seed": prog.get("seed") or meta.get("seed"),
            "route": prog.get("route") or meta.get("route") or "?",
            "pinned": os.path.exists(os.path.join(d, "PIN")),
            "retired": "retired" in d,
            "minutes": round(mins, 1),
            "accepted_per_min": round(acc / mins, 2) if mins else 0.0,
            "dup_pct": round(100 * dup / (acc + dup), 2)
            if acc + dup else 0.0,
            "build_pass_pct": round(100 * (gen - bpsd) / gen, 3)
            if gen else 0.0,
            "screen_rej_pct": round(100 * screened
                                    / (screened + pairs), 2)
            if screened + pairs else 0.0,
            "median_cycle_secs": (secs_list[len(secs_list) // 2]
                                  if secs_list else None),
            "bins_at_cap": prog.get("bins_at_cap", 0),
            "pool_balance": round(balance, 3),
            "verified": verified, "residual_zero": res0, "noisy": noisy,
            "first_evidence_hours": (round(first_ev_h, 2)
                                     if first_ev_h is not None else None),
            "stale": bool(updated and time.time() - updated > 3600
                          and not "retired" in d),
            }, warns


def _score(s):
    W = SEED_WEIGHTS
    pts = 0.0
    pts += s["verified"] * W["verified_hadamard_each"]
    pts += s["residual_zero"] * W["residual_zero_each"]
    pts += min(s["noisy"] * W["noisy_collision_each"],
               W["noisy_collision_cap"])
    pts += min(s["accepted_per_min"] / W["accepted_per_min_ref"], 1.0) \
        * W["accepted_per_min_max_pts"]
    pts += s["dup_pct"] * W["dup_pct_penalty_per_pct"]
    pts += s["bins_at_cap"] * W["capped_bin_penalty"]
    pts += s["pool_balance"] * W["pool_balance_max_pts"]
    pts += min(s["build_pass_pct"] / W["build_pass_ref_pct"], 1.0) \
        * W["build_pass_max_pts"]
    if s["median_cycle_secs"]:
        pts += min(s["median_cycle_secs"] / W["cycle_secs_ref"], 1.0) \
            * W["cycle_secs_penalty_max"]
    if s["first_evidence_hours"] is not None:
        pts += max(0.0, W["first_evidence_max_pts"]
                   - s["first_evidence_hours"])
    return round(pts, 2)


def seed_leaderboard(a):
    """Rank seeds by MATH EVIDENCE first, runtime health second. Raw
    hash-noise collisions are capped at +5 points total; one
    residual_zero record is worth +1000; a verified Hadamard +10000.
    Writes work/seed_leaderboard.json and prints the table."""
    dirs = list(worker_dirs(a.workdir))
    for sub in sorted(os.listdir(a.workdir)):
        p = os.path.join(a.workdir, sub)
        if os.path.isdir(p) and sub.startswith(("retired", "retired_gs_")):
            dirs += [os.path.join(p, x) for x in sorted(os.listdir(p))
                     if x.startswith("worker_")]
    workers, warns = [], []
    for d in dirs:
        s, w = _collect_worker_stats(d, a.cycles)
        s["score"] = _score(s)
        workers.append(s)
        warns += w
    for w in warns:
        print(f"WARNING: {w}")
    seeds = {}
    for s in workers:
        if s["seed"] is None:
            continue
        g = seeds.setdefault(s["seed"], {"seed": s["seed"], "workers": [],
                                         "routes": set()})
        g["workers"].append(s)
        g["routes"].add(s["route"])
    rows = []
    for seed, g in seeds.items():
        ws = g["workers"]
        tot_min = sum(x["minutes"] for x in ws) or 1e-9
        agg = {"seed": seed,
               "routes": sorted(g["routes"]),
               "worker_ids": sorted(x["worker_id"] for x in ws),
               "pinned": any(x["pinned"] for x in ws),
               "stale": any(x["stale"] for x in ws),
               "minutes": round(tot_min, 1),
               "accepted_per_min": round(
                   sum(x["accepted_per_min"] * x["minutes"]
                       for x in ws) / tot_min, 2),
               "dup_pct": round(sum(x["dup_pct"] * x["minutes"]
                                    for x in ws) / tot_min, 2),
               "build_pass_pct": round(
                   sum(x["build_pass_pct"] * x["minutes"]
                       for x in ws) / tot_min, 3),
               "screen_rej_pct": round(
                   sum(x["screen_rej_pct"] * x["minutes"]
                       for x in ws) / tot_min, 2),
               "median_cycle_secs": max(
                   (x["median_cycle_secs"] or 0) for x in ws) or None,
               "bins_at_cap": max(x["bins_at_cap"] for x in ws),
               "pool_balance": round(sum(x["pool_balance"] for x in ws)
                                     / len(ws), 3),
               "verified": sum(x["verified"] for x in ws),
               "residual_zero": sum(x["residual_zero"] for x in ws),
               "noisy": sum(x["noisy"] for x in ws),
               "first_evidence_hours": min(
                   (x["first_evidence_hours"] for x in ws
                    if x["first_evidence_hours"] is not None),
                   default=None)}
        agg["score"] = _score(agg)
        rows.append(agg)
    rows.sort(key=lambda r: -r["score"])
    print(f"{'#':>3} {'seed':>10} {'route':>6} {'score':>9} "
          f"{'acc/min':>8} {'dup%':>6} {'verif':>6} {'res0':>5} "
          f"{'noisy':>6} {'pass%':>6} {'bal':>5} {'@cap':>5} "
          f"{'1stEv(h)':>9}  flags")
    for i, r in enumerate(rows, 1):
        flags = []
        if r["pinned"]:
            flags.append("PINNED(clue owner)")
        if r["stale"]:
            flags.append("STALE")
        if r["bins_at_cap"]:
            flags.append("CAPPED")
        fe = r["first_evidence_hours"]
        fe = fe if fe is not None else "-"
        print(f"{i:>3} {r['seed']:>10} {'/'.join(r['routes']):>6} "
              f"{r['score']:>9,.1f} {r['accepted_per_min']:>8,.1f} "
              f"{r['dup_pct']:>5.1f}% {r['verified']:>6} "
              f"{r['residual_zero']:>5} {r['noisy']:>6} "
              f"{r['build_pass_pct']:>5.2f}% {r['pool_balance']:>5.2f} "
              f"{r['bins_at_cap']:>5} "
              f"{fe:>9}  {' '.join(flags) or '-'}")
    payload = {"schema_version": 1, "generated": time.time(),
               "weights": SEED_WEIGHTS, "warnings": warns,
               "seeds": rows, "workers": workers}
    out = os.path.join(a.workdir, "seed_leaderboard.json")
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, out)
    print(f"\nwritten: {out} ({len(rows)} seeds, {len(workers)} workers "
          f"incl. retired)")
    print("note: raw noisy collisions are capped at "
          f"+{SEED_WEIGHTS['noisy_collision_cap']} pts total; only "
          "residual_zero/verified evidence can dominate the ranking.")


# ======================= safe run supervisor =======================
APPROVED_GS = "0:94054003,4:94091007"
APPROVED_TT = "1,2,3,5,6,7,8,9,10,11"


def _manifest_path(wd):
    return os.path.join(wd, "run_manifest.json")


def _read_manifest(wd):
    try:
        return json.load(open(_manifest_path(wd)))
    except (OSError, ValueError):
        return {"entries": []}


def _log_file(wd, name):
    logs = os.path.join(wd, "logs")
    os.makedirs(logs, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    return open(os.path.join(logs, f"{name}-{ts}.log"), "a")


def _spawn_logged(wd, name, cmd, env=None):
    logf = _log_file(wd, name)
    logf.write(f"[{time.strftime('%F %T')}] spawn: {' '.join(cmd)}\n")
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)


def _all_supervised_pids(wd):
    """Workers + auto-rotate + watch-leaderboard (NOT the dashboard)."""
    pids = [p for lst in ps_workers(wd).values() for p, _ in lst]
    for pf in ("auto_rotate.pid", "watch_leaderboard.pid"):
        d = None
        try:
            d = json.load(open(os.path.join(wd, pf)))
        except (OSError, ValueError):
            pass
        if d and d.get("pid"):
            pids.append(int(d["pid"]))
    return sorted(set(pids))


def _signal_all(pids, sig):
    n = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            n += 1
        except (ProcessLookupError, PermissionError):
            pass
    return n


def _proc_state(pid):
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                             capture_output=True, text=True).stdout
        return out.strip()
    except Exception:
        return ""


def run_safe(a):
    """Start EXACTLY the approved mix (default 10 TT / 2 GS with worker
    004 pinned GS). Never uses launch-missing. Writes run_manifest.json,
    timestamped logs, PINs worker 004, sets H668_SCREEN_K for workers,
    starts auto-rotate-gs --watch and the dashboard."""
    gs = {}
    for part in a.gs.split(","):
        wid, seed = part.split(":")
        gs[int(wid)] = int(seed)
    tt_ids = [int(x) for x in a.tt_workers.split(",") if x != ""]
    if 4 not in gs:
        sys.exit("REFUSED: worker 004 must be GS "
                 "(it is the pinned clue owner); got GS workers "
                 f"{sorted(gs)}")
    if 4 in tt_ids:
        sys.exit("REFUSED: worker 004 listed as TT; it must stay GS")
    overlap = set(gs) & set(tt_ids)
    if overlap:
        sys.exit(f"REFUSED: workers {sorted(overlap)} listed as both "
                 f"GS and TT")
    dup = [w for w in list(gs) + tt_ids
           if lock_holder(os.path.join(a.workdir,
                                       f"worker_{w:03d}"))[0]]
    if dup and not a.replace:
        sys.exit(f"REFUSED: workers {sorted(dup)} already running; "
                 f"pass --replace to stop and restart them")
    seeds = []
    for line in open(a.seeds).read().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(int(line))
    # restart semantics: a TT worker keeps its existing seed (restart is
    # not rotation -- rotation is auto-rotate's job); only fresh worker
    # dirs draw new unused seeds from the file.
    tt_seed_map = {}
    for wid in tt_ids:
        meta = os.path.join(a.workdir, f"worker_{wid:03d}", "meta.json")
        try:
            m = json.load(open(meta))
            if m.get("route") == "tt" and m.get("seed") is not None:
                tt_seed_map[wid] = int(m["seed"])
        except (OSError, ValueError):
            pass
    used = _used_seeds(a.workdir)
    fresh = [s for s in seeds if s not in used]
    need_fresh = [w for w in tt_ids if w not in tt_seed_map]
    if len(fresh) < len(need_fresh):
        sys.exit(f"REFUSED: need {len(need_fresh)} unused TT seeds, "
                 f"only {len(fresh)} available in {a.seeds}")
    for wid, s in zip(need_fresh, fresh):
        tt_seed_map[wid] = s
    if dup:
        print(f"--replace: stopping running workers {sorted(dup)}")
        for w in dup:
            d = os.path.join(a.workdir, f"worker_{w:03d}")
            with open(os.path.join(d, "STOP_WORKER"), "w") as fh:
                fh.write("run-safe --replace\n")
        deadline = time.time() + a.wait_minutes * 60
        while time.time() < deadline and any(
                lock_holder(os.path.join(a.workdir,
                                         f"worker_{w:03d}"))[0]
                for w in dup):
            time.sleep(1)
        for w in dup:
            p = os.path.join(a.workdir, f"worker_{w:03d}", "STOP_WORKER")
            if os.path.exists(p):
                os.remove(p)
    os.makedirs(os.path.join(a.workdir, "worker_004"), exist_ok=True)
    with open(os.path.join(a.workdir, "worker_004", "PIN"), "w") as fh:
        fh.write("clue owner: seed 94091007 stays GS\n")
    env = {**os.environ, "H668_SCREEN_K": str(a.screen_k)}
    entries = []

    def launch(wid, route, n, seed):
        cmd = [sys.executable, "worker.py", "--route", route,
               "--n", str(n), "--worker-id", str(wid),
               "--seed", str(seed), "--workdir", a.workdir,
               "--pool-cap", str(a.pool_cap), "--batch", str(a.batch),
               "--max-pairs", str(a.max_pairs)]
        p = _spawn_logged(a.workdir, f"worker_{wid:03d}", cmd, env)
        entries.append({"route": route, "worker_id": wid, "seed": seed,
                        "pid": p.pid, "command": " ".join(cmd),
                        "started": time.time(),
                        "screen_k": a.screen_k})
        print(f"  worker {wid:>2}: {route.upper()} seed {seed} "
              f"pid {p.pid}")

    print(f"run-safe: starting {len(tt_ids)} TT / {len(gs)} GS")
    for wid in sorted(gs):
        launch(wid, "gs", a.gs_n, gs[wid])
    for wid in tt_ids:
        launch(wid, "tt", a.tt_n, tt_seed_map[wid])
    if not a.no_autorotate:
        cmd = [sys.executable, "ctl.py", "auto-rotate-gs",
               "--workdir", a.workdir, "--seed-file", a.seeds,
               "--replace-stale", "--watch",
               "--cooldown-minutes", "30"]
        p = _spawn_logged(a.workdir, "auto_rotate", cmd, env)
        entries.append({"route": None, "worker_id": None, "seed": None,
                        "pid": p.pid, "command": " ".join(cmd),
                        "started": time.time(), "screen_k": a.screen_k,
                        "role": "auto-rotate"})
        print(f"  auto-rotate-gs watching (pid {p.pid})")
    if not a.no_dashboard:
        denv = {**env, "H668_WORKDIR": a.workdir,
                "H668_WORKERS": str(len(gs) + len(tt_ids)),
                "H668_SEEDS": a.seeds,
                "H668_EXPECT_GS": str(len(gs)),
                "H668_EXPECT_TT": str(len(tt_ids))}
        p = _spawn_logged(a.workdir, "dashboard",
                          [sys.executable, "kid_dashboard_v3.py"], denv)
        with open(os.path.join(a.workdir, "dashboard.pid"), "w") as fh:
            json.dump({"pid": p.pid, "started": time.time()}, fh)
        entries.append({"route": None, "worker_id": None, "seed": None,
                        "pid": p.pid,
                        "command": "kid_dashboard_v3.py",
                        "started": time.time(), "screen_k": a.screen_k,
                        "role": "dashboard"})
        print(f"  dashboard (pid {p.pid})")
    tmp = _manifest_path(a.workdir) + f".tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump({"written": time.time(), "entries": entries}, fh,
                  indent=1)
    os.replace(tmp, _manifest_path(a.workdir))
    print(f"manifest: {_manifest_path(a.workdir)}")


def pause_safe(a):
    """SIGSTOP workers + auto-rotate + watch-leaderboard. The dashboard
    stays alive (same semantics as its Pause button)."""
    pids = _all_supervised_pids(a.workdir)
    n = _signal_all(pids, signal.SIGSTOP)
    print(f"paused {n} supervised process(es) (dashboard untouched)")


def resume_safe(a):
    pids = _all_supervised_pids(a.workdir)
    n = _signal_all(pids, signal.SIGCONT)
    print(f"resumed {n} supervised process(es)")


def stop_safe(a):
    """Resume anything paused (a SIGSTOPped worker cannot see STOP),
    write STOP, wait for clean exits, then stop auto-rotate,
    watch-leaderboard, and the dashboard."""
    pids = _all_supervised_pids(a.workdir)
    _signal_all(pids, signal.SIGCONT)
    print(f"resumed {len(pids)} process(es) so they can exit cleanly")
    with open(os.path.join(a.workdir, "STOP"), "w") as fh:
        fh.write(f"stop-safe {time.strftime('%F %T')}\n")
    deadline = time.time() + a.wait_minutes * 60
    while time.time() < deadline:
        alive = [w for w in ps_workers(a.workdir)]
        if not any(lock_holder(os.path.join(
                a.workdir, f"worker_{w:03d}"))[0] for w in alive):
            break
        time.sleep(1)
    for pf in ("auto_rotate.pid", "watch_leaderboard.pid",
               "dashboard.pid"):
        path = os.path.join(a.workdir, pf)
        try:
            pid = json.load(open(path)).get("pid")
            if pid:
                os.kill(int(pid), signal.SIGTERM)
                print(f"stopped {pf.split('.')[0]} (pid {pid})")
        except (OSError, ValueError, ProcessLookupError):
            pass
        if os.path.exists(path):
            os.remove(path)
    print("stop-safe complete; worker state preserved")


def status_safe(a):
    man = _read_manifest(a.workdir)
    expect = {e["worker_id"]: e["route"] for e in man["entries"]
              if e.get("worker_id") is not None}
    live = ps_workers(a.workdir)
    mix = {"gs": 0, "tt": 0}
    wrong, missing, dups, paused = [], [], [], []
    for wid in sorted(set(expect) | set(live)):
        d = os.path.join(a.workdir, f"worker_{wid:03d}")
        alive = lock_holder(d)[0]
        route = None
        try:
            route = json.load(open(os.path.join(d, "meta.json")))["route"]
        except (OSError, ValueError, KeyError):
            pass
        if alive and route in mix:
            mix[route] += 1
        if wid in expect and not alive:
            missing.append(wid)
        if len(live.get(wid, [])) > 1:
            dups.append(wid)
        if wid in expect and route and route != expect[wid]:
            wrong.append((wid, route, expect[wid]))
        for pid, _ in live.get(wid, []):
            if _proc_state(pid).startswith("T"):
                paused.append(wid)
    print(f"route mix (alive): {mix['tt']} TT / {mix['gs']} GS "
          f"(manifest expects "
          f"{sum(1 for r in expect.values() if r == 'tt')} TT / "
          f"{sum(1 for r in expect.values() if r == 'gs')} GS)")
    d4 = os.path.join(a.workdir, "worker_004")
    r4 = None
    try:
        r4 = json.load(open(os.path.join(d4, "meta.json")))["route"]
    except (OSError, ValueError, KeyError):
        pass
    pin4 = os.path.exists(os.path.join(d4, "PIN"))
    flag4 = "OK" if (r4 == "gs" and pin4) else "*** WRONG ***"
    print(f"worker 004: route={r4} pinned={pin4} {flag4}")
    try:
        import urllib.request
        port = int(os.environ.get("H668_DASH_PORT", "6680"))
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/routing.json", timeout=3) as r:
            print(f"dashboard: HTTP {r.status} on port {port}")
    except Exception as exc:
        print(f"dashboard: NOT RESPONDING ({type(exc).__name__})")
    ar = {}
    try:
        ar = json.load(open(os.path.join(a.workdir, "auto_rotate.pid")))
    except (OSError, ValueError):
        pass
    ar_alive = False
    if ar.get("pid"):
        try:
            os.kill(int(ar["pid"]), 0)
            ar_alive = True
        except OSError:
            pass
    print(f"auto-rotate: {'watching (pid ' + str(ar['pid']) + ')' if ar_alive else 'not running'}")
    print(f"state: {'PAUSED (workers ' + str(sorted(set(paused))) + ')' if paused else 'running'}")
    print(f"missing workers: {missing or '-'}")
    print(f"duplicate workers: {sorted(set(dups)) or '-'}")
    print(f"wrong-route workers: {wrong or '-'}")
    lb = None
    try:
        lb = json.load(open(os.path.join(a.workdir,
                                         "seed_leaderboard.json")))
    except (OSError, ValueError):
        pass
    if lb:
        age = (time.time() - lb.get("generated", 0)) / 60
        top = lb["seeds"][0] if lb.get("seeds") else None
        print(f"leaderboard: {len(lb.get('seeds', []))} seeds, "
              f"refreshed {age:.0f} min ago"
              + (f"; top seed {top['seed']} "
                 f"(score {top['score']:,.0f})" if top else ""))
    else:
        print("leaderboard: not generated "
              "(run: python3 ctl.py refresh-leaderboard)")


def refresh_leaderboard(a):
    a.cycles = getattr(a, "cycles", 200)
    seed_leaderboard(a)


def watch_leaderboard(a):
    """Refresh the leaderboard every --interval seconds. Plain SIGTERM
    stops it cleanly (pid file removed)."""
    pid_path = os.path.join(a.workdir, "watch_leaderboard.pid")

    def _term(signum, frame):
        try:
            os.remove(pid_path)
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    with open(pid_path, "w") as fh:
        json.dump({"pid": os.getpid(), "interval": a.interval,
                   "started": time.time()}, fh)
    tick = 0
    try:
        while True:
            tick += 1
            print(f"--- leaderboard refresh {tick} "
                  f"({time.strftime('%H:%M:%S')}) ---")
            try:
                refresh_leaderboard(a)
            except SystemExit:
                pass
            if a.max_ticks and tick >= a.max_ticks:
                return
            time.sleep(a.interval)
    finally:
        try:
            os.remove(pid_path)
        except OSError:
            pass


def _compatible_pair_ops(route, pools, bins):
    """Exact count of build/probe pairs whose FIRST PAF coordinate sums
    are mutually compatible (a necessary condition for an exact match):
    wt1*(a0+b0) + wt2*(c0+d0) = 0. Computed via histogram convolution."""
    pafs = [route.paf(pools[b].seqs)[:, 0].astype(np.int64)
            for b in bins]
    w1, w2 = route.wt

    def conv(x, y):
        mn = int(x.min() + y.min()) if len(x) and len(y) else 0
        hx = np.bincount(x - x.min()) if len(x) else np.array([0])
        hy = np.bincount(y - y.min()) if len(y) else np.array([0])
        return np.convolve(hx, hy), mn
    h1, m1 = conv(pafs[0], pafs[1])
    h2, m2 = conv(pafs[2], pafs[3])
    build = probe = 0
    for t1_off, cnt1 in enumerate(h1):
        if not cnt1:
            continue
        t1 = m1 + t1_off
        num = -w1 * t1
        if num % w2:
            continue
        t2 = num // w2
        idx = t2 - m2
        if 0 <= idx < len(h2) and h2[idx]:
            build += int(cnt1)
            probe += int(h2[idx])
    return build, probe


def _micro_enum(route, pools, bins, o, L, frag_pairs, wt1, wt2):
    """Deterministically enumerate v7 micro-units for one key: per
    watermark block, the compatible coord-0 partitions and their
    fragment counts under frag_pairs. Mirrors engine._match_partitioned
    exactly. Returns list of (block, t1, frag, build_pairs, probe_pairs)."""
    import engine
    units = []
    blocks = [((o[0], L[0]), (0, L[1]), (0, L[2]), (0, L[3])),
              ((0, o[0]), (o[1], L[1]), (0, L[2]), (0, L[3])),
              ((0, o[0]), (0, o[1]), (o[2], L[2]), (0, L[3])),
              ((0, o[0]), (0, o[1]), (0, o[2]), (o[3], L[3]))]
    pafs = [route.paf(pools[b].seqs) for b in bins]
    for bi, ((al, ah), (bl, bh), (cl, ch), (dl, dh)) in enumerate(blocks):
        if al >= ah or bl >= bh or cl >= ch or dl >= dh:
            continue
        gA = engine._groups_by_coord(pafs[0][al:ah])
        gB = engine._groups_by_coord(pafs[1][bl:bh])
        gC = engine._groups_by_coord(pafs[2][cl:ch])
        gD = engine._groups_by_coord(pafs[3][dl:dh])
        for t1 in sorted({x + y for x in gA for y in gB}):
            num = -wt1 * t1
            if num % wt2:
                continue
            t2 = num // wt2
            bsegs = [(gA[x], gB[t1 - x]) for x in gA if (t1 - x) in gB]
            psegs = [(gC[x], gD[t2 - x]) for x in gC if (t2 - x) in gD]
            if not bsegs or not psegs:
                continue
            bp = sum(len(x) * len(y) for x, y in bsegs)
            pp = sum(len(x) * len(y) for x, y in psegs)
            nfrag = 1
            if bp > frag_pairs:
                nfrag = 0
                cur = 0
                for x, y in bsegs:
                    step = max(1, frag_pairs // max(len(y), 1))
                    for lo in range(0, len(x), step):
                        segp = len(x[lo:lo + step]) * len(y)
                        if cur + segp > frag_pairs and cur:
                            nfrag += 1
                            cur = 0
                        cur += segp
                if cur:
                    nfrag += 1
            per_frag_bp = bp / nfrag
            for fi in range(nfrag):
                units.append((bi, int(t1), fi, per_frag_bp, pp))
    return units


def match_audit(a):
    """Per-key matching cost audit: what the watermarks cover, what is
    genuinely pending, what the CURRENT engine will actually spend on it
    (including the chunk x full-reprobe multiplication), and what a
    partition-compatible engine would spend. READ-ONLY."""
    import worker as wkm
    import engine
    for d in worker_dirs(a.workdir):
        wid = int(os.path.basename(d).split("_")[1])
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
        except (OSError, ValueError):
            continue
        rname = meta.get("route")
        n = meta.get("n") or (167 if rname == "gs" else 56)
        route = wkm.GSRoute(n) if rname == "gs" else wkm.TTRoute(n)
        ppath = os.path.join(d, "pools.npz")
        spath = os.path.join(d, "matchstate.json")
        if not os.path.exists(ppath):
            continue
        data = np.load(ppath, allow_pickle=False)
        pools = {}
        for b in route.bins:
            bk = wkm.bin_key(b)
            p = engine.Pool(route.lengths[b], route.periodic)
            p.seqs = (data[bk].astype(np.int8) if bk in data.files
                      else np.empty((0, route.lengths[b]), np.int8))
            pools[b] = p
        raw_state = {}
        try:
            raw_state = json.load(open(spath))
        except (OSError, ValueError):
            pass
        opt_ok = raw_state.get("opt_version") == wkm.OPT_VERSION
        st = wkm.MatchState(spath)
        fp_status = "no state file"
        if raw_state:
            try:
                st.load(pools, wkm.bin_key)
                fp_status = ("valid" if st.marks else
                             "DISCARDED (fingerprint/opt mismatch)")
            except Exception as exc:
                fp_status = f"load error: {exc}"
        mtime = (time.strftime("%F %T", time.localtime(
            os.path.getmtime(spath))) if os.path.exists(spath) else "-")
        print(f"\n=== worker {wid} route={rname} n={n} ===")
        print(f"matchstate: opt_version "
              f"{'OK' if opt_ok else 'MISMATCH -> discard'}"
              f" ({raw_state.get('opt_version')} vs {wkm.OPT_VERSION}); "
              f"fingerprints {fp_status}; last update {mtime}")
        micro_data, micro_status = {}, "no micro state"
        if a.micro:
            import worker as wkm2
            mpath = os.path.join(d, "micro_matchstate.json")
            if os.path.exists(mpath):
                try:
                    md = json.load(open(mpath))
                    if md.get("micro_version") != wkm.MICRO_VERSION:
                        micro_status = (f"DISCARDED: micro_version "
                                        f"{md.get('micro_version')} != "
                                        f"{wkm.MICRO_VERSION}")
                    elif md.get("opt_version") != wkm.OPT_VERSION:
                        micro_status = (f"DISCARDED: opt_version "
                                        f"{md.get('opt_version')} != "
                                        f"{wkm.OPT_VERSION}")
                    else:
                        micro_data = md.get("keys", {})
                        micro_status = (f"valid header (frag_pairs "
                                        f"{md.get('frag_pairs'):,})")
                except (OSError, ValueError):
                    micro_status = "DISCARDED: unreadable/corrupt"
            print(f"micro state: {micro_status}")
        tot_cur = tot_part = 0
        for pi, pat in enumerate(route.patterns):
            for si, ((b1, b2), (b3, b4)) in enumerate(route.splits(pat)):
                bins = (b1, b2, b3, b4)
                L = [len(pools[b].seqs) for b in bins]
                if min(L) == 0:
                    continue
                key = f"{route.name}|{pi}|{si}"
                got = st.get(key, [str(b) for b in bins])
                o = [got[str(b)] for b in bins]
                tri = (b1 == b2, b3 == b4)
                blocks = [
                    ((o[0], L[0]), (0, L[1]), (0, L[2]), (0, L[3])),
                    ((0, o[0]), (o[1], L[1]), (0, L[2]), (0, L[3])),
                    ((0, o[0]), (0, o[1]), (o[2], L[2]), (0, L[3])),
                    ((0, o[0]), (0, o[1]), (0, o[2]), (o[3], L[3]))]
                pend_build = pend_probe = cur_ops = 0
                eff = max(1, min(4000, a.max_pairs // max(L[1], 1)))
                for (al, ah), (bl, bh), (cl, ch), (dl, dh) in blocks:
                    if al >= ah or bl >= bh or cl >= ch or dl >= dh:
                        continue
                    bp = (ah - al) * (bh - bl)
                    pp = (ch - cl) * (dh - dl)
                    chunks = -(-(ah - al) // eff)
                    pend_build += bp
                    pend_probe += pp
                    cur_ops += bp + chunks * pp
                total_quads = L[0] * L[1] * L[2] * L[3]
                covered = o[0] * o[1] * o[2] * o[3]
                cb, cp = _compatible_pair_ops(route, pools, bins)
                part_ops = min(cb, pend_build) + min(cp, pend_probe) \
                    if pend_build else 0
                changed = [f"{bins[x]}:{o[x]}->{L[x]}"
                           for x in range(4) if L[x] > o[x]]
                if pend_build == 0:
                    reason = ("fully covered -- unchanged/capped key, "
                              "correctly NOT rematched")
                elif max(o) == 0:
                    reason = "BASELINE never completed for this key"
                else:
                    reason = f"growth since watermark: {changed}"
                tot_cur += cur_ops
                tot_part += part_ops
                print(f"key {key} bins={bins} sizes={L} marks={o} "
                      f"tri={tri}")
                print(f"    quads covered {100*covered/max(total_quads,1):.1f}%"
                      f"; pending build {pend_build:,} + probe "
                      f"{pend_probe:,} pair-ops")
                print(f"    CURRENT engine true ops (chunk x reprobe): "
                      f"{cur_ops:,}")
                print(f"    partition-compatible ops (coord-0 exact "
                      f"condition): {part_ops:,}")
                print(f"    reason: {reason}")
                if a.micro and pend_build:
                    fp_default = int(float(os.environ.get(
                        "H668_MICRO_TARGET_SECS", "300"))
                        * float(os.environ.get("H668_MICRO_RATE",
                                               "12e6")))
                    ent = micro_data.get(key, {})
                    frag_pairs = int(ent.get("frag_pairs", 0)) or \
                        fp_default
                    units = _micro_enum(route, pools, bins, o, L,
                                        frag_pairs, *route.wt)
                    kstat, kreason = "fresh (no units)", None
                    done_set = set()
                    if ent:
                        fp_ok = all(
                            len(pools[bb].seqs) == ent["fp"][str(bb)]
                            ["len"] and wkm._fingerprint(
                                pools[bb].seqs) == ent["fp"][str(bb)]
                            ["hash"] for bb in bins
                            if str(bb) in ent.get("fp", {}))
                        if not fp_ok:
                            kstat, kreason = "DISCARDED", \
                                "fingerprint mismatch (pool mutated/grew)"
                        elif ent.get("ranges") != [
                                list(sum(bx, ())) for bx in [
                                    ((o[0], L[0]), (0, L[1]), (0, L[2]),
                                     (0, L[3])),
                                    ((0, o[0]), (o[1], L[1]), (0, L[2]),
                                     (0, L[3])),
                                    ((0, o[0]), (0, o[1]), (o[2], L[2]),
                                     (0, L[3])),
                                    ((0, o[0]), (0, o[1]), (0, o[2]),
                                     (o[3], L[3]))]]:
                            kstat, kreason = "DISCARDED", \
                                "block ranges changed"
                        else:
                            done_set = {u for u, v in
                                        ent.get("units", {}).items()
                                        if v.get("done")}
                            kstat = "valid"
                    pend_units = [u for u in units
                                  if f"b{u[0]}|{u[1]}:{u[2]}"
                                  not in done_set]
                    big = max(pend_units,
                              key=lambda u: u[3] + u[4],
                              default=None)
                    rate2 = 14e6
                    print(f"    micro: {kstat}"
                          + (f" ({kreason})" if kreason else "")
                          + f"; units total {len(units)}, done "
                          f"{len(units) - len(pend_units)}, pending "
                          f"{len(pend_units)}")
                    if big:
                        est = (big[3] + big[4]) / rate2
                        print(f"    largest pending unit: block "
                              f"{big[0]} t1={big[1]} frag {big[2]} "
                              f"~{big[3] + big[4]:,.0f} ops "
                              f"~{est:,.0f}s"
                              + ("  ** UNIT TOO LARGE TO CHECKPOINT "
                                 "WITHIN 10 MIN **" if est > 600
                                 else ""))
        rate = 14e6
        print(f"worker {wid} TOTALS: current engine {tot_cur:,} ops "
              f"(~{tot_cur/rate/3600:.1f} h at 14M/s); partitioned "
              f"{tot_part:,} ops (~{tot_part/rate/3600:.2f} h)")


def verify(a):
    H = np.loadtxt(a.csv, delimiter=",", dtype=np.int64)
    N = core.verify_hadamard(H)
    print(f"OK: {a.csv} is a verified {N}x{N} Hadamard matrix "
          f"(entries +-1, H H^T = {N} I).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "launch-missing"):
        p = sub.add_parser(name)
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--tt-share", type=float, default=2 / 3)
        p.add_argument("--gs-n", type=int, default=167)
        p.add_argument("--tt-m", type=int, default=56)
        p.add_argument("--seed", type=int, default=20260706)
        p.add_argument("--workdir", default="work")
        p.add_argument("--pool-cap", type=int, default=20000)
        p.add_argument("--max-pairs", type=int, default=20_000_000)
        p.add_argument("--wait", action="store_true")
    for name in ("monitor", "watch", "stop", "merge", "status",
                 "clean-bad", "kill-duplicates", "migrate"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default="work")
        if name == "monitor":
            p.add_argument("--once", action="store_true")
    pa = sub.add_parser("acceptance")
    pa.add_argument("--workdir", default="work")
    pa.add_argument("--cycles", type=int, default=50)
    pe = sub.add_parser("experiment")
    pe.add_argument("--workdir", default="work")
    pe.add_argument("--route", default="gs", choices=["gs", "tt"])
    pe.add_argument("--n", type=int, default=167)
    pe.add_argument("--seeds", type=int, default=3)
    pe.add_argument("--base-seed", type=int, default=90000000)
    pe.add_argument("--minutes", type=float, default=3)
    pe.add_argument("--pool-cap", type=int, default=20000)
    pe.add_argument("--batch", type=int, default=200000)
    pr = sub.add_parser("recommend-routes")
    pr.add_argument("--workdir", default="work")
    pr.add_argument("--stale-hours", type=float, default=3.0)
    pg = sub.add_parser("rotate-gs-seeds")
    pg.add_argument("--workdir", default="work")
    pg.add_argument("--seeds", required=True)
    pg.add_argument("--replace-workers", required=True)
    pg.add_argument("--gs-n", type=int, default=167)
    pg.add_argument("--pool-cap", type=int, default=100000)
    pg.add_argument("--max-pairs", type=int, default=20_000_000)
    pg.add_argument("--wait-minutes", type=float, default=10)
    pr2 = sub.add_parser("auto-rotate-gs")
    pr2.add_argument("--workdir", default="work")
    pr2.add_argument("--seed-file", required=True)
    pr2.add_argument("--replace-stale", action="store_true")
    pr2.add_argument("--stale-cycles", type=int, default=10)
    pr2.add_argument("--dup-threshold", type=float, default=95.0)
    pr2.add_argument("--min-accepted-per-min", type=float, default=1.0)
    pr2.add_argument("--cooldown-minutes", type=float, default=30.0)
    pr2.add_argument("--max-rotations", type=int, default=4)
    pr2.add_argument("--dry-run", action="store_true")
    pr2.add_argument("--once", action="store_true")
    pr2.add_argument("--watch", action="store_true")
    pr2.add_argument("--watch-interval-seconds", type=float, default=600.0)
    pr2.add_argument("--max-ticks", type=int, default=0,
                     help="watch mode: stop after N ticks (0 = forever)")
    pr2.add_argument("--gs-n", type=int, default=167)
    pr2.add_argument("--pool-cap", type=int, default=100000)
    pr2.add_argument("--max-pairs", type=int, default=20_000_000)
    pr2.add_argument("--wait-minutes", type=float, default=10)
    for name in ("replay-collision", "replay-cycle"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default="work")
        p.add_argument("--worker-id", type=int, required=True)
        p.add_argument("--cycle", type=int, default=None,
                       required=(name == "replay-cycle"))
        p.add_argument("--max-pairs", type=int, default=20_000_000)
    ps2 = sub.add_parser("seed-leaderboard")
    ps2.add_argument("--workdir", default="work")
    ps2.add_argument("--cycles", type=int, default=200)
    prs = sub.add_parser("run-safe")
    prs.add_argument("--workdir", default="work")
    prs.add_argument("--seeds", required=True)
    prs.add_argument("--gs", default=APPROVED_GS,
                     help="wid:seed pairs, comma separated")
    prs.add_argument("--tt-workers", default=APPROVED_TT)
    prs.add_argument("--gs-n", type=int, default=167)
    prs.add_argument("--tt-n", type=int, default=56)
    prs.add_argument("--screen-k", type=int, default=4)
    prs.add_argument("--pool-cap", type=int, default=100000)
    prs.add_argument("--batch", type=int, default=200000)
    prs.add_argument("--max-pairs", type=int, default=20_000_000)
    prs.add_argument("--replace", action="store_true")
    prs.add_argument("--wait-minutes", type=float, default=10)
    prs.add_argument("--no-dashboard", action="store_true")
    prs.add_argument("--no-autorotate", action="store_true")
    for name in ("stop-safe", "pause-safe", "resume-safe",
                 "status-safe", "refresh-leaderboard"):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default="work")
        if name == "stop-safe":
            p.add_argument("--wait-minutes", type=float, default=15)
        if name == "refresh-leaderboard":
            p.add_argument("--cycles", type=int, default=200)
    pwl = sub.add_parser("watch-leaderboard")
    pwl.add_argument("--workdir", default="work")
    pwl.add_argument("--interval", type=float, default=600)
    pwl.add_argument("--cycles", type=int, default=200)
    pwl.add_argument("--max-ticks", type=int, default=0)
    pma = sub.add_parser("match-audit")
    pma.add_argument("--workdir", default="work")
    pma.add_argument("--max-pairs", type=int, default=20_000_000)
    pma.add_argument("--micro", action="store_true")
    pv = sub.add_parser("verify")
    pv.add_argument("csv")
    a = ap.parse_args()
    {"launch": launch_missing, "launch-missing": launch_missing,
     "status": status, "monitor": monitor, "watch": watch, "stop": stop,
     "merge": merge, "clean-bad": clean_bad, "migrate": migrate,
     "kill-duplicates": kill_duplicates, "verify": verify,
     "acceptance": acceptance, "experiment": experiment,
     "recommend-routes": recommend_routes,
     "rotate-gs-seeds": rotate_gs_seeds,
     "auto-rotate-gs": auto_rotate_gs,
     "replay-collision": replay_collision,
     "replay-cycle": replay_cycle,
     "seed-leaderboard": seed_leaderboard,
     "run-safe": run_safe, "stop-safe": stop_safe,
     "pause-safe": pause_safe, "resume-safe": resume_safe,
     "status-safe": status_safe,
     "refresh-leaderboard": refresh_leaderboard,
     "watch-leaderboard": watch_leaderboard,
     "match-audit": match_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
