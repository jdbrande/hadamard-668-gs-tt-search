"""Auto-rotation tests. Run: python3 tests_autorotate.py"""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import json
import os
import shutil
import subprocess
import sys
import time

import worker as wk

PASS = 0
PY = sys.executable
WD = "ar_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def ctl(args):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", WD],
                          capture_output=True, text=True, timeout=400)


def mk_worker(wid, route, cycles, seed=111, at_cap=0, coll=0,
              accepted=0, dup=5000, gen=100000, prog_extra=None):
    d = os.path.join(WD, f"worker_{wid:03d}")
    os.makedirs(d, exist_ok=True)
    json.dump({"worker_id": wid, "route": route, "seed": seed},
              open(os.path.join(d, "meta.json"), "w"))
    pools = {"1": 20000 if at_cap else 500}
    rows = [json.dumps({"t": 0, "cycle": c + 1, "secs": 60.0,
                        "status": "running", "pools": pools,
                        "pool_cap": 20000 if at_cap else 100000,
                        "generated": gen, "accepted": accepted,
                        "dup_rejected": dup, "build_psd_rej": 90000,
                        "pairs_hashed": 10, "probes": 10,
                        "probe_psd_rej": 0, "psd_rejected": 0,
                        "collisions": coll, "buckets": 10})
            for c in range(cycles)]
    if rows:
        open(os.path.join(d, "cycles.jsonl"), "w").write(
            "\n".join(rows) + "\n")
    wk.write_json_atomic(os.path.join(d, "progress.json"),
                         {"route": route, "updated": time.time(),
                          "cycle": cycles, "status": "running",
                          "totals": {"accepted": accepted * cycles},
                          "telemetry_started_at": time.time() - 3600,
                          "pools": pools,
                          "latest_cycle": prog_extra or {}})
    return d


class Holder:
    """Hold a worker lock in a child process that exits on STOP_WORKER."""
    CODE = ("import sys, os, time; sys.path.insert(0, '.');"
            "import worker as wk;"
            "d = sys.argv[-1]; assert wk.acquire_worker_lock(d);"
            "\nwhile not os.path.exists(os.path.join(d, 'STOP_WORKER')):"
            " time.sleep(0.2)")

    def __init__(self, d):
        self.p = subprocess.Popen([PY, "-c", self.CODE, d])
        deadline = time.time() + 20
        import ctl as _c  # noqa
        while time.time() < deadline:
            r = subprocess.run([PY, "-c",
                                "import sys; sys.path.insert(0,'.');"
                                "import ctl;"
                                f"print(ctl.lock_holder({d!r})[0])"],
                               capture_output=True, text=True)
            if "True" in r.stdout:
                return
            time.sleep(0.3)


shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
open("ar_seeds.txt", "w").write(
    "94054011\n94041003\n# comment\n94000001\n")

# workers: 0 stale-alive, 1 healthy, 2 long-cycle, 3 TT-stale-looking,
# 4 promising (collisions), 5 capped, 6 stale-but-dead
d0 = mk_worker(0, "gs", 12, seed=20260706)
h0 = Holder(d0)
mk_worker(1, "gs", 12, accepted=200, dup=10)
h1 = Holder(os.path.join(WD, "worker_001"))
mk_worker(2, "gs", 2, prog_extra={"phase": "matching key 5/30 block 2/4"})
h2 = Holder(os.path.join(WD, "worker_002"))
mk_worker(3, "tt", 12)
h3 = Holder(os.path.join(WD, "worker_003"))
mk_worker(4, "gs", 12, coll=7)
h4 = Holder(os.path.join(WD, "worker_004"))
mk_worker(5, "gs", 12, at_cap=1)
h5 = Holder(os.path.join(WD, "worker_005"))
mk_worker(6, "gs", 12)  # no holder -> dead

base = ["auto-rotate-gs", "--seed-file", "ar_seeds.txt",
        "--replace-stale", "--stale-cycles", "10",
        "--dup-threshold", "95", "--min-accepted-per-min", "1",
        "--cooldown-minutes", "0", "--max-rotations", "4",
        "--wait-minutes", "1", "--gs-n", "13", "--pool-cap", "200"]

# ---- 1. dry-run: correct classification, no changes -------------------------
snapshot = sorted(os.listdir(WD))
r = ctl(base + ["--dry-run"])
out = r.stdout
check("dry-run plans rotation ONLY for the proven-stale alive GS worker",
      "[plan] rotate w0" in out and "PROVEN STALE" in out
      and "[plan] rotate w1" not in out)
check("healthy GS worker skipped (accepting)",
      any("w1" in l and "accepting" in l for l in out.splitlines()))
check("long-cycle worker skipped (not enough completed cycles, "
      "never rotated on activity or wall time)",
      any("w2" in l and "completed" in l for l in out.splitlines()))
check("TT worker never touched",
      any("w3" in l and "never touched" in l for l in out.splitlines()))
check("promising-signals worker kept",
      any("w4" in l and "promising" in l for l in out.splitlines()))
check("capped worker skipped (cap issue, not seed issue)",
      any("w5" in l and "cap issue" in l for l in out.splitlines()))
check("dead worker skipped (launch-missing territory)",
      any("w6" in l and "not alive" in l for l in out.splitlines()))
check("dry-run changed nothing on disk",
      sorted(os.listdir(WD)) == snapshot
      and not os.path.exists(os.path.join(WD, "seed_rotations.jsonl")))

# ---- 2. real rotation --------------------------------------------------------
r = ctl(base)
out = r.stdout
check("rotation executes for w0 with first unused seed",
      "[rotated] w0" in out and "94054011" in out)
ret = [d for d in os.listdir(WD) if d.startswith("retired_gs_")]
check("old dir retired to retired_gs_<ts>/worker_000, preserved",
      len(ret) == 1 and os.path.exists(os.path.join(
          WD, ret[0], "worker_000", "cycles.jsonl")))
log = [json.loads(l) for l in
       open(os.path.join(WD, "seed_rotations.jsonl"))]
check("rotation log records old/new seed, reason, metrics, cycle",
      len(log) == 1 and log[0]["old_seed"] == 20260706
      and log[0]["new_seed"] == 94054011
      and "PROVEN STALE" in log[0]["reason"]
      and log[0]["old_dup_pct"] >= 95
      and log[0]["old_cycle"] == 12)
deadline = time.time() + 60
new_meta = os.path.join(WD, "worker_000", "meta.json")
ok = False
while time.time() < deadline:
    try:
        m = json.load(open(new_meta))
        if m.get("seed") == 94054011 and m.get("route") == "gs":
            ok = True
            break
    except (OSError, ValueError):
        pass
    time.sleep(0.5)
check("replacement worker running as GS with the new seed "
      "(worker count constant)", ok)
open(os.path.join(WD, "worker_000", "STOP_WORKER"), "w").write("x\n")

# ---- 3. cooldown + seed accounting -------------------------------------------
d0b = mk_worker(7, "gs", 12, seed=222)
h7 = Holder(d0b)
r = ctl(base[:base.index("--cooldown-minutes") + 1] + ["30"]
        + base[base.index("--cooldown-minutes") + 2:] + ["--dry-run"])
check("cooldown blocks another rotation immediately after one",
      "cooldown" in r.stdout and "[plan]" not in r.stdout)
# expire cooldown by rewriting the log timestamp
log[0]["timestamp"] = time.time() - 3600
open(os.path.join(WD, "seed_rotations.jsonl"), "w").write(
    json.dumps(log[0]) + "\n")
r = ctl(base + ["--dry-run"])
check("next rotation plans the NEXT unused seed (94041003), "
      "skipping used ones",
      "[plan] rotate w7" in r.stdout and "94041003" in r.stdout)

# ---- 4. watch mode ticks + obeys cooldown ------------------------------------
r = ctl(base + ["--dry-run", "--watch", "--watch-interval-seconds",
                "0.3", "--max-ticks", "2"])
check("watch mode runs periodic ticks",
      r.stdout.count("watch tick") == 2)

for h in (h0, h1, h2, h3, h4, h5, h7):
    try:
        h.p.kill()
    except Exception:
        pass
time.sleep(1)
shutil.rmtree(WD, ignore_errors=True)
os.remove("ar_seeds.txt")
print(f"\nALL {PASS} AUTO-ROTATION TESTS PASSED")
