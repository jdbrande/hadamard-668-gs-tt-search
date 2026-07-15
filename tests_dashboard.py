"""Dashboard routing-panel tests. Run: python3 tests_dashboard.py"""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import importlib
import json
import os
import shutil
import subprocess
import sys
import time

import worker as wk

PASS = 0
PY = sys.executable
WD = "dash_work"


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


def mk_worker(wid, route, seed, cycles=12, accepted=0, dup=5000,
              status="running", phase="", eta=None, bins_at_cap=0):
    d = os.path.join(WD, f"worker_{wid:03d}")
    os.makedirs(d, exist_ok=True)
    json.dump({"worker_id": wid, "route": route, "seed": seed},
              open(os.path.join(d, "meta.json"), "w"))
    rows = [json.dumps({"t": 0, "cycle": c + 1, "secs": 60.0,
                        "status": "running", "pools": {"1": 500},
                        "pool_cap": 100000, "generated": 100000,
                        "accepted": accepted, "dup_rejected": dup,
                        "build_psd_rej": 90000, "pairs_hashed": 10,
                        "probes": 10, "probe_psd_rej": 0,
                        "psd_rejected": 0, "collisions": 0,
                        "buckets": 10}) for c in range(cycles)]
    if rows:
        open(os.path.join(d, "cycles.jsonl"), "w").write(
            "\n".join(rows) + "\n")
    lc = {"phase": phase}
    if eta:
        lc["eta_key_seconds"] = eta
    wk.write_json_atomic(os.path.join(d, "progress.json"),
                         {"route": route, "updated": time.time(),
                          "cycle": cycles, "status": status,
                          "bins_at_cap": bins_at_cap,
                          "totals": {"accepted": accepted * cycles,
                                     "dup_rejected": dup * cycles},
                          "telemetry_started_at": time.time() - 3600,
                          "pools": {"1": 500}, "pool_cap": 100000,
                          "latest_cycle": lc})


def snapshot(root):
    out = {}
    for base, _, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime)
    return out


shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
mk_worker(0, "gs", 94054011, accepted=200, dup=10,
          status="matching", phase="matching key 3/30 block 1/4",
          eta=5400)
mk_worker(1, "gs", 94041003, accepted=180, dup=10)
mk_worker(2, "gs", 20260706, accepted=0, dup=5000)   # duplicate-saturated
mk_worker(3, "tt", 555, cycles=0, status="running")  # silent TT
open("dash_seeds.txt", "w").write(
    "94054011\n94041003\n94000001\n94047010\n94025015\n")
rots = [{"timestamp": time.time() - 100, "old_worker_id": 2,
         "old_seed": 111, "new_seed": 94000001,
         "reason": "PROVEN STALE: test", "old_accepted_per_min": 0.0,
         "old_dup_pct": 100.0, "old_cycle": 40}]
open(os.path.join(WD, "seed_rotations.jsonl"), "w").write(
    "\n".join(json.dumps(r) for r in rots) + "\n")

os.environ.update(H668_WORKDIR=WD, H668_WORKERS="4",
                  H668_SEEDS="dash_seeds.txt", H668_DASH_CACHE_TTL="0")
sys.path.insert(0, ".")
import dash_routing as dr
importlib.reload(dr)

# ---- 1. route mix card --------------------------------------------------------
mix = dr.route_mix()
check("route mix counts GS/TT/total",
      mix["gs"]["count"] == 3 and mix["tt"]["count"] == 1
      and mix["total"] == 4)
check("route mix reports accepted/min by route",
      mix["gs"]["apm"] > 300 and mix["tt"]["apm"] == 0)

# ---- 2. worker table fields ---------------------------------------------------
rows = {r["id"]: r for r in mix["workers"]}
check("worker rows show route, seed, acc/min, dup%, bins_at_cap",
      rows[0]["route"] == "GS" and rows[0]["seed"] == 94054011
      and rows[0]["accepted_per_min"] > 100
      and rows[2]["dup_pct"] == 100.0 and rows[2]["bins_at_cap"] == 0)
check("status mapping: matching / accepting / duplicate-saturated",
      rows[0]["status"] == "matching" and rows[1]["status"] == "accepting"
      and rows[2]["status"] == "duplicate-saturated")
check("phase and key ETA surfaced",
      rows[0]["phase"].startswith("matching key")
      and rows[0]["eta_key_seconds"] == 5400)

# ---- 3. stale field always present in dashboard progress() --------------------
import kid_dashboard_v3 as dash
importlib.reload(dash)
for wid in (0, 3):
    check(f"progress() always carries a stale field (worker {wid})",
          "stale" in dash.progress(wid))
os.makedirs(os.path.join(WD, "worker_009"))
check("progress() carries stale even with no telemetry at all",
      "stale" in dash.progress(9))

# ---- 4. auto-rotate pid detection ---------------------------------------------
st = dr.autorotate_status()
check("no pid file -> watch not running", st["watch_running"] is False)
json.dump({"pid": os.getpid(), "cooldown_minutes": 30,
           "started": time.time()},
          open(os.path.join(WD, "auto_rotate.pid"), "w"))
dr._cache.clear()
st = dr.autorotate_status()
check("live pid detected as watch mode active",
      st["watch_running"] is True and st["pid"] == os.getpid()
      and st["rotations_today"] == 1
      and st["cooldown_remaining_seconds"] > 0
      and "PROVEN STALE" in st["latest_reason"])
json.dump({"pid": 99999999, "cooldown_minutes": 30},
          open(os.path.join(WD, "auto_rotate.pid"), "w"))
dr._cache.clear()
check("dead pid -> watch not running",
      dr.autorotate_status()["watch_running"] is False)

# ---- 5. seed queue -------------------------------------------------------------
sq = dr.seed_queue()
check("seed queue counts remaining and lists next unused",
      sq["total_in_file"] == 5 and sq["remaining"] == 2
      and sq["next_unused"] == [94047010, 94025015])
check("seed queue separates live-worker seeds from rotation-log seeds",
      94054011 in sq["used_by_live_workers"]
      and 94000001 in sq["used_in_rotation_log"])
check("seed-low badge when fewer than 20 remain", sq["low"] is True)

# ---- 6. rotation history parsing ------------------------------------------------
hist = dr.rotation_history(10)
check("rotation history parses all required columns",
      len(hist) == 1 and hist[0]["worker_id"] == 2
      and hist[0]["old_seed"] == 111 and hist[0]["new_seed"] == 94000001
      and hist[0]["old_dup_pct"] == 100.0 and hist[0]["old_cycle"] == 40)

# ---- 7. recommendation + badges + panel render ----------------------------------
# make the duplicate-saturated worker ALIVE (hold its lock) so the
# assessment path -- which mirrors what auto-rotate would actually do --
# can classify it as a rotation candidate
check("test process holds worker_002 lock (simulates alive worker)",
      wk.acquire_worker_lock(os.path.join(WD, "worker_002")))
dr._cache.clear()
rec = dr.recommendation()
check("recommendation flags stale GS candidates",
      "GS stale candidates found" in rec and "2" in rec)
html_out = dr.render_routing_html()
check("routing panel renders cards, table, history, badges",
      "Route Mix" in html_out and "Auto-Rotate" in html_out
      and "Seed Queue" in html_out and "Rotation history" in html_out
      and "duplicate-saturated" in html_out)
page_html = dash.page()
check("main dashboard page embeds the routing panel",
      "Routing &amp; Auto-Rotation" in page_html)

# ---- 8. endpoints are READ-ONLY --------------------------------------------------
before = snapshot(WD)
_ = dr.routing_payload()
rs = dr.rotate_status_payload()
_ = dr.render_routing_html()
after = snapshot(WD)
check("routing/rotate-status/render make ZERO filesystem changes",
      before == after)
check("rotate-status is an explicit read-only dry-run preview",
      rs["read_only"] is True and "never rotates" in rs["note"]
      and any(p["worker_id"] == 2 and "PROVEN STALE" in p["reason"]
              for p in rs["would_rotate"]))
check("rotate-status lists skips with reasons",
      any(s["worker_id"] == 1 for s in rs["skips"]))
src = open("dash_routing.py").read()
check("dash_routing contains no write/remove/spawn calls",
      all(tok not in src for tok in ('open(pid_path, "w"', '"w")',
                                     "'w')", "os.remove", "os.replace",
                                     "Popen", "os.kill(int(pid), 9")))

shutil.rmtree(WD, ignore_errors=True)
os.remove("dash_seeds.txt")
print(f"\nALL {PASS} DASHBOARD TESTS PASSED")
