"""Supervisor tests. Run: python3 tests_supervisor.py"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

PASS = 0
PY = sys.executable
WD = "sup_work"
sys.path.insert(0, ".")
import ctl as ctlmod  # noqa: E402


def check(name, cond):
    global PASS
    if not cond:
        print(f"FAIL  {name}")
        sys.exit(1)
    PASS += 1
    print(f"ok    {name}")


def ctl(args, timeout=300):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", WD],
                          capture_output=True, text=True, timeout=timeout)


def alive(wid):
    return ctlmod.lock_holder(os.path.join(WD, f"worker_{wid:03d}"))[0]


def wait_for(cond, secs=45):
    deadline = time.time() + secs
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.5)
    return False


shutil.rmtree(WD, ignore_errors=True)
os.makedirs(WD)
open("sup_seeds.txt", "w").write("95100001\n95100002\n95100003\n")
BASE = ["run-safe", "--seeds", "sup_seeds.txt", "--gs-n", "41",
        "--tt-n", "26", "--pool-cap", "500", "--batch", "20000",
        "--max-pairs", "500000", "--no-dashboard", "--no-autorotate"]

# ---- 1. refusals: worker 004 must be GS -------------------------------------
r = ctl(BASE + ["--gs", "0:111", "--tt-workers", "1"])
check("run-safe refuses when worker 004 is not GS",
      r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr)
      and "004" in (r.stdout + r.stderr))
r = ctl(BASE + ["--gs", "0:111,4:222", "--tt-workers", "4,1"])
check("run-safe refuses when worker 004 is listed as TT",
      r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr))
check("refusal left no manifest and no PIN",
      not os.path.exists(os.path.join(WD, "run_manifest.json"))
      and not os.path.exists(os.path.join(WD, "worker_004", "PIN")))

# ---- 2-3. successful launch: PIN, manifest, exact ids/seeds ------------------
r = ctl(BASE + ["--gs", "0:94054003,4:94091007", "--tt-workers", "1,2"])
check("run-safe launches the requested mix", r.returncode == 0
      and "2 TT / 2 GS" in r.stdout)
check("run-safe creates worker_004/PIN",
      os.path.exists(os.path.join(WD, "worker_004", "PIN")))
man = json.load(open(os.path.join(WD, "run_manifest.json")))
got = {(e["worker_id"], e["route"], e["seed"]) for e in man["entries"]}
check("manifest records exact worker IDs, routes, and seeds",
      {(0, "gs", 94054003), (4, "gs", 94091007), (1, "tt", 95100001),
       (2, "tt", 95100002)} == got)
check("manifest records screen_k=4, pid, command, started",
      all(e["screen_k"] == 4 and e["pid"] and "worker.py" in e["command"]
          and e["started"] for e in man["entries"]))
check("timestamped logs written",
      any(f.startswith("worker_000-") for f in
          os.listdir(os.path.join(WD, "logs"))))
check("all four workers took their locks",
      wait_for(lambda: all(alive(w) for w in (0, 1, 2, 4))))

# ---- 4. duplicate refusal / --replace ----------------------------------------
r = ctl(BASE + ["--gs", "0:94054003,4:94091007", "--tt-workers", "1,2"])
check("run-safe refuses duplicate launch without --replace",
      r.returncode != 0 and "already running" in (r.stdout + r.stderr))
old_pids = {e["worker_id"]: e["pid"] for e in man["entries"]}
r = ctl(BASE + ["--gs", "0:94054003,4:94091007", "--tt-workers", "1,2",
                "--replace", "--wait-minutes", "3"], timeout=400)
man2 = json.load(open(os.path.join(WD, "run_manifest.json")))
new_pids = {e["worker_id"]: e["pid"] for e in man2["entries"]}
check("--replace stops and relaunches with fresh pids",
      r.returncode == 0 and all(new_pids[w] != old_pids[w]
                                for w in (0, 1, 2, 4)))
check("replaced workers are alive again",
      wait_for(lambda: all(alive(w) for w in (0, 1, 2, 4))))

# ---- 5. pause-safe leaves the dashboard alone --------------------------------
dash = subprocess.Popen([PY, "-c", "import time; time.sleep(300)"])
json.dump({"pid": dash.pid}, open(os.path.join(WD, "dashboard.pid"), "w"))
r = ctl(["pause-safe"])
def state(pid):
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                          capture_output=True, text=True).stdout.strip()
wpids = [p for lst in ctlmod.ps_workers(WD).values() for p, _ in lst]
check("pause-safe SIGSTOPs all workers",
      wpids and wait_for(lambda: all(state(p).startswith("T")
                                     for p in wpids), 15))
check("pause-safe does NOT pause the dashboard",
      not state(dash.pid).startswith("T"))
r = ctl(["resume-safe"])
check("resume-safe wakes the workers",
      wait_for(lambda: all(not state(p).startswith("T")
                           for p in wpids), 15))

# ---- 6. stop-safe resumes paused workers so they can exit --------------------
ctl(["pause-safe"])
wait_for(lambda: all(state(p).startswith("T") for p in wpids), 15)
r = ctl(["stop-safe", "--wait-minutes", "3"], timeout=400)
check("stop-safe resumed paused workers first (message + clean exits)",
      "resumed" in r.stdout
      and wait_for(lambda: not any(alive(w) for w in (0, 1, 2, 4)), 30))
check("stop-safe wrote STOP and terminated the dashboard",
      os.path.exists(os.path.join(WD, "STOP"))
      and wait_for(lambda: dash.poll() is not None, 15)
      and not os.path.exists(os.path.join(WD, "dashboard.pid")))

# ---- 7-9. status-safe detections ----------------------------------------------
r = ctl(["status-safe"])
check("status-safe detects missing workers after stop",
      "missing workers" in r.stdout
      and all(str(w) in r.stdout.split("missing workers:")[1]
              .splitlines()[0] for w in (0, 1, 2, 4)))
meta4 = os.path.join(WD, "worker_004", "meta.json")
m = json.load(open(meta4))
m["route"] = "tt"
json.dump(m, open(meta4, "w"))
r = ctl(["status-safe"])
check("status-safe detects worker 004 wrong route",
      "*** WRONG ***" in r.stdout)
check("status-safe reports route mix vs manifest expectation",
      "route mix" in r.stdout and "manifest expects" in r.stdout)
m["route"] = "gs"
json.dump(m, open(meta4, "w"))
decoys = [subprocess.Popen(
    [PY, "-c", "import time; time.sleep(60)", "worker.py",
     "--worker-id", "1", "--workdir", WD]) for _ in range(2)]
time.sleep(1)
r = ctl(["status-safe"])
for p in decoys:
    p.kill()
check("status-safe detects duplicate worker pids",
      "duplicate workers: [1]" in r.stdout)

# ---- 10. refresh + watch leaderboard -------------------------------------------
lb_path = os.path.join(WD, "seed_leaderboard.json")
if os.path.exists(lb_path):
    os.remove(lb_path)
r = ctl(["refresh-leaderboard"])
check("refresh-leaderboard writes seed_leaderboard.json",
      os.path.exists(lb_path)
      and json.load(open(lb_path))["schema_version"] == 1)
r = ctl(["watch-leaderboard", "--interval", "0.3", "--max-ticks", "2"])
check("watch-leaderboard ticks on the interval",
      r.stdout.count("leaderboard refresh") == 2)
p = subprocess.Popen([PY, "ctl.py", "watch-leaderboard", "--workdir", WD,
                      "--interval", "60"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
wait_for(lambda: os.path.exists(os.path.join(
    WD, "watch_leaderboard.pid")), 15)
p.send_signal(signal.SIGTERM)
check("watch-leaderboard stops cleanly on SIGTERM and removes its pid "
      "file", wait_for(lambda: p.poll() is not None, 15)
      and p.returncode == 0
      and not os.path.exists(os.path.join(WD, "watch_leaderboard.pid")))

# ---- 11-12. dashboard warnings --------------------------------------------------
lb = json.load(open(lb_path))
lb["generated"] = time.time() - 3600
json.dump(lb, open(lb_path, "w"))
os.environ.update(H668_WORKDIR=WD, H668_WORKERS="4",
                  H668_DASH_CACHE_TTL="0", H668_SEEDS="sup_seeds.txt",
                  H668_EXPECT_GS="2", H668_EXPECT_TT="10")
import importlib
import dash_routing as dr
importlib.reload(dr)
html_out = dr.render_routing_html()
check("dashboard warns when leaderboard is older than 15 minutes",
      "LEADERBOARD STALE" in html_out)
m["route"] = "tt"
json.dump(m, open(meta4, "w"))
dr._cache.clear()
warns = dr.supervisor_warnings()
check("dashboard warns when worker 004 is wrong route",
      any("must be GS" in w for w in warns))
os.remove(os.path.join(WD, "worker_004", "PIN"))
m["route"] = "gs"
json.dump(m, open(meta4, "w"))
dr._cache.clear()
check("dashboard warns when worker 004 is unpinned, and on route-mix "
      "drift",
      any("not PINNED" in w for w in dr.supervisor_warnings())
      and any("route mix" in w for w in dr.supervisor_warnings()))
payload = dr.routing_payload()
check("warnings carried in routing.json payload (read-only)",
      "supervisor_warnings" in payload and payload["read_only"] is True)

shutil.rmtree(WD, ignore_errors=True)
os.remove("sup_seeds.txt")
for f in ("hadamard_164.csv", "hadamard_104.csv"):
    if os.path.exists(f):
        os.remove(f)
print(f"\nALL {PASS} SUPERVISOR TESTS PASSED")
