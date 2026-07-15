"""Test suite (v2). Run: python3 tests.py    (exits nonzero on any failure)."""
# H668 test color hook
try:
    from test_colors import install as _h668_install_test_colors
    _h668_install_test_colors()
except Exception:
    pass

import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

import core
import engine

rng = np.random.default_rng(7)
PASS = 0
PY = sys.executable


def check(name, cond):
    global PASS
    if not cond:
        print(f"\033[31mFAIL  {name}\033[0m")
        sys.exit(1)
    PASS += 1
    print(f"\033[32mok    {name}\033[0m")


# ---- 1. kernels vs slow direct implementations ----------------------------
def slow_paf(seq, s):
    n = len(seq)
    return sum(int(seq[i]) * int(seq[(i + s) % n]) for i in range(n))


x = rng.choice([-1, 1], 13).astype(np.int8)
p = core.paf_half_batch(x[None, :])[0]
check("PAF fft == direct", all(p[s - 1] == slow_paf(x, s) for s in range(1, 7)))

y = rng.choice([-1, 1], 11).astype(np.int8)
npv = core.npaf_batch(y[None, :], 10)[0]
check("NPAF fft == direct",
      all(npv[j - 1] == core.npaf_direct(y, j) for j in range(1, 11)))

F = np.fft.rfft(x.astype(float))
check("PSD sieve == direct",
      bool(core.psd_pass_periodic(x[None, :], 52)[0])
      == bool((np.abs(F[1:]) ** 2).max() <= 52 + 1e-6))

# ---- 2. canonicalization ---------------------------------------------------
a = rng.choice([-1, 1], 9).astype(np.int8)
check("canon: negation (periodic)",
      core.canonical_key(a, True) == core.canonical_key(-a, True))
check("canon: cyclic shift (periodic)",
      core.canonical_key(a, True) == core.canonical_key(np.roll(a, 3), True))
check("canon: reversal (nonperiodic)",
      core.canonical_key(a, False) == core.canonical_key(a[::-1], False))
check("canon: shift NOT merged (nonperiodic)",
      core.canonical_key(a, False) != core.canonical_key(np.roll(a, 1), False)
      or np.array_equal(a, np.roll(a, 1)))

pool = engine.Pool(9, periodic=True)
pool.add(np.stack([a, -a, np.roll(a, 2), a[::-1]]))
check("pool dedup collapses equivalents", len(pool.seqs) == 1)

# ---- 3. exact-sum generation ------------------------------------------------
b = core.symmetric_with_sum(50, 13, 5, rng)
check("symmetric generator: sums", np.all(b.sum(axis=1) == 5))
check("symmetric generator: symmetry",
      all(np.array_equal(r[1:], r[1:][::-1]) for r in b))
c = core.random_with_sum(50, 55, 3, rng)
check("plain generator: sums", np.all(c.sum(axis=1) == 3))

# ---- 4. TT fixture and full TT chain; verification gate ---------------------
X, Y, Z, W = [1, 1, 1, 1], [1, 1, -1, 1], [1, 1, -1, -1], [1, -1, 1]
ok = all(core.npaf_direct(np.array(X), j) + core.npaf_direct(np.array(Y), j)
         + 2 * core.npaf_direct(np.array(Z), j)
         + 2 * core.npaf_direct(np.array(W), j) == 0 for j in range(1, 4))
check("TT(4) fixture satisfies the Turyn-type identity", ok)
H44 = core.tt_build(X, Y, Z, W)
check("TT(4) -> verified Hadamard order 44", core.verify_hadamard(H44) == 44)

Hbad = H44.copy()
Hbad[0, 0] *= -1
try:
    core.verify_hadamard(Hbad)
    check("verify_hadamard rejects a near-miss", False)
except AssertionError:
    check("verify_hadamard rejects a near-miss", True)

# ---- 5. matcher backends agree; end-to-end GS solve at n=13 ------------------
import worker as wk

route = wk.GSRoute(13)
pools = {s: engine.Pool(13, True) for s in route.bins}
for s in route.bins:
    for _ in range(40):
        if len(pools[s].seqs) >= 64:
            break
        pools[s].add(route.generate(s, 20000, rng),
                     need=64 - len(pools[s].seqs))
solved = None
for pat in route.patterns:
    for (b1, b2), (b3, b4) in route.splits(pat):
        P = [pools[b].seqs for b in (b1, b2, b3, b4)]
        pafs = [route.paf(p) for p in P]
        hit = engine.match(*pafs, wt1=1, wt2=1, max_pairs=2_000_000)
        if hit:
            i, j, k, l = hit
            H = core.gs_build([P[0][i], P[1][j], P[2][k], P[3][l]])
            solved = core.verify_hadamard(H)
            break
    if solved:
        break
check("GS n=13 end-to-end -> verified Hadamard order 52", solved == 52)

if engine.load_fastmatch() is not None:
    pa = core.paf_half_batch(pools[3].seqs[:40])
    lib = engine._LIB
    engine._LIB = None
    r_py = engine.match(pa, pa, pa, pa, max_pairs=10_000)
    engine._LIB = lib
    r_cc = engine.match(pa, pa, pa, pa, max_pairs=10_000)
    check("C++ and numpy matchers agree on solvability",
          (r_py is None) == (r_cc is None))
else:
    print("note: libfastmatch not built; skipping backend parity test")

# ---- 6. worker checkpoint resume ---------------------------------------------
wd = "test_work"
shutil.rmtree(wd, ignore_errors=True)
cmd = [PY, "worker.py", "--route", "gs", "--n", "13",
       "--worker-id", "0", "--workdir", wd, "--pool-cap", "300",
       "--batch", "20000", "--max-pairs", "500000"]
subprocess.run(cmd, capture_output=True, text=True, timeout=300)
check("worker run 1 exits after writing STOP+SOLUTION",
      os.path.exists(os.path.join(wd, "SOLUTION.json")))
check("worker wrote pools checkpoint",
      os.path.exists(os.path.join(wd, "worker_000", "pools.npz")))
os.remove(os.path.join(wd, "STOP"))
os.remove(os.path.join(wd, "SOLUTION.json"))
r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
check("worker resumes from checkpoint and solves again",
      "resumed pools" in (r2.stdout + r2.stderr)
      and os.path.exists(os.path.join(wd, "SOLUTION.json")))
H = np.loadtxt("hadamard_52.csv", delimiter=",", dtype=np.int64)
check("solution CSV verifies exactly", core.verify_hadamard(H) == 52)

# ---- 7. HARDENING: duplicate worker id is rejected by the lock ----------------
os.remove(os.path.join(wd, "STOP"))
os.remove(os.path.join(wd, "SOLUTION.json"))
holder = wk.acquire_worker_lock(os.path.join(wd, "worker_000"))
check("test process acquired worker_000 lock", holder)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
check("duplicate worker id exits with code 2",
      r.returncode == 2 and "REFUSING TO START" in (r.stdout + r.stderr))
wk._LOCK_HANDLE.close()  # release for later tests
wk._LOCK_HANDLE = None

# ---- 8. HARDENING: corrupt pools.npz is quarantined, worker continues ---------
pool_file = os.path.join(wd, "worker_000", "pools.npz")
with open(pool_file, "wb") as fh:
    fh.write(b"this is not a zip archive at all \x00\x01\x02")
r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
bad_dir = os.path.join(wd, "bad_npz_backup")
check("corrupt checkpoint quarantined to bad_npz_backup",
      os.path.isdir(bad_dir) and len(os.listdir(bad_dir)) >= 1
      and "quarantine" in (r.stdout + r.stderr))
check("worker recovered fresh and still solved",
      os.path.exists(os.path.join(wd, "SOLUTION.json")))
shutil.rmtree(wd, ignore_errors=True)


def fake_worker(wdir, wid):
    """A sleeper whose argv looks like a worker to the ps scanner."""
    return subprocess.Popen(
        [PY, "-c", "import time; time.sleep(300)",
         "worker.py", "--workdir", wdir, "--worker-id", str(wid)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ctl(args, wdir):
    return subprocess.run([PY, "ctl.py"] + args + ["--workdir", wdir],
                          capture_output=True, text=True, timeout=300)


# ---- 9. HARDENING: launch-missing starts only missing ids ---------------------
lm = "lm_work"
shutil.rmtree(lm, ignore_errors=True)
os.makedirs(lm)
fakes = [fake_worker(lm, 0), fake_worker(lm, 2)]
time.sleep(1)
r = ctl(["launch-missing", "--workers", "3", "--tt-share", "0",
         "--gs-n", "13", "--pool-cap", "200", "--wait"], lm)
out = r.stdout + r.stderr
check("launch-missing reports already-alive ids",
      "already alive: [0, 2]" in out)
check("launch-missing launches only worker 1",
      "launched worker 1" in out and "launched 1 missing" in out
      and "launched worker 0" not in out and "launched worker 2" not in out)
check("gap-filled worker solved and wrote SOLUTION",
      os.path.exists(os.path.join(lm, "SOLUTION.json")))
r = ctl(["launch-missing", "--workers", "3"], lm)
check("launch-missing refuses when SOLUTION exists",
      "not launching" in (r.stdout + r.stderr))

# ---- 10. HARDENING: no growth on repeated launch-missing ----------------------
os.remove(os.path.join(lm, "SOLUTION.json"))
if os.path.exists(os.path.join(lm, "STOP")):
    os.remove(os.path.join(lm, "STOP"))
fakes.append(fake_worker(lm, 1))
time.sleep(1)
before = ctl(["status"], lm).stdout
r1 = ctl(["launch-missing", "--workers", "3"], lm)
r2 = ctl(["launch-missing", "--workers", "3"], lm)
check("repeated launch-missing launches nothing when all alive",
      "nothing to launch" in r1.stdout and "nothing to launch" in r2.stdout)
check("status reports 3 unique alive ids",
      "unique alive worker ids: 3" in ctl(["status"], lm).stdout)

# ---- 11. HARDENING: kill-duplicates ---------------------------------------------
dup = fake_worker(lm, 1)   # second process claiming worker 1
time.sleep(1)
st = ctl(["status"], lm).stdout
check("status flags the duplicate", "DUPLICATE!" in st)
r = ctl(["kill-duplicates"], lm)
time.sleep(1)
check("kill-duplicates removed the extra process",
      "killed 1 duplicate" in r.stdout
      and "DUPLICATE!" not in ctl(["status"], lm).stdout)

# ---- 12. HARDENING: clean-bad quarantines and prunes ------------------------------
os.makedirs(os.path.join(lm, "worker_009"), exist_ok=True)
with open(os.path.join(lm, "worker_009", "pools.npz"), "wb") as fh:
    fh.write(b"garbage")
stale = os.path.join(lm, "worker_009", "pools.npz.tmp.999.dead.npz")
with open(stale, "wb") as fh:
    fh.write(b"tmp")
os.utime(stale, (time.time() - 3600, time.time() - 3600))
r = ctl(["clean-bad"], lm)
check("clean-bad quarantines corrupt npz and removes stale temps",
      "quarantined" in r.stdout and "removed stale temp" in r.stdout
      and not os.path.exists(stale))

for f in fakes + [dup]:
    try:
        f.send_signal(signal.SIGKILL)
    except Exception:
        pass
shutil.rmtree(lm, ignore_errors=True)
shutil.rmtree("test_work", ignore_errors=True)

# ---- 13. run_forever exits and reports when SOLUTION.json exists -------------------
rf = "rf_work"
shutil.rmtree(rf, ignore_errors=True)
os.makedirs(rf)
import json
json.dump({"order": 52, "route": "gs", "worker": 0, "pattern": [5, 3, 3, 3],
           "csv": os.path.abspath("hadamard_52.csv"),
           "sequences": [[1] * 13] * 4}, open(os.path.join(rf, "SOLUTION.json"), "w"))
r = subprocess.run(["bash", "run_forever.sh", "3"], capture_output=True,
                   text=True, timeout=120, env={**os.environ, "WORKDIR": rf})
check("run_forever exits 0 on existing SOLUTION and prints CSV path",
      r.returncode == 0 and "SOLVED" in r.stdout
      and "hadamard_52.csv" in r.stdout)
shutil.rmtree(rf, ignore_errors=True)

print(f"\nALL {PASS} TESTS PASSED")
