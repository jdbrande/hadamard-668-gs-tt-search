# h668 — search framework for a Hadamard matrix of order 668

Order 668 = 4 × 167 is the smallest order for which no Hadamard matrix is
known (open since the 428 case fell in 2004). This framework runs two search
routes through one shared, exactly-verified pipeline:

- **tt** (priority): Turyn-type quadruples TT(56) — sequences X, Y, Z, W of
  lengths 56, 56, 56, 55 with N_X(j) + N_Y(j) + 2N_Z(j) + 2N_W(j) = 0 for all
  j ≥ 1 (nonperiodic autocorrelation). Chain: TT(56) → base sequences
  (111, 111, 56, 56) → T-sequences (167) → Goethals–Seidel → H(668).
  223 free bits. This is the route that produced the order-428 matrix.
- **gs**: symmetric Goethals–Seidel quadruples, four length-167 sequences
  with a[i] = a[167−i], periodic autocorrelations summing to zero.
  336 free bits, cheaper candidates, but rides an existence bet on symmetry.

Both routes use: exact-sum candidate generation → spectral (PSD) sieve →
canonical dedup (negation / reversal / cyclic shift where valid) →
meet-in-the-middle matching on integer autocorrelation vectors, C++ hot path.

**Honesty contract:** `hadamard_668.csv` is written only after
`core.verify_hadamard` proves entries ±1 and H·Hᵀ = 668·I in exact integer
arithmetic. No modular shortcuts. Anything less than that never touches disk.
This is an open research problem; expect pools to grow and matches to be rare
to nonexistent. Everything the framework learns persists across restarts.

## Setup

    python3 -m pip install numpy
    # optional but strongly recommended compiled matcher:
    c++ -O3 -std=c++17 -shared -fPIC fastmatch.cpp -o libfastmatch.so     # Linux
    c++ -O3 -std=c++17 -dynamiclib fastmatch.cpp -o libfastmatch.dylib   # macOS

Python falls back to a slower numpy matcher automatically if the library is
absent; semantics are identical (tested).

## Short test (always do this first)

    python3 tests.py                       # 34 tests, ~3-4 min
    python3 tests_opt.py                   # 15 optimization-equivalence tests
    python3 tests_migration.py             # 18 pool-migration tests
    ./smoke.sh                             # 3 workers on toy orders, ~1 min

## Launch 12 workers (Apple Silicon M3 Max)

    python3 ctl.py launch --workers 12     # 8 tt + 4 gs by default

Each worker gets a deterministic seed (base_seed + 1000·id), a private
directory `work/worker_NNN/`, and checkpoints pools atomically every cycle.

## Run forever

    ./run_forever.sh 12                    # relaunches workers if they die,
                                           # merges pools hourly, exits on solve

## Monitor

    python3 ctl.py monitor                 # live tail of every worker
    python3 ctl.py watch                   # blocks until SOLUTION.json, prints
                                           # the four sequences

## Stop

    python3 ctl.py stop                    # workers exit at cycle boundary
                                           # (pools already checkpointed)

## Merge pools across workers

    python3 ctl.py merge                   # worker pools -> global_pools_*.npz
                                           # workers absorb it automatically

Run this periodically (run_forever.sh does it hourly). Merging matters:
matching probability scales with the product of pool sizes, so shared pools
beat isolated ones.

## Verify (independent, exact)

    python3 ctl.py verify hadamard_668.csv

Loads the CSV fresh and re-proves H·Hᵀ = 668·I. Run this before believing
anyone — including this framework.

## 120-core machine

    python3 ctl.py launch --workers 120 --pool-cap 50000 --max-pairs 60000000

Guidance: memory per worker ≈ max_pairs × 16 bytes for the C++ pair table
plus pools (small). At 60 M pairs that's ~1 GB/worker — size to your RAM.
Keep the tt:gs split (default 2:1) unless you have a reason; diversity
protects against the symmetric-GS existence bet. Increase merge frequency
(`ctl.py merge` in cron every 15 min) since 120 workers generate pools fast.
Workers are independent processes with no shared state except the merged
pool files and the STOP flag, so they scale linearly and survive any subset
being killed.

## Success protocol

1. A worker's matcher returns indices; the quadruple is assembled and passed
   to `core.verify_hadamard` (exact, raises on failure).
2. Only then: `hadamard_668.csv` is written, `SOLUTION.json` (with all four
   sequences) is atomically placed in the workdir, and `STOP` is created.
3. All workers observe STOP at their next cycle boundary and exit.
4. The four sequences are printed in ± notation in the winner's log and by
   `ctl.py watch`.

## Files

    core.py        exact math: PAF/NPAF, PSD sieves, sum patterns,
                   canonicalization, GS/TT assembly, verification
    engine.py      candidate pools with dedup, chunked MITM matcher,
                   ctypes bridge to the C++ backend
    fastmatch.cpp  compiled pair-hash matcher (FNV-1a, open addressing,
                   exact re-verification on every hash hit)
    worker.py      worker loop: seed, checkpoint, resume, match, succeed
    ctl.py         launch / launch-missing / status / monitor / watch /
                   stop / merge / migrate / clean-bad / kill-duplicates /
                   verify
    tests.py       19-test suite incl. end-to-end solves at toy orders
    smoke.sh       3-worker orchestration smoke test
    run_forever.sh supervised nonstop runner with hourly merges

## Design choices, briefly

- **Two routes, one framework.** TT(56) and symmetric GS(167) are both
  "four pools + complementary-vector matching" problems; only the
  autocorrelation kernel (periodic vs nonperiodic), weights (1,1) vs (1,2),
  and assembly differ. The worker treats a route as a plug-in object.
- **Integer matching.** PAF/NPAF vectors are exact int16; matching is on
  byte-identical vectors, so hash hits are re-verified exactly and false
  positives are impossible by construction.
- **Sum patterns as a hard partition.** Solutions must hit a Diophantine
  sum pattern, so pools are binned by |sum| and only viable pattern splits
  are matched — no compute wasted on impossible combinations.
- **Canonical dedup.** Negation always preserves autocorrelation; reversal
  does for both routes; cyclic shifts only in the periodic route. Pools store
  one representative per orbit, so pool size measures genuine diversity.
- **Crash-only design.** All state files are written to a temp path and
  renamed atomically; a kill -9 at any moment loses at most one cycle.

## Recovery (read this before deleting anything)

`ctl.py status` is your first move in any confusion:

    python3 ctl.py status --workdir work
    # id  route     pid  state    ...          log
    #  0     tt    4711  ALIVE                 work/worker_000/log.txt
    #  1     gs    4712  ALIVE    DUPLICATE!   work/worker_001/log.txt
    # unique alive worker ids: 12

**Check worker count.** The last line of `status` shows unique alive IDs.
Liveness is decided by two independent signals: a flock probe on each
`worker_NNN/lock` (a live worker always holds its lock) and a portable
`ps -axo pid=,args=` scan (no pgrep — identical behavior on macOS and Linux).

**Kill duplicates.** If `status` shows `DUPLICATE!`:

    python3 ctl.py kill-duplicates --workdir work

It keeps the process that holds the lock (falling back to the lowest PID),
SIGTERMs the rest, and SIGKILLs stragglers. Note that since v2, duplicates
should not arise at all: a second process started with an occupied
workdir + worker-id exits immediately with code 2.

**Clean bad checkpoints.**

    python3 ctl.py clean-bad --workdir work

Scans every `.npz`, quarantines unreadable ones into `work/bad_npz_backup/`
(never deletes them), and removes temp files older than 5 minutes. Workers
also do this on their own: a corrupt `pools.npz` at resume is quarantined and
that worker simply starts its pools fresh — one bad checkpoint can no longer
kill a route.

**Restart from saved pools.** Pools are the accumulated value of every CPU
hour you've spent. To restart cleanly:

    python3 ctl.py stop --workdir work        # workers exit at cycle boundary
    python3 ctl.py clean-bad --workdir work
    python3 ctl.py launch-missing --workers 12 --workdir work

Workers resume from their own `pools.npz` and absorb `global_pools_*.npz`
automatically. `launch`/`launch-missing` remove a stale `STOP` file for you.

**Why not `rm -rf work`.** The work directory *is* the search state: every
deduplicated, sieve-passing candidate ever found, across all workers and
merges. Deleting it resets the search to zero — matching probability scales
with the product of pool sizes, so throwing pools away costs quadratically.
Delete `work/` only if you genuinely want a from-scratch run (e.g. new
seed-set experiment). Quarantined files in `bad_npz_backup/` are safe to
delete once you've confirmed you don't need forensics.

**Supervisor behavior (v2).** `run_forever.sh` never launches full batches.
Each 30 s cycle it: exits with the CSV path and sequences if `SOLUTION.json`
exists; runs `clean-bad`; runs `launch-missing` (a no-op when all workers are
alive); merges pools hourly. Worker count is therefore monotone up to the
target and can never grow past it.

## Optimization-equivalence tests

    python3 tests_opt.py

Rule of the repo: **no optimization ships without a test proving it returns
the same valid solutions as the complete search.** The suite compares every
fast path against `engine.find_all_matches`, a deliberately slow exhaustive
reference, on small instances with known solutions:

- **PAF dedup is loss-free.** Every raw candidate's exact PAF vector keeps a
  stored representative, and raw vs deduped pools solve identical pattern
  sets (found solutions are assembled and exactly verified).
- **NPAF dedup preserves TT constructibility.** The canonical representatives
  of the known TT(4) quadruple keep exact NPAF vectors and still assemble to
  a verified order-44 Hadamard matrix.
- **The pair-PSD screen is sound.** Theory: an exact match makes the PSDs sum
  to the full budget at every frequency, so its pairs can never fail a
  per-frequency cap. Tests: both pairs of every known solution pass (450
  pairs at n=13, plus the TT(4) fixture), the filtered solution set equals
  the complete solution set on every pattern/split, and the production
  matcher with the screen ON agrees with the exhaustive reference everywhere.
- **Incremental matching equals full rescan.** The four disjoint watermark
  blocks (old-region complement) union with the old region to exactly the
  full cross-product — verified set-equal, with empty overlap, on every
  pattern/split.
- **Watermarks survive checkpoint/resume** and are guarded: mutating a pool
  prefix or bumping `worker.OPT_VERSION` discards them (safe: discarding only
  causes re-checking, never skipping). Matching versions keep them.
- **End-to-end**: the optimized worker still solves both toy routes and its
  CSVs re-verify exactly.

If you change matching, dedup, or filter semantics, bump `OPT_VERSION` in
worker.py — stale incremental caches from older semantics are then discarded
automatically on every worker's next start.

## Migrating existing pools to an optimized version

Pools are the accumulated value of your CPU time; the optimized code reuses
them and never restarts from zero when pools exist. What it will NOT do is
trust old claims about which pairs were already checked.

**Recommended (offline, before relaunching workers):**

    python3 ctl.py stop --workdir work
    python3 ctl.py migrate --workdir work
    python3 ctl.py launch-missing --workers 12 --workdir work

`migrate` walks every worker directory and, per pool file: validates the
.npz (corrupt files are quarantined to `bad_npz_backup/`, never deleted),
reloads candidates through canonical dedup (duplicates and
negation/shift/reversal equivalents collapse to one representative, which is
loss-free -- see tests_opt.py), rebuilds the PAF/NPAF and PSD caches, and
writes `migration.json` + `caches.npz`/`cacheinfo.json`, all stamped with
the current `OPT_VERSION`. It is idempotent: rerunning reports "up to date".

**Automatic (nothing to do):** workers perform the same migration on their
own first startup if `migration.json` is missing or from an older version.

**Checked-pair history rules:**

- Watermarks (which pair regions are already checked) live ONLY in
  `matchstate.json`, stamped with `OPT_VERSION` and per-bin pool
  fingerprints. That is the only machine-readable checked-pair record the
  code trusts.
- Any matchstate from a different version, with mismatched fingerprints, or
  unreadable, is discarded with a logged message. Discarding is always safe:
  it re-checks pairs, never skips them.
- Consequence: the first optimized run over migrated pools performs a
  baseline full pass (watermarks all zero -> the incremental block
  decomposition degenerates to the complete cross-product). Once the
  baseline completes per pattern/split, watermarks advance and subsequent
  cycles are strictly old-x-new incremental. Watermarks persist across
  restarts, so resume knows exactly what has been checked; a restart
  mid-baseline redoes only the pattern/splits whose pass hadn't completed.
- tests_migration.py includes an adversarial test: legacy pools containing a
  valid solution plus a legacy matchstate falsely claiming those pools were
  fully checked. The optimized worker discards the claim, runs the baseline,
  and finds the solution -- proving no potentially valid solution can be
  skipped by stale history.

If you change math filters or pair logic, bump `OPT_VERSION` in worker.py:
watermarks AND caches are then invalidated and rebuilt automatically, and
every worker's next cycle is a fresh baseline over its existing pools.
