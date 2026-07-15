H668 Math Proof Packet

Project: H668 search
Goal: construct a Hadamard matrix of order 668.

This file is the correctness contract for future speed patches.

It does not claim H(668) has been found.
It states what must remain true while optimizing the search.

1. Final verification standard

A real success is not a counter, score, collision, hash, or dashboard signal.

A real success must produce a matrix H with entries ±1 such that:

H H^T = 668 I

The final authority is:

core.verify_hadamard(H)

Expected success artifacts:

work/SOLUTION.json
hadamard_668.csv
VERIFIED Hadamard order 668

Any patch that finds a candidate must still assemble the matrix and pass the verifier.

2. Safe optimization rule

A speed patch is safe only if skipped work is one of these:

mathematically impossible
mathematically redundant
already checked by an older superset scan
only scheduling, not candidate removal

Unsafe:

rejecting candidates because they look unlikely
rejecting because a seed score is low
rejecting because a bin is capped
weakening final verification

A true 1.3x safe patch is better than a fake 2x risky patch.

3. GS route correctness

GS searches four sequences of length n = 167.

The route uses the Goethals-Seidel construction.

For this project:

4 × 167 = 668

A valid GS quadruple must satisfy the route-specific periodic autocorrelation identity used by the implementation.

Plain language:

The nonzero autocorrelation contributions cancel.
Then the Goethals-Seidel array assembles to a Hadamard matrix of order 668.

Any GS optimization must preserve every canonical orbit that could satisfy the PAF identity.

The verifier still checks the final assembled matrix.

4. TT route correctness

TT candidates are not trusted by name alone.

They must still pass:

route assembly
PAF/NPAF matching
core.verify_hadamard(H)

Any TT optimization must preserve every TT object that could assemble into a valid Hadamard matrix.

5. Canonical orbit dedup soundness

A sequence may have mathematically equivalent forms:

cyclic shift
reversal
negation
allowed route-specific symmetries

These forms are in the same orbit.

For the matching equations, all members of the same orbit have the same relevant autocorrelation data.

Therefore:

keeping one canonical representative per orbit is safe

This is only safe if the canonical representative is the same one workers already use.

Required invariant:

canonical key set after merge == canonical key set of raw union

No orbit may be lost.

Existing tests proving this:

tests_canon.py

Known green proof lines:

NO ORBIT LOST: canonical key set of merged global == canonical key set of raw union, every bin
canonical merge is idempotent
route-safe merge remains route-safe

6. Triangle same-bin pair restriction soundness

When matching two sequences from the same bin, the pair equation is symmetric.

Checking both ordered pairs is redundant:

(A, B)
(B, A)

The unordered representative is enough:

i <= j

This is safe only for same-bin pair loops where the two sides are mathematically interchangeable.

Required invariant:

tri-restricted search returns the same solution set as exhaustive same-bin search

Existing tests proving this:

tests_canon.py

Known green proof lines:

REPRESENTATIVE PROOF: for every exhaustive solution, the ordered representative is also a solution
incremental blocks with global offsets hash exactly the same i<=j pairs as one unblocked pass
python and native v6 agree

7. PSD and full-frequency screen soundness

PSD screens are allowed only as necessary-condition filters.

They may reject a candidate only when the candidate cannot possibly satisfy the route autocorrelation identity.

Allowed:

reject mathematically impossible candidates earlier

Not allowed:

reject candidates because they look unlikely
reject candidates because the seed score is low
reject candidates because a bin is annoying

Required invariant:

screened search finds all known small valid solutions
native and Python agree

Existing tests:

tests_screen.py
tests_native.py

8. Native code safety rule

Python is the reference oracle.

Native C++ is an acceleration path only.

Any native function must agree with Python on small exhaustive cases and fixture cases.

If native and Python disagree, native must disable itself loudly and fall back to Python.

Known correct behavior:

native path DISABLED
falling back to pure Python

9. Watermark and OPT_VERSION rule

Watermarks mean:

these pair ranges were already checked under a specific algorithm and pool fingerprint

Watermarks remain valid only when the old work checked a superset of the new required work.

Safe without OPT_VERSION bump:

triangle restriction after exhaustive old scan
screen gets stricter only when old scan already checked more

Requires discard or OPT_VERSION bump:

changed matching semantics
changed hash key meaning
changed route identity
changed candidate representation in a way old watermarks no longer cover

Existing safety tests:

fingerprint mismatch discards watermarks
OPT_VERSION change discards stale watermarks

10. Seed scoring is not math proof

Seed score is scheduling information.

It must never decide mathematical validity.

Raw noisy collisions must not dominate.

Current intended scoring hierarchy:

verified_hadamard: highest
residual_zero: very high
raw noisy collision: near zero and capped
runtime health: low priority scheduling signal

Seed scoring may prioritize workers.
It must not delete valid candidates or change verification.

11. Capped-bin optimization safety rule

Capped-bin handling is scheduling, not math.

Safe:

rotate seeds that are producing mostly capped or duplicate rows
reduce checkpoint churn for unchanged capped bins
prioritize bins with room
avoid repeated matching against unchanged capped bins

Unsafe unless proven:

permanently delete capped-bin candidates
make a bin unreachable forever
reject candidates only because a bin is capped
change the definition of a valid candidate

If a patch avoids capped-bin waste, it must prove:

the full search space remains reachable over time
existing pools are not deleted
valid solutions are not filtered out
worker 004 remains GS and pinned

12. Worker 004 safety rule

Worker 004 is special.

It must stay:

route: GS
seed: 94091007
PIN file present: work/worker_004/PIN

It had the only historical pre-evidence clue.

That clue cannot be replayed, but the worker should remain protected.

Hard fail condition:

worker 004 starts as TT
worker 004 loses PIN
worker 004 is auto-rotated

13. Required proof obligations for any future speed patch

Every speed patch must answer these before implementation:

What exact work is being skipped?
Why is that work mathematically redundant or impossible?
Does the patch remove candidates, or only change scheduling?
Does the patch preserve the final solution set?
Does it affect GS, TT, or both?
Does it affect watermarks?
Does OPT_VERSION need to change?
Does native agree with Python?
Do known small GS and TT fixtures still solve?
Does worker 004 remain GS and pinned?
14. Required test commands

Minimum tests after any speed patch:

python3 build_native.py

for t in tests tests_opt tests_migration tests_telemetry tests_native tests_acceptance tests_routing tests_autorotate tests_dashboard tests_clues tests_screen tests_canon tests_seedscore
do
python3 "$t.py" || exit 1
done

Required benchmark for speed patches:

python3 bench_hotspots.py --workdir work

15. Non-negotiables

Do not weaken final verification.

Do not delete existing pools.

Do not route worker 004 away from GS.

Do not treat noisy hash collisions as strong evidence.

Do not use dashboard scores as mathematical truth.

Do not accept a speedup without proof that the solution set is preserved.
