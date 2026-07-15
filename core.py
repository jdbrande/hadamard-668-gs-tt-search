"""Core mathematics for the order-668 Hadamard search. Pure numpy, exact.

Two engines share this module:
  Route A  "gs" : symmetric Goethals-Seidel quadruples, length n (167).
                  Constraint: sum of periodic autocorrelations = 0, shifts
                  1..(n-1)/2 (symmetry makes PAF(s)=PAF(n-s)).
  Route B  "tt" : Turyn-type quadruples X,Y,Z,W of lengths m,m,m,m-1 (m=56).
                  Constraint: N_X(j)+N_Y(j)+2N_Z(j)+2N_W(j)=0, j=1..m-1
                  (nonperiodic autocorrelation). Chain to a Hadamard matrix:
                  TT(m) -> base(2m-1,2m-1,m,m) -> T-sequences(3m-1)
                  -> Goethals-Seidel with 1x1 blocks -> order 4(3m-1).
                  m=56 gives 4*167 = 668.
"""
import numpy as np

# ---------------------------------------------------------------- autocorr

def paf_full(seq):
    """Periodic autocorrelation vector (all shifts), exact integers."""
    n = len(seq)
    F = np.fft.rfft(seq.astype(np.float64))
    return np.rint(np.fft.irfft(np.abs(F) ** 2, n)).astype(np.int64)


def paf_half_batch(seqs):
    """Shifts 1..(n-1)//2 for a batch (rows), int16."""
    n = seqs.shape[1]
    h = (n - 1) // 2
    F = np.fft.rfft(seqs.astype(np.float64), axis=1)
    p = np.fft.irfft(np.abs(F) ** 2, n, axis=1)
    return np.rint(p[:, 1:h + 1]).astype(np.int16)


def npaf_direct(seq, j):
    if j >= len(seq):
        return 0
    return int(np.dot(seq[:len(seq) - j], seq[j:]))


def npaf_batch(seqs, out_len):
    """Nonperiodic autocorrelation, shifts 1..out_len, via zero-padded FFT."""
    m = seqs.shape[1]
    P = 1
    while P < 2 * m:
        P *= 2
    F = np.fft.rfft(seqs.astype(np.float64), P, axis=1)
    a = np.fft.irfft(np.abs(F) ** 2, P, axis=1)
    res = np.zeros((seqs.shape[0], out_len), dtype=np.int16)
    take = min(out_len, m - 1)
    res[:, :take] = np.rint(a[:, 1:take + 1]).astype(np.int16)
    return res


# ---------------------------------------------------------------- sieves

def psd_pass_periodic(seqs, cap):
    """Keep rows whose nonzero-frequency PSD never exceeds cap (=4n)."""
    F = np.fft.rfft(seqs.astype(np.float64), axis=1)
    psd = np.abs(F) ** 2
    return psd[:, 1:].max(axis=1) <= cap + 1e-6


def psd_pass_padded(seqs, cap, pad):
    """Nonperiodic sieve: padded PSD must stay <= cap at every frequency."""
    F = np.fft.rfft(seqs.astype(np.float64), pad, axis=1)
    psd = np.abs(F) ** 2
    return psd.max(axis=1) <= cap + 1e-6


# ---------------------------------------------------------------- patterns

def gs_sum_patterns(n):
    """(s1>=s2>=s3>=s4) odd with sum of squares = 4n."""
    t = 4 * n
    out = []
    lim = int(t ** 0.5)
    for s1 in range(1, lim + 1, 2):
        for s2 in range(1, s1 + 1, 2):
            for s3 in range(1, s2 + 1, 2):
                r = t - s1 * s1 - s2 * s2 - s3 * s3
                if r < 1:
                    continue
                s4 = int(round(r ** 0.5))
                if s4 * s4 == r and s4 % 2 == 1 and s4 <= s3:
                    out.append((s1, s2, s3, s4))
    return out


def tt_sum_patterns(m):
    """(sx,sy,sz,sw): sx^2+sy^2+2sz^2+2sw^2 = 2(3m-1); parity even,even,even,odd.
    WLOG sx>=sy>=0, sz>=0, sw>=1 (negation is free per sequence)."""
    t = 2 * (3 * m - 1)
    out = []
    lim = int(t ** 0.5)
    for sx in range(0, lim + 1, 2):
        for sy in range(0, sx + 1, 2):
            for sz in range(0, lim + 1, 2):
                r = t - sx * sx - sy * sy - 2 * sz * sz
                if r < 2:
                    continue
                if r % 2:
                    continue
                sw2 = r // 2
                sw = int(round(sw2 ** 0.5))
                if sw * sw == sw2 and sw % 2 == 1:
                    out.append((sx, sy, sz, sw))
    return out


# ---------------------------------------------------------------- dedup

def _canonical_key_py(seq, periodic):
    """Pure-Python reference implementation (correctness oracle for the
    native path; also the runtime fallback)."""
    s = np.asarray(seq, dtype=np.int8)
    cands = []
    variants = [s, s[::-1]]
    for v in variants:
        for sign in (1, -1):
            w = sign * v
            if periodic:
                n = len(w)
                d = np.concatenate([w, w])
                for k in range(n):
                    cands.append(d[k:k + n].tobytes())
            else:
                cands.append(w.tobytes())
    return min(cands)


def canonical_key(seq, periodic):
    """Bytes key: lexicographic minimum over the invariance group.
    periodic (PAF-preserving): negation, cyclic shifts, reversal.
    nonperiodic (NPAF-preserving): negation, reversal only.
    Uses the native implementation when available and checked (identical
    output guaranteed by the oracle in native.py); Python otherwise."""
    import native
    if native.load() is not None:
        return native.canonical_key(seq, periodic)
    return _canonical_key_py(seq, periodic)


# ---------------------------------------------------------------- generation

def random_with_sum(count, length, target_sum, rng):
    """Uniform random +-1 rows with exact sum (sign chosen so sum>=0 later)."""
    if (length + target_sum) % 2:
        raise ValueError("parity mismatch")
    p = (length + target_sum) // 2
    if not (0 <= p <= length):
        return np.empty((0, length), dtype=np.int8)
    out = np.full((count, length), -1, dtype=np.int8)
    for r in range(count):
        idx = rng.choice(length, p, replace=False)
        out[r, idx] = 1
    return out


def symmetric_with_sum(count, n, target_sum, rng):
    """Symmetric length-n sequences (a[i]=a[n-i]) with exact sum.
    sum = a0 + 2*inner  where inner is the sum of h=(n-1)/2 free entries."""
    h = (n - 1) // 2
    rows = []
    for a0 in (1, -1):
        need = target_sum - a0
        if need % 2:
            continue
        inner = need // 2
        if (h + inner) % 2 or abs(inner) > h:
            continue
        half = random_with_sum(count // 2 + 1, h, inner, rng)
        if len(half) == 0:
            continue
        full = np.empty((len(half), n), dtype=np.int8)
        full[:, 0] = a0
        full[:, 1:h + 1] = half
        full[:, h + 1:] = half[:, ::-1]
        rows.append(full)
    if not rows:
        return np.empty((0, n), dtype=np.int8)
    return np.concatenate(rows)[:count]


# ---------------------------------------------------------------- assembly

def circulant(row):
    n = len(row)
    return np.array([np.roll(row, k) for k in range(n)], dtype=np.int64)


def goethals_seidel_array(A, B, C, D):
    n = A.shape[0]
    R = np.fliplr(np.eye(n, dtype=np.int64))
    return np.block([
        [A,        B @ R,      C @ R,      D @ R],
        [-B @ R,   A,          -D.T @ R,   C.T @ R],
        [-C @ R,   D.T @ R,    A,          -B.T @ R],
        [-D @ R,   -C.T @ R,   B.T @ R,    A]])


def gs_build(seqs):
    A, B, C, D = (circulant(np.asarray(s, dtype=np.int64)) for s in seqs)
    return goethals_seidel_array(A, B, C, D)


def tt_build(X, Y, Z, W):
    """TT(m) -> base -> T-sequences(3m-1) -> Hadamard(4(3m-1))."""
    X, Y, Z, W = (list(map(int, s)) for s in (X, Y, Z, W))
    m = len(X)
    A = np.array(Z + W, dtype=np.int64)                # length 2m-1
    B = np.array(Z + [-w for w in W], dtype=np.int64)
    C = np.array(X, dtype=np.int64)
    D = np.array(Y, dtype=np.int64)
    L = 3 * m - 1
    z_m = np.zeros(m, dtype=np.int64)
    z_2m1 = np.zeros(2 * m - 1, dtype=np.int64)
    T1 = np.concatenate([(A + B) // 2, z_m])
    T2 = np.concatenate([(A - B) // 2, z_m])
    T3 = np.concatenate([z_2m1, (C + D) // 2])
    T4 = np.concatenate([z_2m1, (C - D) // 2])
    assert np.all(np.abs(T1) + np.abs(T2) + np.abs(T3) + np.abs(T4) == 1)
    X1, X2, X3, X4 = map(circulant, (T1, T2, T3, T4))
    e1 = goethals_seidel_array(X1, X2, X3, X4)
    e2 = goethals_seidel_array(X2, -X1, X4, -X3)
    e3 = goethals_seidel_array(X3, -X4, -X1, X2)
    e4 = goethals_seidel_array(X4, X3, -X2, -X1)
    return e1 + e2 + e3 + e4


def verify_hadamard(H):
    """Exact check: entries +-1 and H H^T = N I. Returns N or raises."""
    H = np.asarray(H, dtype=np.int64)
    N = H.shape[0]
    if H.shape != (N, N) or not np.all(np.abs(H) == 1):
        raise AssertionError("entries are not all +-1")
    if not np.array_equal(H @ H.T, N * np.eye(N, dtype=np.int64)):
        raise AssertionError("H H^T != N I  --  NOT a Hadamard matrix")
    return N
