"""Single loader + correctness gate for the native module (libfastmatch).

Env switch:
  H668_NATIVE=0  force pure-Python paths
  H668_NATIVE=1  use native (default when the library is compiled)

Rule enforced here: at first load, native functions are cross-checked
against the Python reference on randomized deterministic fixtures. ANY
disagreement disables native for the whole process with a loud message,
recorded in status()["fallback_reason"] and in worker telemetry.
"""
import ctypes
import os
import time

import numpy as np

_STATE = {"lib": None, "checked": False, "enabled": False,
          "backend": "python", "fallback_reason": None,
          "speedup_estimate": None}


def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("libfastmatch.so", "libfastmatch.dylib"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            return p
    return None


def _bind(lib):
    common = [ctypes.POINTER(ctypes.c_int16), ctypes.c_longlong] * 4 + [
        ctypes.c_longlong, ctypes.c_int16, ctypes.c_int16,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong)]
    if hasattr(lib, "match_pairs_v7"):
        P16 = ctypes.POINTER(ctypes.c_int16)
        PF = ctypes.POINTER(ctypes.c_float)
        PI = ctypes.POINTER(ctypes.c_int32)
        PL = ctypes.POINTER(ctypes.c_longlong)
        lib.match_pairs_v7.restype = ctypes.c_longlong
        lib.match_pairs_v7.argtypes = (
            [P16, P16, P16, P16, ctypes.c_longlong,
             ctypes.c_int16, ctypes.c_int16,
             PF, PI, PF, PI, PF, PI, PF, PI,
             ctypes.c_longlong, ctypes.c_longlong,
             ctypes.c_float, ctypes.c_float,
             PL, PL, PL, PL, ctypes.c_longlong,
             PL, PL, PL, PL, ctypes.c_longlong,
             ctypes.c_longlong, ctypes.c_longlong,
             ctypes.c_longlong, ctypes.c_longlong,
             ctypes.c_int32, ctypes.c_int32, ctypes.c_longlong,
             PL, PL, PL, ctypes.c_longlong, PL])
    if hasattr(lib, "match_pairs_v6"):
        lib.match_pairs_v6.restype = ctypes.c_longlong
        lib.match_pairs_v6.argtypes = (
            [ctypes.POINTER(ctypes.c_int16), ctypes.c_longlong] * 4 + [
                ctypes.c_longlong, ctypes.c_int16, ctypes.c_int16,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_int32, ctypes.c_int32,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong), ctypes.c_longlong,
                ctypes.POINTER(ctypes.c_longlong)])
    if hasattr(lib, "match_pairs_v5"):
        lib.match_pairs_v5.restype = ctypes.c_longlong
        lib.match_pairs_v5.argtypes = (
            [ctypes.POINTER(ctypes.c_int16), ctypes.c_longlong] * 4 + [
                ctypes.c_longlong, ctypes.c_int16, ctypes.c_int16,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_longlong, ctypes.c_longlong,
                ctypes.c_float, ctypes.c_float,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong), ctypes.c_longlong,
                ctypes.POINTER(ctypes.c_longlong)])
    if hasattr(lib, "match_pairs_v4"):
        lib.match_pairs_v4.restype = ctypes.c_longlong
        lib.match_pairs_v4.argtypes = (
            [ctypes.POINTER(ctypes.c_int16), ctypes.c_longlong] * 4 + [
                ctypes.c_longlong, ctypes.c_int16, ctypes.c_int16,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_longlong, ctypes.c_float, ctypes.c_float,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong), ctypes.c_longlong,
                ctypes.POINTER(ctypes.c_longlong)])
    for sym in ("match_pairs_v2", "match_pairs_v3"):
        if hasattr(lib, sym):
            fn = getattr(lib, sym)
            fn.restype = ctypes.c_longlong
            fn.argtypes = common
    if hasattr(lib, "canon_key"):
        lib.canon_key.restype = ctypes.c_int
        lib.canon_key.argtypes = [ctypes.POINTER(ctypes.c_int8),
                                  ctypes.c_longlong, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int8)]
    if hasattr(lib, "fm_backend"):
        lib.fm_backend.restype = ctypes.c_char_p


def canonical_key(seq, periodic):
    """Native canonical key. Only call when load() is not None."""
    lib = _STATE["lib"]
    s = np.ascontiguousarray(seq, dtype=np.int8)
    out = np.empty(len(s), dtype=np.int8)
    lib.canon_key(s.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                  len(s), 1 if periodic else 0,
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
    return out.tobytes()


def _fail(reason):
    _STATE.update(lib=None, enabled=False, backend="python",
                  fallback_reason=reason)
    print(f"[native] *** DISAGREEMENT WITH PYTHON: {reason} ***")
    print("[native] native path DISABLED for this process; "
          "falling back to pure Python. Please report this.")


def _self_check(lib):
    """Randomized deterministic oracle. Returns None on success, else a
    loud reason string. Test hook: H668_NATIVE_FORCE_MISMATCH=1 simulates
    a disagreement to prove the fallback path works."""
    import core
    rng = np.random.default_rng(2026)
    if os.environ.get("H668_NATIVE_FORCE_MISMATCH") == "1":
        return "forced mismatch (H668_NATIVE_FORCE_MISMATCH=1)"
    # canonical keys, both semantics, assorted lengths incl. edge n=1,2
    for n in (1, 2, 3, 9, 13, 55, 56, 167):
        for _ in range(30):
            s = rng.choice(np.array([-1, 1], np.int8), n)
            for periodic in (True, False):
                if canonical_key(s, periodic) != \
                        core._canonical_key_py(s, periodic):
                    return f"canon_key mismatch at n={n} periodic={periodic}"
    # matcher results vs python dict reference on random PAF fixtures
    import engine
    for trial in range(6):
        h = int(rng.integers(1, 90))
        na = int(rng.integers(1, 40))
        paf = rng.integers(-50, 50, (na, h)).astype(np.int16)
        ref = engine.find_all_matches(paf, paf, paf, paf,
                                      wt1=1 + trial % 2, wt2=1 + trial % 3)
        got = _raw_match(lib, paf, paf, paf, paf,
                         1 + trial % 2, 1 + trial % 3)
        if (got is None) != (len(ref) == 0):
            return f"match existence mismatch (trial {trial})"
        if got is not None and got not in ref:
            return f"match returned invalid indices (trial {trial})"
    return None


def _raw_match(lib, A, B, C, D, wt1, wt2):
    sym = getattr(lib, "match_pairs_v3",
                  getattr(lib, "match_pairs_v2", None))
    out = (ctypes.c_longlong * 4)()
    st = (ctypes.c_longlong * 5)()

    def P(x):
        return np.ascontiguousarray(x, np.int16).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16))
    h = A.shape[1]
    null_f = ctypes.POINTER(ctypes.c_float)()
    null_i = ctypes.POINTER(ctypes.c_int32)()
    found = sym(P(A), len(A), P(B), len(B), P(C), len(C), P(D), len(D),
                h, wt1, wt2, null_f, null_i, null_f, null_f, null_i,
                null_f, 0, 0.0, 0.0, out, st)
    return tuple(out) if found else None


def load():
    """Returns the checked ctypes lib, or None (python fallback)."""
    if os.environ.get("H668_NATIVE") == "0":
        if _STATE["backend"] != "python":
            _STATE.update(lib=None, enabled=False, backend="python",
                          fallback_reason="H668_NATIVE=0")
        _STATE["fallback_reason"] = _STATE["fallback_reason"] \
            or "H668_NATIVE=0"
        return None
    if _STATE["checked"]:
        return _STATE["lib"]
    _STATE["checked"] = True
    path = _find_lib()
    if path is None:
        _STATE["fallback_reason"] = "libfastmatch not compiled"
        return None
    lib = ctypes.CDLL(path)
    _bind(lib)
    _STATE["lib"] = lib
    reason = _self_check(lib)
    if reason is not None:
        _fail(reason)
        return None
    backend = "scalar-c++"
    if hasattr(lib, "fm_backend"):
        backend = lib.fm_backend().decode()
    _STATE.update(enabled=True, backend=backend, fallback_reason=None)
    return lib


def status(with_speedup=False):
    load()
    if with_speedup and _STATE["enabled"] \
            and _STATE["speedup_estimate"] is None:
        _STATE["speedup_estimate"] = _estimate_speedup()
    return {"native_enabled": _STATE["enabled"],
            "native_backend": _STATE["backend"],
            "native_fallback_reason": _STATE["fallback_reason"],
            "native_speedup_estimate": _STATE["speedup_estimate"]}


def _estimate_speedup():
    """~0.5s micro-benchmark: native vs python matcher on a fixed fixture."""
    import engine
    rng = np.random.default_rng(7)
    paf = rng.integers(-40, 40, (150, 83)).astype(np.int16)
    old = os.environ.get("H668_NATIVE")
    try:
        t0 = time.perf_counter()
        engine.match(paf, paf, paf, paf, max_pairs=10**9)
        t_native = time.perf_counter() - t0
        os.environ["H668_NATIVE"] = "0"
        t0 = time.perf_counter()
        engine.match(paf, paf, paf, paf, max_pairs=10**9)
        t_py = time.perf_counter() - t0
    finally:
        if old is None:
            os.environ.pop("H668_NATIVE", None)
        else:
            os.environ["H668_NATIVE"] = old
    return round(t_py / max(t_native, 1e-9), 1)
