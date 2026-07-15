#!/usr/bin/env python3
"""Build the native module. Usage: python3 build_native.py
macOS/Apple Silicon: Apple clang from Xcode CLT, no Homebrew needed.
Tries the best flags first, falls back gracefully, then runs the
correctness oracle and refuses to bless a build that disagrees."""
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fastmatch.cpp")


def try_build(flags, out):
    cmd = ["c++", "-O3", "-std=c++17"] + flags + [SRC, "-o", out]
    print("$", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    mac = sys.platform == "darwin"
    arm = platform.machine().lower() in ("arm64", "aarch64")
    out = os.path.join(HERE,
                       "libfastmatch.dylib" if mac else "libfastmatch.so")
    shared = ["-dynamiclib"] if mac else ["-shared", "-fPIC"]
    # best -> safest flag sets (-march=native is rejected by Apple clang
    # on arm64; -mcpu=native is its equivalent)
    attempts = []
    if arm:
        attempts = [shared + ["-mcpu=native"], shared + ["-mcpu=apple-m1"],
                    shared]
    else:
        attempts = [shared + ["-march=native"], shared]
    for flags in attempts:
        r = try_build(flags, out)
        if r.returncode == 0:
            print(f"built {out} with flags {flags}")
            break
        print(r.stderr.strip().splitlines()[-1] if r.stderr else "failed")
    else:
        sys.exit("all flag sets failed; check Xcode CLT / g++ install")

    import native
    native._STATE.update(lib=None, checked=False)  # force fresh check
    lib = native.load()
    st = native.status()
    if lib is None:
        sys.exit(f"BUILD REJECTED by correctness oracle: "
                 f"{st['native_fallback_reason']}")
    print(f"correctness oracle passed; backend = {st['native_backend']}")
    print("native module ready.")


if __name__ == "__main__":
    main()
