# H668 shared test output colors
import builtins

_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"

def _color_first_arg(s: str) -> str:
    if "\033[" in s:
        return s

    stripped = s.strip()
    lower = stripped.lower()

    if lower.startswith("ok") or lower.startswith("pass") or "all " in lower and "passed" in lower:
        return _GREEN + s + _RESET

    if (
        lower.startswith("fail")
        or " failed" in lower
        or "traceback" in lower
        or "assertionerror" in lower
        or "syntaxerror" in lower
    ):
        return _RED + s + _RESET

    return s

def install():
    if getattr(builtins.print, "_h668_color_print", False):
        return

    original_print = builtins.print

    def color_print(*args, **kwargs):
        if args and isinstance(args[0], str):
            args = (_color_first_arg(args[0]),) + args[1:]
        return original_print(*args, **kwargs)

    color_print._h668_color_print = True
    builtins.print = color_print
