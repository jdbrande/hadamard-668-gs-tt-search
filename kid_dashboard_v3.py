import tempfile
import subprocess
#!/usr/bin/env python3
import os, re, json, html, time, subprocess, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

WORKDIR = Path(os.environ.get("H668_WORKDIR", "work"))
import dash_routing  # read-only routing/auto-rotation panel logic
import os, signal, subprocess
import os
import sys
import time
EXPECTED = int(os.environ.get("H668_WORKERS", "12"))
PORT = int(os.environ.get("H668_DASH_PORT", "6680"))

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def num(s):
    try:
        return int(str(s).replace(",", ""))
    except Exception:
        return 0

def big(n):
    n = int(n or 0)
    if n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.2f} trillion"
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} billion"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} million"
    return f"{n:,}"

def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

def tail(path, n=300):
    try:
        return Path(path).read_text(errors="ignore").splitlines()[-n:]
    except Exception:
        return []

def get_workers():
    out = sh("ps -axo pid,etime,%cpu,command")
    rows = []
    for line in out.splitlines():
        if "worker.py --route" not in line:
            continue
        m = re.match(r"\s*(\d+)\s+(\S+)\s+([\d.]+)\s+(.*)", line)
        if not m:
            continue
        pid, etime, cpu, cmd = m.groups()
        route = re.search(r"--route\s+(\w+)", cmd)
        wid = re.search(r"--worker-id\s+(\d+)", cmd)
        if not route or not wid:
            continue
        rows.append({
            "pid": pid,
            "etime": etime,
            "cpu": float(cpu),
            "cmd": cmd,
            "route": route.group(1),
            "id": int(wid.group(1)),
        })
    return sorted(rows, key=lambda x: x["id"])

def latest_cycle(wid):
    lines = tail(WORKDIR / f"worker_{wid:03d}" / "log.txt", 1000)

    v3 = re.compile(
        r"\[w(\d+)\s+([0-9:]+)\]\s+cycle\s+(\d+)\s+\(([\d.]+)s\)\s+"
        r"splits\s+(\d+)/(\d+)\s+new_pairs=([\d,]+)\s+"
        r"psd_rej=([\d,]+)/([\d,]+)\s+probes=([\d,]+)\s+"
        r"coll=([\d,]+)\s+rate=([\d,]+)"
    )

    old = re.compile(
        r"\[w(\d+)\s+([0-9:]+)\]\s+cycle\s+(\d+)\s+\(([\d.]+)s\).*?"
        r"pairs=([\d,]+).*?probes=([\d,]+).*?collchk=([\d,]+).*?psdskip=([\d,]+)"
    )

    last = None
    for line in lines:
        m = v3.search(line)
        if m:
            last = {
                "version": "v3",
                "time": m.group(2),
                "cycle": int(m.group(3)),
                "seconds": float(m.group(4)),
                "splits_done": int(m.group(5)),
                "splits_total": int(m.group(6)),
                "new_pairs": num(m.group(7)),
                "build_psd_rej": num(m.group(8)),
                "probe_psd_rej": num(m.group(9)),
                "probes": num(m.group(10)),
                "collisions": num(m.group(11)),
                "rate": num(m.group(12)),
                "raw": line,
            }
            continue

        m = old.search(line)
        if m:
            pairs = num(m.group(5))
            probes = num(m.group(6))
            seconds = float(m.group(4))
            last = {
                "version": "log",
                "time": m.group(2),
                "cycle": int(m.group(3)),
                "seconds": seconds,
                "splits_done": 0,
                "splits_total": 0,
                "new_pairs": pairs,
                "build_psd_rej": 0,
                "probe_psd_rej": num(m.group(8)),
                "probes": probes,
                "collisions": num(m.group(7)),
                "rate": int(pairs / seconds) if seconds > 0 else 0,
                "raw": line,
            }
    return last

STALE_AFTER = 600  # seconds without an update -> quiet


def _progress_raw(wid):
    """Telemetry for one worker. Prefers progress.json (first-class, written
    atomically by the worker each cycle); falls back to parsing the latest
    log line for runs that predate telemetry. Always annotates:
      source   "progress.json" | "log fallback" | "none"
      updated  unix time of last telemetry update (file field or log mtime)
      quiet    True if no update for STALE_AFTER seconds
    """
    ppath = WORKDIR / f"worker_{wid:03d}" / "progress.json"
    p = read_json(ppath)
    if p:
        p["source"] = "progress.json"
        p["updated"] = float(p.get("updated") or ppath.stat().st_mtime)
        p["quiet"] = (time.time() - p["updated"]) > STALE_AFTER
        return p

    c = latest_cycle(wid)
    if not c:
        return {"source": "none", "updated": 0, "quiet": True, "totals": {}}

    lpath = WORKDIR / f"worker_{wid:03d}" / "log.txt"
    try:
        updated = lpath.stat().st_mtime
    except OSError:
        updated = 0
    psd_total = int(c.get("build_psd_rej", 0)) + int(c.get("probe_psd_rej", 0))
    secs = float(c.get("seconds", 0)) or 1.0
    return {
        "source": "log fallback",
        "updated": updated,
        "quiet": (time.time() - updated) > STALE_AFTER,
        "cycle": int(c.get("cycle", 0)),
        "cycle_seconds": secs,
        "partial_telemetry": True,   # logs only show the latest cycle
        "latest_cycle": {
            "pairs_hashed": int(c.get("new_pairs", 0)),
            "probes": int(c.get("probes", 0)),
            "psd_rejected": psd_total,
            "collisions": int(c.get("collisions", 0)),
            "rate_pairs_per_sec": int(c.get("new_pairs", 0) / secs),
            "rate_probes_per_sec": int(c.get("probes", 0) / secs),
            "rate_psd_rej_per_sec": int(psd_total / secs),
        },
        "totals": {
            "pairs_hashed": int(c.get("new_pairs", 0)),
            "probes": int(c.get("probes", 0)),
            "probe_psd_rej": psd_total,
            "collisions": int(c.get("collisions", 0)),
            "generated": 0,
            "accepted": 0,
            "dup_rejected": 0,
        },
    }



def progress(wid):
    d = _progress_raw(wid)
    if isinstance(d, dict):
        updated = d.get("updated") or d.get("mtime") or 0
        try:
            updated = float(updated)
        except (TypeError, ValueError):
            updated = 0
        d["stale"] = bool(updated and (time.time() - updated) > 600)
    return d


def age_str(ts):
    if not ts:
        return "-"
    a = max(0, time.time() - ts)
    if a < 90:
        return f"{a:.0f}s"
    if a < 5400:
        return f"{a/60:.0f}m"
    return f"{a/3600:.1f}h"


def trend_str(p):
    seq = (p or {}).get("recent_cycle_seconds") or []
    if len(seq) < 2:
        return "-"
    return " → ".join(f"{s/60:.0f}m" for s in seq[-5:])

def meta(wid):
    return read_json(WORKDIR / f"worker_{wid:03d}" / "meta.json")

def solution_status():
    sol = WORKDIR / "SOLUTION.json"
    if sol.exists():
        data = read_json(sol)
        return True, data
    return False, {}

def scary_errors():
    hits = []
    for p in list(WORKDIR.glob("launch_w*.out")) + list(WORKDIR.glob("worker_*/log.txt")):
        for line in tail(p, 80):
            if any(x in line for x in ["Traceback", "BadZipFile", "Exception", "Error"]):
                hits.append(f"{p}: {line}")
    return hits[-12:]

def page():
    workers = get_workers()
    alive = len(workers)
    ids_alive = {w["id"] for w in workers}
    tt = [w for w in workers if w["route"] == "tt"]
    gs = [w for w in workers if w["route"] == "gs"]
    found, sol = solution_status()
    errors = scary_errors()

    cycles = {i: latest_cycle(i) for i in range(EXPECTED)}
    metas = {i: meta(i) for i in range(EXPECTED)}
    progs = {i: progress(i) for i in range(EXPECTED)}

    def latest_of(i, key, logkey=None, logextra=None):
        lc = progs[i].get("latest_cycle")
        if lc:
            return int(lc.get(key, 0))
        c = cycles[i] or {}
        v = int(c.get(logkey or key, 0))
        if logextra:
            v += int(c.get(logextra, 0))
        return v

    latest_pairs = sum(latest_of(i, "pairs_hashed", "new_pairs")
                       for i in range(EXPECTED))
    latest_probes = sum(latest_of(i, "probes") for i in range(EXPECTED))
    latest_psd = sum(latest_of(i, "psd_rejected", "build_psd_rej",
                               "probe_psd_rej") for i in range(EXPECTED))
    latest_coll = sum(latest_of(i, "collisions") for i in range(EXPECTED))
    latest_rate = sum(latest_of(i, "rate_pairs_per_sec", "rate")
                      for i in range(EXPECTED))
    latest_probe_rate = sum(latest_of(i, "rate_probes_per_sec")
                            for i in range(EXPECTED))
    latest_psd_rate = sum(latest_of(i, "rate_psd_rej_per_sec")
                          for i in range(EXPECTED))
    quiet_ids = [i for i in range(EXPECTED) if progs[i].get("quiet")]
    n_progress = sum(1 for i in range(EXPECTED)
                     if progs[i].get("source") == "progress.json")

    total_pairs = 0
    total_probes = 0
    total_psd = 0
    total_coll = 0
    total_generated = 0
    total_accepted = 0
    total_dup = 0

    for i in range(EXPECTED):
        t = progs[i].get("totals", {})
        total_pairs += int(t.get("pairs_hashed", 0))
        total_probes += int(t.get("probes", 0))
        bp = int(t.get("build_psd_rej", 0))
        pp = int(t.get("probe_psd_rej", 0))
        total_psd += (bp + pp) if (bp or pp) else int(t.get("psd_rejected", 0))
        total_coll += int(t.get("collisions", 0))
        total_generated += int(t.get("generated", 0))
        total_accepted += int(t.get("accepted", 0))
        total_dup += int(t.get("dup_rejected", 0))

    if found:
        headline = "🎉 TREASURE FOUND"
        explain = "A solution file exists. Verify it before celebrating too hard."
    elif alive == EXPECTED and quiet_ids:
        headline = "🟡 Some workers look quiet"
        explain = (f"All {EXPECTED} are awake but "
                   f"{', '.join('w'+str(i) for i in quiet_ids)} "
                   f"has not updated telemetry in over 10 minutes.")
    elif alive == EXPECTED:
        headline = "🟢 Search looks healthy"
        explain = f"All {EXPECTED} little workers are awake."
    elif alive < EXPECTED:
        headline = "🟡 Some workers are missing"
        explain = f"Only {alive} of {EXPECTED} workers are awake."
    else:
        headline = "🟡 Check status"
        explain = "Something is unusual, but the dashboard is still reading logs."

    def card(title, value, desc):
        return f"""
        <div class="card">
          <div class="title">{html.escape(title)}</div>
          <div class="value">{value}</div>
          <div class="desc">{html.escape(desc)}</div>
        </div>
        """

    worker_rows = ""
    for i in range(EXPECTED):
        w = next((x for x in workers if x["id"] == i), None)
        m = metas[i]
        c = cycles[i]
        p = progs[i]
        route = (w or {}).get("route") or m.get("route") \
            or p.get("route", "?")
        cpu = f"{w['cpu']:.1f}%" if w else "-"
        etime = html.escape(w["etime"]) if w else "-"
        pid = html.escape(w["pid"]) if w else "-"
        state = "ALIVE" if i in ids_alive else "dead"
        if p.get("quiet") and i in ids_alive:
            state = "BUSY?"
        cyc = p.get("cycle") or (c["cycle"] if c else "-")
        lc = p.get("latest_cycle") or {}
        pairs = big(lc.get("pairs_hashed", (c or {}).get("new_pairs", 0)))
        probes = big(lc.get("probes", (c or {}).get("probes", 0)))
        src = p.get("source", "none")
        upd = age_str(p.get("updated"))
        worker_rows += f"""
        <tr>
          <td>w{i}</td><td>{html.escape(str(route).upper())}</td><td>{state}</td>
          <td>{cpu}</td><td>{etime}</td><td>{pid}</td>
          <td>{cyc}</td><td>{pairs}</td><td>{probes}</td>
          <td>{html.escape(src)}</td><td>{upd} ago</td>
          <td>{html.escape(trend_str(p))}</td>
        </tr>
        """

    cycle_rows = ""
    for i in range(EXPECTED):
        c = cycles[i]
        if not c:
            continue
        route = metas[i].get("route", "?")
        cycle_rows += f"""
        <tr>
          <td>w{i}</td>
          <td>{html.escape(str(route).upper())}</td>
          <td>{c['cycle']}</td>
          <td>{c['seconds']/60:.1f} min</td>
          <td>{c.get('splits_done', 0)}/{c.get('splits_total', 0)}</td>
          <td>{c['new_pairs']:,}</td>
          <td>{c['probes']:,}</td>
          <td>{c['build_psd_rej'] + c['probe_psd_rej']:,}</td>
          <td>{c['collisions']:,}</td>
          <td>{c.get('rate', 0):,}</td>
        </tr>
        """

    n_native = sum(1 for i in range(EXPECTED)
                   if progs[i].get("native_enabled"))
    backends = {progs[i].get("native_backend") for i in range(EXPECTED)
                if progs[i].get("native_enabled")}
    fallbacks = [f"w{i}: {progs[i]['native_fallback_reason']}"
                 for i in range(EXPECTED)
                 if progs[i].get("source") == "progress.json"
                 and not progs[i].get("native_enabled")
                 and progs[i].get("native_fallback_reason")]
    native_card = f"{n_native}/{EXPECTED}"
    if backends:
        native_card += " " + "/".join(sorted(b for b in backends if b))
    if fallbacks:
        native_card += " ⚠"

    cumulative_rows = "<table><tr><th>ID</th><th>Generated</th>" \
        "<th>Accepted</th><th>Dup rejects</th><th>Pairs</th><th>Probes</th>" \
        "<th>PSD rejects</th><th>Telemetry</th><th>Native</th></tr>"
    for i in range(EXPECTED):
        t = progs[i].get("totals", {})
        if not t:
            continue
        flag = "partial" if progs[i].get("partial_telemetry") else "full"
        psd_t = (int(t.get("build_psd_rej", 0))
                 + int(t.get("probe_psd_rej", 0))
                 or int(t.get("psd_rejected", 0)))
        cumulative_rows += (
            f"<tr><td>w{i}</td><td>{int(t.get('generated', 0)):,}</td>"
            f"<td>{int(t.get('accepted', 0)):,}</td>"
            f"<td>{int(t.get('dup_rejected', 0)):,}</td>"
            f"<td>{big(t.get('pairs_hashed', 0))}</td>"
            f"<td>{big(t.get('probes', 0))}</td>"
            f"<td>{big(psd_t)}</td><td>{flag}</td>"
            f"<td>{html.escape(str(progs[i].get('native_backend', '-')))}"
            f"{' (est ' + str(progs[i].get('native_speedup_estimate')) + 'x)' if progs[i].get('native_speedup_estimate') else ''}"
            f"</td></tr>")
    cumulative_rows += "</table>"
    if fallbacks:
        cumulative_rows += ("<p><b>⚠ Native fallbacks:</b> "
                            + html.escape("; ".join(fallbacks)) + "</p>")

    err_html = "<p>No scary errors found.</p>" if not errors else "<ul>" + "".join(f"<li>{html.escape(e)}</li>" for e in errors) + "</ul>"

    sol_html = ""
    if found:
        sol_html = "<pre>" + html.escape(json.dumps(sol, indent=2)[:4000]) + "</pre>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>H668 v3 Kid Dashboard</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
  background:#111;
  color:#eee;
  margin:24px;
}}
h1 {{ font-size:42px; margin-bottom:4px; }}
.sub {{ color:#aaa; margin-bottom:22px; }}
.grid {{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:16px;
}}
.card {{
  background:#1d1d1d;
  border:1px solid #333;
  border-radius:18px;
  padding:18px;
}}
.title {{ color:#aaa; font-size:15px; }}
.value {{ font-size:32px; font-weight:800; margin:8px 0; }}
.desc {{ color:#ccc; line-height:1.35; }}
.section {{
  margin-top:26px;
  background:#181818;
  border:1px solid #333;
  border-radius:18px;
  padding:18px;
}}
table {{
  width:100%;
  border-collapse:collapse;
  margin-top:12px;
  background:#151515;
}}
th, td {{
  text-align:left;
  padding:9px;
  border-bottom:1px solid #333;
  font-size:14px;
}}
th {{ color:#aaa; }}
code, pre {{
  background:#101010;
  border:1px solid #333;
  border-radius:10px;
  padding:10px;
  display:block;
  overflow:auto;
}}
button {{
  font-size:20px;
  padding:10px 16px;
  border-radius:10px;
  border:1px solid #555;
  background:#333;
  color:white;
}}
input {{
  font-size:20px;
  padding:10px;
  border-radius:10px;
  border:1px solid #555;
  background:#222;
  color:#eee;
}}

/* PRETTY_DASHBOARD_PATCH_SAFE */
:root {{
  --dash-card: rgba(29, 36, 51, 0.88);
  --dash-card-hover: rgba(36, 47, 68, 0.96);
  --dash-line: rgba(255, 255, 255, 0.12);
  --dash-row-a: rgba(255, 255, 255, 0.035);
  --dash-row-b: rgba(255, 255, 255, 0.075);
  --dash-row-hover: rgba(105, 180, 255, 0.17);
  --dash-blue: rgba(125, 190, 255, 0.9);
}}

body {{
  background:
    radial-gradient(circle at top left, rgba(65, 113, 255, 0.22), transparent 34rem),
    radial-gradient(circle at 85% 8%, rgba(21, 230, 180, 0.13), transparent 28rem),
    linear-gradient(180deg, #0b1020 0%, #101420 60%, #0b1020 100%);
}}

.card {{
  position: relative;
  overflow: hidden;
  background: var(--dash-card);
  border: 1px solid var(--dash-line);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}}

.card::before {{
  content: "";
  position: absolute;
  inset: -1px;
  background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.16), transparent);
  transform: translateX(-130%);
  transition: transform 0.45s ease;
  pointer-events: none;
}}

.card:hover {{
  transform: translateY(-5px) scale(1.015);
  background: var(--dash-card-hover);
  border-color: rgba(125, 190, 255, 0.72);
  box-shadow:
    0 18px 42px rgba(0, 0, 0, 0.36),
    0 0 0 1px rgba(125, 190, 255, 0.18),
    0 0 24px rgba(125, 190, 255, 0.10);
}}

.card:hover::before {{
  transform: translateX(130%);
}}

.card:hover .value {{
  text-shadow: 0 0 18px rgba(125, 190, 255, 0.42);
}}

.section {{
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
  backdrop-filter: blur(8px);
}}

table {{
  border-collapse: separate;
  border-spacing: 0;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid var(--dash-line);
}}

th {{
  position: sticky;
  top: 0;
  z-index: 1;
  background: #182033;
  border-bottom: 1px solid rgba(255, 255, 255, 0.16);
}}

table tr:nth-child(odd) td {{
  background: var(--dash-row-a);
}}

table tr:nth-child(even) td {{
  background: var(--dash-row-b);
}}

table tr:hover td {{
  background: var(--dash-row-hover);
  color: #ffffff;
}}

td {{
  transition: background 0.14s ease, color 0.14s ease;
}}

td:first-child {{
  font-weight: 700;
  color: #d7e8ff;
}}

.value {{
  letter-spacing: -0.03em;
}}

button {{
  transition: transform 0.14s ease, background 0.14s ease, border-color 0.14s ease;
}}

button:hover {{
  transform: translateY(-2px);
  background: #3f526e;
  border-color: var(--dash-blue);
}}

input:focus {{
  outline: none;
  border-color: var(--dash-blue);
  box-shadow: 0 0 0 3px rgba(125, 190, 255, 0.18);
}}

pre, code {{
  border-color: rgba(125, 190, 255, 0.22);
}}

@media (prefers-reduced-motion: reduce) {{
  .card, .card::before, td, button {{
    transition: none;
  }}
  .card:hover, button:hover {{
    transform: none;
  }}
}}


/* HIDE_CLICK_FOR_LOGS_TEXT */
.clickable-card .desc::after {{
  content: "" !important;
}}

</style>
</head>
<body>
<h1>{html.escape(headline)}</h1>
<div class="sub">{html.escape(explain)} Refreshes every 15 seconds.</div>

<div class="grid">
{card("Workers awake", f"{alive}/{EXPECTED}", "You want 12/12.")}
{card("TT quiet thinkers", f"{len(tt)}/8", "High CPU means they are working, even if quiet.")}
{card("GS chatty checkers", f"{len(gs)}/4", "These usually report cycles more often.")}
{card("Latest new pair pile", big(latest_pairs), "New pair hashes in the latest visible cycle logs.")}
{card("Latest rocks checked", big(latest_probes), "Probe checks in the latest visible cycle logs.")}
{card("Latest impossible skips", big(latest_psd), "PSD rejects. These were skipped because math proved they cannot work.")}
{card("Cumulative probes", big(total_probes), "Total probes from progress.json or latest worker logs.")}
{card("Cumulative pairs", big(total_pairs), "Total pair hashes from progress.json or latest worker logs.")}
{card("Pair ops/sec", f"{latest_rate:,}", "Latest-cycle pair hashing speed across workers.")}
{card("Probe ops/sec", f"{latest_probe_rate:,}", "Latest-cycle probe speed across workers.")}
{card("PSD rejects/sec", f"{latest_psd_rate:,}", "How fast impossible pairs are being skipped.")}
{card("Telemetry", f"{n_progress}/{EXPECTED}", "Workers reporting via progress.json (rest are log fallback).")}
{card("Quiet workers", f"{len(quiet_ids)}", "Workers silent for over 10 minutes. 0 is good.")}
{card("Native engine", native_card, "Workers running the compiled monster-mode matcher.")}
{card("Promising clues", f"{total_coll:,}", "Collision checks. Zero is normal. Bigger means near-hit activity.")}
</div>

<div class="section">
<h2>Explain it like I’m 5</h2>
<p><b>new_pairs</b> means the workers made new puzzle-piece pairs.</p>
<p><b>probes</b> means they checked if those pairs fit with other pairs.</p>
<p><b>PSD rejects</b> means the math said “nope, impossible” before wasting time.</p>
<p><b>collisions / promising clues</b> means something got close enough to check harder.</p>
<p><b>MATCH → VERIFIED → SOLUTION</b> means treasure found.</p>
</div>

<div class="section">
{dash_routing.render_routing_html()}

<h2>Workers</h2>
<table>
<tr><th>ID</th><th>Route</th><th>State</th><th>CPU</th><th>Runtime</th><th>PID</th><th>Cycle</th><th>Latest pairs</th><th>Latest probes</th><th>Source</th><th>Updated</th><th>Cycle trend</th></tr>
{worker_rows}
</table>
</div>

<div class="section">
<h2>Latest cycle detail</h2>
<table>
<tr><th>ID</th><th>Route</th><th>Cycle</th><th>Time</th><th>Splits</th><th>New pairs</th><th>Probes</th><th>PSD rejects</th><th>Collisions</th><th>Pair ops/sec</th></tr>
{cycle_rows or '<tr><td colspan="10">No cycle logs yet. Workers may still be inside their first v3 baseline pass.</td></tr>'}
</table>
</div>

<div class="section">
<h2>Cumulative totals (all time, from progress.json)</h2>
<p class="sub">Latest-cycle numbers live in the table above; these are
lifetime sums. Workers marked "partial" predate telemetry, so their
generated/accepted/duplicate history before telemetry began is honestly
unknown and counted from 0.</p>
{cumulative_rows}
<p><b>Generated:</b> {total_generated:,}</p>
<p><b>Accepted:</b> {total_accepted:,}</p>
<p><b>Duplicate rejects:</b> {total_dup:,}</p>
<p><b>PSD / impossible rejects:</b> {total_psd:,}</p>
<p><b>Pair hashes:</b> {total_pairs:,}</p>
<p><b>Probe checks:</b> {total_probes:,}</p>
<p><b>Promising clues:</b> {total_coll:,}</p>
</div>

<div class="section">
<h2>Solution file</h2>
{sol_html or '<p>No solution file yet.</p>'}
</div>

<div class="section">
<h2>Stop Dashboard Only</h2>
<p>This turns off the web page only. It does not stop the H668 workers.</p>
<form method="POST" action="/shutdown" onsubmit="return confirm('Stop only the dashboard? Search workers keep running.');">
  <p>Type <b>DASHBOARD</b>:</p>
  <input name="confirm" autocomplete="off" placeholder="type DASHBOARD">
  <button type="submit">Stop dashboard only</button>
</form>
</div>

<div class="section">
<h2>Recent scary errors</h2>
{err_html}
</div>


<div id="logDrawer" class="log-drawer" aria-hidden="true">
  <div class="log-drawer-backdrop" onclick="closeLogDrawer()"></div>
  <aside class="log-drawer-panel">
    <div class="log-drawer-top">
      <div>
        <div class="log-drawer-label">Live log view</div>
        <h2 id="logDrawerTitle">Logs</h2>
      </div>
      <div class="log-drawer-actions">
        <button type="button" onclick="copyLogDrawer()">Copy</button>
        <button type="button" onclick="closeLogDrawer()">Close</button>
      </div>
    </div>
    <pre id="logDrawerBody">Click a card or worker row.</pre>
  </aside>
</div>

<script>
(function () {{
  const cardMap = new Map([
    ["Workers awake", "workers"],
    ["TT quiet thinkers", "tt"],
    ["GS chatty checkers", "gs"],
    ["Native engine", "native"],
    ["Latest new pair pile", "cycles"],
    ["Latest rocks checked", "cycles"],
    ["Latest impossible skips", "cycles"],
    ["Cumulative probes", "cycles"],
    ["Cumulative pairs", "cycles"],
    ["Pair ops/sec", "cycles"],
    ["Probe ops/sec", "cycles"],
    ["Promising clues", "cycles"],
    ["Generated", "acceptance"],
    ["Accepted", "acceptance"],
    ["Duplicate rejects", "acceptance"],
    ["PSD / impossible rejects", "acceptance"]
  ]);

  window.openLogDrawer = async function (title, url) {{
    const drawer = document.getElementById("logDrawer");
    const titleEl = document.getElementById("logDrawerTitle");
    const bodyEl = document.getElementById("logDrawerBody");
    titleEl.textContent = title;
    bodyEl.textContent = "Loading...";
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    try {{
      const res = await fetch(url, {{ cache: "no-store" }});
      bodyEl.textContent = await res.text();
    }} catch (err) {{
      bodyEl.textContent = "Could not load logs:" + String.fromCharCode(10) + err;
    }}
  }};

  window.closeLogDrawer = function () {{
    const drawer = document.getElementById("logDrawer");
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
  }};

  window.copyLogDrawer = async function () {{
    const bodyEl = document.getElementById("logDrawerBody");
    await navigator.clipboard.writeText(bodyEl.textContent || "");
  }};

  function attachCards() {{
    document.querySelectorAll(".card").forEach(card => {{
      const titleEl = card.querySelector(".title");
      if (!titleEl) return;
      const title = titleEl.textContent.trim();
      const kind = cardMap.get(title);
      if (!kind) return;
      card.classList.add("clickable-card");
      card.addEventListener("click", () => openLogDrawer(title, "/logs?kind=" + encodeURIComponent(kind)));
    }});
  }}

  function attachRows() {{
    document.querySelectorAll("table tr").forEach(row => {{
      const first = row.querySelector("td:first-child");
      if (!first) return;
      const txt = first.textContent.trim();
      const m = /^w(\d+)$/.exec(txt);
      if (!m) return;
      row.classList.add("clickable-row");
      row.addEventListener("click", () => openLogDrawer("Worker " + txt, "/logs?kind=worker&id=" + encodeURIComponent(m[1])));
    }});
  }}

  document.addEventListener("keydown", event => {{
    if (event.key === "Escape") closeLogDrawer();
  }});

  document.addEventListener("DOMContentLoaded", () => {{
    attachCards();
    attachRows();
  }});
}})();
</script>

<style>
/* LOG_DRAWER_PATCH_SAFE */
.clickable-card {{
  cursor: pointer;
}}

.clickable-row {{
  cursor: pointer;
}}

.clickable-card .desc::after {{
  content: "  • click for logs";
  color: rgba(125, 190, 255, 0.8);
}}

.log-drawer {{
  position: fixed;
  inset: 0;
  z-index: 9999;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s ease;
}}

.log-drawer.open {{
  opacity: 1;
  pointer-events: auto;
}}

.log-drawer-backdrop {{
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.58);
  backdrop-filter: blur(4px);
}}

.log-drawer-panel {{
  position: absolute;
  top: 0;
  right: 0;
  width: min(980px, 88vw);
  height: 100%;
  background:
    radial-gradient(circle at top left, rgba(80, 160, 255, 0.16), transparent 22rem),
    #0d1322;
  border-left: 1px solid rgba(125, 190, 255, 0.25);
  box-shadow: -24px 0 60px rgba(0, 0, 0, 0.45);
  transform: translateX(100%);
  transition: transform 0.22s ease;
  display: flex;
  flex-direction: column;
}}

.log-drawer.open .log-drawer-panel {{
  transform: translateX(0);
}}

.log-drawer-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}}

.log-drawer-label {{
  color: #8fbfff;
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}

.log-drawer-top h2 {{
  margin: 4px 0 0;
  font-size: 28px;
}}

.log-drawer-actions {{
  display: flex;
  gap: 10px;
}}

#logDrawerBody {{
  margin: 0;
  flex: 1;
  overflow: auto;
  border: 0;
  border-radius: 0;
  background: #090d16;
  color: #d8e8ff;
  font-size: 13px;
  line-height: 1.45;
  padding: 18px;
  white-space: pre-wrap;
}}

@media (prefers-reduced-motion: reduce) {{
  .log-drawer,
  .log-drawer-panel {{
    transition: none;
  }}
}}
</style>


<div id="quickLogPanel" style="display:none; position:fixed; inset:24px; z-index:99999; background:#08101d; color:#dbeafe; border:1px solid #6aa8ff; border-radius:16px; box-shadow:0 24px 80px rgba(0,0,0,.6); overflow:hidden;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; padding:14px 18px; background:#111b2e; border-bottom:1px solid rgba(255,255,255,.14);">
    <strong id="quickLogTitle">Logs</strong>
    <div>
      <button onclick="navigator.clipboard.writeText(document.getElementById('quickLogBody').textContent)">Copy</button>
      <button onclick="document.getElementById('quickLogPanel').style.display='none'">Close</button>
    </div>
  </div>
  <pre id="quickLogBody" style="margin:0; padding:16px; height:calc(100% - 56px); overflow:auto; white-space:pre-wrap; font-size:13px; line-height:1.4;">Loading...</pre>
</div>

<script>
/* FORCE_CLICK_LOGS */
(function() {{
  function openLogs(title, kind, id) {{
    var panel = document.getElementById("quickLogPanel");
    var titleEl = document.getElementById("quickLogTitle");
    var body = document.getElementById("quickLogBody");
    titleEl.textContent = title;
    body.textContent = "Loading...";
    panel.style.display = "block";
    var url = "/logs?kind=" + encodeURIComponent(kind);
    if (id !== undefined && id !== null) url += "&id=" + encodeURIComponent(id);
    fetch(url, {{cache:"no-store"}})
      .then(function(r) {{ return r.text(); }})
      .then(function(t) {{ body.textContent = t; }})
      .catch(function(e) {{ body.textContent = "Failed to load logs:" + String.fromCharCode(10) + e; }});
  }}

  function cardKind(text) {{
    text = text.toLowerCase();
    if (text.indexOf("workers awake") >= 0) return ["Workers awake", "workers"];
    if (text.indexOf("tt quiet") >= 0) return ["TT workers", "tt"];
    if (text.indexOf("gs chatty") >= 0) return ["GS workers", "gs"];
    if (text.indexOf("native engine") >= 0) return ["Native engine", "native"];
    if (text.indexOf("cumulative") >= 0 || text.indexOf("latest") >= 0 || text.indexOf("ops/sec") >= 0 || text.indexOf("rocks") >= 0 || text.indexOf("pile") >= 0) return ["Cycle logs", "cycles"];
    if (text.indexOf("accepted") >= 0 || text.indexOf("generated") >= 0 || text.indexOf("duplicate") >= 0 || text.indexOf("psd") >= 0) return ["Acceptance", "acceptance"];
    if (text.indexOf("error") >= 0 || text.indexOf("scary") >= 0) return ["Errors", "errors"];
    return null;
  }}

  document.addEventListener("click", function(e) {{
    var card = e.target.closest(".card");
    if (card) {{
      var picked = cardKind(card.textContent || "");
      if (picked) {{
        e.preventDefault();
        e.stopPropagation();
        openLogs(picked[0], picked[1]);
        return false;
      }}
    }}

    var row = e.target.closest("tr");
    if (row) {{
      var txt = row.textContent || "";
      var m = txt.match(/\bw0?(\d{{1,2}})\b/) || txt.match(/\bworker[_ ]?0?(\d{{1,2}})\b/i);
      if (m) {{
        var wid = parseInt(m[1], 10);
        if (wid >= 0 && wid < 12) {{
          e.preventDefault();
          e.stopPropagation();
          openLogs("Worker " + wid, "worker", wid);
          return false;
        }}
      }}
    }}
  }}, true);

  document.addEventListener("keydown", function(e) {{
    if (e.key === "Escape") {{
      var panel = document.getElementById("quickLogPanel");
      if (panel) panel.style.display = "none";
    }}
  }});

  var style = document.createElement("style");
  style.textContent = ".card{{cursor:pointer}}.card:hover{{outline:1px solid rgba(125,190,255,.75)}}tr{{cursor:pointer}}";
  document.head.appendChild(style);
}})();
</script>


<script>
/* POOLS_CARD_PATCH_SAFE */
(function () {{
  function fmt(n) {{
    n = Number(n || 0);
    return n.toLocaleString();
  }}

  function ensurePoolsCard() {{
    if (document.getElementById("poolsCard")) return;

    const firstCard = document.querySelector(".card");
    if (!firstCard || !firstCard.parentElement) return;

    const card = document.createElement("div");
    card.className = "card clickable-card";
    card.id = "poolsCard";
    card.title = "Click to open pool details";
    card.innerHTML =
      '<div class="title">Pools</div>' +
      '<div class="value" id="poolsValue">...</div>' +
      '<div class="desc" id="poolsDesc">loading pool cap fill</div>';

    card.addEventListener("click", function () {{
      if (typeof window.openLogDrawer === "function") {{
        window.openLogDrawer("Pools", "/logs?kind=pools");
      }} else {{
        window.open("/logs?kind=pools", "_blank");
      }}
    }});

    firstCard.parentElement.appendChild(card);
  }}

  async function updatePoolsCard() {{
    ensurePoolsCard();

    const value = document.getElementById("poolsValue");
    const desc = document.getElementById("poolsDesc");
    if (!value || !desc) return;

    try {{
      const res = await fetch("/pools.json", {{ cache: "no-store" }});
      const data = await res.json();

      value.textContent = fmt(data.total);
      desc.textContent =
        data.fill_pct + "% filled • " +
        fmt(data.min_bin) + " min / " +
        fmt(data.avg_bin) + " avg / " +
        fmt(data.max_bin) + " max • " +
        data.at_cap + " bins capped";
    }} catch (err) {{
      value.textContent = "error";
      desc.textContent = "pool summary unavailable";
    }}
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", function () {{
      updatePoolsCard();
      setInterval(updatePoolsCard, 15000);
    }});
  }} else {{
    updatePoolsCard();
    setInterval(updatePoolsCard, 15000);
  }}
}})();
</script>


<style>
#h668ShotBtn {{
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 99999;
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,.25);
  background: rgba(20, 25, 35, .92);
  color: #fff;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 8px 24px rgba(0,0,0,.35);
}}
#h668ShotBtn:hover {{
  transform: translateY(-1px);
}}
</style>

<!-- old floating Save Screenshot button removed; Snapshot now lives in H668 Controls -->

<script>
async function captureFullDashboardScreenshot() {{
  const btn = document.getElementById("h668ShotBtn");
  const oldText = btn.textContent;
  btn.textContent = "Capturing...";
  btn.style.visibility = "hidden";

  try {{
    const width = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth, window.innerWidth);
    const height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight, window.innerHeight);

    const clone = document.documentElement.cloneNode(true);
    const killBtn = clone.querySelector("#h668ShotBtn");
    if (killBtn) killBtn.remove();

    const xml = new XMLSerializer().serializeToString(clone);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${{width}}" height="${{height}}">
      <foreignObject width="100%" height="100%">${{xml}}</foreignObject>
    </svg>`;

    const blob = new Blob([svg], {{type: "image/svg+xml;charset=utf-8"}});
    const url = URL.createObjectURL(blob);

    const img = new Image();
    img.onload = function() {{
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#0b0f17";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0);

      URL.revokeObjectURL(url);

      const a = document.createElement("a");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      a.download = `h668_dashboard_${{stamp}}.png`;
      a.href = canvas.toDataURL("image/png");
      a.click();

      btn.style.visibility = "visible";
      btn.textContent = oldText;
    }};

    img.onerror = function() {{
      URL.revokeObjectURL(url);
      btn.style.visibility = "visible";
      btn.textContent = "Screenshot failed";
      setTimeout(function() {{ btn.textContent = oldText; }}, 2000);
      alert("Screenshot failed. Use Chrome full-page screenshot instead.");
    }};

    img.src = url;
  }} catch (err) {{
    btn.style.visibility = "visible";
    btn.textContent = "Screenshot failed";
    setTimeout(function() {{ btn.textContent = oldText; }}, 2000);
    alert("Screenshot failed: " + err);
  }}
}}
</script>


<!-- server screenshot override -->

<script>
window.captureFullDashboardScreenshot = async function() {{
  const btn = document.getElementById("h668ShotBtn");
  const oldText = btn ? btn.textContent : "Save Screenshot";

  if (btn) {{
    btn.textContent = "Capturing...";
    btn.disabled = true;
  }}

  try {{
    const res = await fetch("/screenshot.png?ts=" + Date.now());
    if (!res.ok) {{
      throw new Error(await res.text());
    }}

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = "h668_dashboard_" + stamp + ".png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    if (btn) {{
      btn.textContent = oldText;
      btn.disabled = false;
    }}
  }} catch (err) {{
    if (btn) {{
      btn.textContent = "Screenshot failed";
      btn.disabled = false;
      setTimeout(function() {{ btn.textContent = oldText; }}, 2000);
    }}
    alert("Screenshot failed: " + err);
  }}
}};
</script>





<!-- H668 popup command modal -->
<div id="h668-popup-controls-root"></div>
<script src="/h668-popup-controls.js"></script>
<script src="/h668-minimize-panel.js"></script>
<script src="/h668-chip-layout.js"></script>
<script src="/h668-snapshot-download.js"></script>
<script src="/h668-remove-snapshot-popup.js"></script>






</body>
</html>
"""


def log_view(raw_path):
    parsed = urlparse(raw_path)
    q = parse_qs(parsed.query)
    kind = (q.get("kind") or ["workers"])[0]
    wid_arg = (q.get("id") or [""])[0]

    def section(title, body):
        return f"=== {title} ===\n{body.strip()}\n"

    def worker_progress(wid):
        p = WORKDIR / f"worker_{wid:03d}" / "progress.json"
        if not p.exists():
            return f"worker_{wid:03d}: no progress.json"
        d = read_json(p)
        age = time.time() - p.stat().st_mtime
        lines = [
            f"worker_{wid:03d}",
            f"route: {d.get('route')}",
            f"cycle: {d.get('cycle')}",
            f"status: {d.get('status')}",
            f"age: {age:.1f}s",
            f"pool_cap: {d.get('pool_cap')}",
            f"bins_at_cap: {d.get('bins_at_cap')}",
            f"native_enabled: {d.get('native_enabled')}",
            f"native_backend: {d.get('native_backend')}",
            f"native_fallback_reason: {d.get('native_fallback_reason')}",
            "",
            "latest_cycle:",
            json.dumps(d.get("latest_cycle", {}), indent=2),
            "",
            "totals:",
            json.dumps(d.get("totals", {}), indent=2),
        ]
        return "\n".join(lines)

    def worker_log(wid, n=220):
        path = WORKDIR / f"worker_{wid:03d}" / "log.txt"
        lines = tail(path, n)
        return "\n".join(lines) if lines else f"no log found for worker_{wid:03d}"

    def route_logs(route):
        chunks = []
        for wid in range(EXPECTED):
            m = meta(wid)
            prog = read_json(WORKDIR / f"worker_{wid:03d}" / "progress.json")
            r = (prog.get("route") or m.get("route") or "").lower()
            if r != route:
                continue
            lines = tail(WORKDIR / f"worker_{wid:03d}" / "log.txt", 80)
            useful = [x for x in lines if "cycle " in x or "absorbed" in x or "MATCH" in x or "SOLUTION" in x]
            chunks.append(section(f"worker_{wid:03d} {route.upper()}", "\n".join(useful[-40:]) or "no recent route lines"))
        return "\n".join(chunks) or f"no {route} workers found"

    if kind == "pools":
        return pools_summary_text()

    if kind == "workers":
        body = sh('ps -axo pid,etime,%cpu,command | grep "[w]orker.py --route" | sort -k1')
        return section("live worker processes", body or "no workers found")

    if kind == "tt":
        return route_logs("tt")

    if kind == "gs":
        return route_logs("gs")

    if kind == "native":
        chunks = []
        for wid in range(EXPECTED):
            chunks.append(worker_progress(wid))
        return section("native status", "\n\n".join(chunks))

    if kind == "acceptance":
        try:
            body = subprocess.check_output(
                ["python3", "ctl.py", "acceptance", "--workdir", str(WORKDIR), "--cycles", "50"],
                text=True,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            body = f"acceptance command failed:\n{e}"
        return section("acceptance report", body)

    if kind == "errors":
        errs = scary_errors()
        return section("recent scary errors", "\n".join(errs) if errs else "No scary errors found.")

    if kind == "cycles":
        chunks = []
        for wid in range(EXPECTED):
            lines = tail(WORKDIR / f"worker_{wid:03d}" / "log.txt", 500)
            useful = [x for x in lines if "cycle " in x or "absorbed" in x or "MATCH" in x or "SOLUTION" in x]
            if useful:
                chunks.append(section(f"worker_{wid:03d} latest cycles", "\n".join(useful[-20:])))
        return "\n".join(chunks) or "no cycle logs found"

    if kind == "worker":
        try:
            wid = int(wid_arg)
        except Exception:
            return "bad worker id"
        if wid < 0 or wid >= EXPECTED:
            return "bad worker id"
        return section(f"worker_{wid:03d} progress", worker_progress(wid)) + "\n" + section(f"worker_{wid:03d} log tail", worker_log(wid))

    return "unknown log view"


def _extract_pool_dict_from_progress(d, wid):
    candidates = [
        d,
        d.get("latest_cycle", {}) if isinstance(d, dict) else {},
        d.get("totals", {}) if isinstance(d, dict) else {},
    ]

    keys = [
        "pools",
        "pool_sizes",
        "pool_size_by_bin",
        "pools_by_bin",
        "bin_sizes",
    ]

    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for key in keys:
            val = obj.get(key)
            if isinstance(val, dict) and val:
                out = {}
                for k, v in val.items():
                    try:
                        out[str(k)] = int(v)
                    except Exception:
                        pass
                if out:
                    return out

    logp = WORKDIR / f"worker_{wid:03d}" / "log.txt"
    lines = tail(logp, 250)
    for line in reversed(lines):
        m = re.search(r"pools=(\{.*?\})\s+pairs=", line)
        if not m:
            continue
        try:
            import ast
            raw = ast.literal_eval(m.group(1))
            return {str(k): int(v) for k, v in raw.items()}
        except Exception:
            pass

    return {}

def pools_summary_payload():
    workers = []
    total = 0
    cap_slots = 0
    at_cap = 0
    all_bins = []
    route_totals = {}

    for wid in range(EXPECTED):
        prog_path = WORKDIR / f"worker_{wid:03d}" / "progress.json"
        d = read_json(prog_path)
        m = meta(wid)
        route = (d.get("route") or m.get("route") or "?").lower()
        cap = d.get("pool_cap") or m.get("pool_cap") or 0
        try:
            cap = int(cap)
        except Exception:
            cap = 0

        pools = _extract_pool_dict_from_progress(d, wid)
        subtotal = sum(pools.values())
        bins = len(pools)

        total += subtotal
        if cap and bins:
            cap_slots += cap * bins
            at_cap += sum(1 for v in pools.values() if v >= cap)

        if pools:
            all_bins.extend(pools.values())

        route_totals.setdefault(route, 0)
        route_totals[route] += subtotal

        workers.append({
            "id": wid,
            "route": route,
            "cycle": d.get("cycle"),
            "pool_cap": cap,
            "bins": bins,
            "total": subtotal,
            "min_bin": min(pools.values()) if pools else 0,
            "max_bin": max(pools.values()) if pools else 0,
            "avg_bin": round(subtotal / bins, 1) if bins else 0,
            "at_cap": sum(1 for v in pools.values() if cap and v >= cap),
            "pools": pools,
        })

    fill_pct = round((100.0 * total / cap_slots), 2) if cap_slots else 0.0

    return {
        "total": total,
        "fill_pct": fill_pct,
        "cap_slots": cap_slots,
        "at_cap": at_cap,
        "workers": workers,
        "routes": route_totals,
        "min_bin": min(all_bins) if all_bins else 0,
        "max_bin": max(all_bins) if all_bins else 0,
        "avg_bin": round(sum(all_bins) / len(all_bins), 1) if all_bins else 0,
        "updated": int(time.time()),
    }

def pools_summary_text():
    payload = pools_summary_payload()
    lines = []
    lines.append("=== pools summary ===")
    lines.append(f"total pool entries: {payload['total']:,}")
    lines.append(f"fill: {payload['fill_pct']}% of current cap slots")
    lines.append(f"bins at cap: {payload['at_cap']}")
    lines.append(f"min bin: {payload['min_bin']:,}")
    lines.append(f"avg bin: {payload['avg_bin']:,}")
    lines.append(f"max bin: {payload['max_bin']:,}")
    lines.append("")
    lines.append("routes:")
    for route, value in sorted(payload["routes"].items()):
        lines.append(f"  {route}: {value:,}")
    lines.append("")
    lines.append("workers:")
    for w in payload["workers"]:
        lines.append(
            f"  w{w['id']:03d} {w['route']:>2} cycle={w['cycle']} "
            f"total={w['total']:,} bins={w['bins']} "
            f"min={w['min_bin']:,} avg={w['avg_bin']:,} max={w['max_bin']:,} "
            f"@cap={w['at_cap']}"
        )
    return "\n".join(lines)

def make_dashboard_screenshot_png():
    """Server-side screenshot using local Chrome headless.

    This is read-only. It does not touch workers. It only asks Chrome to
    render localhost:6680 into a temporary PNG and returns the bytes.
    """
    chrome_candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]

    chrome = None
    for c in chrome_candidates:
        if os.path.exists(c):
            chrome = c
            break

    if chrome is None:
        return None, "Chrome or Chromium was not found in /Applications."

    fd, outpath = tempfile.mkstemp(prefix="h668_dashboard_", suffix=".png")
    os.close(fd)

    url = f"http://127.0.0.1:{PORT}/"

    base = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=3000",
        "--window-size=1800,7000",
        f"--screenshot={outpath}",
        url,
    ]

    try:
        r = subprocess.run(base, capture_output=True, text=True, timeout=45)
        if r.returncode != 0:
            # Older Chrome fallback.
            base[1] = "--headless"
            r = subprocess.run(base, capture_output=True, text=True, timeout=45)

        if r.returncode != 0:
            return None, (r.stderr or r.stdout or "Chrome screenshot failed")

        data = Path(outpath).read_bytes()
        return data, None
    except Exception as e:
        return None, str(e)
    finally:
        try:
            os.remove(outpath)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode(errors="ignore")
        fields = parse_qs(body)
        confirm = fields.get("confirm", [""])[0]
        if self.path == "/shutdown" and confirm == "DASHBOARD":
            data = b"<html><body style='background:#111;color:#eee;font-family:sans-serif'><h1>Dashboard stopped</h1><p>The H668 workers are still running.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        data = b"Type DASHBOARD exactly to stop only the dashboard.\n"
        self.send_response(400)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


    def _h668_control_pids(self):
        """Return PIDs for workers and auto-rotate. Dashboard stays alive."""
        out = subprocess.check_output(
            ["ps", "-axo", "pid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pids = []
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            pid_s, cmd = parts
            if ("worker.py" in cmd or "ctl.py auto-rotate-gs" in cmd
                    or "ctl.py watch-leaderboard" in cmd):
                try:
                    pids.append(int(pid_s))
                except ValueError:
                    pass
        return sorted(set(pids))

    def _h668_control(self, sig):
        count = 0
        for pid in self._h668_control_pids():
            try:
                os.kill(pid, sig)
                count += 1
            except ProcessLookupError:
                pass
        return count

    def _h668_redirect_home(self, msg):
        self.send_response(303)
        self.send_header("Location", "/?control=" + msg)
        self.end_headers()


    # H668 full seed leaderboard endpoint
    def _h668_send_text(self, text, code=200):
        data = text.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _h668_seed_leaderboard_text(self):
        root = os.path.dirname(os.path.abspath(__file__))
        workdir = os.environ.get("H668_WORKDIR", "work")
        cmd = [sys.executable, "ctl.py", "seed-leaderboard", "--workdir", workdir]
        try:
            r = subprocess.run(
                cmd,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            header = "$ " + " ".join(cmd).replace(sys.executable, "python3", 1) + "\n"
            return header + r.stdout
        except subprocess.TimeoutExpired:
            return "seed-leaderboard timed out after 120 seconds\n"


    # H668 daily supervisor command endpoints
    def _h668_supervisor_text(self, text, code=200):
        data = text.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _h668_supervisor_redirect(self, msg):
        self.send_response(303)
        self.send_header("Location", "/?control=" + msg)
        self.end_headers()

    def _h668_supervisor_run(self, args, timeout=120):
        root = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "ctl.py"] + list(args)
        try:
            r = subprocess.run(
                cmd,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            shown = "$ python3 " + " ".join(cmd[1:]) + "\n"
            return shown + r.stdout
        except subprocess.TimeoutExpired:
            return "command timed out: " + " ".join(cmd) + "\n"

    def _h668_supervisor_spawn(self, args, label):
        root = os.path.dirname(os.path.abspath(__file__))
        logs = os.path.join(root, "logs")
        os.makedirs(logs, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = os.path.join(logs, "dashboard_" + label + "_" + stamp + ".log")
        cmd = [sys.executable, "ctl.py"] + list(args)
        fh = open(log, "a")
        subprocess.Popen(
            cmd,
            cwd=root,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return log


    # H668 popup command modal helpers
    def _h668_popup_send(self, text, content_type="text/plain; charset=utf-8", code=200):
        data = text.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _h668_popup_run(self, args, timeout=120):
        root = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "ctl.py"] + list(args)
        try:
            r = subprocess.run(
                cmd,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            shown = "$ python3 " + " ".join(cmd[1:]) + "\n"
            return shown + r.stdout
        except subprocess.TimeoutExpired:
            return "command timed out: " + " ".join(cmd) + "\n"

    def _h668_popup_spawn(self, args, label):
        root = os.path.dirname(os.path.abspath(__file__))
        logs = os.path.join(root, "logs")
        os.makedirs(logs, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log = os.path.join(logs, "dashboard_popup_" + label + "_" + stamp + ".log")
        cmd = [sys.executable, "ctl.py"] + list(args)
        fh = open(log, "a")
        subprocess.Popen(
            cmd,
            cwd=root,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return "$ python3 " + " ".join(cmd[1:]) + "\nstarted in background\nlog: " + log + "\n"

    def _h668_popup_js(self):
        return '\n// H668 popup command modal\n(function () {\n  function el(tag, attrs, text) {\n    const x = document.createElement(tag);\n    if (attrs) {\n      for (const k in attrs) {\n        if (k === "style") x.setAttribute("style", attrs[k]);\n        else if (k === "title") x.setAttribute("title", attrs[k]);\n        else x[k] = attrs[k];\n      }\n    }\n    if (text !== undefined) x.textContent = text;\n    return x;\n  }\n\n  function ensureModal() {\n    let box = document.getElementById("h668-command-modal");\n    if (box) return box;\n\n    box = el("div", {\n      id: "h668-command-modal",\n      style: "display:none;position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:100000;width:min(900px,86vw);max-height:76vh;background:#111827;color:#e5e7eb;border:1px solid #475569;border-radius:14px;box-shadow:0 18px 60px rgba(0,0,0,.55);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"\n    });\n\n    const header = el("div", {\n      style: "display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid #374151;background:#0f172a;border-radius:14px 14px 0 0;"\n    });\n\n    const title = el("div", {\n      id: "h668-command-title",\n      style: "font-weight:800;flex:1;font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;"\n    }, "Command Output");\n\n    const min = el("button", {\n      title: "Minimize this popup",\n      style: "padding:6px 9px;border:0;border-radius:8px;background:#334155;color:white;font-weight:800;cursor:pointer;"\n    }, "Minimize");\n\n    const close = el("button", {\n      title: "Close this popup",\n      style: "padding:6px 9px;border:0;border-radius:8px;background:#7f1d1d;color:white;font-weight:800;cursor:pointer;"\n    }, "Close");\n\n    const pre = el("pre", {\n      id: "h668-command-output",\n      style: "margin:0;padding:12px;overflow:auto;max-height:62vh;white-space:pre-wrap;font-size:12px;line-height:1.35;"\n    }, "");\n\n    min.onclick = function () {\n      box.style.display = "none";\n      let chip = document.getElementById("h668-command-minimized");\n      if (!chip) {\n        chip = el("button", {\n          id: "h668-command-minimized",\n          title: "Restore command popup",\n          style: "position:fixed;right:340px;bottom:12px;z-index:100001;padding:10px 12px;border:1px solid #475569;border-radius:12px;background:#111827;color:white;font-weight:800;cursor:pointer;box-shadow:0 4px 18px rgba(0,0,0,.35);"\n        }, "Command Output");\n        chip.onclick = function () {\n          chip.style.display = "none";\n          box.style.display = "block";\n        };\n        document.body.appendChild(chip);\n      }\n      chip.style.display = "block";\n    };\n\n    close.onclick = function () {\n      box.style.display = "none";\n    };\n\n    header.appendChild(title);\n    header.appendChild(min);\n    header.appendChild(close);\n    box.appendChild(header);\n    box.appendChild(pre);\n    document.body.appendChild(box);\n    return box;\n  }\n\n  async function runCommand(label, url) {\n    const box = ensureModal();\n    const title = document.getElementById("h668-command-title");\n    const out = document.getElementById("h668-command-output");\n    const chip = document.getElementById("h668-command-minimized");\n\n    if (chip) chip.style.display = "none";\n    title.textContent = label;\n    out.textContent = "Running " + label + "...\\n";\n    box.style.display = "block";\n\n    try {\n      const r = await fetch(url, { cache: "no-store" });\n      const text = await r.text();\n      out.textContent = text || "(no output)";\n    } catch (e) {\n      out.textContent = "FAILED: " + e;\n    }\n  }\n\n  function makeButton(label, url, title, bg) {\n    const b = el("button", {\n      title: title,\n      style: "display:block;width:100%;margin-bottom:7px;padding:8px 10px;border:0;border-radius:8px;background:" + bg + ";color:white;text-decoration:none;font-weight:800;text-align:center;cursor:pointer;"\n    }, label);\n    b.onclick = function () { runCommand(label, url); };\n    return b;\n  }\n\n  function addPanel() {\n    if (document.getElementById("h668-controls")) return;\n\n    const panel = el("div", {\n      id: "h668-controls",\n      style: "position:fixed;right:12px;bottom:12px;z-index:99999;width:310px;padding:10px;border:1px solid #555;border-radius:12px;background:rgba(20,20,20,0.94);color:white;font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;box-shadow:0 4px 18px rgba(0,0,0,0.35);"\n    });\n\n    panel.appendChild(el("div", {\n      style: "font-weight:800;margin-bottom:8px;font-size:16px;"\n    }, "H668 Controls"));\n\n    panel.appendChild(makeButton("Status", "/api/safe-status", "Runs status-safe. Shows route mix, worker 004, dashboard, auto-rotate, and warnings.", "#1f2937"));\n    panel.appendChild(makeButton("Full Seed Leaderboard", "/api/seed-leaderboard", "Shows the full seed leaderboard table.", "#334155"));\n    panel.appendChild(makeButton("Refresh Leaderboard", "/api/safe-refresh-leaderboard", "Updates work/seed_leaderboard.json for the dashboard card.", "#475569"));\n\n    const row1 = el("div", { style: "display:flex;gap:7px;margin-bottom:7px;" });\n    row1.appendChild(makeButton("Pause", "/api/safe-pause", "Pauses workers and auto-rotate. Dashboard stays alive.", "#b45309"));\n    row1.appendChild(makeButton("Resume", "/api/safe-resume", "Resumes workers and auto-rotate.", "#087443"));\n    panel.appendChild(row1);\n\n    const row2 = el("div", { style: "display:flex;gap:7px;margin-bottom:7px;" });\n    row2.appendChild(makeButton("Stop", "/api/safe-stop", "Starts stop-safe. This may stop the dashboard after the response.", "#b42318"));\n    row2.appendChild(makeButton("Start", "/api/safe-run", "Starts the exact approved 10 TT / 2 GS mix. Refuses duplicates.", "#2563eb"));\n    panel.appendChild(row2);\n\n    panel.appendChild(makeButton("Restart Safe", "/api/safe-restart", "Runs run-safe --replace using the approved mix. Worker 004 remains GS and pinned.", "#7c3aed"));\n\n    panel.appendChild(el("div", {\n      style: "font-size:11px;margin-top:7px;opacity:.75;text-align:center;"\n    }, "click opens popup, hover for description"));\n\n    document.body.appendChild(panel);\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", addPanel);\n  } else {\n    addPanel();\n  }\n})();\n'


    def _h668_minimize_panel_js(self):
        return '\n// H668 minimize control panel addon\n(function () {\n  function installMinimize() {\n    const panel = document.getElementById("h668-controls");\n    if (!panel || document.getElementById("h668-controls-minimize-btn")) return;\n\n    const btn = document.createElement("button");\n    btn.id = "h668-controls-minimize-btn";\n    btn.textContent = "Minimize Panel";\n    btn.title = "Hide the H668 control panel. A small restore button will stay visible.";\n    btn.style.display = "block";\n    btn.style.width = "100%";\n    btn.style.marginTop = "7px";\n    btn.style.padding = "7px 10px";\n    btn.style.border = "0";\n    btn.style.borderRadius = "8px";\n    btn.style.background = "#0f172a";\n    btn.style.color = "white";\n    btn.style.fontWeight = "800";\n    btn.style.cursor = "pointer";\n\n    let chip = document.getElementById("h668-controls-restore-chip");\n    if (!chip) {\n      chip = document.createElement("button");\n      chip.id = "h668-controls-restore-chip";\n      chip.textContent = "H668 Controls";\n      chip.title = "Restore the H668 control panel";\n      chip.style.display = "none";\n      chip.style.position = "fixed";\n      chip.style.right = "12px";\n      chip.style.bottom = "12px";\n      chip.style.zIndex = "100000";\n      chip.style.padding = "10px 12px";\n      chip.style.border = "1px solid #555";\n      chip.style.borderRadius = "12px";\n      chip.style.background = "rgba(20,20,20,0.94)";\n      chip.style.color = "white";\n      chip.style.fontWeight = "800";\n      chip.style.cursor = "pointer";\n      chip.style.boxShadow = "0 4px 18px rgba(0,0,0,0.35)";\n      chip.onclick = function () {\n        panel.style.display = "block";\n        chip.style.display = "none";\n        try { localStorage.setItem("h668ControlsMinimized", "0"); } catch (e) {}\n      };\n      document.body.appendChild(chip);\n    }\n\n    btn.onclick = function () {\n      panel.style.display = "none";\n      chip.style.display = "block";\n      try { localStorage.setItem("h668ControlsMinimized", "1"); } catch (e) {}\n    };\n\n    panel.appendChild(btn);\n\n    try {\n      if (localStorage.getItem("h668ControlsMinimized") === "1") {\n        panel.style.display = "none";\n        chip.style.display = "block";\n      }\n    } catch (e) {}\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", installMinimize);\n  } else {\n    installMinimize();\n  }\n\n  let tries = 0;\n  const timer = setInterval(function () {\n    installMinimize();\n    tries += 1;\n    if (tries > 20 || document.getElementById("h668-controls-minimize-btn")) {\n      clearInterval(timer);\n    }\n  }, 250);\n})();\n'


    # H668 minimized chip layout fix
    def _h668_chip_layout_js(self):
        return '\n// H668 minimized chip layout fix\n(function () {\n  function fixChips() {\n    const controls = document.getElementById("h668-controls-restore-chip");\n    const command = document.getElementById("h668-command-minimized");\n\n    if (controls) {\n      controls.textContent = "H668 Controls";\n      controls.style.right = "12px";\n      controls.style.bottom = "12px";\n      controls.style.width = "180px";\n      controls.style.textAlign = "center";\n      controls.style.whiteSpace = "nowrap";\n    }\n\n    if (command) {\n      command.textContent = "Command Output";\n      command.style.right = "12px";\n      command.style.bottom = controls && controls.style.display !== "none" ? "62px" : "12px";\n      command.style.width = "180px";\n      command.style.textAlign = "center";\n      command.style.whiteSpace = "nowrap";\n    }\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", fixChips);\n  } else {\n    fixChips();\n  }\n\n  setInterval(fixChips, 300);\n})();\n'


    # H668 snapshot inside controls addon
    def _h668_snapshot_controls_js(self):
        return '\n// H668 snapshot inside controls addon\n(function () {\n  function removeOldSnapshotControls() {\n    const nodes = Array.from(document.querySelectorAll("a,button,div"));\n    for (const n of nodes) {\n      if (n.closest("#h668-controls")) continue;\n      if (n.closest("#h668-screenshot-modal")) continue;\n\n      const txt = (n.textContent || "").trim().toLowerCase();\n      const idc = ((n.id || "") + " " + (n.className || "")).toLowerCase();\n      const style = window.getComputedStyle(n);\n\n      const looksSnapshot =\n        txt === "save screenshot" ||\n        txt === "snapshot" ||\n        txt === "screenshot" ||\n        txt.includes("save screenshot") ||\n        idc.includes("snapshot") ||\n        idc.includes("screenshot");\n\n      const floating =\n        style.position === "fixed" ||\n        style.position === "absolute" ||\n        idc.includes("floating");\n\n      if (looksSnapshot && floating) {\n        let target = n;\n        for (let i = 0; i < 3; i++) {\n          if (\n            target.parentElement &&\n            target.parentElement !== document.body &&\n            window.getComputedStyle(target.parentElement).position === "fixed"\n          ) {\n            target = target.parentElement;\n          }\n        }\n        target.remove();\n      }\n    }\n  }\n\n  function ensureScreenshotModal() {\n    let box = document.getElementById("h668-screenshot-modal");\n    if (box) return box;\n\n    box = document.createElement("div");\n    box.id = "h668-screenshot-modal";\n    box.style.display = "none";\n    box.style.position = "fixed";\n    box.style.left = "50%";\n    box.style.top = "50%";\n    box.style.transform = "translate(-50%,-50%)";\n    box.style.zIndex = "100002";\n    box.style.width = "min(1100px,88vw)";\n    box.style.maxHeight = "82vh";\n    box.style.background = "#111827";\n    box.style.color = "#e5e7eb";\n    box.style.border = "1px solid #475569";\n    box.style.borderRadius = "14px";\n    box.style.boxShadow = "0 18px 60px rgba(0,0,0,.55)";\n    box.style.overflow = "hidden";\n\n    const header = document.createElement("div");\n    header.style.display = "flex";\n    header.style.alignItems = "center";\n    header.style.gap = "8px";\n    header.style.padding = "10px 12px";\n    header.style.borderBottom = "1px solid #374151";\n    header.style.background = "#0f172a";\n\n    const title = document.createElement("div");\n    title.textContent = "Dashboard Snapshot";\n    title.style.fontWeight = "800";\n    title.style.flex = "1";\n    title.style.fontFamily = "system-ui,-apple-system,BlinkMacSystemFont,sans-serif";\n\n    const refresh = document.createElement("button");\n    refresh.textContent = "Refresh";\n    refresh.title = "Take a fresh screenshot";\n    refresh.style.padding = "6px 9px";\n    refresh.style.border = "0";\n    refresh.style.borderRadius = "8px";\n    refresh.style.background = "#2563eb";\n    refresh.style.color = "white";\n    refresh.style.fontWeight = "800";\n    refresh.style.cursor = "pointer";\n\n    const min = document.createElement("button");\n    min.textContent = "Minimize";\n    min.title = "Minimize this screenshot popup";\n    min.style.padding = "6px 9px";\n    min.style.border = "0";\n    min.style.borderRadius = "8px";\n    min.style.background = "#334155";\n    min.style.color = "white";\n    min.style.fontWeight = "800";\n    min.style.cursor = "pointer";\n\n    const close = document.createElement("button");\n    close.textContent = "Close";\n    close.title = "Close this screenshot popup";\n    close.style.padding = "6px 9px";\n    close.style.border = "0";\n    close.style.borderRadius = "8px";\n    close.style.background = "#7f1d1d";\n    close.style.color = "white";\n    close.style.fontWeight = "800";\n    close.style.cursor = "pointer";\n\n    const wrap = document.createElement("div");\n    wrap.style.padding = "10px";\n    wrap.style.overflow = "auto";\n    wrap.style.maxHeight = "72vh";\n    wrap.style.textAlign = "center";\n\n    const img = document.createElement("img");\n    img.id = "h668-screenshot-img";\n    img.alt = "Dashboard snapshot";\n    img.style.maxWidth = "100%";\n    img.style.borderRadius = "10px";\n    img.style.border = "1px solid #374151";\n    img.style.background = "#020617";\n\n    function loadFresh() {\n      img.src = "/screenshot.png?ts=" + Date.now();\n    }\n\n    refresh.onclick = loadFresh;\n\n    min.onclick = function () {\n      box.style.display = "none";\n      let chip = document.getElementById("h668-screenshot-minimized");\n      if (!chip) {\n        chip = document.createElement("button");\n        chip.id = "h668-screenshot-minimized";\n        chip.textContent = "Snapshot";\n        chip.title = "Restore dashboard snapshot";\n        chip.style.position = "fixed";\n        chip.style.right = "12px";\n        chip.style.bottom = "112px";\n        chip.style.width = "180px";\n        chip.style.zIndex = "100003";\n        chip.style.padding = "10px 12px";\n        chip.style.border = "1px solid #475569";\n        chip.style.borderRadius = "12px";\n        chip.style.background = "#111827";\n        chip.style.color = "white";\n        chip.style.fontWeight = "800";\n        chip.style.cursor = "pointer";\n        chip.style.boxShadow = "0 4px 18px rgba(0,0,0,.35)";\n        chip.onclick = function () {\n          chip.style.display = "none";\n          box.style.display = "block";\n        };\n        document.body.appendChild(chip);\n      }\n      chip.style.display = "block";\n    };\n\n    close.onclick = function () {\n      box.style.display = "none";\n    };\n\n    header.appendChild(title);\n    header.appendChild(refresh);\n    header.appendChild(min);\n    header.appendChild(close);\n    wrap.appendChild(img);\n    box.appendChild(header);\n    box.appendChild(wrap);\n    document.body.appendChild(box);\n\n    loadFresh();\n    return box;\n  }\n\n  function openSnapshot() {\n    const box = ensureScreenshotModal();\n    const img = document.getElementById("h668-screenshot-img");\n    const chip = document.getElementById("h668-screenshot-minimized");\n    if (chip) chip.style.display = "none";\n    img.src = "/screenshot.png?ts=" + Date.now();\n    box.style.display = "block";\n  }\n\n  function addSnapshotButton() {\n    removeOldSnapshotControls();\n\n    const panel = document.getElementById("h668-controls");\n    if (!panel || document.getElementById("h668-snapshot-control-btn")) return;\n\n    const btn = document.createElement("button");\n    btn.id = "h668-snapshot-control-btn";\n    btn.textContent = "Snapshot";\n    btn.title = "Opens a dashboard screenshot inside a popup. Uses /screenshot.png.";\n    btn.style.display = "block";\n    btn.style.width = "100%";\n    btn.style.marginBottom = "7px";\n    btn.style.padding = "8px 10px";\n    btn.style.border = "0";\n    btn.style.borderRadius = "8px";\n    btn.style.background = "#0e7490";\n    btn.style.color = "white";\n    btn.style.textDecoration = "none";\n    btn.style.fontWeight = "800";\n    btn.style.textAlign = "center";\n    btn.style.cursor = "pointer";\n    btn.onclick = openSnapshot;\n\n    const leaderboard = Array.from(panel.querySelectorAll("button,a")).find(x =>\n      (x.textContent || "").trim().toLowerCase().includes("full seed leaderboard")\n    );\n\n    if (leaderboard && leaderboard.parentElement === panel) {\n      leaderboard.insertAdjacentElement("afterend", btn);\n    } else {\n      panel.insertBefore(btn, panel.children[1] || null);\n    }\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", addSnapshotButton);\n  } else {\n    addSnapshotButton();\n  }\n\n  let tries = 0;\n  const timer = setInterval(function () {\n    addSnapshotButton();\n    tries += 1;\n    if (tries > 30 && document.getElementById("h668-snapshot-control-btn")) {\n      clearInterval(timer);\n    }\n  }, 250);\n})();\n'


    # H668 snapshot download fix
    def _h668_snapshot_download_js(self):
        return '\n// H668 snapshot download fix\n(function () {\n  function stamp() {\n    const d = new Date();\n    const pad = n => String(n).padStart(2, "0");\n    return d.getFullYear()\n      + pad(d.getMonth() + 1)\n      + pad(d.getDate())\n      + "_"\n      + pad(d.getHours())\n      + pad(d.getMinutes())\n      + pad(d.getSeconds());\n  }\n\n  async function downloadSnapshot() {\n    const url = "/screenshot.png?download=1&ts=" + Date.now();\n    const name = "h668_dashboard_snapshot_" + stamp() + ".png";\n\n    const r = await fetch(url, { cache: "no-store" });\n    if (!r.ok) throw new Error("screenshot HTTP " + r.status);\n\n    const blob = await r.blob();\n    const objectUrl = URL.createObjectURL(blob);\n\n    const a = document.createElement("a");\n    a.href = objectUrl;\n    a.download = name;\n    document.body.appendChild(a);\n    a.click();\n    a.remove();\n\n    setTimeout(function () {\n      URL.revokeObjectURL(objectUrl);\n    }, 1500);\n  }\n\n  function makeBtn(text) {\n    const btn = document.createElement("button");\n    btn.textContent = text;\n    btn.title = "Downloads /screenshot.png to your browser Downloads folder";\n    btn.style.display = "block";\n    btn.style.width = "100%";\n    btn.style.marginBottom = "7px";\n    btn.style.padding = "8px 10px";\n    btn.style.border = "0";\n    btn.style.borderRadius = "8px";\n    btn.style.background = "#0891b2";\n    btn.style.color = "white";\n    btn.style.textDecoration = "none";\n    btn.style.fontWeight = "800";\n    btn.style.textAlign = "center";\n    btn.style.cursor = "pointer";\n    btn.onclick = function () {\n      btn.textContent = "Downloading...";\n      downloadSnapshot()\n        .then(function () { btn.textContent = text; })\n        .catch(function (e) {\n          btn.textContent = "Download failed";\n          alert("Snapshot download failed: " + e);\n          setTimeout(function () { btn.textContent = text; }, 2000);\n        });\n    };\n    return btn;\n  }\n\n  function addDownloadToControls() {\n    const panel = document.getElementById("h668-controls");\n    if (!panel || document.getElementById("h668-download-snapshot-btn")) return;\n\n    const btn = makeBtn("Download Snapshot");\n    btn.id = "h668-download-snapshot-btn";\n\n    const snap = document.getElementById("h668-snapshot-control-btn");\n    if (snap && snap.parentElement === panel) {\n      snap.insertAdjacentElement("afterend", btn);\n    } else {\n      panel.insertBefore(btn, panel.children[2] || null);\n    }\n  }\n\n  function addDownloadToSnapshotPopup() {\n    const modal = document.getElementById("h668-screenshot-modal");\n    if (!modal || document.getElementById("h668-snapshot-modal-download-btn")) return;\n\n    const header = modal.querySelector("div");\n    if (!header) return;\n\n    const btn = document.createElement("button");\n    btn.id = "h668-snapshot-modal-download-btn";\n    btn.textContent = "Download";\n    btn.title = "Save this dashboard snapshot to your Downloads folder";\n    btn.style.padding = "6px 9px";\n    btn.style.border = "0";\n    btn.style.borderRadius = "8px";\n    btn.style.background = "#0891b2";\n    btn.style.color = "white";\n    btn.style.fontWeight = "800";\n    btn.style.cursor = "pointer";\n    btn.onclick = function () {\n      btn.textContent = "Downloading...";\n      downloadSnapshot()\n        .then(function () { btn.textContent = "Download"; })\n        .catch(function (e) {\n          btn.textContent = "Failed";\n          alert("Snapshot download failed: " + e);\n          setTimeout(function () { btn.textContent = "Download"; }, 2000);\n        });\n    };\n\n    const close = Array.from(header.querySelectorAll("button")).find(x => x.textContent === "Close");\n    if (close) {\n      close.insertAdjacentElement("beforebegin", btn);\n    } else {\n      header.appendChild(btn);\n    }\n  }\n\n  function install() {\n    addDownloadToControls();\n    addDownloadToSnapshotPopup();\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", install);\n  } else {\n    install();\n  }\n\n  setInterval(install, 500);\n})();\n'


    # H668 remove snapshot popup cleanup
    def _h668_remove_snapshot_popup_js(self):
        return '\n// H668 remove snapshot popup cleanup\n(function () {\n  function cleanupSnapshotPopup() {\n    const ids = [\n      "h668-screenshot-modal",\n      "h668-screenshot-minimized",\n      "h668-snapshot-control-btn"\n    ];\n    ids.forEach(function (id) {\n      const n = document.getElementById(id);\n      if (n) n.remove();\n    });\n\n    // Rename Download Snapshot to Snapshot if you prefer one simple button.\n    const dl = document.getElementById("h668-download-snapshot-btn");\n    if (dl) {\n      dl.textContent = "Snapshot";\n      dl.title = "Downloads the dashboard screenshot to your browser Downloads folder";\n    }\n  }\n\n  if (document.readyState === "loading") {\n    document.addEventListener("DOMContentLoaded", cleanupSnapshotPopup);\n  } else {\n    cleanupSnapshotPopup();\n  }\n\n  setInterval(cleanupSnapshotPopup, 500);\n})();\n'

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/h668-remove-snapshot-popup.js":
            self._h668_popup_send(self._h668_remove_snapshot_popup_js(), "application/javascript; charset=utf-8")
            return

        path = self.path.split("?", 1)[0]
        if path == "/h668-snapshot-download.js":
            self._h668_popup_send(self._h668_snapshot_download_js(), "application/javascript; charset=utf-8")
            return


        path = self.path.split("?", 1)[0]
        if path == "/h668-chip-layout.js":
            self._h668_popup_send(self._h668_chip_layout_js(), "application/javascript; charset=utf-8")
            return

        path = self.path.split("?", 1)[0]
        if path == "/h668-minimize-panel.js":
            self._h668_popup_send(self._h668_minimize_panel_js(), "application/javascript; charset=utf-8")
            return

        path = self.path.split("?", 1)[0]
        workdir = os.environ.get("H668_WORKDIR", "work")
        seeds = os.environ.get("H668_SEEDS", "seeds.txt")

        if path == "/h668-popup-controls.js":
            self._h668_popup_send(self._h668_popup_js(), "application/javascript; charset=utf-8")
            return
        if path == "/api/safe-status":
            self._h668_popup_send(self._h668_popup_run(["status-safe", "--workdir", workdir]))
            return
        if path == "/api/seed-leaderboard":
            self._h668_popup_send(self._h668_popup_run(["seed-leaderboard", "--workdir", workdir]))
            return
        if path == "/api/safe-refresh-leaderboard":
            self._h668_popup_send(self._h668_popup_run(["refresh-leaderboard", "--workdir", workdir]))
            return
        if path == "/api/safe-pause":
            self._h668_popup_send(self._h668_popup_run(["pause-safe", "--workdir", workdir]))
            return
        if path == "/api/safe-resume":
            self._h668_popup_send(self._h668_popup_run(["resume-safe", "--workdir", workdir]))
            return
        if path == "/api/safe-stop":
            self._h668_popup_send(self._h668_popup_spawn(["stop-safe", "--workdir", workdir], "stop_safe"))
            return
        if path == "/api/safe-run":
            self._h668_popup_send(self._h668_popup_spawn(["run-safe", "--workdir", workdir, "--seeds", seeds], "run_safe"))
            return
        if path == "/api/safe-restart":
            self._h668_popup_send(self._h668_popup_spawn(["run-safe", "--workdir", workdir, "--seeds", seeds, "--replace"], "run_safe_replace"))
            return

        path = self.path.split("?", 1)[0]
        workdir = os.environ.get("H668_WORKDIR", "work")
        seeds = os.environ.get("H668_SEEDS", "seeds.txt")

        if path == "/safe-status.txt":
            self._h668_supervisor_text(self._h668_supervisor_run(["status-safe", "--workdir", workdir]))
            return
        if path == "/safe-refresh-leaderboard":
            self._h668_supervisor_run(["refresh-leaderboard", "--workdir", workdir])
            self._h668_supervisor_redirect("leaderboard-refreshed")
            return
        if path == "/safe-pause":
            self._h668_supervisor_run(["pause-safe", "--workdir", workdir])
            self._h668_supervisor_redirect("paused")
            return
        if path == "/safe-resume":
            self._h668_supervisor_run(["resume-safe", "--workdir", workdir])
            self._h668_supervisor_redirect("resumed")
            return
        if path == "/safe-stop":
            self._h668_supervisor_spawn(["stop-safe", "--workdir", workdir], "stop_safe")
            self._h668_supervisor_redirect("stop-safe-started")
            return
        if path == "/safe-run":
            self._h668_supervisor_spawn(["run-safe", "--workdir", workdir, "--seeds", seeds], "run_safe")
            self._h668_supervisor_redirect("run-safe-started")
            return
        if path == "/safe-restart":
            self._h668_supervisor_spawn(["run-safe", "--workdir", workdir, "--seeds", seeds, "--replace"], "run_safe_replace")
            self._h668_supervisor_redirect("restart-safe-started")
            return

        path = self.path.split("?", 1)[0]
        if path == "/seed-leaderboard.txt":
            self._h668_send_text(self._h668_seed_leaderboard_text())
            return

        path = self.path.split("?", 1)[0]
        if path == "/pause-all":
            n = self._h668_control(signal.SIGSTOP)
            self._h668_redirect_home("paused-" + str(n))
            return
        if path == "/resume-all":
            n = self._h668_control(signal.SIGCONT)
            self._h668_redirect_home("resumed-" + str(n))
            return


        if self.path.startswith("/screenshot.png"):
            data, err = make_dashboard_screenshot_png()
            if data is None:
                msg = ("Screenshot failed: " + str(err)).encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Disposition",
                             "attachment; filename=h668_dashboard.png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith(("/rotate-status.json",
                                 "/autorotate.json")):
            # READ-ONLY preview of what auto-rotate WOULD do; the
            # dashboard never rotates, starts, or stops workers.
            data = json.dumps(
                dash_routing.rotate_status_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type",
                             "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/routing.json"):
            data = json.dumps(dash_routing.routing_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type",
                             "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/pools.json"):
            data = json.dumps(pools_summary_payload()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/logs"):
            data = log_view(self.path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    print(f"H668 v3 kid dashboard running at http://0.0.0.0:{PORT}")
    print("This dashboard is read-only. It does not stop the search workers.")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
