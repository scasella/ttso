"""The pre-registered latency rule (v0.4): tasks whose wall window overlaps a memory-guard pause or a
system sleep are left out of the latency means.

Pauses come from runs/v0.4/memory_guard.log (PAUSE until the next RESUME / STUCK line); sleeps from
`pmset -g log` (a Sleep line until the next Wake / DarkWake line). For each run directory given,
prints the affected tasks and writes latency.json next to its summary: every arm's mean latency
over all tasks and over the unaffected ones.

    python scripts/pauses_v04.py runs/v0.4/gemma-4-E2B/json runs/v0.4/gemma-4-E2B/catalog ...
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "runs" / "v0.4" / "memory_guard.log"


def _intervals(events):
    """[(time, starts)] sorted -> [(start, end)] (an open interval ends now)."""
    out, start = [], None
    for t, starts in sorted(events):
        if starts and start is None:
            start = t
        elif not starts and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, datetime.now().timestamp()))
    return out


def guard_pauses():
    ev = []
    for line in GUARD.read_text().splitlines() if GUARD.exists() else []:
        m = re.match(r"(\S+Z) (PAUSE|RESUME|STUCK)", line)
        if m:
            t = datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            ev.append((t.timestamp(), m[2] == "PAUSE"))
    return _intervals(ev)


def system_sleeps():
    out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True).stdout
    ev = []
    for line in out.splitlines():
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4}) (Sleep|Wake|DarkWake)\s", line)
        if m:
            ev.append((datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S %z").timestamp(), m[2] == "Sleep"))
    return _intervals(ev)


def main(dirs):
    gaps = [("pause", a, b) for a, b in guard_pauses()] + [("sleep", a, b) for a, b in system_sleeps()]
    for d in map(Path, dirs):
        rows = [json.loads(line) for line in open(d / "results.jsonl")]
        hit = {}
        for r in rows:
            s, e = r["wall"]
            ov = [(k, round(min(e, b) - max(s, a), 1)) for k, a, b in gaps if a < e and b > s]
            if ov:
                hit[r["i"]] = ov
        lat = {}
        for k in rows[0]["seconds"]:
            all_ = [r["seconds"][k]["total"] for r in rows]
            ok = [r["seconds"][k]["total"] for r in rows if r["i"] not in hit]
            lat[k] = {"mean_all_s": sum(all_) / len(all_), "mean_s": sum(ok) / len(ok) if ok else None}
        for g in ("greedy", "greedy_masked"):
            if g in rows[0]:
                ok = [r[g]["seconds"] for r in rows if r["i"] not in hit]
                lat[g] = {"mean_s": sum(ok) / len(ok) if ok else None}
        out = {"affected_tasks": hit, "n_left_out": len(hit), "latency": lat}
        (d / "latency.json").write_text(json.dumps(out, indent=1))
        print(d, f"{len(hit)} task(s) overlap a pause or sleep: {hit}")


if __name__ == "__main__":
    main(sys.argv[1:])
