"""v0.4's latency rule (scripts/pauses_v04.py) for the v0.5 runs: tasks whose wall window overlaps a
memory-guard pause (runs/v0.5/memory_guard.log) or a system sleep (`pmset -g log`) are left out of
the latency means. Writes latency.json next to each run's summary, for every v0.5 arm: the repair
regimes (greedy + repair), the two greedy arms, and Best-of-N (its regime's greedy decode plus its
samples), each over all tasks and over the unaffected ones.

    python scripts/pauses_v05.py runs/v0.5/gemma-4-E2B/json
"""

import json
import sys
from pathlib import Path

from scripts import pauses_v04

pauses_v04.GUARD = pauses_v04.ROOT / "runs" / "v0.5" / "memory_guard.log"
BON_GREEDY = {"best_of_n_typemasked": "greedy_typemasked"}


def _seconds(r, arm):
    if arm in r["seconds"]:
        return r["seconds"][arm]["total"]
    if arm in BON_GREEDY:
        return r[BON_GREEDY[arm]]["seconds"] + r[arm].get("seconds", 0.0)
    return r[arm]["seconds"]


def main(dirs):
    gaps = ([("pause", a, b) for a, b in pauses_v04.guard_pauses()]
            + [("sleep", a, b) for a, b in pauses_v04.system_sleeps()])
    for d in map(Path, dirs):
        rows = [json.loads(line) for line in open(d / "results.jsonl")]
        hit = {}
        for r in rows:
            s, e = r["wall"]
            ov = [(k, round(min(e, b) - max(s, a), 1)) for k, a, b in gaps if a < e and b > s]
            if ov:
                hit[r["i"]] = ov
        arms = json.load(open(d / "summary.json"))["arms"]
        lat = {}
        for k in arms:
            all_ = [_seconds(r, k) for r in rows]
            ok = [_seconds(r, k) for r in rows if r["i"] not in hit]
            lat[k] = {"mean_all_s": sum(all_) / len(all_), "mean_s": sum(ok) / len(ok) if ok else None}
        out = {"affected_tasks": hit, "n_left_out": len(hit), "gaps_logged": len(gaps), "latency": lat}
        (d / "latency.json").write_text(json.dumps(out, indent=1))
        print(d, f"{len(hit)} task(s) overlap a pause or sleep: {hit}")


if __name__ == "__main__":
    main(sys.argv[1:])
