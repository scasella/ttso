"""Retroactive audit of the saved v0.2 / v0.3 held-out runs with v0.4's 5-vector scoring.

For every run, task and arm:
  1. Sanity gate: recompute Valid / Correct_1 / Faithful_1 from the saved completion text and
     compare them with the stored values (they must agree before any Faithful_5 delta means
     anything). A Best-of-N row whose greedy completion was already valid stores no text; its
     program is that greedy completion.
  2. Correct_5 / Faithful_5: also right on the task's 5 input vectors (benchmark.vectors,
     catalog_suite.vectors). Programs that are Faithful_1 but not Faithful_5 are listed.
  3. The run's pre-registered tests again, on Faithful_5 (and Correct_5 for the catalog), with
     exact McNemar and Holm.
  4. Best-of-N accounting: samples per task, and tasks where the budget ran out before any
     sample was complete.
Writes runs/v0.4/audit/{audit.json, AUDIT.md}.
"""

import json
from collections import Counter
from pathlib import Path

from ttso.cartridges import bendlite, typed_json_dag
from ttso.engine import benchmark, catalog_suite

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (name, path, suite, model, tasks, primary family, its outcome)
    ("v0.2 base", "runs/v0.2/base", "json", "Qwen/Qwen3-0.6B-Base", lambda: benchmark.make_tasks(60, seed=42), "H", "faithful"),
    ("v0.2 post", "runs/v0.2/post", "json", "Qwen/Qwen3-0.6B", lambda: benchmark.make_tasks(60, seed=42)[:20], "H", "faithful"),
    ("v0.3 json", "runs/v0.3/json", "json", "Qwen/Qwen3.5-0.8B", lambda: benchmark.make_tasks(60, seed=42), "H", "faithful"),
    ("v0.3 catalog", "runs/v0.3/catalog", "catalog", "Qwen/Qwen3.5-0.8B", catalog_suite.heldout_tasks, "C", "correct"),
]
FAMILIES = {
    "H": {"H1": ("ttso_pinned", "greedy"), "H2": ("ttso_pinned", "best_of_n"),
          "H3": ("ttso_pinned_masked", "greedy_masked"), "H4": ("ttso_pinned_masked", "best_of_n_masked")},
    "C": {"C1": ("smart", "best_of_n"), "C2": ("smart_pinned", "best_of_n"),
          "C3": ("local", "best_of_n"), "C4": ("upstream", "best_of_n")},
}
FALLBACK = {"best_of_n": "greedy", "best_of_n_masked": "greedy_masked"}
BON_REGIME = {"json": {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"},
              "catalog": {"best_of_n": "upstream"}}
MAX_NEW = {"json": 320, "catalog": 256}
FIELDS = ("valid", "correct", "correct_5", "faithful", "faithful_5")


def score(suite, task, text):
    if suite == "json":
        return benchmark._outcome(task, typed_json_dag.reference_check(text), text)
    return catalog_suite.CatalogSuite().outcome(task, bendlite.reference_check(text), text)


def complete(suite, text):
    """Did generation finish a document (the cartridge's stop criterion, on text)?"""
    if suite == "json":
        return typed_json_dag.json_end(text) is not None
    return bendlite.program_complete(text)


def tests(rows, family, outcome):
    pairs = FAMILIES[family]
    have = [h for h, (a, b) in pairs.items() if a in rows[0] and b in rows[0]]
    res = {h: benchmark.mcnemar([r[pairs[h][0]][outcome] for r in rows], [r[pairs[h][1]][outcome] for r in rows])
           for h in have}
    adj = benchmark.holm({h: v["p"] for h, v in res.items()})
    return {h: dict(v, pair=pairs[h], p_holm=adj[h]) for h, v in res.items()}


def audit(name, path, suite, model, make, family, prim):
    raw = [json.loads(line) for line in open(ROOT / path / "results.jsonl")]
    arms = json.load(open(ROOT / path / "summary.json"))["arms"]
    tasks = {t["i"]: t for t in make()}
    rows, mismatches, coincidences = [], [], []
    for r in raw:
        task = tasks[r["i"]]
        assert task["text"] == r["task"], (name, r["i"])
        new = {"i": r["i"]}
        for arm in arms:
            text = r[arm].get("text")
            if text is None:
                text = r[FALLBACK[arm]]["text"]
            o = score(suite, task, text)
            for f in ("valid", "correct", "faithful"):
                if o[f] != r[arm][f]:
                    mismatches.append({"i": r["i"], "arm": arm, "field": f, "stored": r[arm][f], "recomputed": o[f]})
            if o["faithful"] and not o["faithful_5"]:
                coincidences.append({"i": r["i"], "arm": arm, "vectors_passed": o["vectors_passed"],
                                     "task": task["text"], "text": text})
            new[arm] = o
        rows.append(new)
    counts = {arm: {f: sum(r[arm][f] for r in rows) for f in FIELDS} for arm in arms}
    out = {"run": name, "path": path, "model": model, "suite": suite, "n": len(rows), "counts": counts,
           "sanity_mismatches": mismatches, "coincidences": coincidences,
           f"family_{prim}_1": tests(rows, family, prim), "family_faithful_5": tests(rows, family, "faithful_5")}
    if suite == "catalog":
        out["family_correct_5"] = tests(rows, family, "correct_5")
        out["smart_pinned_vs_smart_faithful_5"] = benchmark.mcnemar([r["smart_pinned"]["faithful_5"] for r in rows],
                                                                    [r["smart"]["faithful_5"] for r in rows])
    out["best_of_n"] = bon_accounting(raw, suite)
    return out


def bon_accounting(raw, suite):
    """What the saved rows say about token-matched Best-of-N. A sample ends at stop / EOS or at its
    limit min(max_new, budget - spent); sampling stops at the first valid sample or once the
    budget is spent, so a task with one invalid sample spent the whole budget on it. That sample
    is the saved text: when it is incomplete, no complete candidate was drawn on that task."""
    out = {}
    for arm, reg in BON_REGIME[suite].items():
        if arm not in raw[0]:
            continue
        drew = [r for r in raw if r[arm].get("samples", 0) > 0]
        samples = [r[arm]["samples"] for r in drew]
        none_complete = [r["i"] for r in drew if r[arm]["samples"] == 1 and not r[arm]["valid"]
                         and not complete(suite, r[arm]["text"])]
        budget = [r[reg]["extra_tokens"] for r in drew]
        out[arm] = {"tasks": len(raw), "tasks_sampled": len(drew), "samples_total": sum(samples),
                    "samples_per_sampled_task": Counter(samples), "budget_min": min(budget, default=0),
                    "budget_max": max(budget, default=0), "budget_below_max_new": sum(b < MAX_NEW[suite] for b in budget),
                    "tasks_with_no_complete_sample": len(none_complete), "no_complete_sample_tasks": none_complete}
    return out


def table(a):
    lines = [f"### {a['run']}: {a['model']}, {a['suite']}, n = {a['n']}", "",
             "| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 |", "| --- | --- | --- | --- | --- | --- |"]
    for arm, c in a["counts"].items():
        lines.append(f"| `{arm}` | {c['valid']} | {c['correct']} | {c['correct_5']} | {c['faithful']} | {c['faithful_5']} |")
    lines.append("")
    for key in [k for k in a if k.startswith("family_")]:
        lines.append(f"{key}: " + "; ".join(f"{h} {v['only_first']}–{v['only_second']}, p = {v['p']:.3g}, Holm {v['p_holm']:.3g}"
                                            for h, v in a[key].items()))
        lines.append("")
    lines.append(f"Sanity gate: {len(a['sanity_mismatches'])} stored-vs-recomputed mismatches. "
                 f"Faithful_1 but not Faithful_5: {len(a['coincidences'])}.")
    for arm, b in a["best_of_n"].items():
        lines.append(f"`{arm}`: {b['samples_total']} samples on {b['tasks_sampled']} sampled tasks "
                     f"(per task {dict(sorted(b['samples_per_sampled_task'].items()))}); budget {b['budget_min']}–"
                     f"{b['budget_max']} tokens; no complete sample on {b['tasks_with_no_complete_sample']} tasks.")
    return "\n".join(lines) + "\n"


def main():
    res = [audit(*r) for r in RUNS]
    d = ROOT / "runs" / "v0.4" / "audit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "audit.json").write_text(json.dumps(res, indent=1))
    (d / "AUDIT.md").write_text("# v0.2 / v0.3 re-scored with 5 input vectors\n\n" + "\n".join(table(a) for a in res))
    print((d / "AUDIT.md").read_text())


if __name__ == "__main__":
    main()
