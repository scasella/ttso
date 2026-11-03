"""v0.3 benchmarks on Qwen3.5-0.8B (pre-registered in runs/v0.3/PREREGISTRATION.md).

  json      TypedJSONDAG, the v0.2 arms: greedy, best_of_n, ttso_pinned; greedy_masked,
            best_of_n_masked, ttso_pinned_masked; ttso_unpinned (secondary).
            dev: make_tasks(20, seed=0); heldout: make_tasks(60, seed=42).
  catalog   BendLite catalog synthesis: greedy, best_of_n (budget matched to upstream), local,
            upstream, smart (pin_inputs=False, the study's setup), smart_pinned (pin_inputs=True).
            dev: catalog_suite.dev_tasks() (20); heldout: catalog_suite.heldout_tasks() (50).

Both splits run through this one code path. Writes runs/v0.3/[dev/]<suite>/{results.jsonl,
summary.json, analysis.json}. The model is TTSO_HYBRID_MODEL (default Qwen/Qwen3.5-0.8B; pass the
snapshot path when the hub cache has no refs/main).
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.bendlite import BendLiteCatalogCartridge
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark, catalog_suite

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("TTSO_HYBRID_MODEL", "Qwen/Qwen3.5-0.8B")
PRIMARY = {
    "json": {"H1": ("ttso_pinned", "greedy"), "H2": ("ttso_pinned", "best_of_n"),
             "H3": ("ttso_pinned_masked", "greedy_masked"), "H4": ("ttso_pinned_masked", "best_of_n_masked")},
    "catalog": {"C1": ("smart", "best_of_n"), "C2": ("smart_pinned", "best_of_n"),
                "C3": ("local", "best_of_n"), "C4": ("upstream", "best_of_n")},
}
PRIMARY_OUTCOME = {"json": "faithful", "catalog": "correct"}
SECONDARY = {
    "json": {"mask alone (greedy_masked vs greedy)": ("greedy_masked", "greedy"),
             "mask + repair vs repair (ttso_pinned_masked vs ttso_pinned)": ("ttso_pinned_masked", "ttso_pinned"),
             "pinned vs unpinned routing": ("ttso_pinned", "ttso_unpinned"),
             "best_of_n vs greedy": ("best_of_n", "greedy")},
    "catalog": {"smart vs upstream": ("smart", "upstream"), "smart vs local": ("smart", "local"),
                "smart_pinned vs smart": ("smart_pinned", "smart"), "best_of_n vs greedy": ("best_of_n", "greedy")},
}


def setup(be, which):
    specials = be.tok.all_special_ids
    if which == "json":
        mk = lambda **kw: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=specials, **kw)
        regimes = {"ttso_pinned": (mk(), "smart"), "ttso_pinned_masked": (mk(apply_forward_mask=True), "smart"),
                   "ttso_unpinned": (mk(pin_inputs=False), "smart")}
        bon = {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"}
        return regimes, bon, benchmark.JSONDAGSuite(), 320
    mk = lambda **kw: BendLiteCatalogCartridge(be.token_texts(), encode=be.encode, special_ids=specials, **kw)
    regimes = {"local": (mk(), "local"), "upstream": (mk(), "upstream"), "smart": (mk(), "smart"),
               "smart_pinned": (mk(pin_inputs=True), "smart")}
    return regimes, {"best_of_n": "upstream"}, catalog_suite.CatalogSuite(), 256


def tasks_for(which, split):
    if which == "json":
        dev = benchmark.make_tasks(20, seed=0)
        if split == "dev":
            return dev
        held = benchmark.make_tasks(60, seed=42)
        assert not {t["text"] for t in dev} & {t["text"] for t in held}, "held-out tasks overlap the dev tasks"
        return held
    dev = catalog_suite.dev_tasks()
    if split == "dev":
        return dev
    held = catalog_suite.heldout_tasks()
    assert not {t["text"] for t in dev} & {t["text"] for t in held}, "held-out tasks overlap the dev tasks"
    return held


def analyse(which, rows, s):
    f0 = PRIMARY_OUTCOME[which]
    fields = ("valid", "correct", "faithful")
    out = {"counts": {k: {f: s[k][f] for f in fields} for k in s["arms"]},
           "faithful_wilson95": {k: s[k]["faithful_wilson95"] for k in s["arms"]},
           "correct_wilson95": {k: benchmark.wilson(s[k]["correct"], s["n"]) for k in s["arms"]}}
    prim = {h: benchmark.compare(rows, a, b, f0) for h, (a, b) in PRIMARY[which].items()}
    adj = benchmark.holm({h: v["p"] for h, v in prim.items()})
    out[f"primary_{f0}"] = {h: dict(v, pair=PRIMARY[which][h], p_holm=adj[h]) for h, v in prim.items()}
    out["secondary"] = {name: {f: benchmark.compare(rows, a, b, f) for f in fields}
                        for name, (a, b) in SECONDARY[which].items()}
    out["secondary"].update({f"{h} on {f}": benchmark.compare(rows, a, b, f)
                             for h, (a, b) in PRIMARY[which].items() for f in fields if f != f0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("json", "catalog"))
    ap.add_argument("split", choices=("dev", "heldout"))
    ap.add_argument("--n", type=int, default=None, help="only the first n tasks (smoke runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tasks = tasks_for(args.which, args.split)[:args.n]
    be = HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.float16)
    regimes, bon, suite, max_new = setup(be, args.which)
    t0 = time.time()
    be.reset_peak()
    rows, s = benchmark.run(be, regimes, tasks, bon=bon, suite=suite, share_greedy=True, max_new_tokens=max_new)
    m = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]+)", MODEL)  # record a hub id, not a local path
    s.update(model=f"{m[1]}/{m[2]}" if m else MODEL, revision=m[3] if m else None, dtype="float16",
             split=args.split, n_tasks=len(tasks), wall_s=round(time.time() - t0, 1), max_new_tokens=max_new,
             k_max=3, **benchmark.device_info())
    out = Path(args.out or ROOT / "runs" / "v0.3" / ("dev" if args.split == "dev" else "") / args.which)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    a = analyse(args.which, rows, s)
    (out / "analysis.json").write_text(json.dumps(a, indent=1))
    print(json.dumps(a["counts"], indent=1))
    print(json.dumps(a[f"primary_{PRIMARY_OUTCOME[args.which]}"], indent=1))


if __name__ == "__main__":
    main()
