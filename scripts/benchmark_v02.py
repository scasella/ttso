"""v0.2 held-out benchmark (pre-registered in runs/v0.2/PREREGISTRATION.md).

  base   Qwen3-0.6B-Base, 60 held-out tasks (seed 42).
         Arms: greedy, best_of_n, ttso_pinned (backward repair only);
               greedy_masked, best_of_n_masked, ttso_pinned_masked (forward structural mask);
               ttso_unpinned (secondary: the v0.1 spec routing).
  post   Qwen3-0.6B (post-trained), the first 20 of those tasks, same pinned arms (exploratory).

Writes runs/v0.2/<name>/{results.jsonl, summary.json, analysis.json}.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {"H1": ("ttso_pinned", "greedy"), "H2": ("ttso_pinned", "best_of_n"),
           "H3": ("ttso_pinned_masked", "greedy_masked"), "H4": ("ttso_pinned_masked", "best_of_n_masked")}
SECONDARY = {"mask alone (greedy_masked vs greedy)": ("greedy_masked", "greedy"),
             "mask + repair vs repair (ttso_pinned_masked vs ttso_pinned)": ("ttso_pinned_masked", "ttso_pinned"),
             "pinned vs unpinned routing": ("ttso_pinned", "ttso_unpinned")}


def analyse(rows, s):
    out = {"faithful_wilson95": {k: s[k]["faithful_wilson95"] for k in s["arms"]},
           "counts": {k: {f: s[k][f] for f in ("valid", "correct", "faithful")} for k in s["arms"]}}
    prim = {h: benchmark.compare(rows, a, b, "faithful") for h, (a, b) in PRIMARY.items()
            if a in rows[0] and b in rows[0]}
    adj = benchmark.holm({h: v["p"] for h, v in prim.items()})
    out["primary_faithful"] = {h: dict(v, pair=PRIMARY[h], p_holm=adj[h]) for h, v in prim.items()}
    out["secondary"] = {name: {f: benchmark.compare(rows, a, b, f) for f in ("valid", "correct", "faithful")}
                        for name, (a, b) in SECONDARY.items() if a in rows[0] and b in rows[0]}
    out["secondary"].update({f"{h} on {f}": benchmark.compare(rows, a, b, f) for h, (a, b) in PRIMARY.items()
                             for f in ("valid", "correct") if a in rows[0] and b in rows[0]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("base", "post"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = {"base": "Qwen/Qwen3-0.6B-Base", "post": "Qwen/Qwen3-0.6B"}[args.which]
    tasks = benchmark.make_tasks(60, seed=42)
    dev = {t["text"] for t in benchmark.make_tasks(20, seed=0)}
    assert not dev & {t["text"] for t in tasks}, "held-out tasks overlap the development tasks"
    if args.which == "post":
        tasks = tasks[:20]
    be = HuggingFaceMPSBackend(model, device="mps", dtype=torch.float16)
    mk = lambda **kw: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                                            special_ids=be.tok.all_special_ids, **kw)
    regimes = {"ttso_pinned": (mk(), "smart"), "ttso_pinned_masked": (mk(apply_forward_mask=True), "smart")}
    if args.which == "base":
        regimes["ttso_unpinned"] = (mk(pin_inputs=False), "smart")
    bon = {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"}
    t0 = time.time()
    be.reset_peak()
    rows, s = benchmark.run(be, regimes, tasks, bon=bon)
    s.update(model=model, n_tasks=len(tasks), task_seed=42, wall_s=round(time.time() - t0, 1),
             **benchmark.device_info())
    out = Path(args.out or ROOT / "runs" / "v0.2" / args.which)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    a = analyse(rows, s)
    (out / "analysis.json").write_text(json.dumps(a, indent=1))
    print(json.dumps(a["counts"], indent=1))
    print(json.dumps(a["primary_faithful"], indent=1))


if __name__ == "__main__":
    main()
