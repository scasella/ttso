"""v0.4 benchmarks on Gemma 4 E2B (pre-registered in runs/v0.4/PREREGISTRATION.md).

  json      TypedJSONDAG. ttso_pinned / ttso_pinned_masked now pin the task's input contract
            (pin_inputs=True: a wrong input type is caught and repaired unmasked, never decoded
            masked); greedy, best_of_n and the masked twins as before; ttso_unpinned; and two
            secondary arms: ttso_self_pinned (v0.3's pinning, inputs frozen as written) and
            best_of_2_full[_masked] (2 full samples per failing task, no token budget). The
            post-trained model runs the pinned arms, Best-of-N and best_of_2_full only.
            dev: make_tasks(20, seed=0); heldout: make_tasks(60, seed=42).
  catalog   BendLite catalog synthesis: greedy, best_of_n (budget matched to upstream),
            best_of_2_full, local, upstream, smart (pin_inputs=False), smart_pinned (pin_inputs=True;
            this cartridge pins parameter types as written).
            dev: catalog_suite.dev_tasks() (20); heldout: catalog_suite.heldout_tasks() (50).

Every outcome is scored on 1 and 5 input vectors (Correct_1 / Faithful_1, Correct_5 / Faithful_5);
Faithful_5 is primary. Every repair round also runs the symbolic provenance verifier as a shadow of
the DeltaNet (RepairLoop(shadow=True)); the loop follows the DeltaNet.

Models (--model): gemma-4-E2B (base) and gemma-4-E2B-it (post-trained), bf16, text-only, the
per-layer embedding table memory-mapped on the cpu (HuggingFaceMPSBackend map_ple). Prompts: the
JSON suite's raw few-shot prompt for both; for the catalog, the base model gets the same content as
a plain document (catalog_suite.raw_prompt), since under Gemma's chat template it only repeats the
system prompt (dev), and a chat model gets its chat template. Writes runs/v0.4/[dev/]<model>/<suite>/{results.jsonl,
summary.json, analysis.json}.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.bendlite import BendLiteCatalogCartridge
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark, catalog_suite

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # key: (hub id, revision, dtype)
    "gemma-4-E2B": ("google/gemma-4-E2B", "433813a669639fd7eaa83155828e9000ed357d53", torch.bfloat16),
    "gemma-4-E2B-it": ("google/gemma-4-E2B-it", "4adf15748de9e1a0097b6274cee9c15836c41c37", torch.bfloat16),
}
PRIMARY_OUTCOME = "faithful_5"
PRIMARY = {
    "json": {"H1": ("ttso_pinned", "greedy"), "H2": ("ttso_pinned", "best_of_n"),
             "H3": ("ttso_pinned_masked", "greedy_masked"), "H4": ("ttso_pinned_masked", "best_of_n_masked")},
    "catalog": {"C1": ("smart", "best_of_n"), "C2": ("smart_pinned", "best_of_n"),
                "C3": ("local", "best_of_n"), "C4": ("upstream", "best_of_n")},
}
SINGLE = {"json": {}, "catalog": {"C5": ("smart_pinned", "smart", "faithful_5")}}
SECONDARY = {
    "json": {"contract vs self pinning (ttso_pinned vs ttso_self_pinned)": ("ttso_pinned", "ttso_self_pinned"),
             "mask alone (greedy_masked vs greedy)": ("greedy_masked", "greedy"),
             "mask + repair vs repair (ttso_pinned_masked vs ttso_pinned)": ("ttso_pinned_masked", "ttso_pinned"),
             "pinned vs unpinned routing": ("ttso_pinned", "ttso_unpinned"),
             "best_of_n vs greedy": ("best_of_n", "greedy"),
             "ttso_pinned vs best_of_2_full": ("ttso_pinned", "best_of_2_full"),
             "ttso_pinned_masked vs best_of_2_full_masked": ("ttso_pinned_masked", "best_of_2_full_masked"),
             "best_of_2_full vs best_of_n": ("best_of_2_full", "best_of_n")},
    "catalog": {"smart vs upstream": ("smart", "upstream"), "smart vs local": ("smart", "local"),
                "best_of_n vs greedy": ("best_of_n", "greedy"), "smart vs best_of_2_full": ("smart", "best_of_2_full"),
                "smart_pinned vs best_of_2_full": ("smart_pinned", "best_of_2_full"),
                "best_of_2_full vs best_of_n": ("best_of_2_full", "best_of_n")},
}
FIELDS = ("valid", "correct", "correct_5", "faithful", "faithful_5")


def setup(be, which, post_trained=False):
    specials = be.tok.all_special_ids
    if which == "json":
        mk = lambda **kw: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=specials, **kw)
        regimes = {"ttso_pinned": (mk(), "smart"), "ttso_pinned_masked": (mk(apply_forward_mask=True), "smart")}
        if not post_trained:
            regimes["ttso_unpinned"] = (mk(pin_inputs=False), "smart")
            regimes["ttso_self_pinned"] = (mk(pin_inputs="self"), "smart")
        bon = {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"}
        full = {"best_of_2_full": "ttso_pinned", "best_of_2_full_masked": "ttso_pinned_masked"}
        return regimes, bon, full, benchmark.JSONDAGSuite(), 320
    mk = lambda **kw: BendLiteCatalogCartridge(be.token_texts(), encode=be.encode, special_ids=specials, **kw)
    regimes = {"local": (mk(), "local"), "upstream": (mk(), "upstream"), "smart": (mk(), "smart"),
               "smart_pinned": (mk(pin_inputs=True), "smart")}
    suite = catalog_suite.CatalogSuite(raw=not post_trained)  # a base model gets the plain-document prompt
    return regimes, {"best_of_n": "upstream"}, {"best_of_2_full": "upstream"}, suite, 256


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
    has = lambda *arms: all(a in rows[0] for a in arms)
    out = {"counts": {k: {f: s[k][f] for f in FIELDS} for k in s["arms"]},
           "faithful_5_wilson95": {k: s[k]["faithful_5_wilson95"] for k in s["arms"]},
           "correct_5_wilson95": {k: benchmark.wilson(s[k]["correct_5"], s["n"]) for k in s["arms"]}}
    for f in (PRIMARY_OUTCOME, "faithful", "correct_5", "correct"):
        res = {h: benchmark.compare(rows, a, b, f) for h, (a, b) in PRIMARY[which].items() if has(a, b)}
        adj = benchmark.holm({h: v["p"] for h, v in res.items()})
        out[f"family_{f}"] = {h: dict(v, pair=PRIMARY[which][h], p_holm=adj[h]) for h, v in res.items()}
    out["primary"] = out[f"family_{PRIMARY_OUTCOME}"]
    out["single"] = {h: dict(benchmark.compare(rows, a, b, f), pair=(a, b), outcome=f)
                     for h, (a, b, f) in SINGLE[which].items() if has(a, b)}
    out["secondary"] = {name: {f: benchmark.compare(rows, a, b, f) for f in FIELDS}
                        for name, (a, b) in SECONDARY[which].items() if has(a, b)}
    out["coincidences"] = {k: sum(r[k]["faithful"] and not r[k]["faithful_5"] for r in rows) for k in s["arms"]}
    out["shadow"] = {k: s[k]["shadow"] for k in s["arms"] if "shadow" in s[k]}
    out["samples"] = {k: s[k]["samples"] for k in s["arms"] if "samples" in s[k]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("json", "catalog"))
    ap.add_argument("split", choices=("dev", "heldout"))
    ap.add_argument("--model", choices=tuple(MODELS), default="gemma-4-E2B")
    ap.add_argument("--n", type=int, default=None, help="only the first n tasks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hub, rev, dtype = MODELS[args.model]
    tasks = tasks_for(args.which, args.split)[:args.n]
    be = HuggingFaceMPSBackend(hub, device="mps", dtype=dtype, revision=rev)
    post_trained = args.model.endswith("-it")
    regimes, bon, full, suite, max_new = setup(be, args.which, post_trained=post_trained)
    t0 = time.time()
    be.reset_peak()
    rows, s = benchmark.run(be, regimes, tasks, bon=bon, full=full, suite=suite, share_greedy=True,
                            max_new_tokens=max_new, loop_kw={"shadow": True})
    s.update(model=hub, revision=rev, dtype=str(dtype).replace("torch.", ""), split=args.split,
             n_tasks=len(tasks), wall_s=round(time.time() - t0, 1), max_new_tokens=max_new, k_max=3,
             device_weights_gib=be.param_bytes / 2**30, mapped_gib=be.mapped_bytes / 2**30,
             prompt="raw" if args.which == "json" or getattr(suite, "raw", False) else "chat template",
             **benchmark.device_info())
    out = Path(args.out or ROOT / "runs" / "v0.4" / ("dev" if args.split == "dev" else "") / args.model / args.which)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    a = analyse(args.which, rows, s)
    (out / "analysis.json").write_text(json.dumps(a, indent=1))
    print(json.dumps(a["counts"], indent=1))
    print(json.dumps(a["primary"], indent=1))
    print(json.dumps(a["single"], indent=1))
    print(json.dumps({"shadow": a["shadow"], "samples": a["samples"]}, indent=1))


if __name__ == "__main__":
    main()
