"""v0.5: a type-aware forward mask as the baseline for repair on the typed JSON DAG suite (3A),
pre-registered in runs/v0.5/PREREGISTRATION.md.

Arms (Gemma 4 E2B base, bf16 on mps, v0.4's revision, prompts, tokenizer and settings):
  greedy_typemasked        greedy decoding under TypeAwareMask (type_mask.py), contract-pinned
  ttso_pinned_typemasked   the same cartridge, then ttso repair (smart, K_max = 3, as v0.4)
  best_of_n_typemasked     Best-of-N at T = 0.7 under TypeAwareMask, token-matched per task to
                           ttso_pinned_typemasked's extra tokens (= the greedy arm where repair
                           spent nothing), as in v0.4
  greedy_masked, ttso_pinned_masked   v0.4's arms, unchanged (SchemaPrefix mask), the reproduction
dev: make_tasks(20, seed=0); heldout: make_tasks(60, seed=42).

Every program the type-masked cartridge's reference checker sees (greedy, each repair round, each
Best-of-N sample) is audited (TypeMaskedJSONDAGCartridge.audit_program): accepted by the type-aware
mask at every prefix (so the decoder never fell back to a rejected token, i.e. never met an empty
admissible set), failed symbolic-verifier checks and hard errors, and the mask's fallback counts.
Writes runs/v0.5/[dev/]gemma-4-E2B/json/{results.jsonl, audit.jsonl, summary.json, analysis.json}.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.type_mask import FALLBACKS, TypeAwareMask, TypeMaskedJSONDAGCartridge
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark

ROOT = Path(__file__).resolve().parents[1]
MODEL = ("google/gemma-4-E2B", "433813a669639fd7eaa83155828e9000ed357d53", torch.bfloat16)
PRIMARY_OUTCOME = "faithful_5"
PRIMARY = {"T1": ("ttso_pinned_masked", "greedy_typemasked"), "T2": ("ttso_pinned_typemasked", "greedy_typemasked")}
SECONDARY = {"mask alone: greedy_typemasked vs greedy_masked": ("greedy_typemasked", "greedy_masked"),
             "repair under each mask: ttso_pinned_typemasked vs ttso_pinned_masked": ("ttso_pinned_typemasked",
                                                                                       "ttso_pinned_masked"),
             "best_of_n_typemasked vs greedy_typemasked": ("best_of_n_typemasked", "greedy_typemasked"),
             "R: ttso_pinned_masked vs greedy_masked (v0.4's H3)": ("ttso_pinned_masked", "greedy_masked")}
REPRO = {"greedy_masked": 17, "ttso_pinned_masked": 33}  # v0.4 held-out Faithful_5
FIELDS = ("valid", "correct", "correct_5", "faithful", "faithful_5")
FAMILIES = {0: "Nat + U32 + U32 sum", 1: "USD sum + charge", 2: "max of three Nat", 3: "min(U32 + Nat, U32)"}
BON_GREEDY = {"best_of_n_typemasked": "greedy_typemasked"}  # where Best-of-N did not sample, its program is this


def setup(be):
    specials = be.tok.all_special_ids
    plain = TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=specials, apply_forward_mask=True)
    typed = TypeMaskedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=specials)
    typed.audit = []
    regimes = {"ttso_pinned_masked": (plain, "smart"), "ttso_pinned_typemasked": (typed, "smart")}
    return regimes, {"best_of_n_typemasked": "ttso_pinned_typemasked"}, typed


def tasks_for(split):
    dev = benchmark.make_tasks(20, seed=0)
    if split == "dev":
        return dev
    held = benchmark.make_tasks(60, seed=42)
    assert not {t["text"] for t in dev} & {t["text"] for t in held}, "held-out tasks overlap the dev tasks"
    return held


def text_fallbacks(task, text):
    """TypeAwareMask over a row's final text: (accepted, fallback counts of the accepted part)."""
    contract = dict(zip(task["inputs"], task["types"]))
    st = TypeAwareMask(contract).advance(text)
    return st is not None, (st.fallbacks if st is not None else None)


def seconds(rows, arm):
    """Mean seconds per problem. A Best-of-N arm costs its regime's greedy decode plus its samples."""
    if arm in rows[0]["seconds"]:
        return sum(r["seconds"][arm]["total"] for r in rows) / len(rows)
    if arm in BON_GREEDY:
        return sum(r[BON_GREEDY[arm]]["seconds"] + r[arm].get("seconds", 0.0) for r in rows) / len(rows)
    return sum(r[arm]["seconds"] for r in rows) / len(rows)


def gate(audit):
    """The dev gate's checks over every audited program of the type-masked cartridge."""
    complete = [a for a in audit if a["complete"]]
    return {"programs": len(audit), "typed_ok": sum(a["typed_ok"] for a in audit),
            "complete": len(complete), "complete_done": sum(a["done"] for a in complete),
            "complete_valid": sum(a["valid"] for a in complete),
            "complete_failed_checks": sum(len(a["failed_checks"]) for a in complete),
            "complete_hard": sum(len(a["hard"]) for a in complete),
            "incomplete_errors": Counter(a["error"][:40] for a in audit if not a["complete"]),
            "passed": (all(a["typed_ok"] for a in audit)
                       and all(a["done"] and a["valid"] and not a["failed_checks"] and not a["hard"] for a in complete))}


def analyse(rows, s, tasks, split, audit):
    arms = s["arms"]
    has = lambda *a: all(x in rows[0] for x in a)
    out = {"counts": {k: {f: s[k][f] for f in FIELDS} for k in arms},
           "faithful_5_wilson95": {k: benchmark.wilson(s[k]["faithful_5"], s["n"]) for k in arms},
           "correct_5_wilson95": {k: benchmark.wilson(s[k]["correct_5"], s["n"]) for k in arms},
           "seconds_per_problem": {k: seconds(rows, k) for k in arms}}
    for f in (PRIMARY_OUTCOME, "correct_5", "faithful"):
        res = {h: benchmark.compare(rows, a, b, f) for h, (a, b) in PRIMARY.items() if has(a, b)}
        adj = benchmark.holm({h: v["p"] for h, v in res.items()})
        out[f"family_{f}"] = {h: dict(v, pair=PRIMARY[h], p_holm=adj[h]) for h, v in res.items()}
    out["primary"] = out[f"family_{PRIMARY_OUTCOME}"]
    out["secondary"] = {name: {f: benchmark.compare(rows, a, b, f) for f in FIELDS}
                        for name, (a, b) in SECONDARY.items() if has(a, b)}
    by_i = {t["i"]: t for t in tasks}
    out["families"] = {
        FAMILIES[kind]: {k: {f: sum(r[k][f] for r in rows if r["i"] % 4 == kind) for f in ("valid", "correct_5", "faithful_5")}
                         | {"n": sum(r["i"] % 4 == kind for r in rows)} for k in arms}
        for kind in FAMILIES}
    out["families_discordant_faithful_5"] = {
        h: {FAMILIES[kind]: benchmark.compare([r for r in rows if r["i"] % 4 == kind], a, b, PRIMARY_OUTCOME)
            for kind in FAMILIES} for h, (a, b) in PRIMARY.items() if has(a, b)}
    fb = {}
    for k in arms:
        acc = [text_fallbacks(by_i[r["i"]], r[k].get("text") or r[BON_GREEDY[k]]["text"]) for r in rows]
        fb[k] = {"programs": len(acc), "accepted_by_type_mask": sum(ok for ok, _ in acc),
                 **{c: sum(f[c] for ok, f in acc if ok) for c in FALLBACKS}}
    out["final_programs_under_type_mask"] = fb
    out["gate"] = gate(audit)
    out["audit_fallbacks"] = {c: sum(a["fallbacks"][c] for a in audit if a["typed_ok"]) for c in FALLBACKS}
    out["coincidences"] = {k: sum(r[k]["faithful"] and not r[k]["faithful_5"] for r in rows) for k in arms}
    out["shadow"] = {k: s[k]["shadow"] for k in arms if "shadow" in s[k]}
    out["samples"] = {k: s[k]["samples"] for k in arms if "samples" in s[k]}
    v04 = ROOT / "runs" / "v0.4" / ("dev/" if split == "dev" else "") / "gemma-4-E2B" / "json" / "results.jsonl"
    if v04.exists():  # read-only: does the rerun reproduce v0.4's arms token for token?
        old = {r["i"]: r for r in map(json.loads, open(v04))}
        out["reproduction"] = {k: {"v0.4_faithful_5": sum(old[r["i"]][k]["faithful_5"] for r in rows),
                                   "v0.5_faithful_5": s[k]["faithful_5"],
                                   "identical_text": sum(r[k]["text"] == old[r["i"]][k]["text"] for r in rows),
                                   "different_tasks": [r["i"] for r in rows if r[k]["text"] != old[r["i"]][k]["text"]]}
                               for k in REPRO if k in rows[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("dev", "heldout"))
    ap.add_argument("--n", type=int, default=None, help="only the first n tasks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hub, rev, dtype = MODEL
    tasks = tasks_for(args.split)[:args.n]
    be = HuggingFaceMPSBackend(hub, device="mps", dtype=dtype, revision=rev)
    regimes, bon, typed = setup(be)
    t0 = time.time()
    be.reset_peak()
    rows, s = benchmark.run(be, regimes, tasks, bon=bon, suite=benchmark.JSONDAGSuite(), share_greedy=True,
                            max_new_tokens=320, loop_kw={"shadow": True})
    s.update(model=hub, revision=rev, dtype=str(dtype).replace("torch.", ""), split=args.split, n_tasks=len(tasks),
             wall_s=round(time.time() - t0, 1), max_new_tokens=320, k_max=3, temperature=0.7,
             device_weights_gib=be.param_bytes / 2**30, mapped_gib=be.mapped_bytes / 2**30, prompt="raw",
             **benchmark.device_info())
    out = Path(args.out or ROOT / "runs" / "v0.5" / ("dev" if args.split == "dev" else "") / "gemma-4-E2B" / "json")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "audit.jsonl", "w") as f:
        for a in typed.audit:
            f.write(json.dumps(a) + "\n")
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    a = analyse(rows, s, tasks, args.split, typed.audit)
    (out / "analysis.json").write_text(json.dumps(a, indent=1))
    for key in ("counts", "primary", "families", "final_programs_under_type_mask", "gate", "reproduction"):
        print(key, json.dumps(a.get(key), indent=1))


if __name__ == "__main__":
    main()
