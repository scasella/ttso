"""Post-hoc check (v0.2): replay the masked Best-of-N arm with the fixed pick().

The frozen pick() could return an illegal token if, after a rejected draw, every legal token's
sampling probability had underflowed to 0. The fix only changes that exhausted case, and every
draw uses a per-task seeded generator, so identical outputs mean the case never occurred."""

import json
import sys
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark
from ttso.engine.repair_loop import RepairLoop

which = sys.argv[1]
model = {"base": "Qwen/Qwen3-0.6B-Base", "post": "Qwen/Qwen3-0.6B"}[which]
run = Path(__file__).resolve().parents[1] / "runs" / "v0.2" / which
rows = [json.loads(l) for l in open(run / "results.jsonl")]
tasks = {t["i"]: t for t in benchmark.make_tasks(60, seed=42)}
be = HuggingFaceMPSBackend(model, device="mps", dtype=torch.float16)
cart = TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=be.tok.all_special_ids,
                             apply_forward_mask=True)
loop = RepairLoop(be, cart, max_new_tokens=320)
same = diff = 0
for r in rows:
    if r["greedy_masked"]["valid"] or r["ttso_pinned_masked"]["extra_tokens"] == 0:
        continue
    task = tasks[r["i"]]
    (toks, v), n, spent = loop.best_of_n(be.encode(benchmark.prompt_of(task)), r["ttso_pinned_masked"]["extra_tokens"],
                                         temperature=0.7, seed=task["i"])
    old = r["best_of_n_masked"]
    if be.decode(toks) == old["text"] and n == old["samples"] and spent == old["tokens"]:
        same += 1
    else:
        diff += 1
        print("differs:", r["i"])
out = {"identical": same, "different": diff}
(run / "bon_masked_recheck.json").write_text(json.dumps(out))
print(which, out)
