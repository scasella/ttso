"""Task 1 timing (v0.3): the incremental forward mask against v0.2's re-parse acceptor.

  latency   Qwen3-0.6B-Base on mps, the 20 seed-0 tasks, masked greedy + ttso_pinned_masked repair
            with each acceptor, run back to back per task in alternating order (so drift in the
            machine's load hits both alike). Checks that the tokens are identical.
  scaling   CPU only: valid workflow documents with 1-64 nodes, tokenized; the mask work of a masked
            decode (accept() of every token, candidates() at every position) with each acceptor.

Writes runs/v0.3/gate/mask_timing.json.
"""

import json
import random
import time
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark
from ttso.engine.repair_loop import RepairLoop

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B-Base"


def long_doc(n_nodes, rng):
    ins = [{"id": f"in{i}", "type": "U32"} for i in range(3)]
    nodes, prev = [], [x["id"] for x in ins]
    for j in range(n_nodes):
        args = [{"ref": rng.choice(prev)}, {"lit": f"{rng.randint(1, 99)}"}, {"ref": rng.choice(prev)}]
        nodes.append({"id": f"n{j}", "op": rng.choice(["add", "max", "min"]), "out_type": "U32", "args": args})
        prev.append(f"n{j}")
    return json.dumps({"inputs": ins, "nodes": nodes, "output": prev[-1],
                       "call": {x["id"]: "5" for x in ins}})


def main():
    be = HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.float16)
    mk = lambda impl: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=be.tok.all_special_ids,
                                            apply_forward_mask=True, mask_impl=impl)
    loops = {impl: RepairLoop(be, mk(impl), policy="smart", max_new_tokens=320) for impl in ("reparse", "incremental")}
    rows = []
    for task in benchmark.make_tasks(20, seed=0):
        P = be.encode(benchmark.prompt_of(task))
        order = ("reparse", "incremental") if task["i"] % 2 == 0 else ("incremental", "reparse")
        rec = {"i": task["i"]}
        for impl in order:
            loop = loops[impl]
            m0, t0 = loop.cart.mask_seconds, time.time()
            res = loop.repair(P)
            rec[impl] = {"seconds": time.time() - t0, "mask_s": loop.cart.mask_seconds - m0, "tokens": res.tokens,
                         "greedy_tokens": res.greedy_tokens}
        rec["identical"] = rec["reparse"]["tokens"] == rec["incremental"]["tokens"]
        rows.append(rec)
        print(task["i"], {k: round(rec[k]["seconds"], 2) for k in order}, rec["identical"], flush=True)
    lat = {impl: sum(r[impl]["seconds"] for r in rows) / len(rows) for impl in loops}
    mask = {impl: sum(r[impl]["mask_s"] for r in rows) / len(rows) for impl in loops}
    out = {"latency": {"model": MODEL, "n": len(rows), "identical": sum(r["identical"] for r in rows),
                       "latency_mean_s": lat, "mask_mean_s": mask,
                       "per_task": [{k: (v if k in ("i", "identical") else {kk: vv for kk, vv in v.items() if kk != "tokens"})
                                     for k, v in r.items()} for r in rows]}}
    rng, scaling = random.Random(0), []
    carts = {impl: mk(impl) for impl in loops}
    for n_nodes in (1, 4, 16, 64):
        toks = be.encode(long_doc(n_nodes, rng))
        rec = {"nodes": n_nodes, "tokens": len(toks)}
        for impl, cart in carts.items():
            t_acc = t_cand = 0.0
            for t in range(len(toks)):
                t0 = time.perf_counter()
                cart.accept(toks[:t], toks[t])
                t1 = time.perf_counter()
                if impl == "incremental" or len(toks) <= 1000 or t % 10 == 0:  # sample the slow path on long docs
                    cart.candidates(toks[:t])
                    t_cand += (time.perf_counter() - t1) * (1 if impl == "incremental" or len(toks) <= 1000 else 10)
                t_acc += t1 - t0
            rec[impl] = {"accept_us_per_token": t_acc / len(toks) * 1e6, "candidates_ms_per_token": t_cand / len(toks) * 1e3}
        print(rec, flush=True)
        scaling.append(rec)
    out["scaling"] = scaling
    d = ROOT / "runs" / "v0.3" / "gate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mask_timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out["latency"][k] for k in ("identical", "latency_mean_s", "mask_mean_s")}, indent=1))


if __name__ == "__main__":
    main()
