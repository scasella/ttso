"""Shared-greedy determinism check (v0.4), on dev tasks only.

benchmark.run(share_greedy=True) decodes greedy once per greedy arm and hands later regimes a copy of
its KV cache. That is equivalent to re-decoding only if decoding is deterministic and repair from a
copied cache matches repair from a fresh decode. For the first 4 dev tasks of each suite:
  greedy_same   two greedy decodes of the prompt give the same tokens and logits;
  fork_same     repair from a copy of the first decode's cache = repair from the second decode;
  fresh_same    repair that decodes greedy itself gives the same tokens again.
The JSON suite uses ttso_pinned_masked (mask, task contract, rewinds); the catalog suite smart.
Writes runs/v0.4/dev/<model>/determinism.json.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.engine.repair_loop import RepairLoop

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("benchmark_v04", ROOT / "scripts" / "benchmark_v04.py")
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(bv.MODELS), default="gemma-4-E2B")
    args = ap.parse_args()
    hub, rev, dtype = bv.MODELS[args.model]
    be = HuggingFaceMPSBackend(hub, device="mps", dtype=dtype, revision=rev)
    out = []
    for which, regime in (("json", "ttso_pinned_masked"), ("catalog", "smart")):
        regimes, _, _, suite, max_new = bv.setup(be, which, post_trained=args.model.endswith("-it"))
        cart, policy = regimes[regime]
        loop = RepairLoop(be, cart, policy=policy, max_new_tokens=max_new)
        for task in bv.tasks_for(which, "dev")[:4]:
            cart.bind_task(task)
            P = suite.prompt_ids(be, task)
            t1, z1, c1 = loop.greedy(P)
            t2, z2, c2 = loop.greedy(P)
            forked = loop.repair(P, greedy=(t1, z1, be.fork_cache(c1)))
            second = loop.repair(P, greedy=(t2, z2, c2))
            fresh = loop.repair(P)
            rec = {"suite": which, "regime": regime, "i": task["i"],
                   "greedy_same": t1 == t2 and torch.equal(z1, z2),
                   "fork_same": forked.tokens == second.tokens,
                   "fresh_same": fresh.tokens == forked.tokens, "rounds": forked.rounds}
            print(rec, flush=True)
            out.append(rec)
    d = ROOT / "runs" / "v0.4" / "dev" / args.model
    d.mkdir(parents=True, exist_ok=True)
    (d / "determinism.json").write_text(json.dumps(out, indent=1))
    print("all identical:", all(r["greedy_same"] and r["fork_same"] and r["fresh_same"] for r in out))


if __name__ == "__main__":
    main()
