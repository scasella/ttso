"""Gate 4: end-to-end stateful JSON DAG repair on the M4 Pro (HuggingFaceMPSBackend, float16).

20 few-shot TypedJSONDAG prompts, Qwen3-0.6B-Base by default (TTSO_GATE4_MODEL overrides). For each
prompt, from the same greedy completion:
  ttso         Smart repair (<= 3 rounds), every declaration editable (the spec's routing);
  ttso_pinned  Smart repair with the input types frozen as the prompt's contract (LocalOnly
               rewrites a mismatched argument instead of retyping the input);
  best_of_n    resampling at temperature 0.7, matched to `ttso`'s extra tokens on that problem.
Logged to runs/gate4/<timestamp>/ (per-problem jsonl + summary).

Pass: all 20 problems run and are logged; for each ttso regime the mean latency per problem
(greedy + repair) < 8.5 s; peak MPS allocated memory < 3.5 GB. Validity, correctness, faithfulness
and the paired comparisons are reported, not asserted.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("TTSO_GATE4_MODEL", "Qwen/Qwen3-0.6B-Base")
OUT = Path(__file__).resolve().parents[1] / "runs" / "gate4"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple mps")
def test_gate4_json_dag_mps():
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
    from ttso.engine import benchmark

    be = HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.float16)
    mk = lambda pin: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                                           special_ids=be.tok.all_special_ids, pin_inputs=pin)
    regimes = {"ttso": (mk(False), "smart"), "ttso_pinned": (mk(True), "smart")}
    tasks = benchmark.make_tasks(20, seed=0)
    be.reset_peak()
    rows, summary = benchmark.run(be, regimes, tasks, k_max=3)
    summary.update(model=MODEL, device="mps", dtype="float16", **benchmark.device_info())

    run_dir = OUT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    s = summary
    print(f"\n[gate4] {MODEL} on mps/fp16, {s['n']} prompts -> {run_dir}")
    print(f"  {'':12s} {'valid':>6s} {'correct':>8s} {'faithful':>9s} {'extra tok':>10s} {'mean s':>7s} {'max s':>6s}")
    for k in ["greedy", "best_of_n"] + list(regimes):
        x = s[k]
        tok = x.get("extra_tokens_total", x.get("tokens_total", 0))
        lat = f"{x['latency_mean_s']:7.2f}" if "latency_mean_s" in x else " " * 7
        mx = f"{x['latency_max_s']:6.2f}" if "latency_max_s" in x else ""
        print(f"  {k:12s} {x['valid']:6d} {x['correct']:8d} {x['faithful']:9d} {tok:10d} {lat} {mx}")
    for k in regimes:
        print(f"  McNemar {k}: {json.dumps(s[f'mcnemar {k}'])}")
    print(f"  {s['repaired_problems']} problems needed repair; peak MPS allocated {s['peak_mps_allocated_gb']:.2f} GB")

    assert len(rows) == 20
    for k in regimes:
        assert s[k]["valid"] >= s["greedy"]["valid"]
        assert s[k]["latency_mean_s"] < 8.5
    assert s["peak_mps_allocated_gb"] < 3.5
