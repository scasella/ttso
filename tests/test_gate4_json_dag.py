"""Gate 4: end-to-end stateful JSON DAG repair on the M4 Pro.

20 few-shot TypedJSONDAG prompts (the seed-0 development tasks), Qwen3-0.6B-Base by default
(TTSO_GATE4_MODEL overrides). Regimes, each from its own greedy completion:
  ttso_pinned          Smart repair, input types frozen as the prompt's contract (the default)
  ttso_pinned_masked   the same, decoding and repairing inside the forward structural mask
  ttso_unpinned        Smart repair with every declaration editable (the spec's v0.1 routing)
  best_of_n[_masked]   resampling at temperature 0.7, matched per problem to the pinned regime's
                       extra tokens, without / with the forward mask
Logged to runs/gate4/<timestamp>/.

test_gate4_json_dag_mps: all 20 problems run and are logged; every ttso regime's mean latency per
problem (greedy + repair) < 8.5 s; peak MPS allocated memory < 3.5 GB.

test_gate4_llamacpp_parity: the pinned regimes again through LlamaCppBackend (Metal, an f16 GGUF of
the same model). Valid, correct and faithful counts must equal the HuggingFace run's. Per problem,
the greedy completion, the repair trajectory (round, t*, old, new) and the final completion must
match, with one exception: a problem whose greedy completions already differ, at a token where both
backends' top-two logits are within TIE (a float16 tie, about 1-3 ULP at logit ~25). Such ties are
logged. (The strict criterion failed on 1 of 20 problems in the first v0.2 run: a 0.016 margin
between "convert" and "ref", where even HF's incremental decode and a fresh prefill disagree.)
Logs the head-to-head latency and memory.
"""

import json

import pytest
import torch

from tests.gate4_setup import GGUF, MODEL, PARITY, TIE, _regimes, _table, _trajectory


def test_gate4_json_dag_mps(hf_run):
    rows, s, run_dir, _ = hf_run
    print(f"\n[gate4] {MODEL} on mps/fp16, {s['n']} prompts -> {run_dir}")
    print(_table(s, s["arms"]))
    print(f"  peak MPS allocated {s['peak_mps_allocated_gb']:.2f} GB")
    assert len(rows) == 20
    for k in ("ttso_pinned", "ttso_pinned_masked", "ttso_unpinned"):
        g = "greedy_masked" if k.endswith("masked") else "greedy"
        assert s[k]["valid"] >= s[g]["valid"]
        assert s[k]["latency_mean_s"] < 8.5
    assert s["peak_mps_allocated_gb"] < 3.5


@pytest.mark.skipif(not GGUF.exists(), reason=f"no GGUF at {GGUF}")
def test_gate4_llamacpp_parity(hf_run):
    pytest.importorskip("llama_cpp")
    from ttso.backends.llamacpp import LlamaCppBackend
    from ttso.engine import benchmark

    hf_rows, hf_s, run_dir, hf_be = hf_run
    from transformers import AutoTokenizer
    be = LlamaCppBackend(str(GGUF), n_ctx=2048)
    be.special_ids = AutoTokenizer.from_pretrained(MODEL).all_special_ids  # same specials as the HF run
    rows, s = benchmark.run(be, _regimes(be, PARITY), benchmark.make_tasks(20, seed=0), bon={})
    s.update(model=str(GGUF.name), backend="llama.cpp", buffers=be.buffers)
    (run_dir / "llamacpp_summary.json").write_text(json.dumps(s, indent=1))
    with open(run_dir / "llamacpp_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    mismatch = []
    for a, b in zip(hf_rows, rows):
        for k in PARITY:
            g = "greedy_masked" if k.endswith("masked") else "greedy"
            same = (a[g]["text"] == b[g]["text"] and _trajectory(a, k) == _trajectory(b, k)
                    and a[k]["text"] == b[k]["text"]
                    and (a[k]["valid"], a[k]["faithful"]) == (b[k]["valid"], b[k]["faithful"]))
            if not same:
                mismatch.append((a["i"], k))
    print(f"\n[gate4] llama.cpp ({GGUF.name}) vs HuggingFace mps, {len(rows)} prompts")
    print(_table(s, s["arms"]))
    for k in PARITY:
        print(f"  {k}: mean s/problem mps {hf_s[k]['latency_mean_s']:.2f} vs llama.cpp {s[k]['latency_mean_s']:.2f}")
    print(f"  memory: mps allocator peak {hf_s['peak_mps_allocated_gb']:.2f} GB; llama.cpp Metal buffers "
          f"{be.buffers['metal_mib'] / 1024:.2f} GB, CPU buffers {be.buffers['cpu_mib'] / 1024:.2f} GB, "
          f"host logits array {be.buffers['scores_mib'] / 1024:.2f} GB")
    print(f"  per-problem mismatches (problem, regime): {mismatch}")
    for k in PARITY:
        for f in ("valid", "correct", "faithful"):
            assert s[k][f] == hf_s[k][f], (k, f, s[k][f], hf_s[k][f])
    tasks = benchmark.make_tasks(20, seed=0)
    unexplained = []
    for i, k in mismatch:
        g = "greedy_masked" if k.endswith("masked") else "greedy"
        a, b = hf_be.encode(hf_rows[i][g]["text"]), hf_be.encode(rows[i][g]["text"])
        j = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if j is None:  # same greedy completion: the divergence is in repair itself
            unexplained.append((i, k, "repair diverged"))
            continue
        ctx = hf_be.encode(benchmark.prompt_of(tasks[i])) + a[:j]
        margins = []
        for backend in (hf_be, be):
            _, z, _ = backend.generate_from(ctx, None, 1)
            top = torch.topk(z[0], 2).values
            margins.append(float(top[0] - top[1]))
        tie = max(margins) < TIE
        print(f"  problem {i} {k}: greedy diverges at token {j} ({hf_be.decode([a[j]])!r} vs {hf_be.decode([b[j]])!r}), "
              f"top-two margin mps {margins[0]:.4f}, llama.cpp {margins[1]:.4f} -> {'float16 tie' if tie else 'NOT a tie'}")
        if not tie:
            unexplained.append((i, k, margins))
    assert not unexplained, unexplained
