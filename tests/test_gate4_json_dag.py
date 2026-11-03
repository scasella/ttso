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
the same model). Per problem, the greedy completion, the repair trajectory (round, t*, old, new),
the final completion and its valid / faithful verdicts must match the HuggingFace run. Logs the
head-to-head latency and memory.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("TTSO_GATE4_MODEL", "Qwen/Qwen3-0.6B-Base")
ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(os.environ.get("TTSO_GATE4_GGUF", ROOT / "models" / "qwen3-0.6b-base-f16.gguf"))
OUT = ROOT / "runs" / "gate4"
PARITY = ("ttso_pinned", "ttso_pinned_masked")


def _regimes(be, names=None):
    from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
    mk = lambda **kw: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                                            special_ids=sorted(be.eos_ids | set(getattr(be, "special_ids", []))),
                                            **kw)
    r = {"ttso_pinned": (mk(), "smart"), "ttso_pinned_masked": (mk(apply_forward_mask=True), "smart"),
         "ttso_unpinned": (mk(pin_inputs=False), "smart")}
    return {k: v for k, v in r.items() if names is None or k in names}


def _trajectory(row, k):
    return [(h["round"], h.get("t_star"), h.get("old"), h.get("new")) for h in row[k]["history"]]


def _table(s, keys):
    lines = [f"  {'':20s} {'valid':>5s} {'correct':>7s} {'faithful':>8s}  {'faithful 95% CI':>15s} {'extra tok':>9s} {'mean s':>6s}"]
    for k in keys:
        x = s[k]
        tok = x.get("extra_tokens_total", x.get("tokens_total", 0))
        lat = f"{x['latency_mean_s']:6.2f}" if "latency_mean_s" in x else ""
        ci = "{:.2f}-{:.2f}".format(*x["faithful_wilson95"])
        lines.append(f"  {k:20s} {x['valid']:5d} {x['correct']:7d} {x['faithful']:8d}  {ci:>15s} {tok:9d} {lat}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def hf_run():
    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple mps")
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    from ttso.engine import benchmark

    be = HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.float16)
    be.special_ids = be.tok.all_special_ids
    regimes = _regimes(be)
    bon = {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"}
    be.reset_peak()
    rows, s = benchmark.run(be, regimes, benchmark.make_tasks(20, seed=0), bon=bon)
    s.update(model=MODEL, backend="hf_mps", dtype="float16", **benchmark.device_info())
    run_dir = OUT / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return rows, s, run_dir


def test_gate4_json_dag_mps(hf_run):
    rows, s, run_dir = hf_run
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

    hf_rows, hf_s, run_dir = hf_run
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
        for f in ("valid", "faithful"):
            assert s[k][f] == hf_s[k][f], (k, f, s[k][f], hf_s[k][f])
    assert not mismatch
