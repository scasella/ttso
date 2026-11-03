"""Shared Gate 4 setup: the HuggingFace run on the 20 seed-0 tasks (the `hf_run` session fixture in
conftest.py), reused by Gate 4 and by the incremental-mask gate."""

import json
import os
import time
from pathlib import Path

import torch

MODEL = os.environ.get("TTSO_GATE4_MODEL", "Qwen/Qwen3-0.6B-Base")
ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(os.environ.get("TTSO_GATE4_GGUF", ROOT / "models" / "qwen3-0.6b-base-f16.gguf"))
OUT = ROOT / "runs" / "gate4"
PARITY = ("ttso_pinned", "ttso_pinned_masked")
TIE = 0.05


def _regimes(be, names=None, **kw):
    from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
    mk = lambda **k: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                                           special_ids=sorted(be.eos_ids | set(getattr(be, "special_ids", []))),
                                           **k, **kw)
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


def run_hf():
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
    return rows, s, run_dir, be
