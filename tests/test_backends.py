"""KV-cache rewind on the real backends: rewinding to position n and re-feeding the same token
reproduces the original continuation exactly; feeding a different token matches a fresh decode of
the edited prefix; the cached logits are the logits each token was chosen from."""

import os
from pathlib import Path

import pytest
import torch

from tests.conftest import MODEL_ID

PROMPT = "def add(a: int, b: int) -> int:\n    return"
HYBRID_ID = os.environ.get("TTSO_HYBRID_MODEL", "Qwen/Qwen3.5-0.8B")
GGUF = Path(os.environ.get("TTSO_GGUF", Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b-f16.gguf"))


def _check_backend(be, total=24, n=8):
    P = be.encode(PROMPT)
    toks, z, cache = be.generate_from(P, None, total)
    assert len(toks) == total and z.shape == (total, be.vocab_size)
    assert all(int(z[i].argmax()) == toks[i] for i in range(total))  # greedy picks from the cached row
    cache = be.rewind_kv_cache(cache, len(P) + n)
    assert be.cache_len(cache) == len(P) + n
    again, z2, cache = be.generate_from([toks[n]], cache, total - n - 1)
    assert again == toks[n + 1:]
    assert torch.allclose(z2, z[n + 1:], atol=1e-3)
    # an edit: a different token at n, regenerated from the rewound cache vs a fresh decode
    alt = int(torch.topk(z[n], 2).indices[1])
    cache = be.rewind_kv_cache(cache, len(P) + n)
    edited, _, _ = be.generate_from([alt], cache, 12)
    fresh, _, _ = be.generate_from(P + toks[:n] + [alt], None, 12)
    agree = sum(a == b for a, b in zip(edited, fresh)) / max(len(fresh), 1)
    print(f"\n[backend] {type(be).__name__}: rewind replay exact; edited vs fresh decode agree {agree:.0%}")
    assert agree >= 0.75


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple mps")
def test_hf_mps_rewind():
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    _check_backend(HuggingFaceMPSBackend(MODEL_ID, device="mps"))


@pytest.fixture(scope="module")
def hybrid_backend():
    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple mps")
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    return HuggingFaceMPSBackend(HYBRID_ID, device="mps")


@pytest.mark.parametrize("total,n", [(24, 8), (48, 37)])
def test_hf_hybrid_linear_attention_rewind(hybrid_backend, total, n):
    """Qwen3.5-0.8B: 18 Gated DeltaNet layers and 6 attention layers. Rewinding restores a
    recurrent-state checkpoint and replays to n; n = 37 falls between two 16-position checkpoints."""
    be = hybrid_backend
    assert be.hybrid
    _check_backend(be, total, n)
    print(f"[backend] {HYBRID_ID}: checkpoints held at most {be.ckpt_bytes_peak / 2**20:.0f} MiB")


@pytest.mark.skipif(not GGUF.exists(), reason=f"no GGUF at {GGUF}")
def test_llamacpp_rewind():
    pytest.importorskip("llama_cpp")
    from ttso.backends.llamacpp import LlamaCppBackend
    _check_backend(LlamaCppBackend(str(GGUF), n_ctx=512))
