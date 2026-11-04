"""KV-cache rewind gate for Gemma 4 E2B (v0.4).

Gemma 4 E2B has 35 layers: 28 sliding-window (512) and 7 global attention layers, and layers 15-34
reuse the K/V of layers 13 (sliding) and 14 (global), so the cache holds 15 layers. HuggingFace's
`crop` cannot rewind a sliding layer once its sequence passes the window; the backend keeps those
layers in recording mode and slices them itself (ttso/backends/hf_mps.py).

For each case: decode greedily, rewind to prompt + n and re-feed token n, which must reproduce the
tokens and the cached logits exactly; then rewind again, feed a different token (the runner-up at
n) and compare 12 tokens with a fresh decode of the whole edited prefix, which must agree on all 12.
Cases: n = 8 and n = 37 after the 678-token dev prompt (past the window, as in every benchmark
prompt: the brief's gate); n = 8 and n = 37 after a short prompt; and n = 37 after a 560-token
decode, a rewind from past the window back inside it. Another test checks that recording mode
leaves decoding past the window unchanged. The brief proposed checkpoints every 16 positions plus
replay; recording mode needs neither (it keeps 12 KiB per token for the sliding layers) and the
replay is exact, so it was kept.
"""

import os

import pytest
import torch

from ttso.engine import benchmark

MODEL = os.environ.get("TTSO_GEMMA_MODEL", "google/gemma-4-E2B")
PROMPT = "def add(a: int, b: int) -> int:\n    return"
LONG = benchmark.prompt_of(benchmark.make_tasks(1, seed=0)[0])  # a dev prompt, 678 tokens with BOS


@pytest.fixture(scope="module")
def gemma():
    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple mps")
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    be = HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.bfloat16)
    if be.model.config.model_type != "gemma4_text":
        pytest.skip(f"{MODEL} is not Gemma 4")
    return be


def _gate(be, prompt, total, n, edit_len=12):
    P = be.prompt_ids(prompt)
    toks, z, cache = be.generate_from(P, None, total)
    assert len(toks) == total and z.shape == (total, be.vocab_size)
    assert all(int(z[i].argmax()) == toks[i] for i in range(total))  # greedy picks from the cached row
    cache = be.rewind_kv_cache(cache, len(P) + n)
    assert be.cache_len(cache) == len(P) + n
    again, z2, cache = be.generate_from([toks[n]], cache, total - n - 1)
    assert again == toks[n + 1:]
    assert torch.equal(z2, z[n + 1:]), f"max |dz| {(z2 - z[n + 1:]).abs().max():.3g}"
    alt = int(torch.topk(z[n], 2).indices[1])
    cache = be.rewind_kv_cache(cache, len(P) + n)
    edited, _, _ = be.generate_from([alt], cache, edit_len)
    fresh, _, _ = be.generate_from(P + toks[:n] + [alt], None, edit_len)
    agree = sum(a == b for a, b in zip(edited, fresh))
    print(f"\n[gemma4] prompt {len(P)} tokens, decoded {total}, rewound to {n}: replay exact; "
          f"edited vs fresh decode {agree}/{len(fresh)}")
    assert len(edited) == len(fresh) == edit_len and agree == edit_len


@pytest.mark.parametrize("total,n", [(24, 8), (48, 37)])
def test_rewind_short_prompt(gemma, total, n):
    _gate(gemma, PROMPT, total, n)


@pytest.mark.parametrize("total,n", [(24, 8), (48, 37)])
def test_rewind_past_the_window(gemma, total, n):
    """The brief's gate: a 678-token prompt (past the 512 window), rewound at continuation tokens 8
    and 37."""
    _gate(gemma, LONG, total, n)


def test_rewind_back_inside_the_window(gemma):
    _gate(gemma, PROMPT, 560, 37)


def test_recording_mode_leaves_decoding_unchanged(gemma):
    """Past the window, a recording cache gives the same tokens and logits as HuggingFace's own."""
    from transformers import DynamicCache
    be = gemma
    P = be.prompt_ids(LONG)
    ours, z_ours, cache = be.generate_from(P, None, 32)
    theirs, z_theirs, plain = be.generate_from(P, DynamicCache(config=be.model.config), 32)
    assert any(getattr(layer, "record_past", False) for layer in cache.layers)
    assert not any(getattr(layer, "record_past", False) for layer in plain.layers)
    assert ours == theirs and torch.equal(z_ours, z_theirs)


@pytest.mark.parametrize("model_id", [MODEL, MODEL + "-it"])
def test_type_names_have_distinct_first_tokens(model_id):
    """TokenClassManager on the Gemma 4 tokenizer, for both cartridges: building it asserts that no
    two type names share a first token; the heads are those of the Qwen tokenizers."""
    from transformers import AutoTokenizer
    from ttso.cartridges.bendlite import BendLiteCatalogCartridge
    from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
    tok = AutoTokenizer.from_pretrained(model_id)
    texts = tok.batch_decode([[i] for i in range(len(tok))])
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    want = {"U32": {"U", "U32"}, "Nat": {"Nat"}, "Float": {"Float"}, "USD": {"USD"}, "EUR": {"EUR"}}
    for cls in (TypedJSONDAGCartridge, BendLiteCatalogCartridge):
        tcm = cls(texts, encode=enc, special_ids=tok.all_special_ids).tcm
        assert {n: h for n, h in zip(tcm.type_names, tcm.heads)} == {n: want[n] for n in tcm.type_names}
        assert all(int((tcm.type_of == k).sum()) >= 2 for k in range(tcm.K))  # bare and space-led


def test_prompts_start_with_one_bos():
    """Every prompt starts with exactly one BOS: the raw JSON and catalog prompts get it from
    prompt_ids, the chat prompt from the template (rendered here with the -it template on both)."""
    from types import SimpleNamespace

    from transformers import AutoTokenizer
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    from ttso.engine import catalog_suite
    for model_id in (MODEL, MODEL + "-it"):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.chat_template is None:
            tok.chat_template = AutoTokenizer.from_pretrained(MODEL + "-it").chat_template
        be = SimpleNamespace(tok=tok, bos_id=tok.bos_token_id, encode=lambda s: tok(s, add_special_tokens=False)["input_ids"])
        be.prompt_ids = lambda text: HuggingFaceMPSBackend.prompt_ids(be, text)
        for P in (benchmark.JSONDAGSuite().prompt_ids(be, benchmark.make_tasks(1, seed=0)[0]),
                  catalog_suite.CatalogSuite().prompt_ids(be, catalog_suite.dev_tasks()[0]),
                  catalog_suite.CatalogSuite(raw=True).prompt_ids(be, catalog_suite.dev_tasks()[0])):
            assert P[0] == tok.bos_token_id == 2 and P[1] != 2
