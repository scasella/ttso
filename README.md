# ttso — Time-Travel Structured Outputs

`ttso` repairs LLM structured outputs at inference time. It targets errors that a context-free grammar (GBNF, JSON Schema, Outlines) cannot catch, such as a reference 200 tokens after a declaration that disagrees with its type.

It runs a compiled, zero-parameter Gated DeltaNet verifier over the completion. When a check fails, ttso:

1. backpropagates the first failed check into the LLM's cached logits;
2. routes one edit (Smart routing: the declaration or the call site);
3. rewinds the KV cache to that token;
4. regenerates the suffix, for up to 3 rounds.

This is DiffRepair, from the `diffspec` study in the parent directory, packaged for Apple Silicon. Two backends are included: HuggingFace on `mps`, and llama.cpp with Metal.

## Install (M4 Pro, macOS)

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

llama.cpp backend (optional). On this macOS / Command Line Tools combination the source build needs the SDK's libc++ headers:

```bash
SDKROOT=$(xcrun --show-sdk-path) CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path) -DCMAKE_CXX_FLAGS=-isystem\ $(xcrun --show-sdk-path)/usr/include/c++/v1" uv pip install --python .venv/bin/python llama-cpp-python
```

A GGUF can be made from the cached HF weights with the converter that ships in llama.cpp. It needs `gguf-py` from the same source tree, plus `sentencepiece`. The output path `models/qwen3-0.6b-f16.gguf` is what the backend test looks for:

```bash
python <llama.cpp>/convert_hf_to_gguf.py <hf snapshot dir of Qwen/Qwen3-0.6B> --outtype f16 --outfile models/qwen3-0.6b-f16.gguf
```

## Use

```python
from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine.repair_loop import RepairLoop

be = HuggingFaceMPSBackend("Qwen/Qwen3-0.6B-Base")          # float16 on mps
cart = TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                             special_ids=be.tok.all_special_ids, pin_inputs=True)
res = RepairLoop(be, cart, policy="smart", k_max=3).repair(be.encode(prompt))
print(res.verdict.valid, res.rounds, res.extra_tokens, res.history)
```

`LlamaCppBackend("models/qwen3-0.6b-f16.gguf")` is a drop-in replacement for the backend.

## Layout

| Module | What it does |
|---|---|
| `core/deltanet.py` | `CompiledDeltaNet`: memory M ∈ R^{S×K}, ops NOOP / DECLARE / CHECK_VAR / CHECK_CALL, straight-through token bridge, `first_failure_gradients` |
| `core/token_classes.py` | `TokenClassManager`: equivalence classes, type-head tokens, `W_decl`, `W_lit`, `G_type` |
| `core/routing.py` | `diagnose`; `Router` with the policies `smart`, `upstream` (v1), `local`, `blind`; `RepairState` (bans, edited sites) |
| `backends/` | `hf_mps.py` (DynamicCache crop), `llamacpp.py` (`kv_cache_seq_rm`), `scripted.py` (deterministic test LM) |
| `cartridges/` | `bendlite.py` (catalog subset, U32 / Nat), `typed_json_dag.py` (typed tool-call DAGs), `synthetic.py` (SynthLang for Gates 1–2) |
| `engine/` | `repair_loop.py` (the 3-round loop, Best-of-N), `benchmark.py` (Gate 4 tasks and paired statistics) |

## Gates

Run them with:

```bash
.venv/bin/python -m pytest
```

Expect about 7 minutes, most of it Gate 4 (17 tests). Results from 3 November 2026 on an M4 Pro (24 GB), torch 2.15, transformers 5.22.1, llama-cpp-python 0.3.36:

| Gate | Result |
|---|---|
| 1. Exactness and non-fading memory | 200 random programs (27,857 tokens, 3,906 checks) agree with the string-level reference checker on every token. Both the check verdicts and the full memory state after each token match, in float32 and float16, on cpu and mps. The gradient at the declaration is the same at distances 8 and 256 (ratio 1.000000). Tokens in between receive exactly 0. The custom backward equals autograd through the spec's p_t formula (float64). |
| 2. Single-edit type direction (K = 5) | 200 / 200 declarations fixed by one gradient-directed edit. Blind, the model's best other type token at the same site, fixed 45 / 200. |
| 3. Problem 14 | Smart: round 1 changes `+stock: Nat` to `U32`. Round 2 sees `consistent_uses[stock] = 1`, edits the call site (`6n` → `6`) and is valid in 2 rounds. See the notes below for the Upstream comparison. |
| 4. JSON DAG on M4 Pro | Table below. Peak MPS allocated memory was 1.72 GB. |
| Backends | On both HF/mps and llama.cpp/Metal, rewind + replay reproduces the original continuation exactly. An edited token's continuation matches a fresh decode of the edited prefix (12 / 12 tokens). |

Gate 4: 20 prompts, Qwen3-0.6B-Base, float16 on mps. Every regime starts from the same greedy completion. Best-of-N samples at temperature 0.7, with its tokens matched per problem to the unpinned ttso run.

Two runs are logged, `runs/gate4/20261103-084404` (Gate 4 alone) and `20261103-085145` (the full suite). Their outcomes are identical, because decoding is deterministic. Timings are given as the range across the two runs.

| | Valid | Correct | Faithful | Extra tokens | Mean s / problem | Max s |
|---|---|---|---|---|---|---|
| Greedy | 5 | 5 | 5 | 0 | 3.6–4.1 | |
| Best-of-N | 6 | 6 | 5 | 2,670 | | |
| ttso, spec routing | 15 | 11 | 5 | 2,824 | 6.7–7.8 | 12.2–14.7 |
| ttso, `pin_inputs=True` | 14 | 10 | 9 | 2,352 | 5.9–7.2 | 15.5–18.7 |

Column definitions:

- **Correct:** valid, the expected output value, and the task's input names.
- **Faithful:** correct, and every input keeps the type the task stated.

Paired exact McNemar tests on n = 20:

- Correct, ttso vs greedy: 6–0 (p = 0.031).
- Correct, ttso vs Best-of-N: 6–1 (p = 0.125).
- Faithful, pinned vs Best-of-N: 4–0 (p = 0.125).

The 8.5 s latency target holds, but only with a modest margin: 6.7–7.8 s across three runs, including an earlier run of the spec-routing regime alone (`20261103-083639`, same outcomes, 7.63 s). A busy machine could push it past the target. Per-problem logs and summaries are in `runs/gate4/<timestamp>/`.

## What Gate 4 shows

With every declaration editable, the gradient of a failed reference check reaches only the declaration. Smart routing therefore retypes the input. All 6 of its extra correct programs changed an input type that the task had fixed, for example `boxes: Nat` became `U32`. The program is consistent, but the interface changed.

`pin_inputs=True` freezes the input declarations as the prompt's contract. A failed reference then falls back to LocalOnly at the argument's key token. Banning `"ref"` there lets the model write `{"convert": "to_u32(boxes)"}`. That raised faithful programs from 5 to 9.

Two kinds of error are out of reach for either regime: malformed JSON, and literals the schema does not define, such as `"47u32"`. The post-trained Qwen3-0.6B mostly makes structural errors of this kind (10 of 20 malformed arguments in a probe), so the base model is the default.

## Departures from the spec, and why

- **Straight-through bridge.** The spec formula `onehot + softmax − sg(softmax)` is not bit-exact in float16, because the sum rounds before the subtraction. `deltanet._STProject` gathers `W[:, y_t]` forward and applies the softmax Jacobian to `Wᵀg` backward. That is the same gradient, and the Gate 1 test checks the equivalence.
- **Replacement rule.** The pick is `argmax(−g + 1e-5·z)` over tokens with a constructive gradient (−g > 0) only. Otherwise the `1e-5·z` tie-break can outweigh a tiny gradient and choose a token of the wrong type. A site with no constructive token is exhausted, and routing moves on to the next candidate.
- **Bans accumulate per position.** Smart keeps every class it has banned at a position, as B6 Patch 1 did. The spec bans only the current class, which would let a declaration oscillate U32 → Nat → U32 when nothing is rolled back.
- **Type tokens.** Types are recognised by the tokenizer's real first piece: ` U` for `U32` (which tokenizes as U, 3, 2) and ` Nat`. Prefixes such as ` N` are not recognised; they led to `N32` in the study.
- **Problem 14.** The spec's round-2 story (`6n` → `6`) is what happens if the model regenerates `catalog(6n, 38, 47)` after round 1. In the recorded run it regenerated `catalog(6, 38n, 47)`, so round 2 concerned `level`.
  - Gate 3 replays both trajectories with a scripted LM.
  - On the recorded one, v1 Upstream flips `level` to `Nat` and ends invalid, while Smart edits `38n` → `38`. This matches the study.
  - On the spec's trajectory, v1 Upstream also succeeds, because stock's declaration was already tried, so it edits the latest flagged token instead. The Smart/Upstream split there depends on what the model regenerated.
- **JSON DAG schema.**
  - The currency types are `USD` / `EUR`, not `Currency_USD` / `Currency_EUR`. Those share the first token `Currency`, so one token cannot identify the type.
  - Workflows declare typed `inputs` and bind them in a `call`, which gives literals a declared slot to be checked against (`CHECK_CALL`).
  - `out_type` must come before `args`.
- **Peak memory.** torch 2.15 has no MPS peak counter, so peak memory is the maximum of `torch.mps.current_allocated_memory()`, sampled after every forward pass.
- **KV-cache rewind.** In transformers 5.x, `DynamicCache.crop` takes a negative count of tokens to remove; the backend handles both conventions.
