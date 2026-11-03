# ttso — Time-Travel Structured Outputs

`ttso` repairs LLM structured outputs at inference time. It targets errors that a context-free grammar (GBNF, JSON Schema, Outlines) cannot catch, such as a reference 200 tokens after a declaration that disagrees with its type.

It runs a compiled, zero-parameter Gated DeltaNet verifier over the completion. When a check fails, ttso:

1. backpropagates the first failed check into the LLM's cached logits;
2. routes one edit (Smart routing: the declaration or the call site);
3. rewinds the KV cache to that token;
4. regenerates the suffix, for up to 3 rounds.

This is DiffRepair, the inference-time method from the DiffSpec / DiffRepair study (November 2026), packaged for Apple Silicon. Two backends are included: HuggingFace on `mps`, and llama.cpp with Metal.

**v0.2** makes four changes:

- input types are pinned by default;
- an optional forward structural mask keeps decoding inside the workflow schema;
- the HuggingFace backend can rewind hybrid linear-attention models such as Qwen3.5;
- Gate 4 now checks that the llama.cpp backend reproduces the `mps` backend.

**v0.3** makes three changes:

- the forward mask is incremental: each query reads only the new token's characters, and a rewind restores the parse state by lookup;
- the first end-to-end repair benchmarks run on Qwen3.5-0.8B, a hybrid linear-attention model, for both cartridges;
- a frozen BendLite catalog-synthesis suite is added.

The pre-registered held-out benchmarks for v0.3 and v0.2 are below.

## Install (M4 Pro, macOS)

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

**llama.cpp backend (optional).** On this macOS / Command Line Tools combination, the source build needs the SDK's libc++ headers:

```bash
SDKROOT=$(xcrun --show-sdk-path) CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path) -DCMAKE_CXX_FLAGS=-isystem\ $(xcrun --show-sdk-path)/usr/include/c++/v1" uv pip install --python .venv/bin/python llama-cpp-python
```

**GGUF files.** Convert the HF weights with the converter that ships in llama.cpp. It needs `gguf-py` from the same source tree, plus `sentencepiece`. The tests look for two files:

- `models/qwen3-0.6b-f16.gguf`, from Qwen/Qwen3-0.6B, used by the backend test;
- `models/qwen3-0.6b-base-f16.gguf`, from Qwen/Qwen3-0.6B-Base, used by the Gate 4 parity test.

```bash
python <llama.cpp>/convert_hf_to_gguf.py <hf snapshot dir> --outtype f16 --outfile models/<name>.gguf
```

## Use

```python
from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine.repair_loop import RepairLoop

be = HuggingFaceMPSBackend("Qwen/Qwen3-0.6B-Base")          # float16 on mps
cart = TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=be.tok.all_special_ids,
                             apply_forward_mask=True)       # pin_inputs=True is the default
res = RepairLoop(be, cart, policy="smart", k_max=3).repair(be.encode(prompt))
print(res.verdict.valid, res.rounds, res.extra_tokens, res.history)
```

Cartridge options:

- **`pin_inputs=True`** (default): freezes the input types the prompt fixes. A failed reference is then rewritten as a conversion instead of retyping the input. `pin_inputs=False` gives the original spec routing.
- **`apply_forward_mask=True`**: every decoded token and every repair edit must keep the completion a prefix of the workflow schema. The mask covers key order, one key per argument, op and type names, the five literal forms, fresh and declared ids, and complete call bindings. It never checks types; those stay the verifier's job.
- **`mask_impl="incremental"`** (default): the mask runs on `SchemaState`, which advances over each new token's characters, with one state kept per token position. `"reparse"` is v0.2's acceptor, which re-parses the whole completion on every query. The two accept exactly the same tokens.
- **`BendLiteCatalogCartridge(..., pin_inputs=True)`**: freezes every non-main parameter type. A failed operand is then rewritten in place, usually as `U32.from_nat(...)`. The BendLite default stays `False`, the study's setup.

`LlamaCppBackend("models/qwen3-0.6b-base-f16.gguf")` can be used in place of the HuggingFace backend. `HuggingFaceMPSBackend("Qwen/Qwen3.5-0.8B")` rewinds by restoring checkpoints of the linear-attention state; `ckpt_every`, 16 positions by default, trades memory against replay time.

## Layout

| Module | What it does |
|---|---|
| `core/deltanet.py` | `CompiledDeltaNet`: memory M ∈ R^{S×K}, ops NOOP / DECLARE / CHECK_VAR / CHECK_CALL, straight-through token bridge, `first_failure_gradients` |
| `core/token_classes.py` | `TokenClassManager`: equivalence classes, type-head tokens, `W_decl`, `W_lit`, `G_type` |
| `core/routing.py` | `diagnose`; `Router` with the policies `smart`, `upstream` (v1), `local`, `blind`; `RepairState` (bans, edited sites) |
| `backends/` | `hf_mps.py` (crop, or checkpoint-and-replay for hybrid models), `llamacpp.py` (`kv_cache_seq_rm`), `scripted.py` (deterministic test LM), `base.py` (`pick` with a lazy token filter) |
| `cartridges/` | `bendlite.py` (catalog subset, U32 / Nat, optional `pin_inputs`), `typed_json_dag.py` (typed tool-call DAGs; the forward mask as the incremental `SchemaState` / `SchemaTrail`, with v0.2's re-parsing `SchemaPrefix` kept as the reference), `synthetic.py` (SynthLang for Gates 1–2) |
| `engine/` | `repair_loop.py` (the 3-round loop, Best-of-N), `benchmark.py` (JSON DAG tasks, the paired harness with pluggable suites and an optional shared greedy decode, paired statistics, Wilson intervals, Holm), `catalog_suite.py` (the frozen v0.3 BendLite catalog prompts and outcomes) |
| `scripts/` | `benchmark_v03.py` (v0.3, Qwen3.5-0.8B: `{json,catalog} {dev,heldout}`), `benchmark_v02.py` (v0.2's held-out run), `dev_v02.py`, `recheck_bon_masked_v02.py` |

## Held-out benchmarks (v0.3, Qwen3.5-0.8B, pre-registered)

The protocol is in [`runs/v0.3/PREREGISTRATION.md`](runs/v0.3/PREREGISTRATION.md). It was pushed as `34c79c0` before any held-out task ran on Qwen3.5. All prompt, mask and harness settings were fixed on development tasks only.

- **Model:** Qwen/Qwen3.5-0.8B, revision `2fc06364`, float16 on `mps`.
- **Rewind:** after an edit, the KV cache is restored from a linear-state checkpoint and replay.
- **Decoding:** greedy. Arms that share a mask setting reuse one greedy decode; this was checked to be deterministic.
- **Best-of-N:** temperature 0.7, matched per problem to its repair arm's extra tokens.

### 2A. Typed JSON DAG (60 held-out tasks, the v0.2 prompt unchanged)

| Arm | Valid | Correct | Faithful | Faithful 95% CI (Wilson) |
|---|---|---|---|---|
| greedy | 16 | 10 | 8 | 0.07–0.24 |
| best_of_n | 18 | 10 | 8 | 0.07–0.24 |
| ttso_pinned | 31 | 19 | **14** | 0.14–0.35 |
| greedy_masked | 20 | 11 | 9 | 0.08–0.26 |
| best_of_n_masked | 23 | 11 | 9 | 0.08–0.26 |
| ttso_pinned_masked | 32 | 22 | **15** | 0.16–0.37 |
| ttso_unpinned (secondary) | 46 | 17 | 11 | 0.11–0.30 |

**None of the four primary hypotheses is supported.** Each comparison is on Faithful, exact two-sided McNemar, with Holm correction over the four:

| | Comparison | Wins–losses | p | Holm p |
|---|---|---|---|---|
| H1 | ttso_pinned vs greedy | 6–0 | 0.031 | 0.125 |
| H2 | ttso_pinned vs best_of_n | 6–0 | 0.031 | 0.125 |
| H3 | ttso_pinned_masked vs greedy_masked | 6–0 | 0.031 | 0.125 |
| H4 | ttso_pinned_masked vs best_of_n_masked | 6–0 | 0.031 | 0.125 |

- **Every pair runs the expected way, without a single loss.** But the gain is 6 problems in each pair, too few to survive the correction.
- **Valid and Correct (secondary, uncorrected) moved more.** On Correct, repair beats greedy 9–0 unmasked (p = 0.0039) and 11–0 masked (p = 0.00098), with the same counts against Best-of-N.
- **Correct is not Faithful under pinning.** Pinning freezes the input types the model wrote, not the types the task states. Qwen3.5 sometimes declares an input wrongly (`price: Float` for a Nat), and pinning keeps that mistake. As a result, 5 pinned and 7 pinned + mask programs are correct but not faithful.
- **Qwen3.5 writes this schema worse than Qwen3-0.6B-Base did in v0.2** (8 faithful greedy programs vs 15). Its usual errors are `u32` literal suffixes and an invented `charge` input. Many repairs spend all three rounds without reaching a valid program.
- **The mask alone** gives +4 valid, +1 correct and +1 faithful, none significant. No masked output left the schema.
- **Runtime:**
  - Mean s / problem: greedy 5.6, pinned 11.3, pinned + mask 10.4, unpinned 11.7.
  - Peak MPS allocated: 2.86 GB.
  - Linear-state checkpoints held by one cache: at most 302 MiB.
  - Each rewind replays 8.1–8.6 tokens on average.

### 2B. BendLite catalog synthesis (50 held-out tasks)

**The suite is new.** The study's own 50 catalog prompts are not used:
- 36 of them use `*` and 29 use a Bool flag with `match`, which `BendLiteCatalogCartridge` does not parse. Only 3 fit its grammar.
- So `engine/catalog_suite.py` generates a frozen suite in the study's format: 2–4 typed items summed by one def, which is called from `main`. It keeps the study's names, chat prompt and task check.
- These numbers are therefore not comparable with the study's.

| Arm | Valid | Correct | Faithful | Correct 95% CI (Wilson) |
|---|---|---|---|---|
| greedy | 0 | 0 | 0 | 0.00–0.07 |
| best_of_n (matched to upstream) | 1 | 0 | 0 | 0.00–0.07 |
| local | 28 | 19 | 18 | 0.26–0.52 |
| upstream | 28 | 22 | 0 | 0.31–0.58 |
| smart (`pin_inputs=False`) | 29 | **23** | 0 | 0.33–0.60 |
| smart_pinned (`pin_inputs=True`) | 28 | 19 | **18** | 0.26–0.52 |

**All four primary hypotheses are supported.** Each comparison is on Correct, exact two-sided McNemar, with Holm correction over the four:

| | Comparison | Wins–losses | Holm p |
|---|---|---|---|
| C1 | smart vs best_of_n | 23–0 | 9.5e-7 |
| C2 | smart_pinned vs best_of_n | 19–0 | 7.6e-6 |
| C3 | local vs best_of_n | 19–0 | 7.6e-6 |
| C4 | upstream vs best_of_n | 22–0 | 1.4e-6 |

- **Greedy:** 38 of 50 greedy programs put a Nat parameter straight into a U32 sum. Best-of-N drew 100 samples in all and found 1 valid program.
- **Unpinned repair is never faithful.** Every correct program from `smart` or `upstream` retypes a Nat item as U32, and one `upstream` program also retypes a U32 as Nat. The one correct but unfaithful program in `local` and in `smart_pinned` keeps a wrong type the model itself wrote (a U32 item declared Nat).
- **Pinned Smart and LocalOnly are faithful.** They rewrite the operand as `U32.from_nat(...)` instead. On Faithful, `smart_pinned` beats `smart` 18–0 (p = 7.6e-6). On Correct, `smart` leads `smart_pinned` 9–5 (p = 0.42).
- **Some arms ended identical, as the dev set predicted.** `smart_pinned` and `local` ended identical on 49 of 50 tasks, and `smart` and `upstream` on 48.
- **Runtime:**
  - Mean s / problem: greedy 3.0, repair 7.6–8.4.
  - Peak MPS allocated: 2.22 GB.
  - Checkpoints held by one cache: at most 320 MiB.
  - Each rewind replays 6.6–7.3 tokens on average.
  - The Mac went into low-power sleep for 51 minutes during one `local` repair (task 33); that time is removed from the mean. The pause has no effect on outcomes.

### Task 1. Incremental forward mask

On the 20 seed-0 tasks (Qwen3-0.6B-Base), masked greedy, masked repair and masked Best-of-N are token-for-token identical under the re-parse and incremental acceptors. Run back to back in alternating order:
- mask time falls from 0.034 to 0.0065 s per problem, 5.2× faster;
- latency is 4.89 vs 4.64 s per problem.

The mask was only 0.7% of the re-parse latency, so at these lengths (about 150 tokens) almost all of that latency gap is noise.

The gain grows with document length (CPU replay):

| Document tokens | `accept()` µs / token, re-parse → incremental | `candidates()` ms / token |
|---|---|---|
| 131 | 31 → 6.6 | 3.5 → 0.38 |
| 844 | 166 → 7.4 | 14.4 → 0.09 |
| 3,200 | 514 → 13.7 | 55.6 → 0.10 |

## Held-out benchmark (v0.2, pre-registered)

The protocol is in [`runs/v0.2/PREREGISTRATION.md`](runs/v0.2/PREREGISTRATION.md). It was committed and pushed before any held-out task ran.

The forward mask was developed only on Gate 4's 20 seed-0 tasks. The held-out set is 60 new tasks (seed 42), run on Qwen3-0.6B-Base in float16 on `mps`. Each Best-of-N arm samples at temperature 0.7 with its tokens matched per problem to its repair arm; at that budget it drew 1–3 samples per problem.

| Arm | Valid | Correct | Faithful | Faithful 95% CI (Wilson) |
|---|---|---|---|---|
| greedy | 23 | 15 | 15 | 0.16–0.37 |
| best_of_n | 23 | 15 | 15 | 0.16–0.37 |
| ttso_pinned | 44 | 27 | **26** | 0.32–0.56 |
| greedy_masked | 29 | 15 | 15 | 0.16–0.37 |
| best_of_n_masked | 30 | 15 | 15 | 0.16–0.37 |
| ttso_pinned_masked | 51 | 31 | **30** | 0.38–0.62 |
| ttso_unpinned (secondary) | 50 | 35 | 15 | 0.16–0.37 |

All four primary hypotheses are supported (Faithful, exact two-sided McNemar, Holm over the four):

| | Comparison | Wins–losses | p | Holm p |
|---|---|---|---|---|
| H1 | ttso_pinned vs greedy | 11–0 | 0.00098 | 0.0020 |
| H2 | ttso_pinned vs best_of_n | 11–0 | 0.00098 | 0.0020 |
| H3 | ttso_pinned_masked vs greedy_masked | 15–0 | 0.000061 | 0.00024 |
| H4 | ttso_pinned_masked vs best_of_n_masked | 15–0 | 0.000061 | 0.00024 |

Secondary results, not confirmatory:

- **The mask alone** adds valid programs (29 vs 23, 6–0, p = 0.031) but not one correct program. Every program it rescues computes the wrong value.
- **The mask added to repair:** 30 vs 26 faithful, 4–0 (p = 0.125).
- **Unpinned routing** replicates v0.1's finding at n = 60. It has the most correct programs (35), but only 15 are faithful: 20 of its correct programs changed a type the task fixed. Pinned beats it on Faithful 11–0 (p = 0.00098).
- **Mask integrity:** no masked output left the schema.
- **A post-hoc bug check:** a bug fixed after the freeze could only have affected masked sampling. It could return an illegal token when every legal token's probability underflowed. Replaying the masked Best-of-N arm with the fix reproduced all 31 base and 15 post-trained outputs exactly (`runs/v0.2/*/bon_masked_recheck.json`), so the bug never fired.

**Post-trained Qwen3-0.6B** (20 held-out tasks, exploratory):

| Arm | Valid | Correct | Faithful |
|---|---|---|---|
| greedy / best_of_n | 4 / 4 | 4 / 4 | 4 / 4 |
| ttso_pinned | 4 | 4 | 4 |
| greedy_masked / best_of_n_masked | 5 / 5 | 4 / 4 | 4 / 4 |
| ttso_pinned_masked | 12 | 5 | 5 |

- **Without the mask, repair does nothing.** The first failure on almost every problem is a structural error, usually two keys in one argument, and that comes before any type error.
- **With the mask, repair makes 7 more programs valid** (7–0, p = 0.016), but correct and faithful rise by only one. Those programs compute the wrong value: they drop an input, name the wrong output node, or take the wrong steps.

So the mask removes the structural errors, but what remains are logic errors, which type repair cannot fix.

## Gates

Run them with:

```bash
.venv/bin/python -m pytest
```

There are 28 tests; expect about 17 minutes, most of it Gate 4 and the incremental-mask gate. The first run downloads Qwen3-0.6B, Qwen3-0.6B-Base and Qwen3.5-0.8B from the Hugging Face Hub. The parity and GGUF tests skip when their GGUF file is missing. The results below are from 3 November 2026, on an M4 Pro (24 GB) with torch 2.15, transformers 5.22.1 and llama-cpp-python 0.3.36.

| Gate | Result |
|---|---|
| 1. Exactness and non-fading memory | 200 random programs (27,857 tokens, 3,906 checks) agree with the string-level reference checker on every token, in check verdicts and in the full memory state after each token. This holds in float32 and float16, on cpu and mps. The gradient at a declaration is the same at distances 8 and 256 (ratio 1.000000), and tokens in between receive exactly 0. The custom backward equals autograd through the spec's p_t formula (float64). |
| 2. Single-edit type direction (K = 5) | 200 / 200 declarations fixed by one gradient-directed edit. Blind, the model's best other type token at the same site, fixed 45 / 200. |
| 3. Problem 14 | Smart changes `+stock: Nat` to `U32` in round 1. In round 2 it sees `consistent_uses[stock] = 1` and edits the call site (`6n` → `6`), valid in 2 rounds. On the recorded trajectory, v1 Upstream flips `level` to `Nat` and ends invalid. |
| 4a. JSON DAG on mps | 20 development tasks, Qwen3-0.6B-Base. Faithful: greedy 5, Best-of-N 5, pinned 9, pinned + mask 11, unpinned 5; v0.3's incremental mask reproduces these counts exactly. Mean s / problem: pinned 5.5–7.2, pinned + mask 4.7–6.2, unpinned 6.1–7.8 (five runs, two of them in v0.3). Peak MPS allocated 1.48–1.53 GB across the five runs. |
| 4b. llama.cpp parity | The same pinned regimes through llama.cpp (Metal, f16 GGUF). Valid, correct and faithful counts equal the mps run's. 19 of 20 problems match token for token; problem 4 is logged as a float16 tie. llama.cpp holds 1.62 GB of Metal buffers, 0.30 GB of CPU buffers and a 1.16 GB host logits array (required by `logits_all`). Mean s / problem (pinned / pinned + mask) over four runs: llama.cpp 7.27 / 6.55, 3.84 / 3.27, 4.64 / 4.01 and 5.19 / 4.23, against mps 5.51 / 4.72, 6.08 / 5.25, 5.68 / 4.89 and 7.16 / 6.22. llama.cpp was faster in three of the four runs. Other sessions' GPU jobs shared the machine, so the speed comparison is noisy. |
| Backends | On Qwen3 (mps), Qwen3.5-0.8B (mps, hybrid, two rewind depths) and llama.cpp, rewind and replay reproduce the original continuation exactly. An edited token's continuation matches a fresh decode of the edited prefix on 12 of 12 tokens. Qwen3.5 checkpoints held at most 75 MiB in the test (about 19 MiB each). |
| Cartridge edge cases | Two types sharing a first token are rejected. More than `n_slots` ids is a hard error, never an alias. |
| Incremental mask (v0.3) | `SchemaState` agrees with v0.2's re-parse acceptor on all 62,743 prefixes of 150 fuzzed documents, in status and next-character set. `accept()` and `candidates()` agree on 11,093 vocabulary-level queries made out of order, so the per-position trail is rewound and re-extended. On the 20 seed-0 tasks, masked greedy, masked repair and masked Best-of-N are token-for-token identical under both acceptors. Mask time per problem (greedy + repair) fell 0.063 → 0.007 s in this run and 0.035 → 0.010 s in the freeze run. |
| Catalog suite (v0.3) | The dev and held-out task texts (both suites) match their pinned SHA-256 hashes. Every reference program is valid, correct and faithful. The outcomes separate a retyped program (correct, not faithful) from a literal in the def (valid, not correct). `pin_inputs=True` freezes exactly the parameter types, and a failed `from_nat` argument rewrites the whole operand. |

On the parity gate: as first written, it required token-for-token identity and failed on problem 4. At the first differing token, the top two logits ("convert" vs "ref") differ by 0.016 in the mps run and 0.021 in llama.cpp, about one float16 step at that size. HuggingFace's own incremental decode and a fresh prefill disagree there too. The gate now requires equal counts and identical trajectories, except where the greedy completions diverge at a measured tie (top-two margin < 0.05 in both backends), and it logs each tie.

## What the v0.1 Gate 4 run showed

With every declaration editable, the gradient of a failed reference check reaches only the declaration, so Smart routing retypes the input. In v0.1, all 6 extra correct programs changed an input type the task had fixed, for example `boxes: Nat` became `U32`. `pin_inputs=True` freezes those declarations and falls back to LocalOnly at the argument's key token: banning `"ref"` there lets the model write `{"convert": "to_u32(boxes)"}`. v0.2 makes that the default and confirms the effect on held-out tasks.

## Departures from the spec, and why

- **Straight-through bridge.** The spec formula `onehot + softmax − sg(softmax)` is not bit-exact in float16, because the sum rounds before the subtraction. `deltanet._STProject` gathers `W[:, y_t]` in the forward pass and applies the softmax Jacobian to `Wᵀg` in the backward pass. The gradient is the same, and Gate 1 tests the equivalence.
- **Replacement rule.** The pick is `argmax(−g + 1e-5·z)` over tokens with a constructive gradient (−g > 0) only.
  - Otherwise the `1e-5·z` tie-break can outweigh a tiny gradient and choose a token of the wrong type.
  - A site with no constructive token is exhausted, and routing moves on to the next candidate site.
- **Bans accumulate per position.** The spec bans only the current class. Without rollback, that would let a declaration oscillate U32 → Nat → U32.
- **Type tokens.** Types are recognised by the tokenizer's real first piece: ` U` for `U32` (which tokenizes as U, 3, 2) and ` Nat`. Prefixes such as ` N` are not recognised; in the study they led to `N32`.
- **Problem 14.** The spec's round-2 story (`6n` → `6`) assumes the model regenerates `catalog(6n, 38, 47)`. In the recorded run it regenerated `catalog(6, 38n, 47)`. Gate 3 replays both trajectories, and v1 Upstream fails only on the recorded one.
- **JSON DAG schema.**
  - The currency types are `USD` and `EUR`, because `Currency_USD` and `Currency_EUR` share their first token.
  - Workflows declare typed `inputs` and bind them in a `call`.
  - `out_type` precedes `args`.
- **Forward-mask literals.** The v0.2 brief asked to restrict literals to `"<digits>"` or `"<digits>n"`. The mask allows all five literal forms the schema defines, because the USD tasks need `"7usd"`. A `"47u32"` therefore becomes the nearest legal form, `"47usd"`: a type error that repair then fixes.
- **Peak memory.** On `mps`, peak memory is the highest `torch.mps.current_allocated_memory()` read after each forward pass, since torch 2.15 has no MPS peak counter. llama.cpp's figure is the Metal buffer total it logs at load.
- **KV-cache rewind.** In transformers 5.x, `DynamicCache.crop` takes a negative count of tokens to remove. Hybrid linear-attention layers cannot be cropped at all, because their recurrent state is overwritten in place every step; hence the checkpoints.
- **v0.3 catalog prompts.** The v0.3 brief asked to use the study's 50 catalog prompts if available. They are available, but 47 of 50 use `*` or Bool / `match`, which the BendLite cartridge does not parse. Following the brief's fallback, `engine/catalog_suite.py` generates a frozen suite in the study's format.
- **Pinning freezes what the model wrote.** `pin_inputs=True` freezes the model's own input declarations, not the types the prompt states. A model that declares an input wrongly therefore keeps the mistake. On Qwen3.5 this left 5–7 correct but unfaithful JSON programs.
- **Shared greedy decode (v0.3 benchmarks).** Repair arms with the same mask setting start from one greedy decode and a copy of its KV cache instead of re-decoding. On Qwen3.5 this was checked against re-decoding (`runs/v0.3/dev/determinism.json`).

## License

MIT, see [LICENSE](LICENSE).
