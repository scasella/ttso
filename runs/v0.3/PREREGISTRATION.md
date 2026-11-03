# ttso v0.3 pre-registration: Qwen3.5-0.8B held-out benchmarks

Written and committed before any held-out task was run on Qwen3.5-0.8B. Every prompt, mask, routing
and harness setting below was developed and debugged only on the development tasks (20 seed-0 JSON
DAG tasks and 20 seed-0 catalog tasks). The code is frozen at the commit that adds this file. Run
with `scripts/benchmark_v03.py {json,catalog} heldout`.

## Questions

1. **2A, typed JSON DAG.** Do v0.2's four hypotheses hold on a hybrid linear-attention model? That
   is, does pinned repair produce more faithful programs than greedy and Best-of-N at a matched
   budget, with and without the forward structural mask?
2. **2B, BendLite catalog synthesis.** With rewind now exact on Qwen3.5-0.8B, does gradient-routed
   repair (Smart, LocalOnly, Upstream) produce more correct programs than Best-of-N matched to
   Upstream's budget?

## Model, backend and decoding (both parts)

- **Model:** Qwen/Qwen3.5-0.8B, hub snapshot `2fc06364715b967f1860aea9cf38778875588b17`, loaded
  from the local snapshot path (`TTSO_HYBRID_MODEL`).
  - It has 18 Gated DeltaNet layers and 6 full-attention layers.
  - It runs through HuggingFace on `mps` in float16, ttso's default. bf16, which the study used,
    gave the same greedy output on the one development task probed. It was not adopted.
  - Transformers 5.22.1 and torch 2.15.0. HuggingFace falls back to its reference PyTorch kernels
    for the linear layers, since `flash-linear-attention` and `causal_conv1d` are not installed.
- **Rewind:** the hybrid KV rewind from v0.2.
  - It checkpoints the linear-layer state at the end of the prompt and every 16 positions.
  - A rewind restores the latest checkpoint at or before the target, crops the attention layers
    and replays the remaining tokens one at a time.
- **Decoding:** greedy, at most 3 repair rounds. Best-of-N samples at temperature 0.7, seeded per
  task.
- **Shared greedy decode (new in v0.3, a runtime saving).**
  - Regimes with the same forward-mask setting reuse one greedy decode: its tokens, its logits and
    a copy of its KV cache, including the hybrid checkpoints.
  - In v0.2 every regime re-decoded greedy, and the decodes were identical on all 80 tasks
    (`same_greedy`).
  - `runs/v0.3/dev/determinism.json` checks the same on Qwen3.5: a greedy decode repeated on the
    same prompt gives the same tokens, and repair from a copied cache matches repair from a fresh
    decode.
  - A regime that reuses the decode is charged that decode's time.

## Part 2A: TypedJSONDAG (n = 60)

- **Held-out tasks:** `benchmark.make_tasks(60, seed=42)`. These are the v0.2 held-out tasks; they
  have never been run on Qwen3.5. SHA-256 of the task texts:
  `0ef5a0d84e941461d61fb2cf0dae2b2c707ca5ff141be30849811ce5479b7e3c` (`tests/test_catalog_suite.py`).
- **Prompt:** the v0.2 raw few-shot prompt, unchanged, with no chat template. At most 320 new tokens.
- **Arms:** the v0.2 arms, unchanged.
  - `greedy`, `best_of_n` (matched to `ttso_pinned`) and `ttso_pinned`.
  - The masked arms: `greedy_masked`, `best_of_n_masked` (matched to `ttso_pinned_masked`) and
    `ttso_pinned_masked`.
  - `ttso_unpinned` (secondary).
  - The mask now uses the incremental acceptor. It is equivalent to v0.2's mask (Task 1 gate,
    below).
- **Outcomes:** Valid, Correct and Faithful, as in v0.2.

| | Comparison (Faithful) | Direction expected |
| --- | --- | --- |
| H1 | `ttso_pinned` vs `greedy` | ttso higher |
| H2 | `ttso_pinned` vs `best_of_n` | ttso higher |
| H3 | `ttso_pinned_masked` vs `greedy_masked` | ttso higher |
| H4 | `ttso_pinned_masked` vs `best_of_n_masked` | ttso higher |

Each comparison uses an exact two-sided McNemar test on the paired outcomes, with Holm correction
across H1–H4 at α = 0.05. A hypothesis is supported only if its Holm-adjusted p < 0.05 and the
difference goes in the expected direction.

**Secondary analyses (not corrected, not confirmatory):**
- Valid and Correct for the H1–H4 pairs.
- The mask alone: `greedy_masked` vs `greedy`.
- `ttso_pinned_masked` vs `ttso_pinned`.
- Pinned vs unpinned routing.
- `best_of_n` vs `greedy`.

## Part 2B: BendLite catalog synthesis (n = 50)

### Which prompts

The study's own 50 held-out catalog prompts are available locally, in the diffspec repository: the
first 50 of `track2_split(200, seed=777)`. They are not used, because they fall outside the grammar
`BendLiteCatalogCartridge` parses (U32 and Nat, `+` groups, `U32.from_nat`, one call from `main`):
- 36 of the 50 reference programs use `*`.
- 29 use a Bool flag with `match`.
- Only 3 (indices 2, 14 and 22) fit the grammar.

Running them would measure that grammar gap, not repair. This is the brief's fallback.

**The suite used:** `ttso/engine/catalog_suite.py`, a frozen seeded generator in the study's format.
- Each task has 2–4 catalog items, with at least one U32 value (1–60) and one Nat count (1–12). It
  asks for their sum, and 30% of tasks count one U32 item twice.
- It keeps the study's item names and function names.
- The prompt uses the study's chat format, with thinking off: its system prompt, without the Bool,
  `match` and `*` rules and with the example rewritten in the subset, and its user instruction.
- **Held-out:** `heldout_tasks()`, 50 tasks, seed 42, disjoint from the dev tasks. SHA-256:
  `603a59085d75fffc88dce77144a599eb09a30179565540e7fa44a4c963a26bfb`.
- **Dev:** `dev_tasks()`, 20 tasks, seed 0.
- The v0.1 Gate 3 replay uses the study's Problem 14, one of the 3 originals that fit. It is not
  in this suite.

Because the tasks differ, this part's numbers are not comparable with the study's catalog results.

### Arms

All arms see the same prompt, with at most 256 new tokens and 3 rounds. Every arm except
`best_of_n` repairs from the shared greedy completion.

| Arm | Repair |
| --- | --- |
| `greedy` | none |
| `best_of_n` | resampling at T = 0.7; stops at the first valid sample or once its tokens reach `upstream`'s extra tokens on that task (the study's B1) |
| `local` | LocalOnly: memory reads detached; only the failing check's own token can be edited |
| `upstream` | v1 Upstream: the earliest flagged token, then the latest once that declaration was tried |
| `smart` | Smart routing, `pin_inputs=False`: every declaration, parameter types included, is editable (the study's signature-editing setup) |
| `smart_pinned` | Smart routing, `pin_inputs=True`: each parameter type is frozen as written; a failed group operand or `from_nat` argument falls back to a local rewrite of that operand |

`local` and `upstream` use `pin_inputs=False`, as in the study.

### Outcomes

- **Valid:** the cartridge's reference checker accepts the program. This is the repair loop's
  oracle.
- **Correct (primary):** valid, the expected value, and the study's task check (`interface_ok`):
  - every catalog item is a parameter of a non-main def and is used in a non-main body;
  - non-main bodies hold no number literals;
  - `main` passes every catalog value.
- **Faithful:** correct, and every catalog item's parameter has the type the catalog states.

### Primary hypotheses (Correct, n = 50)

| | Comparison | Direction expected |
| --- | --- | --- |
| C1 | `smart` vs `best_of_n` | repair higher |
| C2 | `smart_pinned` vs `best_of_n` | repair higher |
| C3 | `local` vs `best_of_n` | repair higher |
| C4 | `upstream` vs `best_of_n` | repair higher |

Each comparison uses an exact two-sided McNemar test, with Holm correction across C1–C4 at α = 0.05.
The support rule is the same as in 2A.

**Secondary analyses (not corrected, not confirmatory):**
- Valid and Faithful for the C1–C4 pairs.
- `smart` vs `upstream`, `smart` vs `local` and `smart_pinned` vs `smart`, on all three outcomes.
- `best_of_n` vs `greedy`.

## Logged for both parts

- Valid, Correct and Faithful, each with a Wilson 95% interval.
- Extra tokens.
- Mean latency (s/problem, greedy and repair).
- Peak MPS allocated memory.
- Peak checkpoint memory held by one KV cache (MiB).
- Mean replayed tokens per rewind.
- Forward-mask evaluation time.

Other sessions' GPU jobs share this machine (a `delta-cartridge` benchmark on `mps`, among others),
so latencies are noisy and are not comparable with v0.2.

## Task 1 gate: the incremental forward mask (not part of the held-out test)

`SchemaState` replaces the full re-parse in v0.2's `SchemaPrefix`. It is an explicit machine that
follows `SchemaPrefix` instruction by instruction, advances over a new token's characters only and
caches its next-character set. `SchemaTrail` keeps one state per token position, so a rewind to t*
reads `states[t*]`. `tests/test_incremental_schema.py` checks it in three ways:
- a differential fuzz;
- `accept()` / `candidates()` agreement over the Qwen3 vocabulary, with out-of-order queries;
- the model gate: masked greedy, masked ttso and masked Best-of-N on the 20 seed-0 tasks,
  token-for-token against the re-parse acceptor, with both acceptors' mask time and latency logged.

## Development results (dev tasks only; not part of the test)

These come from the frozen code, recorded in `runs/v0.3/dev/{json,catalog}/` and produced by
`scripts/benchmark_v03.py {json,catalog} dev`.

**2A dev, JSON DAG on Qwen3.5-0.8B (n = 20):**

| Arm | Valid | Correct | Faithful |
| --- | --- | --- | --- |
| `greedy` | 3 | 2 | 2 |
| `best_of_n` | 3 | 2 | 2 |
| `ttso_pinned` | 8 | 6 | 6 |
| `greedy_masked` | 5 | 2 | 2 |
| `best_of_n_masked` | 5 | 2 | 2 |
| `ttso_pinned_masked` | 9 | 5 | 5 |
| `ttso_unpinned` | 16 | 6 | 4 |

- On dev, every H1–H4 pair runs in the expected direction, with no losses: 4–0, 4–0, 3–0 and 3–0.
  None is significant at n = 20.
- Qwen3.5 is a weaker writer of this schema than Qwen3-0.6B-Base, which scored 5 faithful greedy
  on the same dev tasks in v0.2.
- Its usual greedy errors are `u32` literal suffixes and an extra `charge` input in the USD tasks.
- Many pinned repairs use all 3 rounds (about 270 extra tokens) without reaching a valid program.
- The prompt was kept identical to v0.2's so the two models can be compared. It was not tuned for
  Qwen3.5.

**2B dev, BendLite catalog on Qwen3.5-0.8B (n = 20):**

| Arm | Valid | Correct | Faithful |
| --- | --- | --- | --- |
| `greedy` | 0 | 0 | 0 |
| `best_of_n` | 0 | 0 | 0 |
| `local` | 12 | 9 | 9 |
| `upstream` | 12 | 11 | 0 |
| `smart` | 13 | 12 | 0 |
| `smart_pinned` | 12 | 9 | 9 |

- **Greedy:** 18 of the 20 greedy programs put a Nat parameter straight into a U32 sum, without
  `U32.from_nat`.
- **Best-of-N:** it drew 1–4 samples per task, and none was valid.
- **Unpinned arms (`smart`, `upstream`):** they fix the sum by retyping the Nat parameters to U32.
  That makes them correct but never faithful.
- **`smart_pinned` and `local`:** they made identical edits and ended with identical programs on
  all 20 dev tasks. Both rewrite the failing operand in place (usually as `U32.from_nat(...)`).
  - Pinned Smart reaches that rewrite because every flagged declaration is frozen.
  - LocalOnly reaches it because its memory reads are detached.
  - C2 and C3 are therefore expected to be nearly identical on the held-out tasks.
- **`smart` and `upstream`:** they differ on 1 of the 20 dev tasks.
- **Plumbing:** none of the dev failures is a harness or parser bug, so no change was made after
  the dev runs. The failures are argument order or arity in `main`, a bare `days + rate` without
  the annotation group, and invented def lines.

**Runtime:**
- Peak MPS allocated: 2.14 GB (catalog), 2.90 GB (JSON).
- Peak checkpoint memory per KV cache: 151 MiB (catalog), 377 MiB (JSON).
- Mean replayed tokens per rewind: about 6–9.
- Mean latency per problem: greedy 2.4 s and repair 5.2–6.0 s (catalog); repair 9.3–9.8 s (JSON).

**Shared-greedy determinism (`runs/v0.3/dev/determinism.json`):** this was checked on the first 4
dev tasks of each suite, on Qwen3.5. On all 8, a second greedy decode gave the same tokens.
Repair from a copied cache matched repair from the original decode, token for token, and so did
repair from a fresh decode.

**Task 1 gate (`runs/v0.3/dev/incremental_gate.log`, Qwen3-0.6B-Base, the 20 seed-0 tasks):**

- **Equivalence:** masked greedy, masked ttso repair and masked Best-of-N are token-for-token
  identical under the re-parse and incremental acceptors on 20 of 20 tasks.
- **Gate 4:** the incremental run reproduces v0.2's counts exactly (faithful: greedy 5, Best-of-N
  5, pinned 9, pinned + mask 11, unpinned 5).

| Mask evaluation | Re-parse | Incremental | Speedup |
| --- | --- | --- | --- |
| Masked greedy, s / problem | 0.014 | 0.003 | 5.1× |
| Masked greedy + repair, s / problem | 0.035 | 0.010 | 3.5× |
| Masked Best-of-N, s / problem | 0.007 | 0.002 | 4.3× |
| CPU replay, `accept()` per token | 55 µs | 6 µs | 9.3× |
| CPU replay, `candidates()` per token | 2.84 ms | 0.23 ms | 12.4× |

The CPU replay runs over the 20 repaired completions (2,939 tokens).

- **Per-problem latency:** no speedup was measured. Masked greedy + repair took 4.74 s with the
  re-parse acceptor against 5.32 s with the incremental one. The incremental run came first,
  about 10 minutes earlier, while other GPU jobs shared the machine.
- At these completion lengths (about 150 tokens), mask evaluation is under 1% of a problem's time,
  so the incremental acceptor cannot change end-to-end latency measurably. Its advantage is in the
  per-query cost: O(token length) instead of O(completion length).



## Analysis plan, fixed now

- Results are reported whatever their direction. Nulls are reported as nulls.
- No task, prompt, mask rule, routing rule or decoding setting changes after the held-out runs
  start. If a run fails for a technical reason, the fix and the rerun are recorded in this file and
  in the v0.3 report.
- Development-set numbers are reported separately from held-out numbers and are never pooled.
