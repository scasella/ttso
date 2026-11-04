# ttso v0.4 pre-registration: audit fixes and Gemma 4 E2B held-out benchmarks

Written and committed before any held-out (seed 42) task was run on Gemma 4 E2B or Gemma 4 E2B-it.
Every prompt, mask, routing and harness setting below was developed and debugged only on the
development tasks (20 seed-0 JSON DAG tasks and 20 seed-0 catalog tasks), and on the saved v0.2 /
v0.3 results (the audit, which runs no model). The code is frozen at the commit that adds this file.
Run with `scripts/benchmark_v04.py {json,catalog} heldout --model <model>`.

v0.4 answers an internal audit of v0.1–v0.3 that raised five issues. Each is addressed below
before the held-out plan.

## 1. Functional scoring on 5 input vectors

**Issue.** Correct and Faithful judged a program on the one input vector the prompt states, so a
program implementing the wrong reusable formula could pass by coincidence.

**Change.** Every program is now also run on 5 deterministic input vectors, defined in
`benchmark.vectors` and `catalog_suite.vectors`:
- the prompt's own vector, substituted into the program's call (not trusted from the model's call);
- 4 varied vectors:
  - JSON: magnitudes 2–9, 10–99, 100–999 and 1000–9999. Catalog: Nat 2–9, 10–40, 41–99 and
    100–250; U32 10–99, 100–999, 1000–9999 and 2–9.
  - Values are distinct within a vector, at least 2, and never an input's stated value, so there
    are no 0 / 1 identities.
  - Each task's vectors come from an RNG seeded with a hash of its text.
  - Coverage is built in: for the max task each input is the largest in some vector, and for
    min(a + b, c), c falls below a + b in two vectors and above it in two.

New metrics:
- **Correct_5** is Correct_1 (v0.2's Correct) and the task's formula value on all 5 vectors.
- **Faithful_5** is Correct_5 with every input's stated name and type kept.
- **Faithful_5 is the primary metric from v0.4.** Correct_1 and Faithful_1 are still recorded, for
  comparison with earlier versions.

Tests:
- `tests/test_functional_scoring.py`: the canonical program of all 80 JSON tasks scores 5/5; the
  vectors cover both branches; wrong formulas that match the stated vector fail Faithful_5.
- `tests/test_catalog_suite.py`: all 70 reference programs score 5/5, and a stated-catalog
  coincidence is caught.

**Retroactive audit** (`scripts/audit_v04.py` → `runs/v0.4/audit/`). The saved v0.2 and v0.3
held-out completions, re-scored. Recomputing Correct_1 and Faithful_1 from the saved text gives the
stored values on every row of every run (0 mismatches).

| Run | Arm | Faithful_1 | Faithful_5 | Correct_1 | Correct_5 |
| --- | --- | --- | --- | --- | --- |
| v0.2 Qwen3-0.6B-Base (n = 60) | `greedy` | 15 | 15 | 15 | 15 |
| | `ttso_pinned` | 26 | 22 | 27 | 23 |
| | `ttso_pinned_masked` | 30 | 23 | 31 | 24 |
| | `ttso_unpinned` | 15 | 15 | 35 | 35 |
| v0.2 Qwen3-0.6B (n = 20) | `ttso_pinned_masked` | 5 | 4 | 5 | 4 |
| v0.3 Qwen3.5-0.8B JSON (n = 60) | `ttso_pinned` | 14 | 14 | 19 | 15 |
| | `ttso_pinned_masked` | 15 | 15 | 22 | 18 |
| | `ttso_unpinned` | 11 | 11 | 17 | 14 |
| v0.3 Qwen3.5-0.8B catalog (n = 50) | every arm | unchanged | | unchanged | |

What the audit shows:
- **v0.2 `ttso_pinned_masked`:** 7 of its 30 Faithful programs are coincidences. Most compute
  min(a + b + c, c), which equals c whenever c is the smaller value. The audit's "7 of 30"
  is confirmed.
- **v0.2 H1–H4 on Faithful_5:**
  - H1 and H2 are 7–0 (p = 0.016) and H3 and H4 are 8–0 (p = 0.0078); Holm p = 0.031 for each.
  - So all four **remain supported**, at smaller margins than on Faithful_1 (11–0 and 15–0, Holm p
    ≤ 0.002).
- **v0.3:** H1–H4 stay 6–0, Holm p = 0.125 (not supported).
- **v0.3 catalog:** C1–C4 are unchanged on Correct_5, and its Faithful counts are unchanged.

## 2. Contract pinning (`TypedJSONDAGCartridge`, `pin_inputs=True`)

**Issue.** `pin_inputs=True` froze whatever input types the model wrote, wrong ones included.

**Change.** With a task bound (`bind_task`), pinned mode now uses the task's stated input contract.
- **Unmasked:** at each named input's type token, the lexer adds `CHECK_VAR(input, stated type)`
  ("input contract") right after the input's `DECLARE`.
  - A wrong written type is the first failed check. Its declaration is flagged (Case B), and the
    gradient edit rewrites it to the stated type.
  - A type that matches is frozen, as before.
- **Masked:** the wrong type is never decoded. The type of a named input admits only the stated
  type, up to its closing quote.
- **Stopping oracle:** `check()` is still the plain reference checker, so Valid means the same in
  every arm and version. The repair loop and Best-of-N stop on `check().valid and contract_ok()`.
- `pin_inputs="self"` reproduces v0.2 / v0.3. Gate 4 uses it, so the v0.2 gate still tests the
  v0.2 arms.
- `BendLiteCatalogCartridge` keeps self-pinning: the audit item names the JSON cartridge.

**Verification on seed 0.**
- **Natural cases:** Gemma 4 E2B never mis-declared an input in greedy or pinned decoding on the
  dev tasks, so there was no natural case to observe.
- **Forced cases** (`tests/test_contract_pinning.py`, Qwen3-0.6B-Base, the 5 seed-0 tasks with a
  Nat input):
  - Each is given a valid, self-consistent program that declares the Nat input as U32.
  - Self-pinning leaves all 5 unrepaired: valid, not faithful.
  - Unmasked contract repair catches all 5. Round 1 is the input-contract check (Case B), which
    rewrites `U` to `Nat`, and the input stays Nat through every later round.
  - Masked contract decoding admits only the stated type at a named input's type slot, and only
    the rest of the name after `U` for U32 (never `SD`).

## 3. Symbolic provenance baseline (`core/symbolic.py`)

**Question.** Does `CompiledDeltaNet` backward routing ever decide differently from a plain
provenance table under the same routing rules?

**`SymbolicProvenanceVerifier`** is pure Python, with no tensors or autograd.
- It reads the same lexer ops (`DECLARE`, `CHECK_VAR`, `CHECK_CALL`).
- For each slot it tracks (declaration token, declared type, consistent uses).
- For the first failed check it returns the provenance Router needs: the declaration read, a
  failed call check's own literal, and which replacement types would satisfy the check.
- `Router.choose()` applies the same Smart (Cases A, B, C), Upstream, LocalOnly and Blind rules to
  either verifier's diagnosis. A symbolic "gradient" edit picks the most likely grammar-valid,
  non-banned token of the needed type.

**Gate** (`tests/test_symbolic_equivalence.py`): diagnoses and edits compared case by case.

| Cases | Diagnoses | Edits compared | Differ | Diagnosis time, DeltaNet vs symbolic |
| --- | --- | --- | --- | --- |
| Gate 1: 200 random SynthLang programs (K = 8), LLM-like random logits, Smart / Upstream / LocalOnly / Blind | 400 | 784 | 0 | 0.37 vs 0.012 ms |
| Gate 2: 200 single-faulty-declaration programs (K = 5) | 400 | 800 | 0 | 0.14 vs 0.005 ms |
| Gate 3: Problem 14's scripted repairs (spec and recorded trajectories; Smart, Upstream, LocalOnly) | 12 rounds | 12 | 0 | |
| Dev: 20 seed-0 JSON tasks, Qwen3-0.6B-Base, full repair runs in 6 regimes (contract-pinned Smart unmasked and masked, self-pinned Smart, unpinned Smart, LocalOnly, Upstream) | 209 rounds | 209 | 0 | 0.8–3.5 vs 0.02 ms |

Gate 1 also checks every check's verdict, consistent-use count and the declaration it read: all
identical. On these cases the DeltaNet's backward routing makes exactly the decisions of the
provenance table, at about 30–150 times the cpu cost of a diagnosis.

The reading the brief anticipated holds here: `CompiledDeltaNet` is an exact algebraic compilation
of symbolic provenance into a recurrent network. Its memory row for a slot is the one-hot type of
the slot's latest declaration, a check's loss is 0 or 1, and its gradient reaches exactly the
declaration it read and the call's own literal, constructive exactly on tokens of the needed type.
For discrete inference-time repair, the provenance table is sufficient. The DeltaNet's added value
is that it is differentiable, for settings that train through it; none of the v0.1–v0.4 results
depends on that.

The two could differ only where float32 rounds (a token probability that underflows to 0, or a type
class whose probability rounds to 1); no such case occurred. The held-out runs run the symbolic
verifier as a shadow at every round and report agreement.

## 4. Resampling accounting

**Issue.** Token-matched Best-of-N's budget is the repair's extra tokens. In v0.3's catalog run it
drew 100 samples across the 50 tasks, and the audit finds that on 18 tasks (the audit item said
17) its only sample was cut off by the budget before it finished.

Other runs, by the same rule (a task's single invalid sample is incomplete):
- v0.2 base: 21 of 37 sampled tasks (`best_of_n`) and 16 of 31 (`best_of_n_masked`).
- v0.3 JSON: 14 of 44 and 12 of 40.

**Change.**
- Every Best-of arm records each sample as complete (EOS or the cartridge's stop), truncated (cut
  off by the token budget) or capped (cut off at the max new tokens).
- Summaries report complete and truncated samples per sampled task, and the number of tasks with
  no complete sample.
- **New secondary arm `best_of_2_full`:** on each task whose greedy completion fails the regime's
  oracle, 2 full samples (no token budget), stopping at the first valid one.
  - It uses the same seed as `best_of_n`, so its first sample is `best_of_n`'s first sample, not
    cut off. This separates the token efficiency of repairing from a reused prefix from that of
    whole-completion resampling.
  - JSON: `best_of_2_full` (`ttso_pinned`'s cartridge) and `best_of_2_full_masked`. Catalog:
    `best_of_2_full` (`upstream`'s cartridge).

## 5. Gemma 4 E2B port

**Text-only load.** The checkpoint is a `Gemma4ForConditionalGeneration` (9.54 GiB, bf16).
`HuggingFaceMPSBackend` loads only the text decoder, as `Gemma4ForCausalLM`, mapping
`model.language_model.*` to `model.*`.
- The vision tower (0.31 GiB), the audio tower (0.57 GiB) and their embedders are never read.
- The text decoder has 4.63B parameters. 2.35B of them (4.38 GiB) are the per-layer embedding (PLE)
  table, a lookup of one 35 × 256 row per token.
- The PLE table stays on the cpu as a read-only memory map of the checkpoint (`map_ple`). A step
  reads only its tokens' rows, and the pages are file-backed, so macOS can drop them under memory
  pressure instead of swapping them.
- On the GPU: 4.25 GiB of weights, against 8.62 GiB without the map. On 3 dev tasks, greedy decodes
  with and without the map give identical tokens and bit-identical logits.

**Memory is logged two ways:**
- peak PyTorch MPS tensor allocation (`torch.mps.current_allocated_memory()`, sampled at every
  decode step), which excludes the mapped table;
- peak process RSS (`ru_maxrss`), which counts cpu memory including the mapped table's touched
  pages.

**dtype: bf16,** the checkpoint's own. fp16 gave identical greedy outputs on 4 dev tasks, with no
NaN or inf (Gemma soft-caps its logits at 30). v0.2 and v0.3 used fp16 for Qwen.

**BOS.** Prompts start with `<bos>` (`prompt_ids`), unless a rendered chat template already starts
with it. Qwen's tokenizers have no BOS, so their prompts are unchanged.

**Sliding-window rewind.** Gemma 4 E2B has 28 sliding-window layers (window 512) and 7 global
layers. Layers 15–34 reuse the K/V of layers 13 and 14, so the cache holds 15 layers.
- HuggingFace's `crop` cannot rewind a sliding layer once its sequence passes the window: it raises
  unless recording mode is on, and even then keeps only the last 511 positions.
- Every benchmark prompt passes the window: the JSON prompt is 678 tokens.
- The backend puts the sliding layers in HuggingFace's recording mode when it builds a cache. They
  keep their full history (12 KiB per token) and still hand attention the same window, so decoding
  is unchanged, and a rewind slices them back exactly.
- **Departure from the brief,** which proposed checkpoints every 16 positions plus replay.
  Recording mode needs neither, and it is exact, so it was kept.

**Rewind gate** (`tests/test_rewind_gemma4.py`):
- **The brief's gate:** on the 678-token prompt, rewinding at continuation tokens 8 and 37.
- **Extra cases:** the same two depths after a short prompt, and a 560-token decode rewound back
  inside the window.
- **Result:** in every case, re-feeding the same token reproduces the tokens and logits bit for bit
  (`torch.equal`), and an edited token matches a fresh decode of the whole edited prefix on 12 of
  12 tokens.
- **Decoding unchanged:** recording mode gives the same tokens and logits as HuggingFace's own
  cache past the window.

**Tokenizer.** `TokenClassManager` builds on the Gemma 4 tokenizer for both cartridges, which
asserts no first-token collision between type names. The heads are U32 → {`U`, `U32`}, and Nat,
Float, USD and EUR → themselves. The base and `-it` tokenizers encode both suites' prompts to
identical ids.

## Tests at the freeze

- **On the frozen code, just before this commit:** the 37 fast tests (everything but the long model
  gates) passed.
- **Earlier today, on the same code paths:**
  - the Gemma rewind gate, 9 of 9;
  - the dev-task symbolic shadow gate, 209 of 209 rounds;
  - the forced contract cases, 5 of 5;
  - Gate 4's HuggingFace run executed as that gate's fixture, but Gate 4's assertions and the
    incremental-mask gate were not evaluated; they run in the full suite.
- **Changes since those runs** are confined to:
  - the catalog prompt for base models;
  - Best-of-N sample accounting and `best_of_k`;
  - summaries and logging.
- **The full 47-test suite** runs after the held-out runs, as the brief asks.

## Held-out plan

**Common to all runs:**
- Greedy decoding, at most 3 repair rounds.
- Best-of-N at T = 0.7, seeded per task.
- Shared greedy decode, guarded: regimes share it only when their decoding constraints are the same.
- The symbolic verifier runs as a shadow of the DeltaNet at every repair round. The loop follows
  the DeltaNet, and agreement is logged.
- Transformers 5.22.1, torch 2.15.0, `mps` on an Apple M4 Pro with 24 GB.

| Model | Revision | dtype |
| --- | --- | --- |
| google/gemma-4-E2B | `433813a669639fd7eaa83155828e9000ed357d53` | bf16 |
| google/gemma-4-E2B-it | `4adf15748de9e1a0097b6274cee9c15836c41c37` | bf16 |

### 3A: TypedJSONDAG on Gemma 4 E2B (n = 60)

- **Tasks:** `benchmark.make_tasks(60, seed=42)`, SHA-256
  `0ef5a0d84e941461d61fb2cf0dae2b2c707ca5ff141be30849811ce5479b7e3c`. These are the v0.2 / v0.3
  held-out tasks, never run on Gemma.
- **Prompt:** v0.2's raw few-shot prompt after `<bos>`. At most 320 new tokens.
- **Arms:**
  - `greedy`, `best_of_n` (matched to `ttso_pinned`) and `ttso_pinned` (contract);
  - `greedy_masked` (which decodes the contract's types), `best_of_n_masked` and
    `ttso_pinned_masked` (contract, masked);
  - `ttso_unpinned`;
  - secondary: `ttso_self_pinned` (v0.3's pinning), `best_of_2_full` and `best_of_2_full_masked`.

**Primary (Faithful_5), Holm over four at α = 0.05; repair expected higher in each:**

| | Comparison |
| --- | --- |
| H1 | `ttso_pinned` vs `greedy` |
| H2 | `ttso_pinned` vs `best_of_n` |
| H3 | `ttso_pinned_masked` vs `greedy_masked` |
| H4 | `ttso_pinned_masked` vs `best_of_n_masked` |

Each test is an exact two-sided McNemar test. A hypothesis is supported only if its Holm p < 0.05
and the difference goes in the expected direction.

**Expectation from dev** (stated so the result can be read against it): the unmasked pair (H1, H2)
is likely null on Gemma, the masked pair likely positive. See the dev results.

**Secondary (not corrected):**
- the H1–H4 pairs on Faithful_1, Correct_5 and Valid;
- contract vs self pinning (`ttso_pinned` vs `ttso_self_pinned`);
- the mask alone; mask + repair vs repair; pinned vs unpinned;
- `best_of_n` vs `greedy`;
- the repair arms vs `best_of_2_full[_masked]`, and `best_of_2_full` vs `best_of_n`;
- Faithful_1-only coincidences per arm; Best-of sample accounting; DeltaNet / symbolic agreement.

### 3A exploratory: Gemma 4 E2B-it (first 20 held-out tasks)

- The same raw prompt, as for post-trained Qwen3-0.6B in v0.2.
- Arms: `greedy`, `best_of_n`, `ttso_pinned`, `greedy_masked`, `best_of_n_masked`,
  `ttso_pinned_masked`, `best_of_2_full` and `best_of_2_full_masked`.
- H1–H4 are computed and reported as exploratory, with no confirmatory claim.

### 3B: BendLite catalog on Gemma 4 E2B (n = 50)

- **Tasks:** `catalog_suite.heldout_tasks()`, the frozen v0.3 suite. SHA-256
  `603a59085d75fffc88dce77144a599eb09a30179565540e7fa44a4c963a26bfb`.
- **Prompt:** the v0.3 prompt's content (the study's system text with its worked example, the task
  and the instruction) as a plain document ending in `Program:` (`catalog_suite.raw_prompt`),
  after `<bos>`. At most 256 new tokens.
  - The base checkpoint ships no chat template. On the dev tasks, under the `-it` checkpoint's chat
    template, Gemma base repeated the system prompt on all 20 tasks, so every arm had 0 valid
    programs (`runs/v0.4/dev/old_design/gemma-4-E2B_catalog_chat_template/`).
  - The plain document is the base-model analogue of the JSON suite's raw few-shot prompt. It was
    the first and only variant tried after the chat template failed, and was not tuned.
  - v0.3's Qwen3.5-0.8B, a chat model, used its chat template, so 3B is not a like-for-like
    comparison with v0.3's prompt.
- **Arms:** `greedy`, `best_of_n` (matched to `upstream`), `local`, `upstream`, `smart`
  (`pin_inputs=False`) and `smart_pinned` (`pin_inputs=True`, parameter types as written);
  secondary `best_of_2_full`.

**Primary (Faithful_5), Holm over four at α = 0.05, vs `best_of_n`:**

| | Comparison | Expected |
| --- | --- | --- |
| C1 | `smart` vs `best_of_n` | **not supported**: unpinned repair retypes Nat parameters as U32, so it is never faithful (0 / 50 in v0.3) |
| C2 | `smart_pinned` vs `best_of_n` | repair higher |
| C3 | `local` vs `best_of_n` | repair higher |
| C4 | `upstream` vs `best_of_n` | **not supported**, as for C1 |

**C5, one test at α = 0.05:** `smart_pinned` vs `smart` on Faithful_5, pinned expected higher.

**Secondary:**
- C1–C4 on Correct_5 (v0.3's primary family, with functional scoring), Holm;
- the same pairs on Faithful_1, Correct_1 and Valid;
- `smart` vs `upstream`, `smart` vs `local`, and `best_of_n` vs `greedy`;
- the repair arms vs `best_of_2_full`, and `best_of_2_full` vs `best_of_n`.

## Logged for every run

- Valid, Correct_1, Correct_5, Faithful_1 and Faithful_5, with Wilson 95% intervals.
- Extra tokens.
- Mean latency in s / problem.
- Peak MPS tensor allocation and peak RSS.
- Weights on the device, and the memory-mapped GiB.
- Rewinds, and replayed tokens (0 on Gemma).
- Mask time.
- Best-of sample kinds.
- Shadow agreement, with each verifier's diagnosis time.
- Each task's wall-clock window.

## Memory, pauses and run order

- Another session's GPU job (about 12 GB) shares the machine, so latencies are noisy.
- A memory guard pauses this session's ttso processes only at critical memory pressure or under 8%
  free memory. Pauses are logged in `runs/v0.4/memory_guard.log`, and system sleeps appear in
  `pmset -g log`.
- **Latency rule:** a task whose wall window overlaps a logged pause or a system sleep is left out
  of the latency means; the number left out is reported. Rows record each task's wall window, not
  each arm's, so the overlap cannot be charged to one arm. Outcomes are unaffected.
- Each held-out run runs under `caffeinate -i`, which prevents idle sleep. It cannot prevent a
  low-battery sleep, so pmset's log is checked afterwards.
- **Run order,** one model resident at a time:
  1. 3A base;
  2. 3B base;
  3. 3A `-it`.

## Development results (dev tasks only; not part of the test)

From the frozen code, `scripts/benchmark_v04.py {json,catalog} dev`, recorded in `runs/v0.4/dev/`.
The JSON runs predate the last change to the catalog prompt, which touches no JSON code path. (An earlier dev pass on a superseded design, with a separate prompt-pinned arm
and single-vector scoring, is kept in `runs/v0.4/dev/old_design/`.)

**JSON DAG, Gemma 4 E2B base (n = 20):**

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 | s / problem |
| --- | --- | --- | --- | --- | --- | --- |
| `greedy` | 10 | 6 | 6 | 6 | 6 | 5.8 |
| `greedy_masked` | 10 | 6 | 6 | 6 | 6 | 5.6 |
| `best_of_n`, `best_of_n_masked` | 10 | 6 | 6 | 6 | 6 | |
| `best_of_2_full`, `best_of_2_full_masked` | 10 | 6 | 6 | 6 | 6 | |
| `ttso_pinned` | 10 | 6 | 6 | 6 | 6 | 9.3 |
| `ttso_pinned_masked` | 15 | 11 | 11 | 11 | 11 | 8.9 |
| `ttso_unpinned` | 10 | 6 | 6 | 6 | 6 | 9.4 |
| `ttso_self_pinned` | 10 | 6 | 6 | 6 | 6 | 9.4 |

- **Unmasked repair gains nothing on Gemma base; masked repair gains 5.** H1 and H2 are 0–0 on
  dev; H3 and H4 are 5–0 (p = 0.0625).
- **Why.** Gemma base writes U32 literals with a `u32` suffix (`"34u32"`), a malformed literal
  and a hard error with no gradient.
  - Unmasked, the Case C fallback bans the literal's first digit, so the model changes the number
    instead of dropping the suffix, and the rounds are spent.
  - The forward mask never admits the `u32` suffix. The error becomes a typed literal mismatch,
    which gradient routing fixes (`u` → `",`).
  - All 5 masked gains are the Nat + U32 + U32 sum tasks.
  - The USD tasks stay valid but not correct (an invented `charge` input), and the min tasks fail
    in every arm.
- **Contract vs self pinning:** identical. Gemma never declared an input with the wrong type on the
  dev tasks.
- **Coincidences:** none (Faithful_1 = Faithful_5 in every arm).
- **Best-of:**
  - `best_of_n` drew 15 samples on 10 tasks, and 10 of them were cut off by the budget.
  - On 5 tasks it drew no complete sample.
  - `best_of_2_full` completed all 20 of its samples and still gained nothing.
- **Shadow:** 116 of 116 repair rounds identical. Diagnosis takes 1.8–4.5 ms (DeltaNet) against
  0.02–0.03 ms (symbolic).
- **Memory:** peak MPS tensor allocation 4.62 GB, peak RSS 2.15 GB.

**Expected on held-out,** from these: H1 and H2 likely null, and H3 and H4 positive if the held-out
tasks behave like dev. 5–0 on dev would need about 8–0 on 60 tasks to pass Holm over four.

**JSON DAG, Gemma 4 E2B-it (n = 20):**

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 | s / problem |
| --- | --- | --- | --- | --- | --- | --- |
| `greedy` | 7 | 5 | 5 | 5 | 5 | 5.9 |
| `greedy_masked` | 10 | 5 | 5 | 5 | 5 | 5.5 |
| `best_of_n` / `best_of_2_full` | 7 | 5 | 5 | 5 | 5 | |
| `best_of_n_masked` / `best_of_2_full_masked` | 10 | 5 | 5 | 5 | 5 | |
| `ttso_pinned` | 18 | 12 | 10 | 12 | 10 | 8.3 |
| `ttso_pinned_masked` | 20 | 14 | 11 | 14 | 11 | 6.9 |

- **The post-trained model gains from repair with and without the mask** (5–0 and 6–0 on
  Faithful_5), unlike post-trained Qwen3-0.6B in v0.2.
- It does not write the `u32` suffix, so its errors are typed.
- 2 pinned and 3 pinned + mask programs are Faithful_1-only coincidences, caught by the 5 vectors.
- Best-of-N drew no complete sample on 11 of the 13 tasks it sampled (10 of 10 masked).
  `best_of_2_full` completed all its samples and gained nothing.
- **Shadow:** 30 of 30 rounds identical.

**BendLite catalog, Gemma 4 E2B base, plain-document prompt (n = 20):**

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 | s / problem |
| --- | --- | --- | --- | --- | --- | --- |
| `greedy` | 1 | 1 | 1 | 1 | 1 | 2.6 |
| `best_of_n` | 2 | 2 | 2 | 2 | 2 | |
| `best_of_2_full` | 3 | 3 | 3 | 2 | 2 | |
| `local` | 5 | 3 | 3 | 3 | 3 | 7.8 |
| `upstream` | 7 | 6 | 6 | 1 | 1 | 7.4 |
| `smart` | 7 | 6 | 6 | 1 | 1 | 7.5 |
| `smart_pinned` | 7 | 5 | 5 | 5 | 5 | 7.2 |

- **Faithful_5, the primary outcome:**
  - C2 (`smart_pinned` vs `best_of_n`) is 4–1, C3 (`local`) 2–1, and C1 and C4 0–1. None is
    significant at n = 20.
  - C5 (`smart_pinned` vs `smart`) is 4–0.
  - On Correct_5, `smart` and `upstream` lead Best-of-N 5–1.
- **Gemma base's catalog errors are mostly structural,** unlike Qwen3.5's in v0.3 (38 of 50 were a
  Nat parameter in a U32 sum):
  - named call arguments (`catalog(base: 7n, ...)`) or a stray `def` inside `main`, both "bad
    operand" (6 of 19);
  - items summed directly in `main` with no def, so the names are undeclared (7);
  - a Nat item in a U32 sum, the type error repair targets (4);
  - a literal type error (1) and a malformed def line (1).
  Undeclared names and parse errors are hard errors with no gradient; the Case C fallback rarely
  fixes them.
- **Best-of:** `best_of_n` drew 47 samples on 19 tasks, 18 of them cut off by the budget, with no
  complete sample on 8 tasks. `best_of_2_full` completed all 38 of its samples.
- **Shadow:** 207 of 207 rounds identical. Diagnosis takes 0.4–1.2 ms (DeltaNet) against
  0.015–0.018 ms (symbolic).
- **Memory:** peak MPS 4.59 GB, peak RSS 2.55 GB.

**Expected on held-out,** from these:
- C2 and C3 positive but small, and possibly not supported after Holm.
- C1 and C4 not supported, as stated above.
- C5 positive.

**Shared-greedy determinism** (`scripts/determinism_v04.py`, 4 dev tasks per suite, Gemma 4 E2B):
- two greedy decodes give identical tokens and bit-identical logits;
- repair from a copied cache matches repair from a second decode and from a fresh one, on 8 of 8
  (`runs/v0.4/dev/gemma-4-E2B/determinism.json`).

