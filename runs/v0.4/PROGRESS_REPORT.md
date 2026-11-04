# ttso v0.4: Progress Report

4 November 2026 · github.com/scasella/ttso · pre-registration `310a2a6`, pushed before any held-out
run · results in the commit after it

## Bottom line

- **The audit's five issues are each addressed, and two of its claims are confirmed exactly.**
  - In v0.2, 7 of the 30 `ttso_pinned_masked` Faithful programs are single-vector coincidences.
  - In v0.3's catalog run, Best-of-N drew 100 samples across the 50 tasks. The audit item said that
    on 17 tasks the budget ran out before any sample finished; it was 18.
- **v0.2 survives re-scoring.** On Faithful_5, H1–H4 are 7–0, 7–0, 8–0 and 8–0, Holm p = 0.031 for
  each, so all four remain supported, at smaller margins than on Faithful_1.
  - v0.3's results are unchanged on Faithful.
- **DeltaNet routing equals symbolic provenance.** Across Gates 1–3, the 20 dev tasks and all 789
  held-out repair rounds, the two never made a different diagnosis or edit.
  - The DeltaNet is an exact compilation of the provenance table.
  - The table costs about 1/20–1/160 as much CPU per diagnosis.
- **Gemma 4 E2B, 3A (typed JSON DAG, n = 60):**
  - **Masked repair is supported:** 33 vs 17 Faithful_5, 16–0 against both masked baselines, Holm
    p = 0.00012.
  - **Unmasked repair is not:** 0–0. Gemma base's errors are a malformed `u32` literal suffix,
    which only the mask turns into something repairable. Dev predicted both.
- **Gemma 4 E2B, 3B (BendLite catalog, n = 50), Faithful_5:**
  - pinned Smart beats Best-of-N 9–0 (Holm p = 0.016) and LocalOnly 7–0 (Holm p = 0.047), so both
    are supported;
  - unpinned Smart and Upstream are 0–0, not supported, as pre-registered;
  - pinned Smart beats unpinned Smart 9–0 (C5, p = 0.0039).
- **Gemma 4 E2B-it (exploratory, 20 tasks):**
  - Repair looks strong on Faithful_1: 7–0 and 10–0, Holm p ≤ 0.031.
  - On Faithful_5 it is only 4–0 (Holm p = 0.5): 3 and 6 of its "faithful" programs are
    coincidences.
- **Contract pinning works but was never exercised by Gemma.** Gemma never declared an input with
  the wrong type, so contract pinning and v0.3's self-pinning scored identically on held-out. The
  mechanism is verified on forced cases.

## Brief vs delivered

| Brief item | Status | Evidence |
| --- | --- | --- |
| 1a. 5-vector functional scoring; Faithful_5 primary; Faithful_1 kept | Done | `engine/benchmark.py`, `engine/catalog_suite.py`, `tests/test_functional_scoring.py` |
| 1a. Retroactive audit of v0.2 / v0.3, side by side, McNemar + Holm on Faithful_5 | Done (0 recompute mismatches) | `scripts/audit_v04.py`, `runs/v0.4/audit/AUDIT.md` |
| 1b. Pinned mode uses the task's contract; caught + repaired unmasked, prevented masked, checked on seed 0 | Done, on forced cases (Gemma never mis-declared) | `cartridges/typed_json_dag.py`, `tests/test_contract_pinning.py` |
| 2. `SymbolicProvenanceVerifier`; 100%-identical gate over Gates 1–3 and the 20 dev tasks; CPU cost | Done: identical everywhere | `core/symbolic.py`, `tests/test_symbolic_equivalence.py` |
| 2. Log complete / truncated samples for token-matched Best-of-N; `best_of_k_full` (K = 2) | Done; run on dev and held-out | `engine/repair_loop.py`, summaries' `samples` |
| 3. Gemma 4 E2B / -it on `mps`, text backbone only | Done; the PLE table is also memory-mapped | `backends/hf_mps.py` |
| 3. Log peak MPS tensor allocation and peak RSS | Done | every v0.4 `summary.json` |
| 3. Rewind gate, 660+ token prompt, depths 8 and 37, exact replay, 12/12 | Done (678 tokens), plus three extra cases | `tests/test_rewind_gemma4.py` |
| 3. Freeze and push before seed 42; 60 JSON + 50 catalog + 20 -it | Done | `310a2a6`; `runs/v0.4/` |
| 4. README corrections, review, full pytest, push, report | Done. The review found 9 issues: 8 fixed after the runs (none changes an outcome), 1 skipped. Full suite 47 / 47 in 19.4 min | `runs/v0.4/final_suite.log`, this file |

## 1. Functional scoring and the audit

- **Vectors.** Each task is scored on 5 vectors: the stated one (substituted into the program's
  call) and 4 varied ones.
  - Magnitudes are 2–9, 10–99, 100–999 and 1000–9999; catalog Nat counts are 2–250.
  - Values are distinct, at least 2, and never an input's stated value.
  - Each task's vectors come from an RNG seeded with a hash of its text.
  - Coverage is built in: every input is the maximum somewhere, and both sides of min(a + b, c)
    occur.
- **Metrics.** Correct_5 is Correct_1 and the formula's value on all 5 vectors. Faithful_5 also
  keeps the stated input names and types.

**Re-scored runs.** Recomputing from the saved text reproduces every stored value first.

| Run | Arm | Faithful_1 → Faithful_5 | Correct_1 → Correct_5 |
| --- | --- | --- | --- |
| v0.2 Qwen3-0.6B-Base, n = 60 | `ttso_pinned` | 26 → 22 | 27 → 23 |
| | `ttso_pinned_masked` | 30 → 23 | 31 → 24 |
| | greedy, Best-of-N, `ttso_unpinned` | unchanged | unchanged |
| v0.2 Qwen3-0.6B, n = 20 | `ttso_pinned_masked` | 5 → 4 | 5 → 4 |
| v0.3 Qwen3.5-0.8B JSON, n = 60 | all arms | unchanged | e.g. `ttso_pinned_masked` 22 → 18 |
| v0.3 Qwen3.5-0.8B catalog, n = 50 | all arms | unchanged | unchanged |

- **What the coincidences compute.** Most are min(a + b + c, c), which equals c whenever c is the
  smaller value. They come only from repair arms: a repaired program is pushed to validity, not
  to the right formula.
- **v0.2 on Faithful_5.** H1–H4 remain supported: 7–0, 7–0, 8–0 and 8–0, Holm p = 0.031 for each.
  On Faithful_1 they were 11–0 and 15–0, Holm p ≤ 0.002.
- **v0.3 on Faithful_5.** H1–H4 are 6–0 each, Holm p = 0.125, so still not supported. C1–C4 are
  unchanged on Correct_5.

## 2. Contract pinning

`pin_inputs=True` now reads the task's stated inputs (`bind_task`).
- **Unmasked:** a CHECK_VAR(input, stated type) at each named input's type token makes a wrong type
  the first failed check. Case B then rewrites it.
- **Masked:** the wrong type cannot be decoded.
- **The repair loop and Best-of-N** stop on `check().valid and contract_ok()`. `check()` and Valid
  are unchanged.
- **Legacy and scope:** `pin_inputs="self"` reproduces v0.2 / v0.3, and Gate 4 uses it. The
  BendLite cartridge keeps self-pinning, since the brief scoped this item to the JSON cartridge.

**Seed-0 verification:**
- **Forced cases** (Qwen3-0.6B-Base, the 5 Nat-input dev tasks, the Nat input written as U32 in an
  otherwise consistent, valid program):
  - self-pinning leaves all 5 unrepaired;
  - contract repair catches all 5 in round 1, rewriting `U` → `Nat` (Case B), and keeps the type
    through later rounds;
  - whether the regenerated rest is right is up to the model: 2 of 5 ended faithful.
- **Masked:** a named input's type admits only the stated type, through its closing quote.
- **On Gemma:** neither the base model nor the -it model ever mis-declared an input, on dev or
  held-out. So `ttso_pinned` and `ttso_self_pinned` are identical on all 60 held-out tasks. The
  v0.3 Qwen3.5 failure this item targets did not occur on Gemma.

## 3. Symbolic provenance baseline

`SymbolicProvenanceVerifier` is pure Python.
- It reads the lexer's ops, keeps (declaration token, declared type, consistent uses) per slot, and
  for the first failed check returns the declaration it read, a failed call's own literal, and the
  type a replacement must have.
- `Router.choose()` applies the same Smart / Upstream / LocalOnly / Blind rules. Its "gradient" edit
  picks the most likely grammar-valid, non-banned token of the needed type.

| Cases | Compared | Differ | Diagnosis, DeltaNet vs symbolic |
| --- | --- | --- | --- |
| Gate 1 (200 programs; also every check's verdict, uses and declaration) | 400 diagnoses, 784 edits | 0 | 0.37 vs 0.012 ms |
| Gate 2 (200 faulty-declaration programs) | 400 diagnoses, 800 edits | 0 | 0.14 vs 0.005 ms |
| Gate 3 (Problem 14, scripted) | 12 rounds | 0 | |
| Dev, 20 tasks × 6 regimes (Qwen3-0.6B-Base) | 209 rounds | 0 | 0.8–3.5 vs 0.02 ms |
| Held-out shadow, Gemma (3A, 3B, 3A-it) | 789 rounds | 0 | 0.3–4.4 vs 0.013–0.031 ms |

**Why they agree.** The DeltaNet's memory row is the one-hot type of a slot's latest declaration,
and a check's loss is 0 or 1. At a declaration, the gradient of the failed check is
−s_v(1[type(v) = k] − S_k). It is constructive exactly on type-k tokens, ranked by −g_v +
1e-5·z_v, which grows with z_v: the most likely type-k token. The two can differ only where float32
rounds a probability to exactly 0 or 1, and that never happened.

**Plainly:** for discrete inference-time repair, symbolic provenance is sufficient. The DeltaNet is
correct, and differentiable for settings that would train through it, but nothing in v0.1–v0.4
needed the gradient.

## 4. Resampling accounting

| Run | Best-of-N samples | Cut off by the budget | Tasks with no complete sample | `best_of_2_full`: samples completed, gain |
| --- | --- | --- | --- | --- |
| 3A base (unmasked / masked) | 44 / 45 on 30 tasks | 30 / 30 | 16 / 16 | 60 / 60 completed, +0 / +0 |
| 3B base | 80 on 42 tasks | 41 | 25 | 82 of 83 completed, +1 |
| 3A -it (unmasked / masked) | 24 / 10 | 14 / 10 | 8 / 10 | 28 / 20 completed, +0 / +0 |

- **Token-matched Best-of-N mostly measures how far a cut-off sample gets.** Its last sample is cut
  off on essentially every task it samples.
- **Giving it whole samples did not change the picture.** `best_of_2_full` spent more tokens than
  repair (3A: 8,715 vs 5,113 unmasked) and gained 0 programs on JSON and 1 on the catalog.
- On these tasks, whole-completion resampling at T = 0.7 rarely finds a fix the greedy completion
  missed.

## 5. Gemma 4 E2B port

- **Load.** The text decoder only: the vision and audio towers (0.89 GiB) are never read.
  - The 2.35B-parameter per-layer embedding table (4.38 GiB) is memory-mapped on the cpu.
  - 4.25 GiB of weights sit on the GPU.
  - Mapped and unmapped decodes are bit-identical (3 dev tasks).
  - bf16, which matches fp16 on 4 dev tasks.
- **Memory, every held-out run:**
  - peak PyTorch MPS tensor allocation 4.59–4.62 GB;
  - peak process RSS 2.7–2.8 GB, which includes the mapped table's touched pages;
  - for comparison, the unmapped backend allocated 8.6 GiB on the GPU, and its process footprint
    was 11 GB.
- **Rewind.** The 28 sliding-window layers keep only 511 positions past the window, so HuggingFace's
  `crop` cannot rewind them.
  - The backend keeps them in recording mode (12 KiB per token) and slices them exactly.
  - Gate: on a 678-token prompt, at continuation tokens 8 and 37, re-feeding reproduces tokens and
    logits bit for bit, and an edited token matches a fresh full-prefix decode on 12 of 12 tokens.
    Three extra cases pass too.
  - Every held-out rewind replayed 0 tokens.
- **Tokenizer.** No first-token collisions between type names for either cartridge, on base or -it.
  Prompts start with exactly one `<bos>`.

## Held-out results

**3A, typed JSON DAG, Gemma 4 E2B base (n = 60), Faithful_5 primary:**

| Arm | Valid | Faithful_5 | s / problem |
| --- | --- | --- | --- |
| greedy / masked | 30 / 30 | 17 / 17 | 5.8 / 5.7 |
| best_of_n / masked; best_of_2_full / masked | 30 | 17 | |
| ttso_pinned (contract) / masked | 30 / **46** | 17 / **33** | 9.0 / 8.8 |
| ttso_unpinned / ttso_self_pinned | 30 / 30 | 17 / 17 | 9.3 / 8.9 |

- **H1 and H2:** 0–0, not supported. **H3 and H4:** 16–0, Holm p = 0.00012, supported.
- **The mechanism.** All 30 invalid greedy programs write `"Nu32"`, a hard error.
  - Unmasked, the fallback edits the number instead of the suffix.
  - The mask forbids the suffix, so the error becomes a typed mismatch the gradient edit fixes.
- **Masked gains:** all 15 Nat + U32 + U32 tasks, and 1 min task.
- **Out of reach for every arm:** the USD tasks (an invented `charge` input) and the min tasks.

**3A exploratory, Gemma 4 E2B-it (first 20):**
- Faithful_5 is greedy 5, pinned 9, pinned + mask 9: 4–0 against both baselines, Holm p = 0.5.
- On Faithful_1 the same comparisons are 7–0 and 10–0 (Holm p ≤ 0.031).
- The gap is 3 + 6 coincidental programs.
- Unlike post-trained Qwen3-0.6B in v0.2, unmasked repair does help here.

**3B, BendLite catalog, Gemma 4 E2B base (n = 50):**

| Arm | Valid | Correct_5 | Faithful_5 |
| --- | --- | --- | --- |
| greedy | 8 | 4 | 4 |
| best_of_n | 8 | 4 | 4 |
| best_of_2_full | 10 | 5 | 5 |
| local | 21 | 11 | 11 |
| upstream | 24 | 17 | 4 |
| smart | 24 | 17 | 4 |
| smart_pinned | 23 | 13 | 13 |

Tests on Faithful_5:
- **C1** (smart) and **C4** (upstream), 0–0: not supported, as pre-registered. Unpinned repair
  retypes Nat items as U32.
- **C2** (smart_pinned), 9–0, Holm p = 0.016: supported.
- **C3** (local), 7–0, Holm p = 0.047: supported.
- **C5** (smart_pinned vs smart), 9–0, p = 0.0039: supported.

On Correct_5 all four C comparisons are supported: 13–0, 9–0, 7–0, 13–0.

Gemma base's catalog errors are mostly structural, which repair can rarely fix:
- 21 parse errors (named call arguments, a stray `def` in `main`);
- 8 undeclared names;
- 10 Nat-in-U32 type errors.

## Departures and disclosures

- **Rewind method.** The brief proposed checkpoints every 16 positions plus replay for the sliding
  layers. Recording-mode slicing is exact, needs no replay, and passed the same gate.
- **Catalog prompt for the base model.** The base model has no chat template.
  - Under Gemma's (borrowed from -it) it echoed the system prompt on all 20 dev tasks, with 0 valid
    programs in every arm.
  - The same content as a plain document was the first and only variant tried. Dev only.
  - 3B therefore does not use v0.3's chat prompt.
- **Pre-registered expectations.** Dev predicted both H3/H4 (masked) and the null H1/H2. They were
  written into the pre-registration before the runs.
- **Latency.** The symbolic shadow runs at every repair round, so v0.4 latencies include it. No task
  overlapped a memory-guard pause or a sleep; runs used `caffeinate -i`.
- **Code changes after the freeze.** Review fixes, none of which changes a recorded outcome
  (listed in the pre-registration's post-run record).
- **Report filename.** The brief asked for "ttso v0.4: Progress Report.md". A colon breaks Windows
  checkouts, so the file is `PROGRESS_REPORT.md`, as in v0.3.

## Limitations

- **Contract pinning is untested by natural errors on Gemma.** Its held-out effect is zero
  because the situation never arose.
- **Contract pinning is JSON-only.** The BendLite cartridge still pins parameter types as written.
- **The JSON suite's remaining failures are not type errors** (an invented input, the min tasks), so
  type repair cannot reach them, on any model.
- **3B on Gemma base is limited by structural errors.** They have no gradient; extending the
  cartridge's grammar or the fallback is future work.
- **Small samples.** The -it run is 20 tasks and exploratory.

## Reproduce

```bash
python scripts/audit_v04.py
```

```bash
python scripts/benchmark_v04.py json heldout --model gemma-4-E2B
```

```bash
python scripts/benchmark_v04.py catalog heldout --model gemma-4-E2B
```

```bash
python scripts/benchmark_v04.py json heldout --model gemma-4-E2B-it --n 20
```

```bash
python -m pytest
```
