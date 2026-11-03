# ttso v0.3: Progress Report

3 November 2026 · github.com/scasella/ttso · pre-registration `34c79c0` (pushed before any
held-out run) · results in the commit after it

## Bottom line

- **Task 1 (incremental mask): done.** `SchemaState` accepts exactly what v0.2's re-parse acceptor
  accepts. Masked outputs are identical, token for token, on 20 of 20 tasks.
  - Mask evaluation is 5.2× faster in a real masked decode, and 38–570× faster per query on
    3,200-token documents.
  - End-to-end latency is unchanged: the mask was under 1% of it at these lengths.
- **Task 2A (JSON DAG on Qwen3.5-0.8B, n = 60): null under the pre-registered test.**
  - Pinned repair beat greedy and Best-of-N 6–0 on Faithful in all four comparisons, with and
    without the mask. The exact p is 0.031 each, but the Holm-adjusted p is 0.125, so H1–H4 are
    **not supported**.
  - The larger gains are on Valid and Correct, which are secondary outcomes.
- **Task 2B (BendLite catalog on Qwen3.5-0.8B, n = 50): supported.**
  - Smart 23, Upstream 22, LocalOnly 19 and pinned Smart 19 correct, against 0 for Best-of-N and
    0 for greedy. All four Holm p ≤ 7.6e-6.
  - The suite is a frozen stand-in. The study's own prompts exceed the cartridge's grammar.
- **Task 3: done.** Code review (5 findings: 1 test fix, 3 documentation fixes, 1 skipped); full suite 28 / 28 passed (16.5 min); pushed.

## Brief vs delivered

| Brief item | Status | Evidence |
| --- | --- | --- |
| `SchemaState.advance(token) -> Optional[SchemaState]`, copyable, cached next-allowed set | Done. States are immutable; `next_chars()` is cached | `cartridges/typed_json_dag.py` |
| Filtering at step t starts from `state[t-1]`; a rewind restores `state[t*-1]` in O(1) | Done via `SchemaTrail`: one state per position, so a rewind is a lookup. Finding the shared prefix is a C-level list compare | same file |
| 100% token equivalence on the 20 seed-0 tasks, masked greedy and masked ttso | 20 / 20, masked Best-of-N included. Also a differential fuzz (62,743 prefixes) and a vocabulary-level `accept` / `candidates` check | `tests/test_incremental_schema.py` |
| Report the mask-time and latency speedups | Mask time 0.034 → 0.0065 s / problem (5.2×). Latency 4.89 → 4.64 s, which is noise | `runs/v0.3/gate/mask_timing.json` |
| 2A: freeze on seed 0; pre-register; 60 held-out; H1–H4 with McNemar + Holm; full metrics | Done. H1–H4 not supported | `runs/v0.3/json/` |
| 2B: 50-prompt BendLite benchmark with Greedy, Best-of-N (matched to Upstream), LocalOnly, Upstream, Smart (unpinned and pinned) | Done on a frozen suite, with the reason stated. C1–C4 supported | `runs/v0.3/catalog/` |
| Push the pre-registration before the held-out runs | `34c79c0`, pushed before either run | GitHub history |
| Review, full pytest, push results, report | Done | this file |

## Task 1: incremental forward mask

- **Design.** `SchemaPrefix`'s recursive descent is flattened into instructions (whitespace,
  literal text, name / word up to a stop character, literal, separators, branch). `SchemaState`
  runs those instructions one character at a time and keeps the scope in tuples (inputs, nodes,
  bound call inputs).
- **Legacy quirks it reproduces.** Each of these is covered by the fuzz:
  - whitespace is only `" \n"` mid-document, while the final strip accepts any whitespace;
  - prefix checks against allowed names run after every character;
  - the "taken name" check runs only at the closing quote;
  - `str.isdigit` accepts Unicode digits.
- **Kept for comparison.** `mask_impl="reparse"` keeps v0.2's acceptor as the reference.
- **Timing.** Qwen3-0.6B-Base, the 20 seed-0 tasks, both acceptors run back to back in
  alternating order:
  - masked greedy + repair spends 0.034 s / problem in the mask with re-parse, against 0.0065 s
    incremental;
  - per-problem latency is 4.89 vs 4.64 s, and the incremental run was faster on 14 of 20 tasks;
  - the mask accounts for only 0.03 s of that difference, so the rest is machine noise.
- **Scaling (CPU replay of a masked decode):**

  | Tokens | `accept()` µs / token | `candidates()` ms / token |
  | --- | --- | --- |
  | 131 | 31 → 6.6 | 3.5 → 0.38 |
  | 844 | 166 → 7.4 | 14.4 → 0.09 |
  | 3,200 | 514 → 13.7 | 55.6 → 0.10 |

## Task 2A: typed JSON DAG on Qwen3.5-0.8B (60 held-out tasks)

The held-out tasks are the v0.2 seed-42 tasks, which had never been run on Qwen3.5. The prompt is
v0.2's, unchanged.

| Arm | Valid | Correct | Faithful (Wilson 95%) |
| --- | --- | --- | --- |
| greedy | 16 | 10 | 8 (0.07–0.24) |
| best_of_n | 18 | 10 | 8 (0.07–0.24) |
| ttso_pinned | 31 | 19 | 14 (0.14–0.35) |
| greedy_masked | 20 | 11 | 9 (0.08–0.26) |
| best_of_n_masked | 23 | 11 | 9 (0.08–0.26) |
| ttso_pinned_masked | 32 | 22 | 15 (0.16–0.37) |
| ttso_unpinned (secondary) | 46 | 17 | 11 (0.11–0.30) |

**Primary (Faithful).** H1–H4 are each 6–0, exact p = 0.031, Holm p = 0.125. None is supported.

**Why the Faithful gain is small.**
- **Weaker writer.** Qwen3.5 writes this schema worse than Qwen3-0.6B-Base: 8 faithful greedy
  programs here, against 15 in v0.2. Many pinned repairs use all three rounds (about 270 tokens)
  without reaching a valid program.
- **Pinning keeps the model's own input types.** It freezes the types the model wrote, not the
  types the task states. In 5 pinned and 7 pinned + mask programs, the model's own wrong input
  type (for example `price: Float` for a Nat) survives into a correct but unfaithful program.

**Secondary, uncorrected.**
- On Correct, repair beats greedy 9–0 (p = 0.0039) and 11–0 masked (p = 0.00098), with the same
  counts against Best-of-N. On Valid it wins 15–0 and 12–0.
- The mask alone gives +4 valid, +1 correct and +1 faithful, none significant.
- Unpinned routing makes the most valid programs (46) but the fewest faithful among the repair
  arms.

**Runtime.**
- Mean s / problem: greedy 5.6, pinned 11.3, pinned + mask 10.4, unpinned 11.7.
- Peak MPS: 2.86 GB.
- Checkpoint memory per cache: at most 302 MiB.
- Rewinds replay 8.1–8.6 tokens on average (102–114 rewinds per arm).

## Task 2B: BendLite catalog synthesis on Qwen3.5-0.8B (50 held-out tasks)

**Prompts.** The study's 50 held-out catalog prompts exist locally (the first 50 of
`track2_split(200, seed=777)` in diffspec), but they do not fit the cartridge:
- 36 use `*` and 29 use a Bool flag with `match`; `BendLiteCatalogCartridge` parses neither;
- only 3 fit its grammar.

So, as the brief allows, `engine/catalog_suite.py` generates a frozen suite in the study's format:
2–4 typed U32 / Nat items summed by one def, which is called from `main`. It keeps the study's
item and function names, its chat prompt (thinking off) and its task check (`interface_ok`). The
held-out set is 50 tasks (seed 42), the dev set 20 (seed 0), and SHA-256 hashes are pinned in a
test.

| Arm | Valid | Correct (Wilson 95%) | Faithful |
| --- | --- | --- | --- |
| greedy | 0 | 0 (0.00–0.07) | 0 |
| best_of_n (matched to upstream) | 1 | 0 (0.00–0.07) | 0 |
| local | 28 | 19 (0.26–0.52) | 18 |
| upstream | 28 | 22 (0.31–0.58) | 0 |
| smart (`pin_inputs=False`) | 29 | 23 (0.33–0.60) | 0 |
| smart_pinned (`pin_inputs=True`) | 28 | 19 (0.26–0.52) | 18 |

**Primary (Correct, Holm over four): all four supported.**

| | Comparison | Wins–losses | Holm p |
| --- | --- | --- | --- |
| C1 | smart vs best_of_n | 23–0 | 9.5e-7 |
| C2 | smart_pinned vs best_of_n | 19–0 | 7.6e-6 |
| C3 | local vs best_of_n | 19–0 | 7.6e-6 |
| C4 | upstream vs best_of_n | 22–0 | 1.4e-6 |

**What the arms did.**
- **Greedy and Best-of-N.** 38 of 50 greedy programs put a Nat parameter straight into a U32 sum.
  Best-of-N drew 100 samples and found 1 valid program, which was not correct.
- **Unpinned repair (Smart, Upstream).** It fixes the sum by retyping the Nat parameters as U32
  and the call literals to match. That is correct under the study's task check but never
  faithful.
- **Pinned Smart and LocalOnly.** They rewrite the operand as `U32.from_nat(...)`, and 18 of 19
  correct programs are faithful. On Faithful, `smart_pinned` beats `smart` 18–0
  (p = 7.6e-6); on Correct, `smart` leads 9–5 (p = 0.42).
- **Arms that end up identical, as the dev set predicted.**
  - `smart_pinned` and `local` produced identical programs on 49 of 50 tasks. With every flagged
    declaration frozen, pinned Smart falls back to the same in-place operand rewrite LocalOnly
    makes.
  - `smart` and `upstream` produced identical programs on 48 of 50.

**Runtime.**
- Mean s / problem: greedy 3.0, repair 7.6–8.4.
- Peak MPS: 2.22 GB.
- Checkpoint memory per cache: at most 320 MiB.
- Rewinds replay 6.6–7.3 tokens on average.
- A 51-minute low-power sleep (battery at 1%) fell inside one `local` repair. It is removed from
  the mean and has no effect on outcomes.

## Departures, disclosed

- **2B prompt suite.** The study's prompts were not used; reason and suite above. The 2B numbers
  are therefore not comparable with the study's catalog results.
- **2A held-out tasks.** These are the v0.2 seed-42 tasks, as the brief specifies, reused on a new
  model.
- **Shared greedy decode.** In v0.3 each mask setting decodes greedy once, and the other repair
  arms reuse it via a cache copy. This was checked on 8 dev tasks: a re-decode is identical, and
  repair from the copy matches repair from a fresh decode.
- **dtype.** float16, ttso's default. The study used bf16; bf16 gave the same greedy output on
  the one task probed.
- **Speedup claim.** The brief asked for the overall per-problem latency speedup. It is within
  noise; the measurement is reported as is.

## Limitations and open items

- **2A is underpowered for Faithful on Qwen3.5.** With four equal comparisons, Holm needs about 8–0 per pair to pass; each pair here was 6–0.
- **Pinning freezes what the model wrote.** A pin to the task's stated types (from the prompt)
  instead could address the 5–7 unfaithful 2A programs. It would also need the prompt's types to be
  parsed, so it is a design change for a future version.
- **The 2B cartridge covers only the U32 / Nat `+` subset.** Extending it to Bool, `match` and
  `*` would allow a run on the study's own prompts.
- **Latency.** Qwen3.5 on `mps` runs HuggingFace's reference PyTorch kernels for its linear
  layers, since `flash-linear-attention` and `causal_conv1d` are not installed. Other sessions'
  GPU jobs shared the machine, so latencies are noisy.

## Reproduce

```bash
export TTSO_HYBRID_MODEL=<path to the Qwen3.5-0.8B snapshot 2fc06364…>
python scripts/benchmark_v03.py json heldout
python scripts/benchmark_v03.py catalog heldout
python scripts/mask_timing_v03.py
python -m pytest
```
