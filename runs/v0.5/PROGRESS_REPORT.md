# ttso v0.5: Progress Report

**Question.** v0.4's masked-repair win on 3A was `ttso_pinned_masked` 33 / 60 against
`greedy_masked` 17 / 60 (Faithful_5). Does it survive against a decoder mask that already knows the
types repair uses?

**Answer: yes, on Faithful_5. T1 is supported, 16–0, Holm p = 6.1 × 10⁻⁵.**
- The type-aware mask removes every type error, raising Valid from 30 to 57 / 60.
- But it adds no faithful program: 17 / 60, the same 17 tasks as `greedy_masked`.
- Repair on top of the type-aware mask (T2) is 0–0.
- Under the pre-registered decision rule the first branch applies: **the headline stands.**

There are three qualifiers, all below:
- the whole win is one error pattern;
- on single-vector scoring it would not be significant;
- a decoder with lookahead was not tested.

- **Pre-registration:** `runs/v0.5/PREREGISTRATION.md`, public ttso `ac8a2f9`. It was pushed at
  12:45:21 UTC, before the first held-out task at 12:45:25.
- **Run:** Gemma 4 E2B base, revision `433813a…`, bf16 on `mps`, on the 60 v0.4 held-out tasks
  (seed 42).
- **Results:** `runs/v0.5/gemma-4-E2B/json/`.

## Results (n = 60)

| Arm | Valid | Correct_5 | Faithful_1 | Faithful_5 | Faithful_5 95% CI | s / problem | Extra tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `greedy_typemasked` | 57 | 17 | 26 | **17** | 18.5–40.8% | 5.98 | |
| `ttso_pinned_typemasked` | 57 | 17 | 26 | 17 | 18.5–40.8% | 5.99 | 9 |
| `best_of_n_typemasked` | 57 | 17 | 26 | 17 | 18.5–40.8% | 6.01 | 9 (3 samples) |
| `greedy_masked` (R) | 30 | 17 | 17 | 17 | 18.5–40.8% | 5.42 | |
| `ttso_pinned_masked` (R) | 46 | 33 | 33 | **33** | 42.5–66.9% | 8.46 | 4,877 |

- **Seconds per problem:** these are means over all 60 tasks. The guard made no pause and the
  machine did not sleep, so the latency rule left out 0 tasks (`latency.json`).
- **Best-of-N cost:** a Best-of-N arm costs its regime's greedy decode plus its samples.
- **Correct_5 interval:** Correct_5 equals Faithful_5 in every arm, so its 95% interval is the
  same.

**Pre-registered tests (exact two-sided McNemar on Faithful_5, Holm over two):**

| | Comparison | Wins–losses | p | Holm p | Supported |
| --- | --- | --- | --- | --- | --- |
| T1 | `ttso_pinned_masked` vs `greedy_typemasked` | 16–0 | 3.1 × 10⁻⁵ | 6.1 × 10⁻⁵ | **yes** |
| T2 | `ttso_pinned_typemasked` vs `greedy_typemasked` | 0–0 | 1 | 1 | no (predicted) |

**Secondary outcomes (not corrected):**
- **On Correct_5:** T1 16–0 and T2 0–0, the same as on Faithful_5.
- **On Faithful_1** (single-vector scoring):
  - T1 is **15–8, p = 0.21**. That is not significant; see "Three qualifiers".
  - T2 is 0–0.

**R (reproduction): exact.** `greedy_masked` is 17 and `ttso_pinned_masked` 33, and their final
texts are identical to v0.4's on all 60 tasks. So there is no difference to explain. v0.4's H3
pair is 16–0 again.

**Predictions against results** (pre-registered from the 20 dev tasks):

| | Predicted | Observed |
| --- | --- | --- |
| `greedy_typemasked` Faithful_5 | 18 (dev rate; 17 if it repeats `greedy_masked` task for task) | 17, task for task the same as `greedy_masked` |
| `greedy_typemasked` Valid | about 57 | 57 |
| T1 | supported, about 16–0 | 16–0 |
| T2 | 0–0 | 0–0 |
| `best_of_n_typemasked` | equal to greedy, or within 1 | equal (3 samples, all truncated) |
| R | exact | exact |

## Which families each arm fixes (Valid / Faithful_5, 15 tasks each)

| Family | `greedy_masked` | `ttso_pinned_masked` | `greedy_typemasked` | `ttso_pinned_typemasked` | `best_of_n_typemasked` |
| --- | --- | --- | --- | --- | --- |
| Nat + U32 + U32 sum | 0 / 0 | **15 / 15** | 13 / 0 | 13 / 0 | 13 / 0 |
| USD sum + charge | 15 / 2 | 15 / 2 | 15 / 2 | 15 / 2 | 15 / 2 |
| max of three Nat | 15 / 15 | 15 / 15 | 15 / 15 | 15 / 15 | 15 / 15 |
| min(U32 + Nat, U32) | 0 / 0 | 1 / 1 | 14 / 0 | 14 / 0 | 14 / 0 |

T1's 16 wins split as 15 in the sum family and 1 in the min family (task 39). T2 has no
discordant pair in any family.

**Sum and min families.** These are the two families where a Nat input feeds a U32 node.
- Masked repair fixes the sum family completely. Under v0.4's mask the greedy program writes
  `{"ref": <Nat input>}` in the U32 node. Repair rewinds to the argument's key and writes
  `{"convert": "to_u32(<Nat input>)"}`, then removes the `usd` suffix from a call literal.
- The type-aware mask makes 27 of these 30 programs valid but fixes none:
  - **Valid but wrong:** in all 27 valid programs the Nat input is never used. The mask admits
    `"ref"` at the key, because U32 ids exist, and then refuses the Nat id. A forward mask cannot
    revisit the key, so the model references the U32 inputs instead, often more than once, and
    sometimes adds nodes; e.g. task 0 computes
    `max(stock + boxes + stock + stock, stock + boxes + stock)`. In 2 min-family programs the Nat
    input's stated value is written in as a U32 literal.
  - **Cut off:** the other 3 run into a literal that never ends and hit the 320-token limit.
- **Min family, masked repair:** it fails 14 of 15, and every min-family task used all 3 rounds.
  - **13 of the 15:** round 1 replaced the reference's key `ref` with `lit` (the model's next
    choice there), not `convert`.
    - The regenerated literal failed its node's type check. The intermediate texts are not
      stored, so the literal itself cannot be shown.
    - Round 2 retyped the node's out_type to the literal's type: USD on 8 tasks, Nat on 5.
    - Round 3 rewrote another `ref` key as `convert`.
    - A `usd` call literal was still left when the rounds ran out.
  - **Task 39, the one fixed:** it followed the sum-family path: `convert` in round 1, then the two
    `usd` call suffixes removed.
  - **The last task:** `convert` in round 1, then a retyped conversion result.

**USD family.** Every arm is faithful on only 2 of 15. Gemma declares the "4 USD charge" as a third
input, `charge`, which the task does not name. That is outside Correct by definition and outside
any type check, and no arm fixes it.

**Max family.** All arms are faithful on all 15.

## Why T2 is null: the mask hides the error from the verifier

Under the type-aware mask the wrong reference is never decoded. What is decoded instead is a
type-correct program with the wrong formula, which no type checker (and so no verifier signal)
can flag.

The audit shows that's the whole story. The type-masked cartridge's reference checker made 132
checks: each task's greedy program twice, plus every repair round and every Best-of-N sample.
- **All 132 checks** are accepted by the type-aware mask at every prefix. So the admissible set
  was never empty: `pick` never fell back to a rejected token.
- **114 complete:** all end in the mask's done state and are valid, with 0 failed
  symbolic-verifier checks and 0 hard errors.
- **18 incomplete:** 3 tasks' programs cut off at 320 tokens (2 greedy checks, 3 repair rounds and
  1 Best-of-N sample each).
- **Repair's only work under the type-aware mask:** 9 one-token edits on those 3 truncated
  programs, at token 319, which cannot help.
- **Best-of-N:** its budget was 3 tokens on each of those tasks, so it drew 3 samples, all cut off.

So the type-aware mask turns a type error that the verifier can see and locate into a semantic
error that it cannot. Repair wins in v0.5 because v0.4's mask lets the error be written, and the
verifier then finds the place to fix it.

## Three qualifiers on T1

1. **One error pattern.** 15 of the 16 wins are the same pattern: a Nat input inside a U32 sum,
   fixed by a conversion. The other families are either already solved by both (max), failed
   identically by both (USD), or mostly failed by both (min). The claim supported here is narrow:
   for this pattern, rewind-and-repair beats the strongest forward mask that uses only decode-time
   type facts.
2. **Five-vector scoring is what shows it.** On Faithful_1, the single-vector metric of v0.2–v0.3,
   T1 is 15–8, p = 0.21.
   - **The 8 losses:** all are min-family tasks where `greedy_typemasked`'s program drops the Nat
     input yet matches the stated vector by coincidence. `ttso_pinned_masked` is invalid on them
     (a `usd` literal is left).
     - Task 3 computes min(orders + seats, seats) = 30, which is what the task's
       min(28 + 16, 30) gives.
     - `greedy_typemasked` has 9 Faithful_1-only coincidences; the masked repair arm has 0.
   - **So:** v0.4's move to 5-vector scoring is what makes this comparison measurable at all.
3. **Not every decoder.** As pre-registered, a decoder with lookahead or backtracking search, one
   that could prefer `"convert"` at the key, was not tested. The stated output type was also not
   used. T1 supports "repair beats an equally informed forward mask", not "repair beats any
   decoder".

## Fallbacks

Where no type is fixed at decode time, the type-aware mask falls back to SchemaPrefix. Counted
over `greedy_typemasked`'s 60 final programs:
- **`out_type`: 85** — one per node: all three ops are polymorphic, so nothing fixes a node's type.
- **`output`: 57** — one per complete program.
- **`input_type`: 13** — the USD family's invented `charge` input, on 13 of 15 USD tasks. The
  contract has no entry for it.

Over all 132 audited checks: 179 `out_type`, 114 `output` and 26 `input_type`.

Read through the type-aware mask, the SchemaPrefix arms' final programs would have been refused
30 / 60 times (`greedy_masked`) and 14 / 60 (`ttso_pinned_masked`). That is every invalid program
in each.

## Cost and resources

- **Mask time:** about 4 ms per problem for both masks. The type-aware mask costs nothing
  measurable over SchemaPrefix.
- **Greedy latency:** `greedy_typemasked` is 0.56 s per problem slower than `greedy_masked`
  because it decodes more tokens, 8,696 against 7,805. Its programs are longer, and three run to
  the 320-token limit.
- **Masked repair:** 8.46 s per problem and 4,877 extra tokens, against 5.98 s for the type-aware
  greedy.
- **Shadow verifier:** the DeltaNet and the symbolic verifier agreed on every round, 75 / 75 under
  v0.4's mask and 9 / 9 under the type-aware mask. Diagnosis took 4.5 ms against 0.026 ms per
  round.
- **Memory:** 4.25 GiB of weights on the device, plus the 4.38 GiB per-layer embedding table
  memory-mapped on the cpu. Peak MPS allocation was 4.63 GB and peak RSS 2.88 GB.
- **Run:** 868.9 s of benchmark wall time. During the run the guard logged no pause, only 7 LOW
  notices at pressure 2 with 28–32% free, and the machine did not sleep.

## What changed in the code

| Item | Files |
| --- | --- |
| TypeAwareMask, its trail and the type-masked cartridge, with a per-program audit | `ttso/cartridges/type_mask.py` |
| A typed variant of the SchemaPrefix machine. The v0.4 program is byte-identical, and `SchemaState` runs `self._prog` | `ttso/cartridges/typed_json_dag.py` |
| `greedy_arm` / `_decode_key` read `mask_name`, so the type-masked regime cannot reuse the SchemaPrefix decode | `ttso/engine/benchmark.py` |
| 6 tests: type-correct fuzz; one-type-error rejection; typed ⊆ schema; never-empty walks; Gemma single-char coverage; forced `16u32` / `16usd` / wrong-ref decodes | `tests/test_type_mask.py` |
| v0.5 harness and latency rule | `scripts/benchmark_v05.py`, `scripts/pauses_v05.py` |

**Tests:** 53 / 53 at the freeze, in 19.5 min (`runs/v0.5/freeze_suite.log`). The README still
says 47 tests and is otherwise untouched, as asked, until you have read this report.

## Departures and notes

- **3B left out, but not because it is clean.** v0.4's 3B greedy failures are mostly structural
  (23 of 42), but 8 of C2's 9 wins and all 7 of C3's wins came on tasks whose greedy program
  failed a type check. So the catalog's repair results face the same question, and v0.5 does not
  answer it.
- **Report filename:** `PROGRESS_REPORT.md`, as in v0.4, rather than a name with a colon.
- **Nothing else** departed from the pre-registration. No task, seed, scorer or v0.4 arm changed.

## Not done; for you to decide

- **README:** not changed. If you keep the headline, the T1 qualifiers above belong next to it.
- **A follow-up hypothesis, untested:** a mask that types literals only (refusing `16usd` in a U32
  slot) but leaves references to repair.
  - It would keep the sum-family fix: the wrong reference stays visible to the verifier.
  - It would remove the `usd` call literals that use up the min family's later rounds.
  - It would not change round 1's `ref` → `lit` choice, which is where most min-family repairs go
    wrong. Whether repair then recovers is an open question.
  - It needs its own pre-registration.
