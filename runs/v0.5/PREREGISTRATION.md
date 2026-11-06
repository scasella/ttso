# ttso v0.5 pre-registration: a type-aware forward mask as the baseline for repair (3A)

Written and committed before any held-out (seed 42) task was run with the type-aware mask. The
mask, the arms and the harness were developed and checked only on the 20 seed-0 development
tasks. The code is frozen at the commit that adds this file. Run with
`scripts/benchmark_v05.py heldout`.

**Question.** v0.4's one positive JSON result was masked repair: on Gemma 4 E2B base,
`ttso_pinned_masked` was Faithful_5 on 33 / 60 held-out tasks against 17 / 60 for `greedy_masked`
(H3, 16–0). But v0.4's forward mask (SchemaPrefix) never looks at types, so its greedy baseline
decodes type errors that a decoder could have refused. Does masked repair still win against a
decoder mask that already knows every type the repair uses?

**Scope.** JSON 3A only. The catalog suite (3B) is out of scope, as the brief directs, because
its greedy errors are mostly structural.

On v0.4's held-out 3B, `greedy` failed on 42 of 50 tasks. By its first error:
- 23 structural (21 bad operands, 2 bad def lines);
- 8 undeclared names;
- 11 type errors (10 type mismatches, 1 literal of the wrong type).

But the catalog's positive results came almost entirely from that type-error minority:
- 8 of C2's 9 wins (`smart_pinned` over `best_of_n`) were on tasks whose greedy program failed a
  type check;
- so were all 7 of C3's wins (`local`).

So 3B's repair results are exposed to the same question as 3A's. v0.5 does not test them, and
they should not be read as having survived a type-aware baseline.

## 1. TypeAwareMask (`ttso/cartridges/type_mask.py`)

It extends SchemaState, the incremental form of SchemaPrefix. The two accept the same strings;
that is checked by differential fuzz in `tests/test_incremental_schema.py`, which still passes. It runs the same machine
(`typed_json_dag._program(typed=True)`: the same 124 instructions in the same places, with the
nine typed slots renamed; the v0.4 program is byte-identical, checked against v0.4's file) and
adds types.

**What it knows at decode time:**
- the schema;
- each op's signature: add / max / min take arguments of the node's out_type; to_u32 is
  Nat → U32, to_nat U32 → Nat, to_float U32 → Float;
- the task's stated input types (the contract, bound per task);
- the types declared earlier in the prefix.

It never reads the expected value, the test vectors or a reference program.

**What it does not know:**
- **No lookahead past the current slot.** At an arg key it admits every key that some
  type-correct argument could follow. It cannot know which id the model is about to write, so it
  cannot prefer `"convert"` over `"ref"`.
- **No rewind.** Once a key is decoded, a forward mask can only restrict what follows it.

This is exactly the limit T1 tests: repair rewinds to the key, and a forward mask cannot. A
supported T1 shows that repair beats this mask, the strongest one that uses only decode-time type
facts. It does not show that repair beats every decoder: one with lookahead or backtracking
search, for instance, is not tested.

**What it admits:**

| Slot | Admitted | Type fixed by |
| --- | --- | --- |
| an input's type | the stated type, if the task names the id | the contract |
| arg key | `"ref"` only if a declared id has the node's out_type; `"convert"` only if a converter yields the out_type from a declared id; `"lit"` always | declarations, signatures |
| `{"ref": x}` | declared ids of the node's out_type | declarations, op signature |
| `{"convert": "f(x)"}` | f whose result is the node's out_type and that has a declared id of its source type; x a declared id of f's source type | declarations, signatures |
| `{"lit": v}` | v in the out_type's literal form | op signature |
| `"call": {x: v}` | v in x's declared type's literal form | x's declaration |

- **Literal forms:** U32 is digits then the closing quote, with no suffix; Nat `7n`, Float `7.5`,
  USD `7usd`, EUR `7eur`. So neither `"16u32"` nor `"16usd"` can be decoded in a U32 slot.
  - `"16u32"` was already outside SchemaPrefix.
  - `"16usd"` in a U32 slot was not, and it is v0.4's masked greedy failure.
- **Never empty:** the arg-key rule keeps every position non-empty. A key is admitted only when a
  type-correct argument can follow it.

**Fallbacks.** Where no type is fixed yet, the mask falls back to SchemaPrefix and counts it, by
kind:
- **`out_type`:** every node's out_type. No op signature or earlier declaration fixes it, since all
  three ops are polymorphic and a literal of any type exists.
- **`output`:** the output id (any node; nothing consumes it).
- **`input_type`:** the type of an input the task does not name. The contract has no entry for it.

A program therefore has at least two fallbacks (one node, one output). The task's stated output
type is not used: it is not part of the contract, and the brief lists what the mask may know.

**How the contract is enforced.** The input-type slot is held to the stated type twice,
identically:
- by v0.4's `_stated` logits mask in `TypedJSONDAGCartridge.logits_processor`, which
  `greedy_masked` also decodes with;
- by the type-aware state itself, so the state alone is a complete type-aware acceptor.

The cartridge (`TypeMaskedJSONDAGCartridge`) uses the mask for decoding, for Best-of-N samples
and for every repair edit (the router's edits must pass the cartridge's `accept`). Its trail is
rebuilt on `bind_task`, so the contract cannot go stale across tasks.

**Harness change (additive).** `benchmark.greedy_arm` / `_decode_key` read the cartridge's
`mask_name`. Without that the type-masked regime would be named `greedy_masked` and, with the
shared greedy decode, silently reuse the SchemaPrefix decode. v0.4's names and keys are otherwise
unchanged.

## 2. Arms (held-out: `benchmark.make_tasks(60, seed=42)`, the v0.4 tasks)

- **`greedy_typemasked`** (the primary comparator): greedy decoding under TypeAwareMask, with
  contract pinning.
- **`ttso_pinned_typemasked`:** the same cartridge, then ttso repair. The policy is smart, at
  most K = 3 rounds, with the symbolic shadow, exactly as `ttso_pinned_masked` in v0.4.
- **`best_of_n_typemasked`:** Best-of-N at T = 0.7 under TypeAwareMask, seeded per task, matched
  per task to `ttso_pinned_typemasked`'s extra tokens, as in v0.4. Where repair spends no token
  (its greedy program is valid and meets the contract), Best-of-N draws nothing and is the greedy
  arm. Under the type-aware mask that is expected on most tasks.
- **The reproduction arms:** `greedy_masked` and `ttso_pinned_masked`, v0.4's arms rerun
  unchanged: the same cartridge, SchemaPrefix, contract pinning and smart repair.

**Common settings, all as in v0.4:**
- **Model:** google/gemma-4-E2B at revision `433813a669639fd7eaa83155828e9000ed357d53`, bf16 on
  `mps`. Text-only, with the per-layer embedding table memory-mapped on the cpu.
- **Prompt:** v0.2's raw few-shot prompt after `<bos>`, with the same tokenizer.
- **Decoding:** at most 320 new tokens, shadow verifier on.
- **Tasks:** SHA-256 of `json.dumps([t["text"] for t in tasks])` is
  `0ef5a0d84e941461d61fb2cf0dae2b2c707ca5ff141be30849811ce5479b7e3c`, the same as v0.4.

## 3. Hypotheses

**Primary outcome:** Faithful_5. Secondary: Correct_5 and Faithful_1.

| | Comparison | Exact two-sided McNemar, Holm over T1 and T2, α = 0.05 |
| --- | --- | --- |
| T1 | `ttso_pinned_masked` vs `greedy_typemasked` | repair (v0.4's mask) vs an equally informed decoder mask |
| T2 | `ttso_pinned_typemasked` vs `greedy_typemasked` | repair on top of the type-aware mask |

A test is supported only if its Holm p < 0.05 and the difference favours repair.

**R (reproduction):** `greedy_masked` = 17 and `ttso_pinned_masked` = 33 on Faithful_5, as in
v0.4. v0.4's decode was deterministic (`runs/v0.4/dev/determinism.log`), so the rerun is expected
to match v0.4 token for token. The analysis compares every task's text with
`runs/v0.4/gemma-4-E2B/json/results.jsonl`, read-only. Any difference is explained before any
result is reported.

## 4. Dev gate and prediction (the 20 seed-0 dev tasks; not part of the test)

`scripts/benchmark_v05.py dev`, with all five arms. Results are in
`runs/v0.5/dev/gemma-4-E2B/json/` and the log in `runs/v0.5/dev/gemma-4-E2B_json.log`.

After this run, no mask, arm or harness code changed. The later edits were:
- comments and a blank line in `tests/test_type_mask.py`;
- `scripts/pauses_v05.py`, which now covers the v0.5 arms;
- this file.

**Gate: passed.** The type-masked cartridge's reference checker made 44 checks. That is every
task's greedy program, checked twice (once for its row, once at the start of repair), plus task
7's three repair rounds and its one Best-of-N sample.
- All 44 are accepted by the type-aware mask at every prefix, so the decoder never met an empty
  admissible set. (`pick` would otherwise have fallen back to a rejected token, and the program
  would leave the mask.)
- 38 are complete. All 38 end in the mask's done state and are valid, with 0 failed
  symbolic-verifier checks and 0 hard errors. So no wrong-type literal or reference was
  admissible.
- The other 6 are all task 7's, cut off at 320 tokens (`json: truncated`): its greedy program
  (twice), its three repair rounds and its Best-of-N sample.

Unit tests (`tests/test_type_mask.py`, 6 tests):
- 300 type-correct random documents are accepted at every prefix, with fallbacks equal to their
  free slots.
- 2,228 one-type-error documents: SchemaPrefix accepts each one and TypeAwareMask rejects it.
- Typed next-character sets lie within SchemaPrefix's.
- 400 random walks through the admissible set never met an empty set; all 400 finished, and all
  are valid.
- Every admitted character is a single Gemma token.
- Under `pick` with logits forced onto them, `"16u32"`, `"16usd"` and a wrong-type reference are
  never decoded. SchemaPrefix decodes `usd` and the wrong reference.

**Dev results (n = 20):**

| Arm | Valid | Correct_5 | Faithful_1 | Faithful_5 | s / problem |
| --- | --- | --- | --- | --- | --- |
| `greedy_masked` | 10 | 6 | 6 | 6 | 5.0 |
| `ttso_pinned_masked` | 15 | 11 | 11 | 11 | 7.9 |
| `greedy_typemasked` | 19 | 6 | 8 | 6 | 5.5 |
| `ttso_pinned_typemasked` | 19 | 6 | 8 | 6 | 5.5 |
| `best_of_n_typemasked` | 19 | 6 | 8 | 6 | 5.5 |

- **T1 on dev:** 5–0 (exact p = 0.0625; Holm 0.125). T2 on dev: 0–0.
- **Reproduction on dev:** both v0.4 arms match `runs/v0.4/dev` token for token on all 20 tasks
  (`greedy_masked` 6, `ttso_pinned_masked` 11).
- **Same tasks:** `greedy_typemasked` is Faithful_5 on exactly the same 6 tasks as `greedy_masked`.

**By family (Valid / Faithful_5, 5 dev tasks each):**

| Family | `greedy_masked` | `ttso_pinned_masked` | `greedy_typemasked` |
| --- | --- | --- | --- |
| Nat + U32 + U32 sum | 0 / 0 | 5 / 5 | 5 / 0 |
| USD sum + charge | 5 / 1 | 5 / 1 | 5 / 1 |
| max of three Nat | 5 / 5 | 5 / 5 | 5 / 5 |
| min(U32 + Nat, U32) | 0 / 0 | 0 / 0 | 4 / 0 |

**What the type-aware mask does.** Gemma writes `{"ref": <the Nat input>}` inside a U32 node. The
mask admits `"ref"` at the key, since U32 ids exist, then refuses the Nat id. A forward mask
cannot go back to the key. So the model, already committed:
- references a U32 input twice (sum family, e.g. `deposit + level + deposit`);
- or writes the Nat input's stated value as a U32 literal (min family, `{"lit": "53"}`). That is
  right on the stated vector only, hence 2 Faithful_1-only coincidences;
- or, once (task 7), writes a 232-digit literal that runs out the 320-token budget.

Every complete one of these programs is type-correct, so the verifier finds nothing to repair. That makes
T2 null by construction. Repair under v0.4's mask instead rewinds to the arg key and writes
`{"convert": "to_u32(...)"}`.

**Fallbacks on dev:** `greedy_typemasked`'s 20 final programs have:
- 28 `out_type` fallbacks (one per node);
- 19 `output` fallbacks (one per complete program);
- 4 `input_type` fallbacks: the USD family's invented `charge` input, which the task does not
  name.

Over all 44 audited checks: 59 `out_type`, 38 `output`, 8 `input_type`.

**Pre-registered predictions for held-out (written before the run):**
- **`greedy_typemasked`, Faithful_5:** 18 / 60.
  - This is the dev rate (6 / 20) times 60; the dev Wilson 95% interval scales to 9–31.
  - Per family, dev × 3: sum 0, USD 3, max 15, min 0.
  - If held-out repeats dev's task-for-task equality with `greedy_masked`, it is 17.
- **Valid:** about 57 / 60 (dev 19 / 20), against `greedy_masked`'s 30.
- **T1:** supported, about 16–0.
  - `ttso_pinned_masked`'s 16 wins over `greedy_masked` in v0.4 were 15 in the Nat + U32 + U32 sum
    family and 1 in the min family (task 39).
  - The dev pattern (`greedy_typemasked` faithful on exactly `greedy_masked`'s tasks) predicts the
    same 16 wins and no losses.
- **T2:** 0–0, not supported.
- **`best_of_n_typemasked` = `greedy_typemasked`,** or within 1. On dev it sampled only the one
  truncated task, with a budget of 3 tokens.
- **R:** exact reproduction (17 and 33, identical texts).

## 5. Decision rule (fixed now)

- **T1 supported:** the headline stands. Repair beats an equally informed decoder mask.
- **T1 not supported, or `greedy_typemasked` at or above `ttso_pinned_masked` on Faithful_5:**
  reframe.
  - Errors whose type is decidable while decoding belong to the mask.
  - Repair's claim narrows to errors that are checkable only after decoding.
  - The paper leads with the resource results: exact rewind, the symbolic verifier matching the
    DeltaNet at a fraction of the CPU cost, and memory.
- **T2** is reported either way. It is expected null by construction. A completed program under
  the type-aware mask has no type error (the random-walk test and the dev gate above), so the
  verifier has nothing to repair.
  Repair can act only on a program cut off at 320 tokens, or on a contract failure, and the mask
  prevents the latter.

## Logged for every run

- **Outcomes:** Valid, Correct_1, Correct_5, Faithful_1 and Faithful_5 per arm, with Wilson 95%
  intervals.
- **Cost:** seconds per problem; extra tokens; mask time.
- **Repair internals:** rewinds; Best-of-N sample kinds; DeltaNet / symbolic shadow agreement.
- **By family:**
  - Valid, Correct_5 and Faithful_5 for each of the four task families (15 held-out tasks each):
    Nat + U32 + U32 sum (kind 0), USD sum + charge (1), max of three Nat (2) and min(U32 + Nat, U32)
    (3);
  - T1 and T2's discordant pairs per family.
- **The mask's audit:**
  - **Every program the type-masked cartridge checks** (greedy, every repair round, every Best-of-N
    sample, `runs/v0.5/gemma-4-E2B/json/audit.jsonl`):
    - whether the type-aware mask accepts it at every prefix;
    - failed symbolic-verifier checks and hard errors;
    - fallback counts by kind.
  - **Every arm's final program, read through the type-aware mask:**
    - accepted or not (the SchemaPrefix arms show how many of their programs the type-aware mask
      would have refused);
    - fallback counts.
- **Reproduction:** for `greedy_masked` and `ttso_pinned_masked`, the tasks whose text differs
  from v0.4.
- **Resources:** peak MPS allocation and peak RSS; each task's wall-clock window.

## Memory, pauses and sleep

- **Memory guard:** the held-out run runs under `caffeinate -i` with the memory guard. The guard
  pauses this session's ttso processes only at critical memory pressure or under 8% free memory,
  and logs to `runs/v0.5/memory_guard.log`.
- **Sleeps** are read from `pmset -g log`.
- **Latency rule (v0.4's):** a task whose wall window overlaps a pause or a sleep is left out of
  the latency means (`scripts/pauses_v05.py`). The number left out is reported, and nothing is
  excluded silently. Outcomes are unaffected.
- One model is resident at a time.

## Tests at the freeze

`python -m pytest -q -s`: **53 passed in 1172.6 s (19.5 min)**, on 6 November 2026. The log is
`runs/v0.5/freeze_suite.log`, with private paths redacted to `<repo>/`; the Gate 4 run it wrote is
`runs/gate4/20261106-072956/`.
- v0.4's 47 tests all still pass, including v0.3's `test_incremental_schema` (SchemaState ==
  SchemaPrefix by differential fuzz, and the masked dev gate token for token). Together with the
  byte-identical program check, that is the evidence v0.4's mask is unchanged.
- The 6 new tests are `tests/test_type_mask.py`.
- Transformers 5.22.1, torch 2.15.0, `mps` on an Apple M4 Pro with 24 GB.

## Post-run record (added after the held-out run)

- **Pre-registration commit (public ttso):** _filled in after the push._
