# ttso v0.2 pre-registration: held-out Gate 4 benchmark

Written and committed before any held-out task was run. The forward structural mask and every
benchmark setting were developed and debugged only on the 20 seed-0 development tasks (Gate 4's
tasks). The code is frozen at the commit that adds this file. Run with `scripts/benchmark_v02.py`.

## Question

Does pinned repair (`pin_inputs=True`) produce more faithful programs than greedy decoding and than
Best-of-N at a matched token budget? Does that still hold once a forward structural mask removes
the syntax and schema errors that backward repair cannot fix?

## Tasks and models

- **Held-out tasks:** `make_tasks(60, seed=42)`, the same 4 templates as Gate 4, with new names and
  values. The script asserts that no held-out task text equals a development task.
- **Primary model:** Qwen/Qwen3-0.6B-Base, float16 on `mps`, greedy decoding, at most 320 new tokens,
  up to 3 repair rounds.
- **Exploratory model:** Qwen/Qwen3-0.6B (post-trained), on the first 20 held-out tasks, with the
  same pinned arms.

## Arms

All arms of a task see the same prompt.

| Arm | Decoding | Repair |
| --- | --- | --- |
| `greedy` | greedy, type-slot mask only (as in v0.1) | none |
| `best_of_n` | temperature 0.7, same decoding constraints as `greedy` | none; stops at the first valid sample or when its tokens equal `ttso_pinned`'s extra tokens on that task |
| `ttso_pinned` | as `greedy` | Smart routing, input types frozen, LocalOnly at the argument key |
| `greedy_masked` | greedy inside the forward structural mask | none |
| `best_of_n_masked` | temperature 0.7 inside the mask | none; budget matched to `ttso_pinned_masked` |
| `ttso_pinned_masked` | inside the mask | as `ttso_pinned`, with every edit also inside the mask |
| `ttso_unpinned` | as `greedy` | the v0.1 spec routing, every declaration editable (secondary, base model only) |

When a repair arm spends no tokens because its greedy completion is already valid, its Best-of-N
arm is that greedy completion.

The forward mask fixes structure and scope, never types:

- key order;
- one key per argument;
- op and type names;
- the five literal forms (`"7"`, `"7n"`, `"7.5"`, `"7usd"`, `"7eur"`);
- fresh ids for new inputs and nodes;
- refs only to ids declared earlier;
- an output that names a node;
- a call that binds every input exactly once.

## Outcomes

- **Valid:** the reference checker accepts the completion.
- **Correct:** valid, with the expected output value and exactly the task's input names.
- **Faithful (primary):** correct, with every input keeping the type the task states.

## Primary hypotheses

All on Faithful, base model, n = 60, exact two-sided McNemar on paired outcomes. Holm correction
across the four, at α = 0.05.

| | Comparison | Direction expected |
| --- | --- | --- |
| H1 | `ttso_pinned` vs `greedy` | ttso higher |
| H2 | `ttso_pinned` vs `best_of_n` | ttso higher |
| H3 | `ttso_pinned_masked` vs `greedy_masked` | ttso higher |
| H4 | `ttso_pinned_masked` vs `best_of_n_masked` | ttso higher |

A hypothesis is supported only if its Holm-adjusted p < 0.05 and the difference is in the expected
direction. Wilson 95% intervals are reported for every arm's Faithful count.

## Secondary and exploratory analyses

These are reported without correction and are not confirmatory:

- Valid and Correct for the H1–H4 pairs.
- The mask alone: `greedy_masked` vs `greedy`.
- Repair with the mask against repair without it: `ttso_pinned_masked` vs `ttso_pinned`.
- Pinned against unpinned routing, on Correct and on Faithful. This replicates the v0.1 finding
  that unpinned repair retypes inputs.
- On the post-trained model, the same arms on 20 tasks, with no hypothesis tests treated as
  confirmatory.

Latency and peak MPS memory are logged.

## Analysis plan, fixed now

- Results are reported whatever their direction. Nulls are reported as nulls.
- No task, prompt, mask rule, routing rule or decoding setting changes after the held-out run
  starts. If the run fails for a technical reason, the fix and the rerun are recorded in this file
  and in the v0.2 report.
- The development-set numbers are reported separately from the held-out numbers. They are never
  pooled.

## Development results (seed-0 tasks, n = 20; not part of the test)

These come from the frozen code, recorded in `runs/v0.2/dev/seed0_base.json` (produced by
`scripts/dev_v02.py`).

| Arm | Valid | Correct | Faithful |
| --- | --- | --- | --- |
| `greedy` | 5 | 5 | 5 |
| `best_of_n` | 5 | 5 | 5 |
| `ttso_pinned` | 14 | 10 | 9 |
| `greedy_masked` | 8 | 5 | 5 |
| `best_of_n_masked` | 8 | 5 | 5 |
| `ttso_pinned_masked` | 17 | 12 | 11 |

- With the mask, all 12 greedy failures are type errors; no structural errors remain.
- One development bug was fixed before this freeze. The first version of the mask scanned only the
  model's top 8,192 tokens. When none was legal, it fell back to an illegal token and left the
  schema; this happened, for example, after `"47u"`, where only `s` is legal. The fix scans only the
  tokens that start with a character the schema allows, so it never leaves the schema.
- Literals such as `"47u32"` are not in the schema. Under the mask the model therefore writes the
  closest legal form, `"47usd"`, which is a type error left for repair.
- Timings were measured while three unrelated GPU jobs shared the machine, so they are not
  comparable with v0.1.

## Post-run record (added after the held-out runs; nothing above was changed)

- **Runs.** Both held-out runs completed on the frozen code (`835f4cb` in github.com/scasella/ttso), with no reruns: `runs/v0.2/base`, `runs/v0.2/post`.
- **Bug found after the freeze.** A code review found a bug in `pick()` that could only affect masked sampling. If the first draw was illegal and every legal token's sampling probability had underflowed to 0, it returned the illegal token.
  - The fix changes only that case. Every draw uses a per-task seeded generator, so replaying the masked Best-of-N arm with the fix shows whether the case ever occurred.
  - The replay reproduced all 31 base and 15 post-trained outputs exactly (`bon_masked_recheck.json`), so the reported numbers stand.
- **Mask integrity.** No masked output in either run left the schema.
