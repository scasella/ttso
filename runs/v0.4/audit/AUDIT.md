# v0.2 / v0.3 re-scored with 5 input vectors

### v0.2 base: Qwen/Qwen3-0.6B-Base, json, n = 60

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 |
| --- | --- | --- | --- | --- | --- |
| `greedy` | 23 | 15 | 15 | 15 | 15 |
| `greedy_masked` | 29 | 15 | 15 | 15 | 15 |
| `best_of_n` | 23 | 15 | 15 | 15 | 15 |
| `best_of_n_masked` | 30 | 15 | 15 | 15 | 15 |
| `ttso_pinned` | 44 | 27 | 23 | 26 | 22 |
| `ttso_pinned_masked` | 51 | 31 | 24 | 30 | 23 |
| `ttso_unpinned` | 50 | 35 | 35 | 15 | 15 |

family_faithful_1: H1 11–0, p = 0.000977, Holm 0.00195; H2 11–0, p = 0.000977, Holm 0.00195; H3 15–0, p = 6.1e-05, Holm 0.000244; H4 15–0, p = 6.1e-05, Holm 0.000244

family_faithful_5: H1 7–0, p = 0.0156, Holm 0.0312; H2 7–0, p = 0.0156, Holm 0.0312; H3 8–0, p = 0.00781, Holm 0.0312; H4 8–0, p = 0.00781, Holm 0.0312

Sanity gate: 0 stored-vs-recomputed mismatches. Faithful_1 but not Faithful_5: 11.
`best_of_n`: 54 samples on 37 sampled tasks (per task {1: 21, 2: 15, 3: 1}); budget 10–328 tokens; no complete sample on 21 tasks.
`best_of_n_masked`: 47 samples on 31 sampled tasks (per task {1: 16, 2: 14, 3: 1}); budget 50–404 tokens; no complete sample on 16 tasks.

### v0.2 post: Qwen/Qwen3-0.6B, json, n = 20

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 |
| --- | --- | --- | --- | --- | --- |
| `greedy` | 4 | 4 | 4 | 4 | 4 |
| `greedy_masked` | 5 | 4 | 4 | 4 | 4 |
| `best_of_n` | 4 | 4 | 4 | 4 | 4 |
| `best_of_n_masked` | 5 | 4 | 4 | 4 | 4 |
| `ttso_pinned` | 4 | 4 | 4 | 4 | 4 |
| `ttso_pinned_masked` | 12 | 5 | 4 | 5 | 4 |

family_faithful_1: H1 0–0, p = 1, Holm 1; H2 0–0, p = 1, Holm 1; H3 1–0, p = 1, Holm 1; H4 1–0, p = 1, Holm 1

family_faithful_5: H1 0–0, p = 1, Holm 1; H2 0–0, p = 1, Holm 1; H3 0–0, p = 1, Holm 1; H4 0–0, p = 1, Holm 1

Sanity gate: 0 stored-vs-recomputed mismatches. Faithful_1 but not Faithful_5: 1.
`best_of_n`: 34 samples on 16 sampled tasks (per task {2: 14, 3: 2}); budget 157–352 tokens; no complete sample on 0 tasks.
`best_of_n_masked`: 24 samples on 15 sampled tasks (per task {1: 6, 2: 9}); budget 45–293 tokens; no complete sample on 6 tasks.

### v0.3 json: Qwen/Qwen3.5-0.8B, json, n = 60

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 |
| --- | --- | --- | --- | --- | --- |
| `greedy` | 16 | 10 | 8 | 8 | 8 |
| `greedy_masked` | 20 | 11 | 9 | 9 | 9 |
| `best_of_n` | 18 | 10 | 8 | 8 | 8 |
| `best_of_n_masked` | 23 | 11 | 9 | 9 | 9 |
| `ttso_pinned` | 31 | 19 | 15 | 14 | 14 |
| `ttso_pinned_masked` | 32 | 22 | 18 | 15 | 15 |
| `ttso_unpinned` | 46 | 17 | 14 | 11 | 11 |

family_faithful_1: H1 6–0, p = 0.0312, Holm 0.125; H2 6–0, p = 0.0312, Holm 0.125; H3 6–0, p = 0.0312, Holm 0.125; H4 6–0, p = 0.0312, Holm 0.125

family_faithful_5: H1 6–0, p = 0.0312, Holm 0.125; H2 6–0, p = 0.0312, Holm 0.125; H3 6–0, p = 0.0312, Holm 0.125; H4 6–0, p = 0.0312, Holm 0.125

Sanity gate: 0 stored-vs-recomputed mismatches. Faithful_1 but not Faithful_5: 0.
`best_of_n`: 78 samples on 44 sampled tasks (per task {1: 15, 2: 24, 3: 5}); budget 17–306 tokens; no complete sample on 14 tasks.
`best_of_n_masked`: 70 samples on 40 sampled tasks (per task {1: 14, 2: 22, 3: 4}); budget 8–300 tokens; no complete sample on 12 tasks.

### v0.3 catalog: Qwen/Qwen3.5-0.8B, catalog, n = 50

| Arm | Valid | Correct_1 | Correct_5 | Faithful_1 | Faithful_5 |
| --- | --- | --- | --- | --- | --- |
| `greedy` | 0 | 0 | 0 | 0 | 0 |
| `best_of_n` | 1 | 0 | 0 | 0 | 0 |
| `local` | 28 | 19 | 19 | 18 | 18 |
| `upstream` | 28 | 22 | 22 | 0 | 0 |
| `smart` | 29 | 23 | 23 | 0 | 0 |
| `smart_pinned` | 28 | 19 | 19 | 18 | 18 |

family_correct_1: C1 23–0, p = 2.38e-07, Holm 9.54e-07; C2 19–0, p = 3.81e-06, Holm 7.63e-06; C3 19–0, p = 3.81e-06, Holm 7.63e-06; C4 22–0, p = 4.77e-07, Holm 1.43e-06

family_faithful_5: C1 0–0, p = 1, Holm 1; C2 18–0, p = 7.63e-06, Holm 3.05e-05; C3 18–0, p = 7.63e-06, Holm 3.05e-05; C4 0–0, p = 1, Holm 1

family_correct_5: C1 23–0, p = 2.38e-07, Holm 9.54e-07; C2 19–0, p = 3.81e-06, Holm 7.63e-06; C3 19–0, p = 3.81e-06, Holm 7.63e-06; C4 22–0, p = 4.77e-07, Holm 1.43e-06

Sanity gate: 0 stored-vs-recomputed mismatches. Faithful_1 but not Faithful_5: 0.
`best_of_n`: 100 samples on 50 sampled tasks (per task {1: 19, 2: 15, 3: 14, 4: 1, 5: 1}); budget 40–300 tokens; no complete sample on 18 tasks.
