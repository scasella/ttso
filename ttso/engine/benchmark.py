"""Typed JSON DAG benchmark: few-shot tasks, and a paired Greedy / Best-of-N / ttso comparison.

Each task names typed inputs, the output type and the call values; the model writes the workflow
JSON. A completion is correct (Correct_1) when the reference checker accepts it, its output value is
the expected one and it declares exactly the task's inputs; faithful (Faithful_1) when it is also
correct with every input keeping the type the task gave it. Those two judge one input vector, the
one the prompt states, so a program can pass them with the wrong reusable formula. v0.4 adds
Correct_5 / Faithful_5: Correct_1 / Faithful_1, and the program's output equals the task's formula
on 5 input vectors (the stated one and 4 varied ones, `vectors`), each substituted into the
program's call. Faithful_5 is the primary metric from v0.4. Best-of-N samples whole completions at temperature 0.7
until it has spent the same number of extra tokens as the first ttso regime spent on that problem.
"""

import hashlib
import json
import random
import time
from math import comb, sqrt

import torch

from ttso.cartridges.typed_json_dag import TYPES, json_end, reference_check
from ttso.engine.repair_loop import RepairLoop

HEADER = """Typed workflow JSON. Inputs and nodes declare a type: U32, Nat, Float, USD or EUR.
Each arg is one of {"ref": id}, {"lit": literal} or {"convert": "fn(id)"}. A node's args must all have the node's out_type.
Literals: "7" is U32 (no suffix), "7n" is Nat, "7.5" is Float, "7usd" is USD, "7eur" is EUR.
Conversions: "to_u32(x)" turns a Nat x into U32, "to_nat(x)" turns a U32 x into Nat, "to_float(x)" turns a U32 x into Float.
The call binds every input to a literal of the input's type.
"""

SHOTS = [
    ("Add a Nat count and a U32 bonus into a U32 total. Call with count 4 and bonus 9.",
     {"inputs": [{"id": "count", "type": "Nat"}, {"id": "bonus", "type": "U32"}],
      "nodes": [{"id": "total", "op": "add", "out_type": "U32",
                 "args": [{"convert": "to_u32(count)"}, {"ref": "bonus"}]}],
      "output": "total", "call": {"count": "4n", "bonus": "9"}}),
    ("Take the larger of two EUR prices, base and promo. Call with base 30 and promo 25.",
     {"inputs": [{"id": "base", "type": "EUR"}, {"id": "promo", "type": "EUR"}],
      "nodes": [{"id": "best", "op": "max", "out_type": "EUR", "args": [{"ref": "base"}, {"ref": "promo"}]}],
      "output": "best", "call": {"base": "30eur", "promo": "25eur"}}),
    ("Add a Nat stock, a Nat reserve and 2 more units, then take the larger of that sum and a Nat floor. "
     "Call with stock 11, reserve 5 and floor 20.",
     {"inputs": [{"id": "stock", "type": "Nat"}, {"id": "reserve", "type": "Nat"}, {"id": "floor", "type": "Nat"}],
      "nodes": [{"id": "units", "op": "add", "out_type": "Nat",
                 "args": [{"ref": "stock"}, {"ref": "reserve"}, {"lit": "2n"}]},
                {"id": "result", "op": "max", "out_type": "Nat", "args": [{"ref": "units"}, {"ref": "floor"}]}],
      "output": "result", "call": {"stock": "11n", "reserve": "5n", "floor": "20n"}}),
]

NAMES = ["stock", "level", "qty", "price", "fee", "tax", "rent", "items", "boxes", "crates",
         "shift", "hours", "bonus", "deposit", "refund", "count", "parts", "orders", "pallets", "seats"]


def _task(rng, i):
    """One task. `kind`, `const` and `values` (added in v0.4; the text and the RNG draws are those of
    v0.2) define the task's formula, for scoring on other input vectors (formula_value)."""
    kind = i % 4
    a, b, c = rng.sample(NAMES, 3)
    va, vb, vc = rng.randint(2, 60), rng.randint(2, 60), rng.randint(2, 60)
    if kind == 0:  # catalog: one Nat item inside a U32 total
        text = (f"Add a Nat {a}, a U32 {b} and a U32 {c} into a U32 total. "
                f"Call with {a} {va}, {b} {vb} and {c} {vc}.")
        return {"text": text, "inputs": [a, b, c], "types": ["Nat", "U32", "U32"], "expected": va + vb + vc,
                "kind": kind, "const": 0, "values": [va, vb, vc]}
    if kind == 1:  # currency: USD inputs and a USD constant
        k = rng.randint(1, 9)
        text = f"Add a USD {a}, a USD {b} and a {k} USD charge into a USD total. Call with {a} {va} and {b} {vb}."
        return {"text": text, "inputs": [a, b], "types": ["USD", "USD"], "expected": va + vb + k,
                "kind": kind, "const": k, "values": [va, vb]}
    if kind == 2:  # max over Nat values with a Nat result
        text = f"Take the largest of three Nat values {a}, {b} and {c}, as a Nat. Call with {a} {va}, {b} {vb} and {c} {vc}."
        return {"text": text, "inputs": [a, b, c], "types": ["Nat", "Nat", "Nat"], "expected": max(va, vb, vc),
                "kind": kind, "const": 0, "values": [va, vb, vc]}
    text = (f"Add a U32 {a} and a Nat {b} into a U32 sum, then take the smaller of that sum and a U32 {c}. "
            f"Call with {a} {va}, {b} {vb} and {c} {vc}.")
    return {"text": text, "inputs": [a, b, c], "types": ["U32", "Nat", "U32"], "expected": min(va + vb, vc),
            "kind": kind, "const": 0, "values": [va, vb, vc]}


def formula_value(task, vec):
    """The task's output for an input vector {input name: value}."""
    x = [vec[n] for n in task["inputs"]]
    kind = task["kind"]
    if kind == 0:
        return sum(x)
    if kind == 1:
        return x[0] + x[1] + task["const"]
    if kind == 2:
        return max(x)
    return min(x[0] + x[1], x[2])


MAGNITUDES = ((2, 9), (10, 99), (100, 999), (1000, 9999))


def task_rng(task):
    """A per-task RNG seeded from the task text (not its index)."""
    return random.Random(int.from_bytes(hashlib.sha256(task["text"].encode()).digest()[:8], "big"))


def vectors(task):
    """5 input vectors: the one the prompt states, then 4 with values of growing magnitude (2-9,
    10-99, 100-999, 1000-9999), distinct, >= 2 and none of the stated values. Coverage by construction: for the max task
    each input is the largest in at least one varied vector; for min(a + b, c), c is below a + b in
    two vectors and above it in two."""
    rng, n, out = task_rng(task), len(task["inputs"]), [dict(zip(task["inputs"], task["values"]))]
    for j, (lo, hi) in enumerate(MAGNITUDES):
        v = rng.sample([x for x in range(lo, hi + 1) if x not in task["values"]], n)
        if task["kind"] == 2:  # the largest value goes to input j % n
            v.sort()
            top = v.pop()
            v.insert(j % n, top)
        elif task["kind"] == 3:
            s, avoid = v[0] + v[1], set(v[:2]) | set(task["values"])
            side = range(max(2, s // 2), s) if j % 2 == 0 else range(s + 1, s + hi + 1)  # c < a + b, c > a + b
            v[2] = rng.choice([c for c in side if c not in avoid])
        out.append(dict(zip(task["inputs"], v)))
    return out


def _literal(value, T):
    return {"U32": f"{value}", "Nat": f"{value}n", "Float": f"{value}.0", "USD": f"{value}usd",
            "EUR": f"{value}eur"}[T]


def vectors_passed(task, text):
    """How many of the task's 5 vectors the program maps to the formula's value, each substituted
    into the program's call as a literal of the input's declared type (0 for an invalid program
    or one whose inputs are not the task's)."""
    try:
        obj = json.loads(text[:json_end(text)])
        declared = {i["id"]: i["type"] for i in obj["inputs"]}
    except Exception:  # noqa: BLE001
        return 0
    if sorted(declared) != sorted(task["inputs"]) or any(T not in TYPES for T in declared.values()):
        return 0
    passed = 0
    for vec in vectors(task):
        obj["call"] = {x: _literal(vec[x], declared[x]) for x in declared}
        v = reference_check(json.dumps(obj))
        passed += bool(v.valid and v.value == formula_value(task, vec))
    return passed


def make_tasks(n=20, seed=0):
    rng = random.Random(seed)
    return [dict(_task(rng, i), i=i) for i in range(n)]


def prompt_of(task):
    parts = [HEADER]
    for text, obj in SHOTS:
        parts.append(f"\nTask: {text}\n{json.dumps(obj)}\n")
    parts.append(f"\nTask: {task['text']}\n")
    return "".join(parts)


def _inputs(text):
    try:
        obj = json.loads(text[:json_end(text)])
        return {i["id"]: i["type"] for i in obj["inputs"]}
    except Exception:  # noqa: BLE001
        return None


def correct(task, verdict, text):
    """Valid, the expected output value, and exactly the task's input names."""
    ins = _inputs(text) if verdict.valid else None
    return bool(ins is not None and verdict.value == task["expected"] and sorted(ins) == sorted(task["inputs"]))


def faithful(task, verdict, text):
    """Correct and every input keeps the type the task gave it."""
    ins = _inputs(text) if verdict.valid else None
    return correct(task, verdict, text) and ins == dict(zip(task["inputs"], task["types"]))


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans."""
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    n, k = x + y, min(x, y)
    return {"only_first": x, "only_second": y,
            "p": min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def holm(ps):
    """Holm-adjusted p-values for a {name: p} dict."""
    order = sorted(ps, key=ps.get)
    out, run_max = {}, 0.0
    for j, k in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - j) * ps[k]))
        out[k] = run_max
    return out


def _outcome(task, verdict, text):
    """Valid, Correct_1 / Faithful_1 ("correct" / "faithful", v0.2's metrics) and Correct_5 /
    Faithful_5 (also right on all 5 vectors)."""
    c1, f1 = correct(task, verdict, text), faithful(task, verdict, text)
    k = vectors_passed(task, text) if c1 else 0
    return {"valid": verdict.valid, "correct": c1, "faithful": f1, "correct_5": c1 and k == 5,
            "faithful_5": f1 and k == 5, "vectors_passed": k, "error": verdict.error}


class JSONDAGSuite:
    """The typed JSON DAG tasks: a raw few-shot prompt, outcomes judged by correct() / faithful()."""

    def prompt_ids(self, backend, task):
        return backend.prompt_ids(prompt_of(task))

    def outcome(self, task, verdict, text):
        return _outcome(task, verdict, text)


def _counters(backend):
    return getattr(backend, "rewinds", 0), getattr(backend, "replayed", 0)


def _mask_s(cart):
    return getattr(cart, "mask_seconds", 0.0)  # time in the forward mask's accept() / candidates()


GREEDY_ARMS = ("greedy", "greedy_masked")
_OUTCOME_KEYS = ("valid", "correct", "faithful", "correct_5", "faithful_5", "vectors_passed", "error")


def greedy_arm(cart):
    return "greedy_masked" if getattr(cart, "apply_forward_mask", False) else "greedy"


def _decode_key(cart):
    """What constrains a cartridge's greedy decode: the forward mask, and with it the task contract."""
    mask = bool(getattr(cart, "apply_forward_mask", False))
    return mask, mask and getattr(cart, "pin_inputs", None) is True


def run(backend, regimes, tasks, bon=None, k_max=3, max_new_tokens=320, temperature=0.7, log=print,
        suite=None, share_greedy=False, loop_kw=None, full=None, k_full=2):
    """Paired benchmark.

    regimes: {name: (cartridge, policy)}. Each regime decodes greedy with its own cartridge's
    decoding constraints (so its latency includes it), then repairs. Greedy arms: "greedy" and
    "greedy_masked", one per forward-mask setting, from the first regime with that setting
    (`same_greedy` records whether later regimes reproduced it). Each cartridge is bound to the
    task (bind_task) before the task is decoded; with the mask, a contract-pinned cartridge decodes
    the task's stated input types, so the masked greedy arm carries the same forward constraints as
    the masked repair arm. With share_greedy, later regimes
    with the same mask setting reuse that decode (tokens, logits and a copy of its KV cache)
    instead of re-decoding, and their latency counts the shared decode's time; this assumes those
    regimes' cartridges constrain decoding alike (backend.fork_cache must be supported). A regime
    that reuses the decode records same_greedy=True by construction and shared_greedy=True.
    bon: {arm: regime}, a Best-of-N arm decoded with that regime's cartridge and matched per
    problem to its extra tokens (default: {"best_of_n": first regime}). When the regime spent
    nothing (greedy valid), the arm is greedy. suite: prompts and outcomes (default: the typed
    JSON DAG tasks). loop_kw: extra RepairLoop arguments (e.g. shadow=True). full: {arm: regime},
    a best_of_k arm (k_full full samples, no token budget) with that regime's cartridge, skipped
    (the greedy arm) on tasks whose greedy completion passes the regime's oracle. Best-of arms record
    each sample as complete, truncated (cut off by the budget) or capped (cut off at max_new_tokens).
    """
    suite = suite or JSONDAGSuite()
    loops = {k: RepairLoop(backend, c, policy=p, k_max=k_max, max_new_tokens=max_new_tokens, **(loop_kw or {}))
             for k, (c, p) in regimes.items()}
    bon = {"best_of_n": next(iter(regimes))} if bon is None else bon
    gname = lambda k: greedy_arm(regimes[k][0])
    rows = []
    for task in tasks:
        P = suite.prompt_ids(backend, task)
        for cart, _ in regimes.values():
            cart.bind_task(task)
        row = {"i": task["i"], "task": task["text"], "expected": task["expected"], "seconds": {},
               "wall": [round(time.time(), 1), None]}  # epoch start / end, to match logged pauses
        greedy_toks, shared = {}, {}
        for name, loop in loops.items():
            cart = regimes[name][0]
            arm = gname(name)
            t0, reused = time.time(), arm in shared
            if reused:
                toks, z, cache, t_greedy, m_greedy, key = shared[arm]
                assert key == _decode_key(cart), f"{name} decodes differently from {arm}'s first regime"
                g = (toks, z, backend.fork_cache(cache))
            else:
                m0 = _mask_s(cart)
                g = loop.greedy(P)
                t_greedy, m_greedy = time.time() - t0, _mask_s(cart) - m0
                if share_greedy:
                    shared[arm] = (g[0], g[1], backend.fork_cache(g[2]), t_greedy, m_greedy, _decode_key(cart))
                t0 += t_greedy  # t0 now marks the start of repair
            if arm not in greedy_toks:
                greedy_toks[arm] = g[0]
                text = backend.decode(g[0])
                row[arm] = suite.outcome(task, cart.check(g[0]), text) | {
                    "tokens": len(g[0]), "text": text, "seconds": round(t_greedy, 3), "mask_s": m_greedy,
                    "contract_ok": cart.contract_ok(g[0])}
            c0, m0 = _counters(backend), _mask_s(cart)
            res = loop.repair(P, greedy=g)
            t_total = t_greedy + time.time() - t0
            c1 = _counters(backend)
            row[name] = suite.outcome(task, res.verdict, res.text) | {
                "rounds": res.rounds, "extra_tokens": res.extra_tokens, "history": res.history,
                "text": res.text, "same_greedy": g[0] == greedy_toks[arm], "shared_greedy": reused,
                "rewinds": c1[0] - c0[0], "replayed": c1[1] - c0[1], "mask_s": m_greedy + _mask_s(cart) - m0}
            row["seconds"][name] = {"greedy": round(t_greedy, 3), "total": round(t_total, 3)}
        for arm, reg in bon.items():
            g_arm = row[gname(reg)]
            budget = row[reg]["extra_tokens"]
            if (g_arm["valid"] and g_arm.get("contract_ok", True)) or budget == 0:  # the regime's oracle
                row[arm] = {k: g_arm[k] for k in g_arm if k in _OUTCOME_KEYS} | {"samples": 0, "tokens": 0,
                                                                              "sample_kinds": []}
                continue
            t0, m0 = time.time(), _mask_s(regimes[reg][0])
            (toks, v), n, spent = loops[reg].best_of_n(P, budget, temperature=temperature, seed=task["i"])
            text = backend.decode(toks)
            row[arm] = suite.outcome(task, v, text) | {"samples": n, "tokens": spent, "text": text,
                                                        "sample_kinds": loops[reg].last_samples,
                                                        "seconds": round(time.time() - t0, 3),
                                                        "mask_s": _mask_s(regimes[reg][0]) - m0}
        for arm, reg in (full or {}).items():
            g_arm = row[gname(reg)]
            if g_arm["valid"] and g_arm.get("contract_ok", True):
                row[arm] = {k: g_arm[k] for k in g_arm if k in _OUTCOME_KEYS} | {"samples": 0, "tokens": 0,
                                                                              "sample_kinds": []}
                continue
            t0 = time.time()
            (toks, v), n, spent = loops[reg].best_of_k(P, k_full, temperature=temperature, seed=task["i"])
            text = backend.decode(toks)
            row[arm] = suite.outcome(task, v, text) | {"samples": n, "tokens": spent, "text": text,
                                                        "sample_kinds": loops[reg].last_samples,
                                                        "seconds": round(time.time() - t0, 3)}
        row["wall"][1] = round(time.time(), 1)
        rows.append(row)
        mark = lambda r: ("F" if r.get("faithful_5") else "f" if r["faithful"] else "C" if r["correct"]
                          else "v" if r["valid"] else "x")  # f: Faithful_1 only
        arms = sorted(greedy_toks) + list(loops) + list(bon) + list(full or {})
        log(f"[{task['i']:2d}] " + " | ".join(
            f"{k} {mark(row[k])}" + (f" r={row[k]['rounds']} +{row[k]['extra_tokens']}" if k in loops else "")
            for k in arms))
    return rows, summarize(rows, list(regimes), list(bon) + list(full or {}), backend)


def summarize(rows, regimes, bons=("best_of_n",), backend=None):
    n = len(rows)
    arms = [k for k in GREEDY_ARMS if k in rows[0]] + list(bons) + list(regimes)
    get = lambda k, f: [r[k][f] for r in rows]
    s = {"n": n, "arms": arms}
    for k in arms:
        s[k] = {f: sum(r[k].get(f, False) for r in rows) for f in ("valid", "correct", "faithful", "correct_5", "faithful_5")}
        s[k]["faithful_wilson95"] = wilson(s[k]["faithful"], n)
        s[k]["faithful_5_wilson95"] = wilson(s[k]["faithful_5"], n)
    for k in regimes:
        sec = [r["seconds"][k]["total"] for r in rows]
        rw, rp = sum(r[k].get("rewinds", 0) for r in rows), sum(r[k].get("replayed", 0) for r in rows)
        s[k].update(extra_tokens_total=sum(get(k, "extra_tokens")),
                    latency_mean_s=sum(sec) / n, latency_max_s=max(sec),
                    same_greedy=all(get(k, "same_greedy")), rewinds=rw, replayed_tokens=rp,
                    replayed_per_rewind=rp / rw if rw else None)
        sh = [h["shadow"] for r in rows for h in r[k]["history"] if "shadow" in h]
        if sh:  # DeltaNet vs symbolic provenance, at every diagnosed round
            main = "deltanet_s" if "deltanet_s" in sh[0] else "symbolic_s"
            s[k]["shadow"] = {"rounds": len(sh), "same_diagnosis": sum(h["same_diagnosis"] for h in sh),
                              "same_edit": sum(h["same_edit"] for h in sh),
                              "deltanet_ms": 1e3 * sum(h["deltanet_s"] for h in sh) / len(sh),
                              "symbolic_ms": 1e3 * sum(h["symbolic_s"] for h in sh) / len(sh), "main": main[:-2]}
    for k in bons:
        s[k]["tokens_total"] = sum(get(k, "tokens"))
        kinds = [r[k].get("sample_kinds", []) for r in rows]
        sampled = [x for x in kinds if x]
        s[k]["samples"] = {"tasks_sampled": len(sampled), "total": sum(map(len, kinds)),
                           "complete": sum(x.count("complete") for x in kinds),
                           "truncated": sum(x.count("truncated") for x in kinds),
                           "capped": sum(x.count("capped") for x in kinds),
                           "complete_per_sampled_task": (sum(x.count("complete") for x in sampled) / len(sampled)
                                                         if sampled else None),
                           "truncated_per_sampled_task": (sum(x.count("truncated") for x in sampled) / len(sampled)
                                                          if sampled else None),
                           "tasks_with_no_complete_sample": sum("complete" not in x for x in sampled)}
    for g in GREEDY_ARMS:
        if g in rows[0]:
            s[g]["latency_mean_s"] = sum(r[g]["seconds"] for r in rows) / n
    if backend is not None and backend.peak_memory() is not None:
        s["peak_mps_allocated_gb"] = backend.peak_memory() / 1e9  # PyTorch MPS tensors
    if backend is not None and hasattr(backend, "peak_rss"):
        s["peak_rss_gb"] = backend.peak_rss() / 1e9  # process RSS (cpu side, mapped pages included)
    if getattr(backend, "hybrid", False):
        s["ckpt_peak_mib"] = backend.ckpt_bytes_peak / 2**20
    return s


def compare(rows, a, b, f="faithful"):
    return mcnemar([r[a][f] for r in rows], [r[b][f] for r in rows])


def device_info():
    return {"torch": torch.__version__, "mps": torch.backends.mps.is_available()}
