"""Typed JSON DAG benchmark: few-shot tasks, and a paired Greedy / Best-of-N / ttso comparison.

Each task names typed inputs, the output type and the call values; the model writes the workflow
JSON. A completion is correct when the reference checker accepts it, its output value is the
expected one and it declares exactly the task's inputs; faithful when it is also correct with every
input keeping the type the task gave it. Best-of-N samples whole completions at temperature 0.7
until it has spent the same number of extra tokens as the first ttso regime spent on that problem.
"""

import json
import random
import time
from math import comb, sqrt

import torch

from ttso.cartridges.typed_json_dag import json_end
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
    kind = i % 4
    a, b, c = rng.sample(NAMES, 3)
    va, vb, vc = rng.randint(2, 60), rng.randint(2, 60), rng.randint(2, 60)
    if kind == 0:  # catalog: one Nat item inside a U32 total
        text = (f"Add a Nat {a}, a U32 {b} and a U32 {c} into a U32 total. "
                f"Call with {a} {va}, {b} {vb} and {c} {vc}.")
        return {"text": text, "inputs": [a, b, c], "types": ["Nat", "U32", "U32"], "expected": va + vb + vc}
    if kind == 1:  # currency: USD inputs and a USD constant
        k = rng.randint(1, 9)
        text = f"Add a USD {a}, a USD {b} and a {k} USD charge into a USD total. Call with {a} {va} and {b} {vb}."
        return {"text": text, "inputs": [a, b], "types": ["USD", "USD"], "expected": va + vb + k}
    if kind == 2:  # max over Nat values with a Nat result
        text = f"Take the largest of three Nat values {a}, {b} and {c}, as a Nat. Call with {a} {va}, {b} {vb} and {c} {vc}."
        return {"text": text, "inputs": [a, b, c], "types": ["Nat", "Nat", "Nat"], "expected": max(va, vb, vc)}
    text = (f"Add a U32 {a} and a Nat {b} into a U32 sum, then take the smaller of that sum and a U32 {c}. "
            f"Call with {a} {va}, {b} {vb} and {c} {vc}.")
    return {"text": text, "inputs": [a, b, c], "types": ["U32", "Nat", "U32"], "expected": min(va + vb, vc)}


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
    return {"valid": verdict.valid, "correct": correct(task, verdict, text),
            "faithful": faithful(task, verdict, text), "error": verdict.error}


class JSONDAGSuite:
    """The typed JSON DAG tasks: a raw few-shot prompt, outcomes judged by correct() / faithful()."""

    def prompt_ids(self, backend, task):
        return backend.encode(prompt_of(task))

    def outcome(self, task, verdict, text):
        return _outcome(task, verdict, text)


def _counters(backend):
    return getattr(backend, "rewinds", 0), getattr(backend, "replayed", 0)


def _mask_s(cart):
    return getattr(cart, "mask_seconds", 0.0)  # time in the forward mask's accept() / candidates()


def run(backend, regimes, tasks, bon=None, k_max=3, max_new_tokens=320, temperature=0.7, log=print,
        suite=None, share_greedy=False):
    """Paired benchmark.

    regimes: {name: (cartridge, policy)}. Each regime decodes greedy with its own cartridge's
    decoding constraints (so its latency includes it), then repairs. Greedy arms: "greedy" and
    "greedy_masked", one per forward-mask setting, from the first regime with that setting
    (`same_greedy` records whether later regimes reproduced it). With share_greedy, later regimes
    with the same mask setting reuse that decode (tokens, logits and a copy of its KV cache)
    instead of re-decoding, and their latency counts the shared decode's time; this assumes those
    regimes' cartridges constrain decoding alike (backend.fork_cache must be supported). A regime
    that reuses the decode records same_greedy=True by construction and shared_greedy=True.
    bon: {arm: regime}, a Best-of-N arm decoded with that regime's cartridge and matched per
    problem to its extra tokens (default: {"best_of_n": first regime}). When the regime spent
    nothing (greedy valid), the arm is greedy. suite: prompts and outcomes (default: the typed
    JSON DAG tasks).
    """
    suite = suite or JSONDAGSuite()
    loops = {k: RepairLoop(backend, c, policy=p, k_max=k_max, max_new_tokens=max_new_tokens)
             for k, (c, p) in regimes.items()}
    bon = {"best_of_n": next(iter(regimes))} if bon is None else bon
    gname = lambda k: "greedy_masked" if getattr(regimes[k][0], "apply_forward_mask", False) else "greedy"
    rows = []
    for task in tasks:
        P = suite.prompt_ids(backend, task)
        row = {"i": task["i"], "task": task["text"], "expected": task["expected"], "seconds": {}}
        greedy_toks, shared = {}, {}
        for name, loop in loops.items():
            cart = regimes[name][0]
            arm = gname(name)
            t0, reused = time.time(), arm in shared
            if reused:
                toks, z, cache, t_greedy, m_greedy = shared[arm]
                g = (toks, z, backend.fork_cache(cache))
            else:
                m0 = _mask_s(cart)
                g = loop.greedy(P)
                t_greedy, m_greedy = time.time() - t0, _mask_s(cart) - m0
                if share_greedy:
                    shared[arm] = (g[0], g[1], backend.fork_cache(g[2]), t_greedy, m_greedy)
                t0 += t_greedy  # t0 now marks the start of repair
            if arm not in greedy_toks:
                greedy_toks[arm] = g[0]
                text = backend.decode(g[0])
                row[arm] = suite.outcome(task, cart.check(g[0]), text) | {
                    "tokens": len(g[0]), "text": text, "seconds": round(t_greedy, 3), "mask_s": m_greedy}
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
            if g_arm["valid"] or budget == 0:
                row[arm] = {k: g_arm[k] for k in ("valid", "correct", "faithful", "error")} | {"samples": 0, "tokens": 0}
                continue
            t0, m0 = time.time(), _mask_s(regimes[reg][0])
            (toks, v), n, spent = loops[reg].best_of_n(P, budget, temperature=temperature, seed=task["i"])
            text = backend.decode(toks)
            row[arm] = suite.outcome(task, v, text) | {"samples": n, "tokens": spent, "text": text,
                                                        "seconds": round(time.time() - t0, 3),
                                                        "mask_s": _mask_s(regimes[reg][0]) - m0}
        rows.append(row)
        mark = lambda r: "F" if r["faithful"] else ("C" if r["correct"] else ("v" if r["valid"] else "x"))
        arms = sorted(greedy_toks) + list(loops) + list(bon)
        log(f"[{task['i']:2d}] " + " | ".join(
            f"{k} {mark(row[k])}" + (f" r={row[k]['rounds']} +{row[k]['extra_tokens']}" if k in loops else "")
            for k in arms))
    return rows, summarize(rows, list(regimes), list(bon), backend)


def summarize(rows, regimes, bons=("best_of_n",), backend=None):
    n = len(rows)
    arms = [k for k in ("greedy", "greedy_masked") if k in rows[0]] + list(bons) + list(regimes)
    get = lambda k, f: [r[k][f] for r in rows]
    s = {"n": n, "arms": arms}
    for k in arms:
        s[k] = {f: sum(get(k, f)) for f in ("valid", "correct", "faithful")}
        s[k]["faithful_wilson95"] = wilson(s[k]["faithful"], n)
    for k in regimes:
        sec = [r["seconds"][k]["total"] for r in rows]
        rw, rp = sum(r[k].get("rewinds", 0) for r in rows), sum(r[k].get("replayed", 0) for r in rows)
        s[k].update(extra_tokens_total=sum(get(k, "extra_tokens")),
                    latency_mean_s=sum(sec) / n, latency_max_s=max(sec),
                    same_greedy=all(get(k, "same_greedy")), rewinds=rw, replayed_tokens=rp,
                    replayed_per_rewind=rp / rw if rw else None)
    for k in bons:
        s[k]["tokens_total"] = sum(get(k, "tokens"))
    for g in ("greedy", "greedy_masked"):
        if g in rows[0]:
            s[g]["latency_mean_s"] = sum(r[g]["seconds"] for r in rows) / n
    if backend is not None and backend.peak_memory() is not None:
        s["peak_mps_allocated_gb"] = backend.peak_memory() / 1e9
    if getattr(backend, "hybrid", False):
        s["ckpt_peak_mib"] = backend.ckpt_bytes_peak / 2**20
    return s


def compare(rows, a, b, f="faithful"):
    return mcnemar([r[a][f] for r in rows], [r[b][f] for r in rows])


def device_info():
    return {"torch": torch.__version__, "mps": torch.backends.mps.is_available()}
