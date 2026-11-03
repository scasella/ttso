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
from math import comb

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
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    n, k = x + y, min(x, y)
    return {"only_first": x, "only_second": y,
            "p": min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0}


def run(backend, regimes, tasks, k_max=3, max_new_tokens=320, temperature=0.7, log=print):
    """regimes: {name: (cartridge, policy)}. The first regime sets Best-of-N's token budget. Every
    regime re-decodes the greedy completion, so its latency includes it; `same_greedy` records
    whether it reproduced the first regime's (decoding on mps is expected to be deterministic)."""
    loops = {k: RepairLoop(backend, c, policy=p, k_max=k_max, max_new_tokens=max_new_tokens)
             for k, (c, p) in regimes.items()}
    first = next(iter(regimes))
    cart = regimes[first][0]
    rows = []
    for task in tasks:
        P = backend.encode(prompt_of(task))
        row = {"i": task["i"], "task": task["text"], "expected": task["expected"], "seconds": {}}
        g0 = None
        for name, loop in loops.items():
            t0 = time.time()
            g = loop.greedy(P)
            t_greedy = time.time() - t0
            if g0 is None:
                g0 = g[0]
                text = backend.decode(g[0])
                v = cart.check(g[0])
                row["greedy"] = {"valid": v.valid, "correct": correct(task, v, text), "faithful": faithful(task, v, text),
                                 "error": v.error, "tokens": len(g[0]), "text": text}
            res = loop.repair(P, greedy=g)
            t_total = time.time() - t0
            row[name] = {"valid": res.verdict.valid, "correct": correct(task, res.verdict, res.text),
                         "faithful": faithful(task, res.verdict, res.text), "error": res.verdict.error,
                         "rounds": res.rounds, "extra_tokens": res.extra_tokens, "history": res.history,
                         "text": res.text, "same_greedy": g[0] == g0}
            row["seconds"][name] = {"greedy": round(t_greedy, 3), "total": round(t_total, 3)}
        budget = row[first]["extra_tokens"]
        if row["greedy"]["valid"] or budget == 0:
            bon = {k: row["greedy"][k] for k in ("valid", "correct", "faithful")} | {"samples": 0, "tokens": 0}
        else:
            (toks, v), n, spent = loops[first].best_of_n(P, budget, temperature=temperature, seed=task["i"])
            text = backend.decode(toks)
            bon = {"valid": v.valid, "correct": correct(task, v, text), "faithful": faithful(task, v, text),
                   "samples": n, "tokens": spent, "text": text}
        row["best_of_n"] = bon
        rows.append(row)
        mark = lambda r: "F" if r["faithful"] else ("C" if r["correct"] else ("v" if r["valid"] else "x"))
        log(f"[{task['i']:2d}] greedy {mark(row['greedy'])} | " +
            " | ".join(f"{k} {mark(row[k])} r={row[k]['rounds']} +{row[k]['extra_tokens']} "
                       f"{row['seconds'][k]['total']:.1f}s" for k in loops) +
            f" | BoN {mark(bon)} n={bon['samples']}   {row['greedy']['error']}")
    return rows, summarize(rows, list(loops), backend)


def summarize(rows, regimes, backend=None):
    n = len(rows)
    get = lambda k, f: [r[k][f] for r in rows]
    s = {"n": n, "regimes": regimes}
    for k in ["greedy", "best_of_n"] + regimes:
        s[k] = {f: sum(get(k, f)) for f in ("valid", "correct", "faithful")}
    for k in regimes:
        sec = [r["seconds"][k]["total"] for r in rows]
        s[k].update(extra_tokens_total=sum(get(k, "extra_tokens")),
                    latency_mean_s=sum(sec) / n, latency_max_s=max(sec))
        s[f"mcnemar {k}"] = {f: {"vs greedy": mcnemar(get(k, f), get("greedy", f)),
                                 "vs best_of_n": mcnemar(get(k, f), get("best_of_n", f))}
                             for f in ("correct", "faithful")}
    s["repaired_problems"] = sum(1 for r in rows if not r["greedy"]["valid"])
    s["greedy_reproduced"] = all(r[k]["same_greedy"] for r in rows for k in regimes)
    s["best_of_n"]["tokens_total"] = sum(get("best_of_n", "tokens"))
    s["greedy"]["latency_mean_s"] = sum(r["seconds"][regimes[0]]["greedy"] for r in rows) / n
    if backend is not None and backend.peak_memory() is not None:
        s["peak_mps_allocated_gb"] = backend.peak_memory() / 1e9
    return s


def device_info():
    return {"torch": torch.__version__, "mps": torch.backends.mps.is_available()}
