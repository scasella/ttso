"""5-vector functional scoring (v0.4) of the typed JSON DAG tasks: the canonical program of every dev
and held-out task is right on all 5 vectors; a wrong formula that happens to give the stated
vector's value passes Faithful_1 and fails Faithful_5; the vectors cover both sides of max / min."""

import json

from ttso.cartridges.typed_json_dag import reference_check
from ttso.engine import benchmark as b

LIT = {"U32": "{}", "Nat": "{}n", "USD": "{}usd"}


def program(task, formula=None):
    """The canonical faithful program for a task (or one computing `formula` instead: a node list)."""
    a, bb, *c = task["inputs"]
    ins = [{"id": x, "type": T} for x, T in zip(task["inputs"], task["types"])]
    nodes = formula or [
        [{"id": "total", "op": "add", "out_type": "U32",
          "args": [{"convert": f"to_u32({a})"}, {"ref": bb}, {"ref": c[0] if c else a}]}],
        [{"id": "total", "op": "add", "out_type": "USD",
          "args": [{"ref": a}, {"ref": bb}, {"lit": f"{task['const']}usd"}]}],
        [{"id": "best", "op": "max", "out_type": "Nat", "args": [{"ref": x} for x in task["inputs"]]}],
        [{"id": "sum", "op": "add", "out_type": "U32", "args": [{"ref": a}, {"convert": f"to_u32({bb})"}]},
         {"id": "low", "op": "min", "out_type": "U32", "args": [{"ref": "sum"}, {"ref": c[0] if c else a}]}],
    ][task["kind"]]
    call = {x: LIT[T].format(v) for x, T, v in zip(task["inputs"], task["types"], task["values"])}
    return json.dumps({"inputs": ins, "nodes": nodes, "output": nodes[-1]["id"], "call": call})


def outcome(task, text):
    return b._outcome(task, reference_check(text), text)


def test_canonical_programs_pass_all_five():
    for task in b.make_tasks(20, seed=0) + b.make_tasks(60, seed=42):
        o = outcome(task, program(task))
        assert o["faithful"] and o["faithful_5"] and o["vectors_passed"] == 5, (task["text"], o)


def test_vectors_are_varied_and_cover_both_branches():
    for task in b.make_tasks(20, seed=0) + b.make_tasks(60, seed=42):
        V = b.vectors(task)
        assert len(V) == 5 and b.formula_value(task, V[0]) == task["expected"]
        for v in V[1:]:
            xs = list(v.values())
            assert len(set(xs)) == len(xs) and min(xs) >= 2 and not set(xs) & set(task["values"])
        if task["kind"] == 2:  # each input is the largest somewhere
            assert {max(v, key=v.get) for v in V[1:]} == set(task["inputs"])
        if task["kind"] == 3:  # min(a + b, c) takes each side
            a, bb, c = task["inputs"]
            assert {v[c] < v[a] + v[bb] for v in V[1:]} == {True, False}


def test_coincidences_fail_faithful_5():
    """For each task, programs computing a wrong formula: whenever one matches the stated vector
    (a coincidence Faithful_1 accepts), Faithful_5 rejects it."""
    seen = 0
    for task in b.make_tasks(20, seed=0) + b.make_tasks(60, seed=42):
        a, bb, *c = task["inputs"]
        T = {"U32": "U32", "Nat": "Nat", "USD": "USD"}[task["types"][0]] if task["kind"] != 0 else "U32"
        wrong = []
        if task["kind"] == 2:  # one input only
            wrong = [[{"id": "best", "op": "max", "out_type": "Nat", "args": [{"ref": x}]}] for x in task["inputs"]]
        elif task["kind"] == 3:  # the sum alone, or c alone
            wrong = [[{"id": "sum", "op": "add", "out_type": "U32", "args": [{"ref": a}, {"convert": f"to_u32({bb})"}]}],
                     [{"id": "low", "op": "min", "out_type": "U32", "args": [{"ref": c[0]}]}]]
        for nodes in wrong:
            o = outcome(task, program(task, nodes))
            assert not o["faithful_5"], (task["text"], nodes)
            seen += o["faithful"]
    assert seen > 0  # some wrong formula did match its stated vector
