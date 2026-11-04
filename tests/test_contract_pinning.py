"""Contract pinning (v0.4, pin_inputs=True with a task bound).

Unmasked, a wrong input type the model writes is caught: the lexer checks each named input's type
token against the task's stated type, the check fails first, the declaration is flagged and the
gradient edit rewrites it to the stated type. Masked, the wrong type is never decoded: the type
slot of a named input admits only the stated type, through its closing quote. pin_inputs="self"
(v0.2 / v0.3) leaves a wrong but consistent input type alone. Model-free tests use Qwen3's
tokenizer with flat logits; the forced-case test runs Qwen3-0.6B-Base on the seed-0 dev tasks.
"""

import json

import pytest
import torch

from ttso.cartridges.typed_json_dag import TYPES, TypedJSONDAGCartridge
from ttso.core.routing import RepairState, Router, diagnose
from ttso.engine import benchmark

TASK = {"inputs": ["price", "qty"], "types": ["Nat", "U32"]}
WRONG = ('{"inputs": [{"id": "price", "type": "U32"}, {"id": "qty", "type": "U32"}], "nodes": [{"id": "t", '
         '"op": "add", "out_type": "U32", "args": [{"ref": "price"}, {"ref": "qty"}]}], "output": "t", '
         '"call": {"price": "6", "qty": "38"}}')


def _carts(tok, texts, **kw):
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    mk = lambda **k: TypedJSONDAGCartridge(texts, encode=enc, special_ids=tok.all_special_ids, **k, **kw)
    return enc, mk


def _allowed(cart, enc, text):
    z = cart.logits_processor(enc(text), torch.zeros(len(cart.texts)))
    ok = torch.isfinite(z)
    return {TYPES[k] for k in range(len(TYPES)) if (ok & (cart.tcm.type_of == k)).any()}


def test_masked_contract_holds_the_stated_type(qwen_tokenizer, qwen_texts):
    enc, mk = _carts(qwen_tokenizer, qwen_texts)
    masked, unmasked, selfpin = mk(apply_forward_mask=True), mk(), mk(pin_inputs="self", apply_forward_mask=True)
    for c in (masked, unmasked, selfpin):
        c.bind_task(TASK)
    assert selfpin.contract is None and masked.contract == unmasked.contract == {"price": 1, "qty": 0}
    head = '{"inputs": [{"id": "price", "type": "'
    assert _allowed(masked, enc, head) == {"Nat"}
    assert _allowed(unmasked, enc, head) == _allowed(selfpin, enc, head) == set(TYPES)  # caught later, not prevented
    second = '{"inputs": [{"id": "price", "type": "Nat"}, {"id": "qty", "type": "'
    assert _allowed(masked, enc, second) == {"U32"}
    ok = torch.isfinite(masked.logits_processor(enc(second + "U"), torch.zeros(len(qwen_texts))))
    allowed = {qwen_texts[i] for i in torch.nonzero(ok).flatten().tolist()}
    assert allowed and all("32".startswith(t) or t.startswith('32"') for t in allowed)  # never "SD"
    unnamed = '{"inputs": [{"id": "cost", "type": "'
    assert _allowed(masked, enc, unnamed) == set(TYPES)
    node = second + 'U32"}], "nodes": [{"id": "t", "op": "add", "out_type": "'
    assert _allowed(masked, enc, node) == set(TYPES)


def test_unmasked_contract_catches_and_repairs_a_wrong_input_type(qwen_tokenizer, qwen_texts):
    enc, mk = _carts(qwen_tokenizer, qwen_texts)
    contract, selfpin = mk(), mk(pin_inputs="self")
    for c in (contract, selfpin):
        c.bind_task(TASK)
    toks = enc(WRONG)
    assert contract.check(toks).valid and selfpin.check(toks).valid  # Valid is the plain checker in both
    assert not contract.contract_ok(toks) and selfpin.contract_ok(toks)
    lx = contract.lex(toks)
    chk = [o for o in lx.ops if o.tag == "input contract"]
    assert len(chk) == 2 and chk[0].expected == TYPES.index("Nat")
    assert chk[0].t not in lx.frozen and chk[1].t in lx.frozen  # wrong type editable, right one frozen
    assert selfpin.lex(toks).frozen == {chk[0].t, chk[1].t}
    z = torch.zeros(len(toks), len(qwen_texts))
    d = diagnose(contract.compile(), contract, toks, z)
    assert d.kind == "check" and d.why == "input contract" and d.t_error == chk[0].t and d.t_decl == chk[0].t
    e = Router(contract, "smart").choose(d, toks, z, RepairState())
    assert e.t == chk[0].t and e.case == "B" and int(contract.tcm.type_of[e.new]) == TYPES.index("Nat")
    assert diagnose(selfpin.compile(), selfpin, toks, z).kind == "valid"  # self-pinning never sees it


def _consistent_wrong(task):
    """A dev task's faithful program with its Nat input declared U32 and used as one (valid, not
    faithful): the case v0.3's self-pinning froze."""
    a, b, c = task["inputs"]
    ins = [{"id": a, "type": "U32"}, {"id": b, "type": "U32"}, {"id": c, "type": "U32"}]
    nodes = [{"id": "total", "op": "add", "out_type": "U32", "args": [{"ref": a}, {"ref": b}, {"ref": c}]}]
    call = {x: str(v) for x, v in zip(task["inputs"], task["values"])}
    return json.dumps({"inputs": ins, "nodes": nodes, "output": "total", "call": call})


@pytest.fixture(scope="module")
def qwen_base():
    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple mps")
    from tests.gate4_setup import MODEL
    from ttso.backends.hf_mps import HuggingFaceMPSBackend
    return HuggingFaceMPSBackend(MODEL, device="mps", dtype=torch.float16)


def _teacher_force(be, P, toks):
    """(toks, cached logits, KV cache) as if `toks` had been decoded after prompt P."""
    cache = be._new_cache()
    with torch.inference_mode():
        o = be.model(input_ids=torch.tensor([P + toks[:-1]], device=be.device), past_key_values=cache, use_cache=True)
    return toks, o.logits[0, len(P) - 1:].float().cpu(), o.past_key_values


def test_forced_wrong_input_type_on_dev_tasks(qwen_base):
    """Seed-0 dev tasks with a Nat input (kind 0): from the consistent wrong program, contract repair
    (unmasked) rewrites the input type to Nat in round 1 and keeps it; self-pinning stops at once,
    unfaithful. Whether the regenerated rest of the program is right is up to the model."""
    from ttso.engine.repair_loop import RepairLoop
    be = qwen_base
    mk = lambda **k: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=be.tok.all_special_ids, **k)
    contract, selfpin = mk(), mk(pin_inputs="self")
    tasks = [t for t in benchmark.make_tasks(20, seed=0) if t["kind"] == 0]
    fixed = 0
    for task in tasks:
        P = be.prompt_ids(benchmark.prompt_of(task))
        toks = be.encode(_consistent_wrong(task))
        for c in (contract, selfpin):
            c.bind_task(task)
        r_self = RepairLoop(be, selfpin, "smart", max_new_tokens=320).repair(P, greedy=_teacher_force(be, P, toks))
        assert r_self.rounds == 0 and r_self.verdict.valid
        assert not benchmark.faithful(task, r_self.verdict, r_self.text)
        r = RepairLoop(be, contract, "smart", max_new_tokens=320).repair(P, greedy=_teacher_force(be, P, toks))
        h = r.history[0]
        assert h["why"] == "input contract" and h["case"] == "B" and h["new"].strip() == "Nat", h
        fixed += f'{{"id": "{task["inputs"][0]}", "type": "Nat"}}' in r.text  # kept through later rounds
        print(f"\n[contract] task {task['i']}: round 1 {h['old']!r} -> {h['new']!r}; after {r.rounds} rounds "
              f"valid={r.verdict.valid} faithful={benchmark.faithful(task, r.verdict, r.text)}")
    assert fixed == len(tasks) == 5
