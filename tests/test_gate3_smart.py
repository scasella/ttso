"""Gate 3: Problem 14 (catalog synthesis, untrained Qwen3.5-0.8B, DiffRepair v2.1 run of 2 Nov 2026).

The language model is a ScriptedBackend: a weighted mixture of whole programs, so regeneration after
an edit is deterministic and the cached logits rank the alternatives the scripts contain.

1. The spec's account: after round 1 the model regenerates `catalog(6n, 38, 47)`. Smart flips
   `+stock: Nat` -> `U32` (Case B, declaration), then sees stock already used consistently in the
   body (consistent_uses > 0) and edits the call site, 6n -> 6: valid in 2 rounds.
2. The recorded run: after round 1 the model actually regenerated `catalog(6, 38n, 47)`, so round 2
   is about `level` and `38n`. Smart edits the call site (38n -> 38, valid); v1 Upstream (flagged
   declaration first, the latest flagged token once that declaration was tried) flips `level` to
   Nat, breaks the body, and ends invalid -- the split recorded in the study.
3. On the spec's account, v1 Upstream also repairs the program: stock's declaration was tried in
   round 1, so round 2 edits the latest flagged token, the call site. The Smart/Upstream split on
   problem 14 therefore depends on the trajectory the model regenerated, not on routing alone.
"""

import pytest
import torch

from ttso.backends.scripted import ScriptedBackend
from ttso.cartridges.bendlite import BendLiteCatalogCartridge
from ttso.engine.repair_loop import RepairLoop

PROMPT = "Write a BendLite catalog program.\n"
HEAD = "import Base\n\ndef catalog(+stock: {s}, +level: {l}, +qty: U32) -> U32:\n  (stock + qty + level : U32)\n\n"
MAIN = "def main() -> U32:\n  catalog({a})\n"
GREEDY = HEAD.format(s="Nat", l="U32") + MAIN.format(a="6n, 38, 47")
SPEC_R1 = HEAD.format(s="U32", l="U32") + MAIN.format(a="6n, 38, 47")
REC_R1 = HEAD.format(s="U32", l="U32") + MAIN.format(a="6, 38n, 47")
REC_UP2 = HEAD.format(s="U32", l="Nat") + MAIN.format(a="6, 38n, 47")
FINAL = HEAD.format(s="U32", l="U32") + MAIN.format(a="6, 38, 47")


@pytest.fixture(scope="module")
def setup(qwen_tokenizer, qwen_texts):
    enc = lambda s: qwen_tokenizer(s, add_special_tokens=False)["input_ids"]
    cart = BendLiteCatalogCartridge(qwen_texts, encode=enc, special_ids=qwen_tokenizer.all_special_ids)
    return qwen_tokenizer, qwen_texts, enc, cart


def run(setup, scripts, policy):
    tok, texts, enc, cart = setup
    P = enc(PROMPT)
    be = ScriptedBackend(tok, [(w, P + enc(s)) for w, s in scripts], texts=texts)
    loop = RepairLoop(be, cart, policy=policy, k_max=3, max_new_tokens=200)
    res = loop.repair(P)
    return res, be


def show(name, res):
    print(f"\n[gate3] {name}: valid={res.verdict.valid} rounds={res.rounds} extra={res.extra_tokens}")
    for h in res.history:
        print(f"   round {h['round']}: first failure t={h['t_error']} ({h['why']}), flagged {h['flagged']}, "
              f"uses={h['consistent_uses']} -> case {h.get('case')} {h.get('route')} t*={h.get('t_star')} "
              f"{h.get('old')!r}->{h.get('new')!r} valid_after={h.get('valid_after')}")


SPEC = [(0.55, GREEDY), (0.30, SPEC_R1), (0.15, FINAL)]
RECORDED = [(0.55, GREEDY), (0.25, REC_R1), (0.12, FINAL), (0.08, REC_UP2)]


def test_reference_and_verifier_agree(setup):
    _, _, enc, cart = setup
    net = cart.compile()
    for src, ok in ((GREEDY, False), (SPEC_R1, False), (REC_R1, False), (REC_UP2, False), (FINAL, True)):
        ids = enc(src)
        v = cart.check(ids)
        assert v.valid == ok, (src, v)
        lx = cart.lex(ids)
        tr = net(ids, lx.ops)
        assert (tr.first_failure is None and not lx.hard) == ok
    assert cart.check(enc(FINAL)).value == 6 + 38 + 47


def test_smart_spec_account(setup):
    res, be = run(setup, SPEC, "smart")
    show("Smart, spec account", res)
    r1, r2 = res.history
    assert r1["why"] == "group operand" and r1["case"] == "B" and r1["route"] == "decl"
    assert (r1["old"], r1["new"]) == (" Nat", " U")  # +stock: Nat -> U32
    assert r2["why"] == "call arg" and r2["consistent_uses"] > 0
    assert r2["case"] == "A" and r2["route"] == "call" and r2["old"] == "n"  # 6n -> 6
    assert res.verdict.valid and res.rounds == 2 and "catalog(6, 38, 47)" in res.text
    assert be.off_script == 0


def test_recorded_trajectory_smart_vs_upstream(setup):
    smart, _ = run(setup, RECORDED, "smart")
    show("Smart, recorded trajectory", smart)
    assert smart.verdict.valid and smart.rounds == 2 and "catalog(6, 38, 47)" in smart.text
    s2 = smart.history[1]
    assert s2["route"] == "call" and s2["consistent_uses"] > 0 and s2["old"] == "n"  # 38n -> 38

    up, _ = run(setup, RECORDED, "upstream")
    show("Upstream (v1), recorded trajectory", up)
    u1, u2 = up.history[:2]
    assert (u1["old"], u1["new"]) == (" Nat", " U")  # same first edit as Smart
    assert u2["t_star"] == u2["t_decl"] and (u2["old"], u2["new"]) == (" U", " Nat")  # +level: U32 -> Nat
    assert not up.verdict.valid and up.rounds == 3


def test_upstream_spec_account_also_repairs(setup):
    up, _ = run(setup, SPEC, "upstream")
    show("Upstream (v1), spec account", up)
    assert up.verdict.valid and up.rounds == 2 and up.history[1]["route"] != "decl"
