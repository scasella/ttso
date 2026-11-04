"""Does CompiledDeltaNet backward routing ever decide differently from a plain symbolic provenance
table (SymbolicProvenanceVerifier) under the same routing rules? (v0.4)

For every case, both verifiers diagnose the same tokens and cached logits, and Router.choose()
picks an edit from each (fresh routing state): Smart (Cases A, B, C), Upstream, LocalOnly (memory
reads detached) and Blind. The diagnosis compared is (kind, t_error, slot, t_decl, t_call, flagged,
consistent_uses); the edit is (t*, y*, case, route).
  Gate 1   200 random SynthLang programs (K = 8, <= 300 tokens), LLM-like random logits; also every
           check's verdict, consistent-use count and declaration.
  Gate 2   the 200 single-faulty-declaration programs (K = 5).
  Gate 3   Problem 14's scripted repairs (BendLite), with the symbolic verifier as a shadow at
           every round of Smart and Upstream.
  Dev      the 20 seed-0 JSON DAG tasks on Qwen3-0.6B-Base, full repair runs with the shadow on,
           for contract-pinned Smart (unmasked and masked), self-pinned Smart, unpinned Smart,
           LocalOnly and Upstream.
Each test also reports the mean CPU time of a diagnosis (lexing excluded) for each verifier.
"""

import json
import time

import pytest
import torch

from tests.test_gate2_direction import llm_like_logits, make_case
from ttso.cartridges.synthetic import SynthLang, rng_for
from ttso.core.deltanet import first_failure_gradients
from ttso.core.routing import RepairState, Router, diagnose
from ttso.core.symbolic import SymbolicProvenanceVerifier

POLICIES = ("smart", "upstream", "local", "blind")
KEY = lambda d: (d.kind, d.t_error, d.slot, d.t_decl, d.t_call, tuple(d.flagged), d.consistent_uses)
EDIT = lambda e: None if e is None else (e.t, e.new, e.case, e.route)


def _compare(lang, cases):
    """cases: (tokens, logits). Returns (#diagnoses, #edits compared, mismatches, ms deltanet, ms symbolic)."""
    net, sym = lang.compile("cpu"), SymbolicProvenanceVerifier.for_cartridge(lang)
    routers = {p: Router(lang, p) for p in POLICIES}
    n_diag = n_edit = 0
    bad, t_net, t_sym = [], 0.0, 0.0
    for i, (prog, z) in enumerate(cases):
        lx = lang.lex(prog)
        for detach in (False, True):
            t0 = time.perf_counter()
            d1 = diagnose(net, lang, prog, z, detach_state=detach, lexed=lx)
            t1 = time.perf_counter()
            d2 = sym.diagnose(lang, prog, detach_state=detach, lexed=lx)
            t_net, t_sym = t_net + t1 - t0, t_sym + time.perf_counter() - t1
            n_diag += 1
            if KEY(d1) != KEY(d2):
                bad.append((i, "diagnosis", detach, KEY(d1), KEY(d2)))
                continue
            for p, r in routers.items():
                if r.detach_state != detach or d1.kind == "valid":
                    continue
                a = EDIT(r.choose(diagnose(net, lang, prog, z, detach_state=detach, lexed=lx), prog, z, RepairState()))
                b = EDIT(r.choose(sym.diagnose(lang, prog, detach_state=detach, lexed=lx), prog, z, RepairState()))
                n_edit += 1
                if a != b:
                    bad.append((i, p, a, b))
    return n_diag, n_edit, bad, 1e3 * t_net / n_diag, 1e3 * t_sym / n_diag


def test_gate1_programs_verdicts_and_decisions():
    lang = SynthLang(n_types=8, n_vars=64)
    rng, gen = rng_for(1), torch.Generator().manual_seed(1)
    progs = [lang.random_program(rng, max_len=300) for _ in range(200)]
    net, sym = lang.compile("cpu"), SymbolicProvenanceVerifier.for_cartridge(lang)
    for prog in progs:  # every check: verdict, consistent uses and the declaration it read
        ops = lang.lex(prog).ops
        tr = net(prog, ops)
        sc = sym.run(prog, ops)
        assert [(c.op.t, c.passed, c.uses_before, tr.decl_of[j]) for j, c in enumerate(tr.checks)] == \
               [(c.t, c.passed, c.uses_before, c.decl_t) for c in sc]
    cases = [(p, llm_like_logits(lang, p, gen)) for p in progs]
    n_diag, n_edit, bad, ms_net, ms_sym = _compare(lang, cases)
    print(f"\n[symbolic] gate 1: {n_diag} diagnoses, {n_edit} edits; {len(bad)} differ. "
          f"Diagnosis {ms_net:.2f} ms (DeltaNet) vs {ms_sym:.3f} ms (symbolic)")
    assert not bad, bad[:5]


def test_gate2_cases_decisions():
    lang = SynthLang(n_types=5, n_vars=64)
    rng, gen = rng_for(2), torch.Generator().manual_seed(2)
    cases = []
    for _ in range(200):
        prog, *_ = make_case(lang, rng)
        cases.append((prog, llm_like_logits(lang, prog, gen)))
    n_diag, n_edit, bad, ms_net, ms_sym = _compare(lang, cases)
    print(f"\n[symbolic] gate 2: {n_diag} diagnoses, {n_edit} edits; {len(bad)} differ. "
          f"Diagnosis {ms_net:.2f} ms vs {ms_sym:.3f} ms")
    assert not bad, bad[:5]


def test_gate3_scripted_repairs_in_shadow(qwen_tokenizer, qwen_texts):
    from tests import test_gate3_smart as g3
    from ttso.backends.scripted import ScriptedBackend
    from ttso.cartridges.bendlite import BendLiteCatalogCartridge
    from ttso.engine.repair_loop import RepairLoop
    enc = lambda s: qwen_tokenizer(s, add_special_tokens=False)["input_ids"]
    cart = BendLiteCatalogCartridge(qwen_texts, encode=enc, special_ids=qwen_tokenizer.all_special_ids)
    P = enc(g3.PROMPT)
    scripts = {"spec": [(5.0, g3.GREEDY), (3.0, g3.SPEC_R1), (1.0, g3.FINAL)],
               "recorded": [(5.0, g3.GREEDY), (3.0, g3.REC_R1), (2.0, g3.REC_UP2), (1.0, g3.FINAL)]}
    rounds = 0
    for name, sc in scripts.items():
        for policy in ("smart", "upstream", "local"):
            be = ScriptedBackend(qwen_tokenizer, [(w, P + enc(s)) for w, s in sc], texts=qwen_texts)
            res = RepairLoop(be, cart, policy=policy, k_max=3, max_new_tokens=200, shadow=True).repair(P)
            for h in res.history:
                if "shadow" in h:
                    rounds += 1
                    assert h["shadow"]["same_diagnosis"] and h["shadow"]["same_edit"], (name, policy, h)
    print(f"\n[symbolic] gate 3: {rounds} repair rounds, all identical")
    assert rounds >= 6


def test_dev_tasks_full_repairs_in_shadow(hf_run):
    from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
    from ttso.engine import benchmark
    *_, run_dir, be = hf_run
    mk = lambda **k: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode,
                                           special_ids=be.tok.all_special_ids, **k)
    regimes = {"contract_smart": (mk(), "smart"), "contract_smart_masked": (mk(apply_forward_mask=True), "smart"),
               "self_smart": (mk(pin_inputs="self"), "smart"), "unpinned_smart": (mk(pin_inputs=False), "smart"),
               "local": (mk(pin_inputs=False), "local"), "upstream": (mk(pin_inputs=False), "upstream")}
    rows, s = benchmark.run(be, regimes, benchmark.make_tasks(20, seed=0), bon={}, loop_kw={"shadow": True},
                            log=lambda *_: None)
    stats = {k: s[k].get("shadow", {"rounds": 0, "same_diagnosis": 0, "same_edit": 0}) for k in regimes}
    (run_dir / "symbolic_shadow.json").write_text(json.dumps(stats, indent=1))
    diff = [(r["i"], k, h["round"], h["shadow"]) for r in rows for k in regimes for h in r[k]["history"]
            if "shadow" in h and not (h["shadow"]["same_diagnosis"] and h["shadow"]["same_edit"])]
    total = sum(v["rounds"] for v in stats.values())
    print(f"\n[symbolic] dev tasks: {total} repair rounds across {len(regimes)} regimes; {len(diff)} differ")
    for k, v in stats.items():
        if v["rounds"]:
            print(f"  {k}: {v['same_edit']}/{v['rounds']} same edit; diagnosis {v['deltanet_ms']:.1f} ms "
                  f"(DeltaNet) vs {v['symbolic_ms']:.2f} ms (symbolic)")
    assert total > 0 and not diff, diff[:3]
