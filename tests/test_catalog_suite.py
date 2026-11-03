"""The v0.3 BendLite catalog suite and BendLite's pin_inputs (CPU).

The held-out task texts are frozen by hash (runs/v0.3/PREREGISTRATION.md), as are the JSON DAG
held-out tasks; the reference programs are faithful; the outcome definitions separate a retyped
program (correct, not faithful) and a literal in the def (valid, not correct); pinning freezes the
parameter declarations and routes a failed group operand to a local rewrite.
"""

import hashlib
import json

from ttso.cartridges.bendlite import BendLiteCatalogCartridge, reference_check
from ttso.engine import benchmark, catalog_suite as cs

HASH = {"catalog_heldout": "603a59085d75fffc88dce77144a599eb09a30179565540e7fa44a4c963a26bfb",
        "catalog_dev": "8b44503105aa605c02a6eb7e4b865fd64e31e0d9a3efa206973710cb7ac19863",
        "json_heldout": "0ef5a0d84e941461d61fb2cf0dae2b2c707ca5ff141be30849811ce5479b7e3c",
        "json_dev": "add6ef18154bc5b70d265013e7ddc00d233bf151e363d7b928d2ce0677ed8b16"}


def _h(tasks):
    return hashlib.sha256(json.dumps([t["text"] for t in tasks]).encode()).hexdigest()


def test_frozen_suites():
    dev, held = cs.dev_tasks(), cs.heldout_tasks()
    assert (len(dev), len(held)) == (20, 50)
    assert not {t["text"] for t in dev} & {t["text"] for t in held}
    assert _h(held) == HASH["catalog_heldout"] and _h(dev) == HASH["catalog_dev"]
    assert _h(benchmark.make_tasks(60, seed=42)) == HASH["json_heldout"]
    assert _h(benchmark.make_tasks(20, seed=0)) == HASH["json_dev"]
    for t in dev + held:
        assert 2 <= len(t["items"]) <= 4 and "Nat" in t["types"] and "U32" in t["types"]
        o = cs.CatalogSuite().outcome(t, reference_check(t["reference"]), t["reference"])
        assert o == {"valid": True, "correct": True, "faithful": True, "error": ""}, t


def test_outcomes_retyped_and_literal():
    t = {"items": ["stock", "level"], "types": ["Nat", "U32"], "values": [6, 38], "expected": 44}
    out = lambda p: cs.CatalogSuite().outcome(t, reference_check(p), p)
    head = "import Base\n\ndef quote({}) -> U32:\n  ({} : U32)\n\ndef main() -> U32:\n  quote({})\n"
    faithful = head.format("stock: Nat, level: U32", "U32.from_nat(stock) + level", "6n, 38")
    retyped = head.format("stock: U32, level: U32", "stock + level", "6, 38")
    literal = head.format("stock: U32, level: U32", "stock + 38", "6, 38")
    assert out(faithful) == {"valid": True, "correct": True, "faithful": True, "error": ""}
    assert out(retyped)["correct"] and not out(retyped)["faithful"]
    assert out(literal)["valid"] and not out(literal)["correct"]


def test_bendlite_pinned_lexing(qwen_tokenizer, qwen_texts):
    enc = lambda s: qwen_tokenizer(s, add_special_tokens=False)["input_ids"]
    prog = "import Base\n\ndef quote(stock: Nat, level: U32) -> U32:\n  (stock + U32.from_nat(level) : U32)\n\ndef main() -> U32:\n  quote(6n, 38)\n"
    toks = enc(prog)
    texts = [qwen_texts[i] for i in toks]
    pinned = BendLiteCatalogCartridge(qwen_texts, encode=enc, special_ids=qwen_tokenizer.all_special_ids, pin_inputs=True)
    free = BendLiteCatalogCartridge(qwen_texts, encode=enc, special_ids=qwen_tokenizer.all_special_ids)
    lp, lf = pinned.lex(toks), free.lex(toks)
    params = [o.t for o in lp.ops if o.tag == "param"]
    assert set(params) == lp.frozen and len(params) == 2 and not lf.frozen
    assert lp.ops == lf.ops
    fn = next(o for o in lp.ops if o.tag == "from_nat arg")
    assert texts[lp.local[fn.t]].strip() == "U"  # a failed from_nat arg rewrites the whole operand
    assert fn.t not in lf.local
