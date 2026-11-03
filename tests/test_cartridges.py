"""Cartridge and token-class edge cases."""

import json

import pytest

from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.core.token_classes import TokenClassManager


def test_two_types_sharing_a_first_token_are_rejected():
    vocab = ["U", "32", "SD", "Nat", " "]
    enc = {"U32": [0, 1], "USD": [0, 2], "Nat": [3]}
    encode = lambda s: enc.get(s.lstrip(' :"'), [4])
    with pytest.raises(AssertionError, match="U32 and USD share a first token"):
        TokenClassManager(vocab, ["U32", "Nat", "USD"], encode=encode)


def test_json_dag_never_aliases_slots():
    chars = sorted(set('{}[]":, _-0123456789abcdefghijklmnopqrstuvwxyzUNFSDE'))
    cart = TypedJSONDAGCartridge(chars, n_slots=4)
    doc = {"inputs": [{"id": f"x{i}", "type": "U32"} for i in range(6)],
           "nodes": [{"id": "s", "op": "add", "out_type": "U32", "args": [{"ref": "x0"}]}],
           "output": "s", "call": {f"x{i}": "1" for i in range(6)}}
    ids = [chars.index(c) for c in json.dumps(doc)]
    lx = cart.lex(ids)
    assert all(o.slot < 4 for o in lx.ops)
    assert any("more than 4 ids" in why for _, why in lx.hard)
