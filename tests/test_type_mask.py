"""TypeAwareMask (v0.5): the type-aware forward mask.

1. Type-correct random documents (random contracts) are accepted at every prefix and end done; the
   reference checker accepts them; the fallback counts are the free slots (one out_type per node,
   the output, each input the contract does not name).
2. One type error (a wrong-type ref, convert or literal, a call literal of the wrong type, an input
   type against the contract) is rejected, while SchemaPrefix (v0.4's mask) still accepts it.
3. Typed acceptance is a subset of SchemaPrefix's on random (also malformed) documents.
4. No position is left with zero admissible characters: guided random walks through the mask's own
   admissible set; every live, unfinished state has a next character, and every walk that finishes
   is accepted by the reference checker (no wrong-type literal or reference was admissible).
5. Token level (Gemma 4 E2B vocabulary, cpu): every character the mask admits is a single
   non-special token, so a non-empty character set is a non-empty token set; and the decoding
   filter (backends.base.pick with the cartridge's ForwardMask) never decodes "16u32", "16usd" or a
   wrong-type reference, even when the logits put them on top. "16u32" is already outside
   SchemaPrefix; "16usd" in a U32 slot and the wrong-type reference are what is new.
"""

import copy
import json
import random
import re

import pytest
import torch

from ttso.cartridges.type_mask import TypeAwareMask, TypeMaskedJSONDAGCartridge
from ttso.cartridges.typed_json_dag import CONVERT, OPS, TYPES, SchemaState, TypedJSONDAGCartridge, reference_check

GEMMA = ("google/gemma-4-E2B", "433813a669639fd7eaa83155828e9000ed357d53")
FORM = {"U32": "{}", "Nat": "{}n", "Float": "{}.5", "USD": "{}usd", "EUR": "{}eur"}
CONV = {f: (TYPES[s], TYPES[d]) for f, (s, d) in CONVERT.items()}
NAMES = ["a", "b", "stock", "x1", "_q", "level", "total", "lv", "price", "boxes"]


def _typed_doc(rng):
    """(document, contract): a type-correct workflow with random whitespace; the contract names a
    random subset of the inputs with their declared types."""
    ws = lambda: rng.choice(["", "", " ", "\n", "  "])
    kv = lambda k, v: ws() + f'"{k}"' + ws() + ":" + ws() + v
    pool = rng.sample(NAMES, len(NAMES))
    ins = [(pool.pop(), rng.choice(TYPES)) for _ in range(rng.randint(1, 3))]
    decl, nodes, parts = list(ins), [], []
    for x, T in ins:
        parts.append(ws() + "{" + kv("id", f'"{x}"') + ws() + "," + kv("type", f'"{T}"') + ws() + "}")
    for _ in range(rng.randint(1, 3)):
        x, T, args = pool.pop(), rng.choice(TYPES), []
        refs = [y for y, U in decl if U == T]
        convs = [(f, [y for y, U in decl if U == s]) for f, (s, d) in CONV.items() if d == T]
        convs = [(f, ys) for f, ys in convs if ys]
        for _ in range(rng.randint(1, 3)):
            k = rng.choice(["ref"] * bool(refs) + ["lit"] + ["convert"] * bool(convs))
            if k == "ref":
                v = f'"{rng.choice(refs)}"'
            elif k == "lit":
                v = '"' + FORM[T].format(rng.randint(0, 99)) + '"'
            else:
                f, ys = rng.choice(convs)
                v = f'"{f}({rng.choice(ys)})"'
            args.append(ws() + "{" + kv(k, v) + ws() + "}")
        nodes.append(ws() + "{" + kv("id", f'"{x}"') + ws() + "," + kv("op", f'"{rng.choice(OPS)}"') + ws() + ","
                     + kv("out_type", f'"{T}"') + ws() + "," + kv("args", "[" + ",".join(args) + ws() + "]") + ws() + "}")
        decl.append((x, T))
    order = ins[:]
    rng.shuffle(order)
    call = "{" + ",".join(ws() + f'"{x}"' + ws() + ":" + ws() + '"' + FORM[T].format(rng.randint(0, 99)) + '"'
                          for x, T in order) + ws() + "}"
    out = rng.choice([x for x, _ in decl[len(ins):]])
    doc = (ws() + "{" + kv("inputs", "[" + ",".join(parts) + ws() + "]") + ws() + "," + kv("nodes", "[" + ",".join(nodes)
           + ws() + "]") + ws() + "," + kv("output", f'"{out}"') + ws() + "," + kv("call", call) + ws() + "}")
    contract = {x: T for x, T in ins if rng.random() < 0.7}
    return doc, contract, len(ins) - len(contract), len(decl) - len(ins)


def test_typed_documents_accepted():
    rng = random.Random(0)
    n = 0
    for _ in range(300):
        d, c, unnamed, n_nodes = _typed_doc(rng)
        st = TypeAwareMask(c)
        for i, ch in enumerate(d):
            st = st.advance(ch)
            assert st is not None, (d[:i + 1], c)
            n += 1
        assert st.done, d
        assert reference_check(d).valid, d
        assert st.fallbacks == {"input_type": unnamed, "out_type": n_nodes, "output": 1}
    print(f"\n[type mask] 300 type-correct documents, {n} prefixes accepted; fallbacks = free slots")


def _other(T):
    return TYPES[(TYPES.index(T) + 1) % len(TYPES)]


def _mutations(d, c):
    """One-type-error variants of a type-correct document, as canonical JSON (each one is inside
    the schema, so SchemaPrefix accepts it)."""
    obj = json.loads(d)
    types = {i["id"]: i["type"] for i in obj["inputs"]} | {n["id"]: n["out_type"] for n in obj["nodes"]}
    order = list(types)
    digits = lambda v: re.match(r"\d+", v).group(0)
    out = []
    for j, i in enumerate(obj["inputs"]):  # an input type against the contract
        if i["id"] in c:
            out.append(_set(obj, ("inputs", j, "type"), _other(i["type"])))
    for j, n in enumerate(obj["nodes"]):
        T, before = n["out_type"], order[:order.index(n["id"])]
        for a_j, a in enumerate(n["args"]):
            at = ("nodes", j, "args", a_j)
            if "ref" in a:  # a ref to a declared id of another type
                out += [_set(obj, at, {"ref": y}) for y in before if types[y] != T][:1]
            elif "lit" in a:  # a literal of another type
                out.append(_set(obj, at, {"lit": FORM[_other(T)].format(digits(a["lit"]))}))
            else:  # a converter applied to an id that is not of its source type
                f = a["convert"].split("(")[0]
                out += [_set(obj, at, {"convert": f"{f}({z})"}) for z in before if types[z] != CONV[f][0]][:1]
    for x, v in obj["call"].items():  # a call literal of another type ("16usd" for a U32 input, ...)
        out.append(_set(obj, ("call", x), FORM[_other(types[x])].format(digits(v))))
        if types[x] == "U32":
            out.append(_set(obj, ("call", x), f"{digits(v)}usd"))
    return out


def _set(obj, path, value):
    o = copy.deepcopy(obj)
    t = o
    for k in path[:-1]:
        t = t[k]
    t[path[-1]] = value
    return json.dumps(o)


def test_one_type_error_rejected():
    rng = random.Random(1)
    n = 0
    for _ in range(300):
        d, c, _, _ = _typed_doc(rng)
        d = json.dumps(json.loads(d))
        for m in _mutations(d, c):
            assert not reference_check(m).valid, m  # a real type error
            assert SchemaState().advance(m) is not None, m  # inside the schema: only the types are wrong
            assert TypeAwareMask(c).advance(m) is None, (m, c)
            n += 1
    assert n > 300
    print(f"\n[type mask] {n} one-type-error documents: SchemaPrefix accepts, TypeAwareMask rejects")


def test_typed_subset_of_schema():
    from tests.test_incremental_schema import _doc
    rng = random.Random(2)
    n = 0
    for _ in range(200):
        d = _doc(rng)
        c = {} if rng.random() < 0.5 else {x: rng.choice(TYPES) for x in ("a", "b", "stock", "level")}
        ty, sc = TypeAwareMask(c), SchemaState()
        for ch in d:
            if ty is not None:
                assert sc is not None
                assert ty.next_chars() <= sc.next_chars()
                n += 1
            ty = None if ty is None else ty.advance(ch)
            sc = None if sc is None else sc.advance(ch)
    print(f"\n[type mask] {n} prefixes of 200 random documents: typed next-character set within SchemaPrefix's")


def _walk(rng, contract, cap=4000):
    """A random walk through the mask's admissible characters, biased to finish (short names, little
    whitespace). -> (text, final state, every next-character set seen)."""
    st, text, seen = TypeAwareMask(contract), "", []
    for _ in range(cap):
        if st.done:
            break
        nxt = sorted(st.next_chars())
        assert nxt, f"empty admissible set after {text!r}"
        seen.append(frozenset(nxt))
        w = [0.02 if ch in " \n" else 6.0 if ch in '"}]),' and len(st.buf) >= 1 else 1.0 for ch in nxt]
        ch = rng.choices(nxt, weights=w)[0]
        st, text = st.advance(ch), text + ch
        assert st is not None
    return text, st, seen


def test_never_empty_random_walks():
    rng = random.Random(3)
    done = 0
    for j in range(400):
        names = rng.sample(NAMES, rng.randint(0, 3))
        contract = {x: rng.choice(TYPES) for x in names}
        text, st, _ = _walk(rng, contract)
        if st.done:
            done += 1
            v = reference_check(text)
            assert v.valid, (text, v.error)
    assert done > 200
    print(f"\n[type mask] 400 random walks, none reached an empty admissible set; {done} finished, all valid")


@pytest.fixture(scope="module")
def gemma():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA[0], revision=GEMMA[1])
    texts = tok.batch_decode([[i] for i in range(len(tok))])
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return tok, texts, enc


def test_admitted_characters_are_tokens(gemma):
    tok, texts, _ = gemma
    single = {t for i, t in enumerate(texts) if len(t) == 1 and i not in set(tok.all_special_ids)}
    rng = random.Random(4)
    chars = set()
    for _ in range(200):
        contract = {x: rng.choice(TYPES) for x in rng.sample(NAMES, rng.randint(0, 3))}
        for s in _walk(rng, contract)[2]:
            chars |= s
    missing = chars - single
    print(f"\n[type mask] {len(chars)} distinct admitted characters, all single Gemma tokens: {not missing}")
    assert not missing, missing


PREFIX = ('{"inputs": [{"id": "price", "type": "Nat"}, {"id": "stock", "type": "U32"}, {"id": "boxes", "type": "U32"}], '
          '"nodes": [{"id": "total", "op": "add", "out_type": "U32", "args": [')
TASK = {"i": 0, "inputs": ["price", "stock", "boxes"], "types": ["Nat", "U32", "U32"]}


def _decode_next(cart, enc, texts, prefix, top):
    """The token pick() decodes after `prefix` when the logits put tokens whose text starts with one
    of `top` highest (each with its own score, in order), under the cartridge's forward mask."""
    from ttso.backends.base import bind, pick
    from ttso.engine.repair_loop import ForwardMask
    head = enc(prefix)
    z = torch.zeros(len(texts))
    for r, s in enumerate(top):
        for i, t in enumerate(texts):
            if t and t.startswith(s) and z[i] == 0:
                z[i] = 100.0 - r
    return texts[pick(cart.logits_processor(head, z), 0.0, None, bind(ForwardMask(cart, head), []))]


def test_forced_bad_tokens_cannot_be_decoded(gemma):
    tok, texts, enc = gemma
    sp = tok.all_special_ids
    typed = TypeMaskedJSONDAGCartridge(texts, encode=enc, special_ids=sp)
    schema = TypedJSONDAGCartridge(texts, encode=enc, special_ids=sp, apply_forward_mask=True)
    for c in (typed, schema):
        c.bind_task(TASK)
    call = PREFIX + '{"convert": "to_u32(price)"}, {"ref": "stock"}, {"ref": "boxes"}]}], "output": "total", "call": {"price": "17n", "stock": "16'
    got_t = _decode_next(typed, enc, texts, call, ["u32", "usd", "u", "n"])
    got_s = _decode_next(schema, enc, texts, call, ["u32", "usd", "u", "n"])
    print(f"\n[type mask] after '\"stock\": \"16', logits on u32 > usd > u > n: SchemaPrefix decodes {got_s!r}, "
          f"TypeAwareMask {got_t!r}")
    assert not got_t[:1] in ("u", "n") and (got_t[:1] == '"' or got_t[:1].isdigit()), got_t
    assert got_s[:1] == "u" and not got_s.startswith("u3")  # v0.4: "16u32" never, "16usd" yes
    ref = PREFIX + '{"ref": "'
    got_t = _decode_next(typed, enc, texts, ref, ["price", "pr", "p"])
    got_s = _decode_next(schema, enc, texts, ref, ["price", "pr", "p"])
    print(f"[type mask] at a U32 node's ref, logits on the Nat input price: SchemaPrefix decodes {got_s!r}, "
          f"TypeAwareMask {got_t!r}")
    assert got_s.startswith("p") and not got_t.startswith("p") and got_t[:1] in "sb", got_t
    lit = PREFIX + '{"lit": "3'
    got_t = _decode_next(typed, enc, texts, lit, ["n", "usd", "u", "eur", "."])
    assert got_t[:1] == '"' or got_t[:1].isdigit(), got_t
