"""Cartridge 2: stateful typed JSON tool-call DAGs.

    {"inputs": [{"id": "stock", "type": "Nat"}, {"id": "level", "type": "U32"}],
     "nodes": [{"id": "total", "op": "add", "out_type": "U32",
                "args": [{"convert": "to_u32(stock)"}, {"ref": "level"}, {"lit": "5"}]}],
     "output": "total",
     "call": {"stock": "6n", "level": "38"}}

A JSON Schema / GBNF grammar can force this shape and force each type to be one of the five names,
but not that a reference agrees with a declaration made 200 tokens earlier. Types: U32, Nat, Float,
USD, EUR. Literals: "7" U32, "7n" Nat, "7.5" Float, "7usd" USD, "7eur" EUR. Ops add / max / min take
arguments of the node's out_type. Conversions: to_u32 (Nat -> U32), to_nat (U32 -> Nat),
to_float (U32 -> Float).

Ops compiled into the DeltaNet (slot = input or node id):
    input type / node out_type      DECLARE(id)                     at the type's first token (decl)
    {"ref": "x"} in node n          CHECK_VAR(x, out_type(n))       at x                  "ref arg"
    {"lit": "5n"} in node n         CHECK_CALL(n)                   at the suffix token   "lit arg"
    {"convert": "to_u32(x)"} in n   CHECK_VAR(x, Nat)               at x                  "convert arg"
                                    CHECK_VAR(n, U32)               at the converter      "convert result"
    "call": {"x": "6n"}             CHECK_CALL(x)                   at the suffix token   "call arg"
The suffix token is the token right after a literal's digits (the closing quote for a plain U32).
With pin_inputs=True the input types are frozen (the prompt's contract); a failed ref / convert
check then falls back to banning the arg's key token ("ref" / "convert"), so the model can rewrite
the argument as a conversion (LocalOnly).
Hard errors (no gradient): malformed JSON or schema, unknown ids or ops, bad literals, unbound or
unknown call inputs, an out_type given after its args.
"""

import json
import re

import torch

from ttso.cartridges.base import Cartridge, Lexed, TextAligner, Verdict, as_list
from ttso.core.deltanet import CHECK_CALL, CHECK_VAR, DECLARE, Op
from ttso.core.token_classes import TokenClassManager

TYPES = ("U32", "Nat", "Float", "USD", "EUR")
U32, NAT, FLOAT, USD, EUR = range(5)
OPS = ("add", "max", "min")
CONVERT = {"to_u32": (NAT, U32), "to_nat": (U32, NAT), "to_float": (U32, FLOAT)}
LIT = re.compile(r"(\d+)(n|\.\d+|usd|eur)?")
LIT_TYPE = {None: U32, "n": NAT, "usd": USD, "eur": EUR}
SUFFIX_OK = re.compile(r'(?:n|\.\d*|usd?|eur)?(?:"[,}\]\s]*)?')
TYPE_SLOT = re.compile(r'"(?:type|out_type)":\s*"$')


def lit_type(s):
    m = LIT.fullmatch(s)
    if not m:
        return None
    return FLOAT if (m.group(2) or "").startswith(".") else LIT_TYPE[m.group(2)]


def _lit_rule(text):
    c = text[:1]
    return {"n": NAT, ".": FLOAT, "u": USD, "e": EUR}.get(c)


class S(str):
    """A JSON string value that remembers the offset of its first content character."""

    def __new__(cls, value, off):
        o = super().__new__(cls, value)
        o.off = off
        return o


class _Err(Exception):
    def __init__(self, off, why):
        self.off, self.why = off, why


class _P:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def expect(self, c):
        self.ws()
        if self.i >= len(self.s):
            raise _Err(len(self.s), "truncated")
        if self.s[self.i] != c:
            raise _Err(self.i, f"expected {c!r}")
        self.i += 1

    def value(self):
        self.ws()
        if self.i >= len(self.s):
            raise _Err(len(self.s), "truncated")
        c = self.s[self.i]
        if c == "{":
            self.i += 1
            out, keys = {}, []
            self.ws()
            if self.s[self.i:self.i + 1] == "}":
                self.i += 1
                return out
            while True:
                self.ws()
                k = self.string()
                self.expect(":")
                out[str(k)] = self.value()
                keys.append(k)
                self.ws()
                if self.s[self.i:self.i + 1] == ",":
                    self.i += 1
                    continue
                self.expect("}")
                out["__keys__"], out["__end__"] = keys, S("}", self.i - 1)
                return out
        if c == "[":
            self.i += 1
            out = []
            self.ws()
            if self.s[self.i:self.i + 1] == "]":
                self.i += 1
                return out
            while True:
                out.append(self.value())
                self.ws()
                if self.s[self.i:self.i + 1] == ",":
                    self.i += 1
                    continue
                self.expect("]")
                return out
        if c == '"':
            return self.string()
        m = re.compile(r"-?\d+(?:\.\d+)?|true|false|null").match(self.s, self.i)
        if not m:
            raise _Err(self.i, "bad value")
        self.i = m.end()
        return json.loads(m.group(0))

    def string(self):
        if self.i >= len(self.s):
            raise _Err(len(self.s), "truncated")
        if self.s[self.i] != '"':
            raise _Err(self.i, "expected a string")
        j = self.i + 1
        while j < len(self.s) and self.s[j] != '"':
            j += 2 if self.s[j] == "\\" else 1
        if j >= len(self.s):
            raise _Err(len(self.s), "truncated")
        v = S(json.loads(self.s[self.i:j + 1]), self.i + 1)
        self.i = j + 1
        return v


def parse_json(text):
    """-> (object or None, (offset, why) or None). Text after the first complete value is ignored."""
    p = _P(text)
    try:
        return p.value(), None
    except _Err as e:
        return None, (e.off, e.why)


def json_end(text):
    """Offset just past the first complete top-level JSON object, or None."""
    depth, in_str, esc, started = 0, False, False, False
    for i, c in enumerate(text):
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth, started = depth + 1, True
        elif c in "}]":
            depth -= 1
            if started and depth == 0:
                return i + 1
    return None


class TypedJSONDAGCartridge(Cartridge):
    type_names = list(TYPES)

    def __init__(self, token_texts, encode=None, special_ids=(), n_slots=64, pin_inputs=False):
        self.texts = token_texts
        self.pin_inputs = pin_inputs  # input types are the prompt's contract: never edit them
        self.n_slots = n_slots
        self._tcm = TokenClassManager(token_texts, TYPES, lit_rule=_lit_rule, lit_default=U32,
                                      special_ids=special_ids, encode=encode)
        self._lit_ok = torch.tensor([bool(t) and SUFFIX_OK.fullmatch(t) is not None for t in token_texts])
        self._lit_ok &= ~self._tcm.special
        self._g_type = self._tcm.G_type(k=None)
        bare = torch.tensor([t == t.lstrip(' \t\n"') for t in token_texts])
        self._g_type[~bare] = float("-inf")

    @property
    def tcm(self):
        return self._tcm

    def text(self, tokens):
        return "".join(self.texts[i] for i in as_list(tokens))

    def stop(self, tokens):
        return json_end(self.text(tokens)) is not None

    def logits_processor(self, tokens, z):
        """Schema-level constrained decoding: a type slot admits only the five type names."""
        tail = "".join(self.texts[i] for i in as_list(tokens)[-6:])
        return z + self._g_type if TYPE_SLOT.search(tail) else z

    def valid_mask(self, tokens, t, site):
        if site == "decl":
            return self.tcm.type_mask(like=int(tokens[t]))
        if site == "lit":
            return self._lit_ok.clone()
        return ~self.tcm.special

    # --- lexer -> DeltaNet ops ---------------------------------------------------------------
    def lex(self, tokens):
        al = TextAligner([self.texts[i] for i in as_list(tokens)])
        lx = Lexed()
        last = max(len(al.texts) - 1, 0)
        tok = lambda off: min(al.tok(off), last)
        obj, err = parse_json(al.text)
        if err is not None:
            lx.hard.append((tok(err[0]), f"json: {err[1]}"))
            return lx
        slot = lambda name: lx.slots.setdefault(str(name), len(lx.slots) % self.n_slots)
        ty = lambda s: TYPES.index(s) if isinstance(s, str) and s in TYPES else None
        hard = lambda s, why: lx.hard.append((tok(s.off) if isinstance(s, S) else last, why))
        if not isinstance(obj, dict):
            hard(None, "not an object")
            return lx
        declared = set()
        for inp in obj.get("inputs", []) if isinstance(obj.get("inputs"), list) else []:
            if not isinstance(inp, dict) or not isinstance(inp.get("id"), S) or not isinstance(inp.get("type"), S):
                hard(None, "bad input")
                continue
            self._declare(lx, tok, slot, inp["id"], inp["type"], ty, hard)
            if self.pin_inputs:
                lx.frozen.add(tok(inp["type"].off))
            declared.add(str(inp["id"]))
        for node in obj.get("nodes", []) if isinstance(obj.get("nodes"), list) else []:
            if not isinstance(node, dict) or not isinstance(node.get("id"), S):
                hard(None, "bad node")
                continue
            keys = node.get("__keys__", [])
            T = ty(node.get("out_type"))
            if isinstance(node.get("out_type"), S):
                self._declare(lx, tok, slot, node["id"], node["out_type"], ty, hard)
            else:
                hard(node["id"], "node without out_type")
            if "args" in keys and "out_type" in keys and keys.index("args") < keys.index("out_type"):
                hard(node["id"], "out_type must precede args")
            if node.get("op") not in OPS:
                hard(node.get("op") if isinstance(node.get("op"), S) else node["id"], "unknown op")
            if str(node["id"]) in declared:
                hard(node["id"], f"duplicate id {str(node['id'])!r}")
            for a in node.get("args", []) if isinstance(node.get("args"), list) else []:
                self._arg(lx, tok, slot, node, T, a, declared, hard)
            declared.add(str(node["id"]))
        call = obj.get("call")
        if isinstance(call, dict):
            for name in call.get("__keys__", []):
                v = call[name]
                if name not in {str(i.get("id")) for i in obj.get("inputs", []) if isinstance(i, dict)}:
                    hard(v, f"unknown call input {name!r}")
                    continue
                self._literal(lx, tok, slot(name), v, hard, "call arg")
            names = [str(i.get("id")) for i in obj.get("inputs", []) if isinstance(i, dict)]
            if any(x not in call for x in names):
                hard(call["__end__"], "unbound input")
        else:
            hard(None, "no call")
        lx.ops.sort(key=lambda o: o.t)
        return lx

    def _declare(self, lx, tok, slot, name, tstr, ty, hard):
        t = tok(tstr.off)
        lx.ops.append(Op(t, DECLARE, slot(name), tag="declare"))
        lx.sites[t] = "decl"
        if ty(tstr) is None:
            hard(tstr, f"bad type {str(tstr)!r}")

    def _literal(self, lx, tok, s, v, hard, tag):
        if not isinstance(v, S) or lit_type(v) is None:
            hard(v, "bad literal")
            return
        m = LIT.fullmatch(v)
        t = tok(v.off + m.end(1))
        lx.ops.append(Op(t, CHECK_CALL, s, tag=tag))
        lx.sites[t] = "lit"

    def _arg(self, lx, tok, slot, node, T, a, declared, hard):
        keys = a.get("__keys__", []) if isinstance(a, dict) else []
        a = {k: v for k, v in a.items() if k not in ("__keys__", "__end__")} if isinstance(a, dict) else a
        if len(keys) != 1:
            hard(keys[1] if len(keys) > 1 else node["id"], "bad arg")
            return
        t_key = tok(keys[0].off)
        if "ref" in a:
            x = a["ref"]
            if str(x) not in declared:
                hard(x, f"unknown ref {str(x)!r}")
            elif T is not None:
                lx.ops.append(Op(tok(x.off), CHECK_VAR, slot(x), T, tag="ref arg"))
                lx.local[tok(x.off)] = t_key
        elif "lit" in a:
            self._literal(lx, tok, slot(node["id"]), a["lit"], hard, "lit arg")
        elif "convert" in a:
            c = a["convert"]
            m = re.fullmatch(r"(\w+)\((\w+)\)", c) if isinstance(c, S) else None
            if not m or m.group(1) not in CONVERT:
                hard(c, "bad convert")
                return
            src, dst = CONVERT[m.group(1)]
            if m.group(2) not in declared:
                hard(c, f"unknown ref {m.group(2)!r}")
                return
            lx.ops.append(Op(tok(c.off + m.start(2)), CHECK_VAR, slot(m.group(2)), src, tag="convert arg"))
            lx.ops.append(Op(tok(c.off), CHECK_VAR, slot(node["id"]), dst, tag="convert result"))
            lx.local[tok(c.off + m.start(2))] = lx.local[tok(c.off)] = t_key
        else:
            hard(keys[0], "bad arg")

    # --- reference checker --------------------------------------------------------------------------
    def check(self, tokens):
        return reference_check(self.text(tokens))


def reference_check(text):
    """Pure-Python checker and evaluator. value = the output node's value under the call."""
    end = json_end(text)
    try:
        obj = json.loads(text[:end] if end else text)
    except Exception as e:  # noqa: BLE001
        return Verdict(False, f"json: {e.__class__.__name__}")
    try:
        inputs = {i["id"]: i["type"] for i in obj["inputs"]}
        call = obj["call"]
        nodes = obj["nodes"]
        out = obj["output"]
    except Exception:  # noqa: BLE001
        return Verdict(False, "schema")
    if any(T not in TYPES for T in inputs.values()):
        return Verdict(False, "bad input type")
    if set(call) != set(inputs):
        return Verdict(False, "call must bind every input exactly")
    env_t, env_v = dict(inputs), {}
    for x, v in call.items():
        if not isinstance(v, str) or lit_type(v) is None:
            return Verdict(False, f"bad literal {v!r}")
        if TYPES[lit_type(v)] != inputs[x]:
            return Verdict(False, f"call: {x} is {inputs[x]}, literal {v!r} is {TYPES[lit_type(v)]}")
        env_v[x] = _lit_value(v)
    for n in nodes:
        try:
            nid, op, T, args = n["id"], n["op"], n["out_type"], n["args"]
        except Exception:  # noqa: BLE001
            return Verdict(False, "schema")
        if T not in TYPES or op not in OPS or not args or nid in env_t:
            return Verdict(False, f"node {nid}: bad node")
        vals = []
        for a in args:
            if not isinstance(a, dict) or len(a) != 1:
                return Verdict(False, f"node {nid}: bad arg")
            (k, v), = a.items()
            if k == "ref":
                if env_t.get(v) != T:
                    return Verdict(False, f"node {nid}: ref {v} is {env_t.get(v)}, node is {T}")
                vals.append(env_v[v])
            elif k == "lit":
                if not isinstance(v, str) or lit_type(v) is None or TYPES[lit_type(v)] != T:
                    return Verdict(False, f"node {nid}: literal {v!r} is not {T}")
                vals.append(_lit_value(v))
            elif k == "convert":
                m = re.fullmatch(r"(\w+)\((\w+)\)", v) if isinstance(v, str) else None
                if not m or m.group(1) not in CONVERT:
                    return Verdict(False, f"node {nid}: bad convert")
                src, dst = CONVERT[m.group(1)]
                if env_t.get(m.group(2)) != TYPES[src] or TYPES[dst] != T:
                    return Verdict(False, f"node {nid}: {m.group(1)}({m.group(2)}) in a {T} node")
                vals.append(env_v[m.group(2)])
            else:
                return Verdict(False, f"node {nid}: bad arg")
        env_t[nid] = T
        env_v[nid] = {"add": sum, "max": max, "min": min}[op](vals)
    if out not in env_v or out in inputs:
        return Verdict(False, "bad output")
    return Verdict(True, "", env_v[out])


def _lit_value(v):
    m = LIT.fullmatch(v)
    return float(m.group(1) + m.group(2)) if (m.group(2) or "").startswith(".") else int(m.group(1))
