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
Input pinning (pin_inputs), v0.4:
  True     (the default) the task's input contract. bind_task(task) gives the cartridge the task's
           stated inputs and types. An input the task names gets, at its type token, a
           CHECK_VAR(id, stated type) right after its DECLARE ("input contract"): a wrong written
           type is the first failed check, its declaration is flagged (Case B) and the gradient
           edit rewrites it to the stated type. A type that matches is frozen. With the forward
           mask the wrong type is never decoded: the type of a named input is held to the stated
           type, through its closing quote (U32 is two tokens, U and 32). contract_ok() is the
           repair loop's extra stopping test; check() stays the plain reference checker, so Valid
           means the same in every arm. With no task bound, as in v0.2 / v0.3: "self".
  "self"   v0.2 / v0.3: freeze each input type as the model wrote it, right or wrong.
  False    the spec's routing: every declaration is editable.
With a frozen input, a failed ref / convert check falls back to banning the arg's key token ("ref" /
"convert"), so the model can rewrite the argument as a conversion (LocalOnly). Input ids are never
constrained, and inputs the task does not name are pinned as written.

apply_forward_mask=True adds a forward structural mask (SchemaPrefix): every decoded token, and every
repair edit, must keep the completion a prefix of the schema. It enforces structure and scope (key
order, literal forms, fresh / declared ids, complete call bindings), never types, so the type
errors are still the verifier's to repair. The mask runs on SchemaState, an incremental form of
SchemaPrefix that advances over each new token's characters, with one state kept per token
position (SchemaTrail), so a rewind restores the state at t* by lookup. mask_impl="reparse" keeps
v0.2's acceptor, which re-parses the whole completion on every query (the equivalence gate's
reference).
Hard errors (no gradient): malformed JSON or schema, unknown ids or ops, bad literals, unbound or
unknown call inputs, an out_type given after its args.
"""

import json
import re
import time

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
INPUT_TYPE_SLOT = re.compile(r'\{\s*"id":\s*"([^"]*)",\s*"type":\s*"(\w*)$')  # an input's type, so far


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


class _More(Exception):
    """The text ended inside a valid prefix."""


class _Bad(Exception):
    """The text left the schema."""


NAME_HEAD = set("abcdefghijklmnopqrstuvwxyz_")
NAME_CHARS = NAME_HEAD | set("0123456789")
ARG_KEYS = ("ref", "lit", "convert")


class SchemaPrefix:
    """Character-level acceptor for the workflow schema (the forward structural mask).

    It fixes the key order (inputs, nodes, output, call; id, type; id, op, out_type, args), one key per
    argument, op and type names, the five literal forms ("7", "7n", "7.5", "7usd", "7eur"), fresh ids
    for new inputs and nodes, refs and conversions only to ids declared earlier, an output naming a
    node, and a call binding every input exactly once. It never looks at types: whether a ref or a
    literal has the right type is left to the verifier. check(text) -> "more" | "done" | "bad".
    """

    def __init__(self, text):
        self.s, self.i = text, 0
        self.inputs, self.nodes = [], []

    def check(self):
        try:
            self._doc()
        except _More:
            return "more"
        except _Bad:
            return "bad"
        return "done" if not self.s[self.i:].strip() else "bad"

    def _c(self):
        if self.i >= len(self.s):
            raise _More
        return self.s[self.i]

    def _ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \n":
            self.i += 1

    def _lit(self, w):
        for ch in w:
            if self._c() != ch:
                raise _Bad
            self.i += 1

    def _key(self, k):
        self._ws()
        self._lit(f'"{k}"')
        self._ws()
        self._lit(":")
        self._ws()

    def _until(self, stop, allowed, charset=None):
        start = self.i
        while self.i < len(self.s) and self.s[self.i] != stop:
            if charset is not None and self.s[self.i] not in (NAME_HEAD if self.i == start else charset):
                raise _Bad
            self.i += 1
        part = self.s[start:self.i]
        if allowed is not None and not any(a.startswith(part) for a in allowed):
            raise _Bad
        if self.i >= len(self.s):
            raise _More
        if not part or (allowed is not None and part not in allowed):
            raise _Bad
        return part

    def _word(self, allowed):
        self._lit('"')
        w = self._until('"', allowed)
        self.i += 1
        return w

    def _name(self, allowed=None, taken=()):
        self._lit('"')
        w = self._until('"', allowed, NAME_CHARS)
        if w in taken:
            raise _Bad
        self.i += 1
        return w

    def _literal(self):
        self._lit('"')
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        c = self._c()
        if self.i == start:
            raise _Bad
        if c == '"':
            self.i += 1
        elif c == "n":
            self._lit('n"')
        elif c == ".":
            self.i += 1
            d = self.i
            while self.i < len(self.s) and self.s[self.i].isdigit():
                self.i += 1
            if self._c() != '"' or self.i == d:
                raise _Bad
            self.i += 1
        elif c == "u":
            self._lit('usd"')
        elif c == "e":
            self._lit('eur"')
        else:
            raise _Bad

    def _sep(self, more, close):
        """After a list item: True to continue (`more` read), False when `close` ends the list."""
        self._ws()
        c = self._c()
        if c in (more, close):
            self.i += 1
            return c == more
        raise _Bad

    def _doc(self):
        self._ws()
        self._lit("{")
        self._key("inputs")
        self._lit("[")
        while True:
            self._ws()
            self._lit("{")
            self._key("id")
            x = self._name(taken=self.inputs)
            self._ws()
            self._lit(",")
            self._key("type")
            self._word(TYPES)
            self._ws()
            self._lit("}")
            self.inputs.append(x)
            if not self._sep(",", "]"):
                break
        self._ws()
        self._lit(",")
        self._key("nodes")
        self._lit("[")
        while True:
            self._ws()
            self._lit("{")
            self._key("id")
            x = self._name(taken=self.inputs + self.nodes)
            for k, allowed in (("op", OPS), ("out_type", TYPES)):
                self._ws()
                self._lit(",")
                self._key(k)
                self._word(allowed)
            self._ws()
            self._lit(",")
            self._key("args")
            self._lit("[")
            while True:
                self._ws()
                self._lit("{")
                self._ws()
                k = self._word(ARG_KEYS)
                self._ws()
                self._lit(":")
                self._ws()
                if k == "ref":
                    self._name(allowed=self.inputs + self.nodes)
                elif k == "lit":
                    self._literal()
                else:
                    self._lit('"')
                    self._until("(", tuple(CONVERT))
                    self.i += 1
                    self._until(")", self.inputs + self.nodes, NAME_CHARS)
                    self.i += 1
                    self._lit('"')
                self._ws()
                self._lit("}")
                if not self._sep(",", "]"):
                    break
            self._ws()
            self._lit("}")
            self.nodes.append(x)
            if not self._sep(",", "]"):
                break
        self._ws()
        self._lit(",")
        self._key("output")
        self._name(allowed=self.nodes)
        self._ws()
        self._lit(",")
        self._key("call")
        self._lit("{")
        bound = []
        while True:
            self._ws()
            bound.append(self._name(allowed=[a for a in self.inputs if a not in bound]))
            self._ws()
            self._lit(":")
            self._ws()
            self._literal()
            self._ws()
            c = self._c()
            if c == "," and len(bound) < len(self.inputs):
                self.i += 1
            elif c == "}" and len(bound) == len(self.inputs):
                self.i += 1
                break
            else:
                raise _Bad
        self._ws()
        self._lit("}")


def schema_state(text):
    return SchemaPrefix(text).check()


# --- incremental acceptor ------------------------------------------------------------------------
# SchemaPrefix as an explicit machine: _doc flattened into instructions, run one character at a time.
_WS, _CH, _UNTIL, _LITERAL, _SEP, _CALLSEP, _BRANCH, _JUMP, _END = range(9)
_ASCII = tuple(chr(i) for i in range(128))


def _program():
    P, at = [], {}
    emit = P.append

    def key(k):
        emit((_WS,)), emit((_CH, f'"{k}"', None)), emit((_WS,)), emit((_CH, ":", None)), emit((_WS,))

    def name(allowed, taken=None, act=None):  # _name: '"', then up to the closing quote
        emit((_CH, '"', None)), emit((_UNTIL, '"', allowed, True, taken, act))

    def word(allowed, act=None):
        emit((_CH, '"', None)), emit((_UNTIL, '"', allowed, False, None, act))

    emit((_WS,)), emit((_CH, "{", None)), key("inputs"), emit((_CH, "[", None))
    at["input"] = len(P)
    emit((_WS,)), emit((_CH, "{", None)), key("id"), name(None, "inputs", "cur")
    emit((_WS,)), emit((_CH, ",", None)), key("type"), word("types")
    emit((_WS,)), emit((_CH, "}", "push_input")), emit((_SEP, ",", "]", "input"))
    emit((_WS,)), emit((_CH, ",", None)), key("nodes"), emit((_CH, "[", None))
    at["node"] = len(P)
    emit((_WS,)), emit((_CH, "{", None)), key("id"), name(None, "decls", "cur")
    for k, allowed in (("op", "ops"), ("out_type", "types")):
        emit((_WS,)), emit((_CH, ",", None)), key(k), word(allowed)
    emit((_WS,)), emit((_CH, ",", None)), key("args"), emit((_CH, "[", None))
    at["arg"] = len(P)
    emit((_WS,)), emit((_CH, "{", None)), emit((_WS,)), word("argkeys", "argk")
    emit((_WS,)), emit((_CH, ":", None)), emit((_WS,)), emit((_BRANCH, {"ref": "ref", "lit": "lit", "convert": "convert"}))
    at["ref"] = len(P)
    name("declared"), emit((_JUMP, "arg_end"))
    at["lit"] = len(P)
    emit((_LITERAL,)), emit((_JUMP, "arg_end"))
    at["convert"] = len(P)
    emit((_CH, '"', None)), emit((_UNTIL, "(", "convert", False, None, None))
    emit((_UNTIL, ")", "declared", True, None, None)), emit((_CH, '"', None))
    at["arg_end"] = len(P)
    emit((_WS,)), emit((_CH, "}", None)), emit((_SEP, ",", "]", "arg"))
    emit((_WS,)), emit((_CH, "}", "push_node")), emit((_SEP, ",", "]", "node"))
    emit((_WS,)), emit((_CH, ",", None)), key("output"), name("nodes")
    emit((_WS,)), emit((_CH, ",", None)), key("call"), emit((_CH, "{", None))
    at["call"] = len(P)
    emit((_WS,)), name("unbound", None, "bound"), emit((_WS,)), emit((_CH, ":", None)), emit((_WS,))
    emit((_LITERAL,)), emit((_CALLSEP, "call"))
    emit((_WS,)), emit((_CH, "}", None)), emit((_END,))
    resolve = lambda x: at[x] if isinstance(x, str) else x
    out = []
    for ins in P:
        if ins[0] in (_SEP, _CALLSEP, _JUMP):
            ins = ins[:-1] + (resolve(ins[-1]),)
        elif ins[0] == _BRANCH:
            ins = (_BRANCH, {k: at[v] for k, v in ins[1].items()})
        out.append(ins)
    return tuple(out)


_PROG = _program()
_FIXED = {"types": TYPES, "ops": OPS, "argkeys": ARG_KEYS, "convert": tuple(CONVERT)}


class SchemaState:
    """The forward mask's parse state after a prefix: the same acceptor as SchemaPrefix, but resumable.

    advance(text) reads only the new characters and returns the state after them, or None once the
    text has left the schema (SchemaPrefix's "bad"). States are immutable, so one per token position
    can be kept and a rewind is a lookup. next_chars() is the cached set of ASCII characters the
    schema allows next.
    """

    __slots__ = ("pc", "k", "buf", "inputs", "nodes", "bound", "cur", "argk", "_next")

    def __init__(self, pc=0, k=0, buf="", inputs=(), nodes=(), bound=(), cur=None, argk=None):
        self.pc, self.k, self.buf = pc, k, buf
        self.inputs, self.nodes, self.bound, self.cur, self.argk = inputs, nodes, bound, cur, argk
        self._next = None

    @property
    def done(self):
        return _PROG[self.pc][0] == _END

    @property
    def status(self):
        return "done" if self.done else "more"

    def _copy(self):
        return SchemaState(self.pc, self.k, self.buf, self.inputs, self.nodes, self.bound, self.cur, self.argk)

    def advance(self, text):
        if not text:
            return self
        s = self._copy()
        for c in text:
            if not s._step(c):
                return None
        return s

    def next_chars(self):
        if self._next is None:
            self._next = frozenset(c for c in _ASCII if self._copy()._step(c))
        return self._next

    def _src(self, name):
        if name in _FIXED:
            return _FIXED[name]
        if name == "inputs":
            return self.inputs
        if name in ("decls", "declared"):
            return self.inputs + self.nodes
        if name == "nodes":
            return self.nodes
        return tuple(a for a in self.inputs if a not in self.bound)  # "unbound"

    def _act(self, act, part=None):
        if act == "cur":
            self.cur = part
        elif act == "argk":
            self.argk = part
        elif act == "bound":
            self.bound = self.bound + (part,)
        elif act == "push_input":
            self.inputs, self.cur = self.inputs + (self.cur,), None
        elif act == "push_node":
            self.nodes, self.cur = self.nodes + (self.cur,), None

    def _enter(self):
        """SchemaPrefix._until fails on an empty allowed set even before its first character."""
        ins = _PROG[self.pc]
        return not (ins[0] == _UNTIL and ins[2] is not None and not self._src(ins[2]))

    def _done_with(self, pc):
        self.k, self.buf, self.pc = 0, "", pc
        return self._enter()

    def _step(self, c):
        while True:
            ins = _PROG[self.pc]
            op = ins[0]
            if op == _WS:
                if c == " " or c == "\n":
                    return True
                self.pc += 1
            elif op == _CH:
                w = ins[1]
                if c != w[self.k]:
                    return False
                self.k += 1
                if self.k < len(w):
                    return True
                self._act(ins[2])
                return self._done_with(self.pc + 1)
            elif op == _UNTIL:
                _, stop, allowed, charset, taken, act = ins
                allowed = None if allowed is None else self._src(allowed)
                if c == stop:
                    part = self.buf
                    if not part or (allowed is not None and part not in allowed) or (taken and part in self._src(taken)):
                        return False
                    self._act(act, part)
                    return self._done_with(self.pc + 1)
                if charset and c not in (NAME_CHARS if self.buf else NAME_HEAD):
                    return False
                self.buf += c
                return allowed is None or any(a.startswith(self.buf) for a in allowed)
            elif op == _LITERAL:
                return self._literal(c)
            elif op == _SEP:
                if c == " " or c == "\n":
                    return True
                if c == ins[1]:
                    return self._done_with(ins[3])
                return c == ins[2] and self._done_with(self.pc + 1)
            elif op == _CALLSEP:
                if c == " " or c == "\n":
                    return True
                if c == "," and len(self.bound) < len(self.inputs):
                    return self._done_with(ins[1])
                return c == "}" and len(self.bound) == len(self.inputs) and self._done_with(self.pc + 1)
            elif op == _BRANCH:
                self.pc = ins[1][self.argk]
            elif op == _JUMP:
                self.pc = ins[1]
            else:  # _END: SchemaPrefix.check strips trailing whitespace
                return c.isspace()

    def _literal(self, c):
        # k: 0 opening quote, 1 no digit yet, 2 digits, 3 after ".", 4 fraction digits, 5 rest of a suffix (buf)
        k = self.k
        if k == 0:
            self.k = 1
            return c == '"'
        if k in (1, 3):
            self.k += 1
            return c.isdigit()
        if k in (2, 4) and c.isdigit():
            return True
        if c == '"' and k in (2, 4):
            return self._done_with(self.pc + 1)
        if k == 2 and c in "n.ue":
            self.k, self.buf = (3, "") if c == "." else (5, {"n": '"', "u": 'sd"', "e": 'ur"'}[c])
            return True
        if k == 5 and c == self.buf[0]:
            self.buf = self.buf[1:]
            return bool(self.buf) or self._done_with(self.pc + 1)
        return False


class SchemaTrail:
    """SchemaState per token position of the trajectory last queried: states[i] is the state after
    the first i tokens (None once a token left the schema). A query for a prefix of that trajectory
    is a lookup (a rewind to t* reads states[t*]); one that extends it advances over the new tokens'
    characters only. The id comparison that finds the shared prefix is a C-level list compare."""

    def __init__(self, texts):
        self.texts = texts
        self.ids, self.states = [], [SchemaState()]

    def state(self, tokens):
        tokens = tokens if type(tokens) is list else as_list(tokens)
        n, ids = len(tokens), self.ids
        if n <= len(ids) and ids[:n] == tokens:
            return self.states[n]
        k = min(n, len(ids))
        if ids[:k] != tokens[:k]:
            k = next(j for j in range(k) if ids[j] != tokens[j])
        del ids[k:], self.states[k + 1:]
        for v in tokens[k:]:
            s = self.states[-1]
            self.states.append(None if s is None else s.advance(self.texts[v]))
            ids.append(v)
        return self.states[n]


class TypedJSONDAGCartridge(Cartridge):
    type_names = list(TYPES)

    def __init__(self, token_texts, encode=None, special_ids=(), n_slots=64, pin_inputs=True,
                 apply_forward_mask=False, mask_impl="incremental"):
        assert mask_impl in ("incremental", "reparse"), mask_impl
        assert pin_inputs in (True, False, "self"), pin_inputs
        self.texts = token_texts
        self.pin_inputs = pin_inputs  # True: the task's input contract; "self": as written; False: free
        self.contract = None  # {input id: type index} of the bound task (pin_inputs=True)
        self.apply_forward_mask = apply_forward_mask  # decode inside the schema (SchemaPrefix)
        self.mask_impl = mask_impl  # "reparse": the v0.2 acceptor, re-parsing the completion each query
        self._trail = SchemaTrail(token_texts)
        self._cand_cache = {}
        self.mask_seconds, self.mask_calls = 0.0, 0  # time spent in accept() / candidates()
        self._special = set(special_ids)
        self._first = None
        self.n_slots = n_slots
        self._tcm = TokenClassManager(token_texts, TYPES, lit_rule=_lit_rule, lit_default=U32,
                                      special_ids=special_ids, encode=encode)
        self._lit_ok = torch.tensor([bool(t) and SUFFIX_OK.fullmatch(t) is not None for t in token_texts])
        self._lit_ok &= ~self._tcm.special
        self._g_type = self._tcm.G_type(k=None)
        bare = torch.tensor([t == t.lstrip(' \t\n"') for t in token_texts])
        self._g_type[~bare] = float("-inf")
        self._g_stated = {(k, ""): self._tcm.G_type(k=k) for k in range(len(TYPES))}  # (type, typed so far)
        for g in self._g_stated.values():
            g[~bare] = float("-inf")

    @property
    def tcm(self):
        return self._tcm

    def text(self, tokens):
        return "".join(self.texts[i] for i in as_list(tokens))

    def stop(self, tokens):
        return json_end(self.text(tokens)) is not None

    def candidates(self, tokens):
        """A cheap superset of the tokens `accept` admits: those whose first character the schema
        allows next (every admitted token starts with one), plus EOS once the document is done."""
        t0 = time.perf_counter()
        if self._first is None:
            self._first = {}
            for i, t in enumerate(self.texts):
                if t and i not in self._special:
                    self._first.setdefault(t[0], []).append(i)
            self._first = {c: torch.tensor(ids) for c, ids in self._first.items()}
        if self.mask_impl == "reparse":
            head = self.text(tokens)
            chars = frozenset(c for c in self._first if c.isascii() and schema_state(head + c) != "bad")
            done = schema_state(head) == "done"
        else:
            s = self._trail.state(tokens)
            chars = frozenset() if s is None else frozenset(s.next_chars() & self._first.keys())
            done = s is not None and s.done
        if (chars, done) not in self._cand_cache:
            m = torch.zeros(len(self.texts), dtype=torch.bool)
            for c in chars:
                m[self._first[c]] = True
            if done:
                m[list(self._special)] = True
            self._cand_cache[chars, done] = m
        self.mask_seconds += time.perf_counter() - t0
        self.mask_calls += 1
        return self._cand_cache[chars, done].clone()

    def accept(self, tokens, cand):
        """Forward structural mask: may token `cand` follow completion `tokens`? (EOS only once the
        document is complete; other special tokens never.)"""
        t0 = time.perf_counter()
        cand = int(cand)
        if self.mask_impl == "reparse":
            head = self.text(tokens)
            ok = (schema_state(head) == "done" if cand in self._special
                  else schema_state(head + self.texts[cand]) != "bad")
        else:
            s = self._trail.state(tokens)
            ok = s is not None and (s.done if cand in self._special else s.advance(self.texts[cand]) is not None)
        self.mask_seconds += time.perf_counter() - t0
        self.mask_calls += 1
        return ok

    def bind_task(self, task):
        if self.pin_inputs is True:
            self.contract = {x: TYPES.index(T) for x, T in zip(task["inputs"], task["types"])}

    def contract_ok(self, tokens):
        """Every input the bound task names is declared with its stated type (True with no contract)."""
        if not self.contract:
            return True
        try:
            obj = json.loads(self.text(tokens)[:json_end(self.text(tokens))])
            declared = {i["id"]: i["type"] for i in obj["inputs"]}
        except Exception:  # noqa: BLE001  (not a document: the reference checker rejects it anyway)
            return True
        return all(declared[x] == TYPES[k] for x, k in self.contract.items() if x in declared)

    def _stated(self, k, typed):
        """Mask for the next token of an input declared with type k, `typed` written so far: the
        rest of the name, then the closing quote."""
        if (k, typed) not in self._g_stated:
            rest = TYPES[k][len(typed):]
            ok = [bool(t) and i not in self._special and (rest.startswith(t) if rest and len(t) <= len(rest)
                                                          else t.startswith(rest + '"'))
                  for i, t in enumerate(self.texts)]
            g = torch.full((len(self.texts),), float("-inf"))
            g[torch.tensor(ok)] = 0.0
            self._g_stated[k, typed] = g
        return self._g_stated[k, typed]

    def logits_processor(self, tokens, z):
        """Schema-level constrained decoding: a type slot admits only the five type names; with a
        contract and the forward mask, an input the task names is held to its stated type."""
        if self.contract and self.apply_forward_mask:
            text = self.text(tokens)
            m = INPUT_TYPE_SLOT.search(text)
            if m and m.group(1) in self.contract and '"nodes"' not in text:
                k = self.contract[m.group(1)]
                if TYPES[k].startswith(m.group(2)):
                    return z + self._stated(k, m.group(2))
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
        slot = lambda name: lx.slots.setdefault(str(name), len(lx.slots))
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
            t_type = tok(inp["type"].off)
            stated = (self.contract or {}).get(str(inp["id"])) if self.pin_inputs is True else None
            if stated is not None:  # the task's contract: check the written type against it
                lx.ops.append(Op(t_type, CHECK_VAR, slot(inp["id"]), stated, tag="input contract"))
                if ty(inp["type"]) == stated:
                    lx.frozen.add(t_type)
            elif self.pin_inputs:
                lx.frozen.add(t_type)
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
        if len(lx.slots) > self.n_slots:  # never alias two symbols onto one memory row
            lx.ops = [o for o in lx.ops if o.slot < self.n_slots]
            hard(None, f"more than {self.n_slots} ids")
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
