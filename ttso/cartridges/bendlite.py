"""Cartridge 1: the BendLite catalog subset of the DiffRepair study (types U32 and Nat).

    import Base
    def catalog(+stock: Nat, +level: U32, +qty: U32) -> U32:
      (stock + qty + level : U32)
    def main() -> U32:
      catalog(6n, 38, 47)

Ops (slots are (function, name) pairs; `__ret__` is a function's declared return type):
    parameter type           DECLARE(f.p)                    at the type's first token   (site decl)
    return type              DECLARE(f.__ret__)              at the type's first token   (site decl)
    group operand x          CHECK_VAR(f.x, T_annotation)    at x                        "group operand"
    U32.from_nat(x)          CHECK_VAR(f.x, Nat)             at x                        "from_nat arg"
    group annotation T       CHECK_VAR(f.__ret__, T)         at T's first token          "body type"
    call literal i of g      CHECK_CALL(g.p_i)               at the token after the digits (site lit)
    call result              CHECK_VAR(g.__ret__, T_ret(f))  at g                        "call result"
A literal's type is read from the token right after its digits: one starting with `n` is Nat, any
other is U32. Hard errors (no gradient): parse failures, unknown names, arity, a `+` group not
annotated U32 (BendLite's operators are typed U32), from_nat inside a non-U32 group.
"""

import re
from dataclasses import dataclass, field

import torch

from ttso.cartridges.base import Cartridge, Lexed, TextAligner, Verdict, as_list
from ttso.core.deltanet import CHECK_CALL, CHECK_VAR, DECLARE, Op
from ttso.core.token_classes import TokenClassManager

TYPES = ("U32", "Nat")
U32, NAT = 0, 1
NAME = r"[A-Za-z_][A-Za-z0-9_]*"
DEF = re.compile(rf"def ({NAME})\((.*)\) -> (\S+):\s*$")
PARAM = re.compile(rf"\s*\+?({NAME}): (\S+)\s*$")
LIT = re.compile(r"(\d+)(n?)$")
SUFFIX_OK = re.compile(r"n?[,)]?[\s,)]*")


@dataclass
class Fn:
    name: str
    params: list  # (name, type text, type offset)
    ret: tuple  # (type text, offset)
    body: dict = field(default_factory=dict)


class _ParseError(Exception):
    def __init__(self, off, why):
        self.off, self.why = off, why


def _split(s, off, sep):
    """Split s on sep, returning (piece, absolute offset of the piece) pairs."""
    out, start = [], 0
    for m in re.finditer(re.escape(sep), s):
        out.append((s[start:m.start()], off + start))
        start = m.end()
    out.append((s[start:], off + start))
    return out


def _strip(piece, off):
    lead = len(piece) - len(piece.lstrip())
    return piece.strip(), off + lead


def parse(text):
    """-> (list of Fn, first parse error (offset, why) or None). Parsing stops at the first error."""
    fns = []
    lines = text.split("\n")
    offs, o = [], 0
    for ln in lines:
        offs.append(o)
        o += len(ln) + 1
    i = 0
    try:
        while i < len(lines):
            ln, off = lines[i], offs[i]
            if not ln.strip() or ln.startswith("import ") or ln.strip().startswith("```"):
                i += 1
                continue
            m = DEF.match(ln)
            if not m:
                raise _ParseError(off, "bad def line")
            params = []
            if m.group(2).strip():
                for piece, po in _split(m.group(2), off + m.start(2), ","):
                    pm = PARAM.match(piece)
                    if not pm:
                        raise _ParseError(po, "bad parameter")
                    params.append((pm.group(1), pm.group(2), po + pm.start(2)))
            f = Fn(m.group(1), params, (m.group(3), off + m.start(3)))
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or not lines[j].startswith("  "):
                raise _ParseError(offs[j] if j < len(lines) else len(text), "expected an indented body")
            f.body = _parse_body(*_strip(lines[j], offs[j]))
            fns.append(f)
            i = j + 1
    except _ParseError as e:
        return fns, (e.off, e.why)
    return fns, None


def _parse_operand(s, off):
    s, off = _strip(s, off)
    if re.fullmatch(NAME, s):
        return {"kind": "var", "name": s, "off": off}
    m = re.fullmatch(rf"U32\.from_nat\(\s*({NAME})\s*\)", s)
    if m:
        return {"kind": "from_nat", "name": m.group(1), "off": off + m.start(1)}
    m = LIT.fullmatch(s)
    if m:
        return {"kind": "lit", "value": int(m.group(1)), "type": NAT if m.group(2) else U32,
                "suffix_off": off + m.end(1), "off": off}
    raise _ParseError(off, "bad operand")


def _parse_body(s, off):
    if s.startswith("(") and s.endswith(")"):
        inner, io = s[1:-1], off + 1
        k = inner.rfind(" : ")
        if k < 0:
            raise _ParseError(off, "bad annotation")
        ann, ao = _strip(inner[k + 3:], io + k + 3)
        ops = [_parse_operand(p, po) for p, po in _split(inner[:k], io, "+")]
        return {"kind": "group", "operands": ops, "ann": ann, "ann_off": ao, "plus": len(ops) > 1}
    m = re.fullmatch(rf"({NAME})\((.*)\)", s)
    if m:
        args = [] if not m.group(2).strip() else [_parse_operand(p, po) for p, po in _split(m.group(2), off + m.start(2), ",")]
        return {"kind": "call", "fn": m.group(1), "fn_off": off, "args": args}
    return {"kind": "operand", "operand": _parse_operand(s, off)}


def _lit_rule(text):
    return NAT if text.startswith("n") else None


class BendLiteCatalogCartridge(Cartridge):
    type_names = list(TYPES)

    def __init__(self, token_texts, encode=None, special_ids=(), n_slots=64):
        self.texts = token_texts
        self.n_slots = n_slots
        self._tcm = TokenClassManager(token_texts, TYPES, lit_rule=_lit_rule, lit_default=U32,
                                      special_ids=special_ids, encode=encode)
        self._lit_ok = torch.tensor([bool(t) and SUFFIX_OK.fullmatch(t) is not None for t in token_texts])
        self._lit_ok &= ~self._tcm.special

    @property
    def tcm(self):
        return self._tcm

    def text(self, tokens):
        return "".join(self.texts[i] for i in as_list(tokens))

    # --- lexer -> DeltaNet ops ---------------------------------------------------------------
    def lex(self, tokens):
        al = TextAligner([self.texts[i] for i in as_list(tokens)])
        fns, err = parse(al.text)
        lx = Lexed()
        slot = lambda f, x: lx.slots.setdefault((f, x), len(lx.slots))
        ty = lambda s: TYPES.index(s) if s in TYPES else None
        sigs = {}
        for f in fns:
            for p, T, o in f.params:
                t = al.tok(o)
                lx.ops.append(Op(t, DECLARE, slot(f.name, p), tag="param"))
                lx.sites[t] = "decl"
                if ty(T) is None:
                    lx.hard.append((t, f"bad type {T!r}"))
            t = al.tok(f.ret[1])
            lx.ops.append(Op(t, DECLARE, slot(f.name, "__ret__"), tag="return type"))
            lx.sites[t] = "decl"
            if ty(f.ret[0]) is None:
                lx.hard.append((t, f"bad type {f.ret[0]!r}"))
            sigs[f.name] = f
        for f in fns:
            self._lex_body(f, sigs, lx, al, slot, ty)
        if err is not None:
            lx.hard.append((min(al.tok(err[0]), max(len(al.texts) - 1, 0)), err[1]))
        if fns and "main" not in sigs and err is None:
            lx.hard.append((len(al.texts) - 1, "no main"))
        lx.ops.sort(key=lambda o: o.t)
        return lx

    def _lex_body(self, f, sigs, lx, al, slot, ty):
        b, names = f.body, {p for p, _, _ in f.params}
        if b["kind"] == "group":
            T = ty(b["ann"])
            if T is None:
                lx.hard.append((al.tok(b["ann_off"]), f"bad annotation {b['ann']!r}"))
                return
            if b["plus"] and T != U32:
                lx.hard.append((al.tok(b["ann_off"]), "operators are typed U32"))
            for o in b["operands"]:
                if o["kind"] in ("var", "from_nat"):
                    if o["name"] not in names:
                        lx.hard.append((al.tok(o["off"]), f"unknown name {o['name']!r}"))
                        continue
                    want = T if o["kind"] == "var" else NAT
                    lx.ops.append(Op(al.tok(o["off"]), CHECK_VAR, slot(f.name, o["name"]), want,
                                     tag="group operand" if o["kind"] == "var" else "from_nat arg"))
                    if o["kind"] == "from_nat" and T != U32:
                        lx.hard.append((al.tok(o["off"]), "from_nat yields U32"))
                elif o["type"] != T:
                    lx.hard.append((al.tok(o["suffix_off"]), "literal type"))
            lx.ops.append(Op(al.tok(b["ann_off"]), CHECK_VAR, slot(f.name, "__ret__"), T, tag="body type"))
        elif b["kind"] == "call":
            g = sigs.get(b["fn"])
            t_fn = al.tok(b["fn_off"])
            if g is None:
                lx.hard.append((t_fn, f"unknown function {b['fn']!r}"))
                return
            if len(b["args"]) != len(g.params):
                lx.hard.append((t_fn, "arity"))
            for a, (p, _, _) in zip(b["args"], g.params):
                if a["kind"] != "lit":
                    lx.hard.append((al.tok(a["off"]), "call arguments are literals"))
                    continue
                t = al.tok(a["suffix_off"]) if a["suffix_off"] < len(al.text) else len(al.texts) - 1
                lx.ops.append(Op(t, CHECK_CALL, slot(g.name, p), tag="call arg"))
                lx.sites[t] = "lit"
            if ty(f.ret[0]) is not None:
                lx.ops.append(Op(t_fn, CHECK_VAR, slot(g.name, "__ret__"), ty(f.ret[0]), tag="call result"))
        else:
            lx.hard.append((al.tok(b["operand"]["off"]), "unsupported body"))

    # --- reference checker -------------------------------------------------------------------------
    def check(self, tokens):
        return reference_check(self.text(tokens))

    def valid_mask(self, tokens, t, site):
        if site == "decl":
            return self.tcm.type_mask(like=int(tokens[t]))
        if site == "lit":
            return self._lit_ok.clone()
        return ~self.tcm.special

    def stop(self, tokens):
        s = self.text(tokens)
        k = s.find("def main(")
        if k < 0:
            return False
        lines = s[k:].split("\n")
        return len(lines) >= 3 and lines[1].strip() != ""


def reference_check(text):
    """Pure-Python BendLite subset checker and evaluator (independent of the DeltaNet)."""
    fns, err = parse(text)
    if err is not None:
        return Verdict(False, err[1])
    sig = {f.name: f for f in fns}
    if "main" not in sig or sig["main"].params:
        return Verdict(False, "no main")
    for f in fns:
        for _, T, _ in f.params:
            if T not in TYPES:
                return Verdict(False, f"bad parameter type {T!r}")
        if f.ret[0] not in TYPES:
            return Verdict(False, f"bad return type {f.ret[0]!r}")
    for f in fns:
        env = {p: T for p, T, _ in f.params}
        b = f.body
        if b["kind"] == "group":
            if b["ann"] not in TYPES:
                return Verdict(False, "bad annotation")
            if b["plus"] and b["ann"] != "U32":
                return Verdict(False, "operators are typed U32")
            for o in b["operands"]:
                if o["kind"] == "var":
                    if env.get(o["name"]) != b["ann"]:
                        return Verdict(False, f"{f.name}: {o['name']} is {env.get(o['name'])}, group is {b['ann']}")
                elif o["kind"] == "from_nat":
                    if env.get(o["name"]) != "Nat" or b["ann"] != "U32":
                        return Verdict(False, f"{f.name}: from_nat({o['name']})")
                elif TYPES[o["type"]] != b["ann"]:
                    return Verdict(False, f"{f.name}: literal type")
            if b["ann"] != f.ret[0]:
                return Verdict(False, f"{f.name}: body type {b['ann']} != {f.ret[0]}")
        elif b["kind"] == "call":
            g = sig.get(b["fn"])
            if g is None or len(b["args"]) != len(g.params):
                return Verdict(False, f"{f.name}: bad call")
            for a, (p, T, _) in zip(b["args"], g.params):
                if a["kind"] != "lit" or TYPES[a["type"]] != T:
                    return Verdict(False, f"{f.name}: argument {p} expects {T}")
            if g.ret[0] != f.ret[0]:
                return Verdict(False, f"{f.name}: call result {g.ret[0]} != {f.ret[0]}")
        else:
            return Verdict(False, "unsupported body")
    return Verdict(True, "", _eval(sig, "main", {}))


def _eval(sig, name, env):
    b = sig[name].body
    if b["kind"] == "group":
        v = 0
        for o in b["operands"]:
            v += env[o["name"]] if o["kind"] in ("var", "from_nat") else o["value"]
        return v % 2**32 if b["ann"] == "U32" else v
    g = sig[b["fn"]]
    return _eval(sig, g.name, {p: a["value"] for a, (p, _, _) in zip(b["args"], g.params)})
