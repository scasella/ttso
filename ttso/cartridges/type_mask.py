"""v0.5: TypeAwareMask, the forward mask that also knows the types (a baseline for repair).

SchemaPrefix (v0.2; SchemaState is its incremental form) keeps a completion inside the workflow
schema but never looks at types, so a wrong-type reference or literal is decoded and left to the
verifier. TypeAwareMask runs the same machine (typed_json_dag._program(typed=True): the same
instructions, with the typed slots renamed) and, at every position whose type is fixed by what is
known at decode time, admits only what fits that type. What it knows: the schema, each op's
signature (add / max / min take arguments of the node's out_type; to_u32: Nat -> U32, to_nat:
U32 -> Nat, to_float: U32 -> Float), the task's stated input types (the contract, from bind_task)
and the types declared earlier in the prefix. It never reads the expected value, the test vectors
or a reference program.

    slot                      admitted                                   type fixed by
    input type                the stated type, if the task names the id  the contract
    arg key                   "ref" only if a declared id has the node's out_type; "convert" only
                              if a converter yields the out_type from a declared id; "lit" always
    {"ref": x}                declared ids of the node's out_type        declarations, op signature
    {"convert": "f(x)"}       f with dst = the node's out_type and a declared id of f's src type;
                              x: declared ids of f's src type            declarations, signatures
    {"lit": v}                v in the node's out_type's literal form    op signature
    "call": {x: v}            v in x's declared type's literal form      x's declaration
Literal forms: U32 digits then the closing quote (never a suffix, so neither "16u32" nor "16usd"),
Nat "7n", Float "7.5", USD "7usd", EUR "7eur". The arg-key rule is what keeps every position
non-empty: a key is admitted only when a type-correct argument can follow it.

Where no type is fixed yet the mask falls back to SchemaPrefix, and counts it (state.fb):
    "input_type"  the type of an input the task does not name (no contract entry)
    "out_type"    every node's out_type: no op signature or earlier declaration fixes it
    "output"      the output id (any node; nothing consumes it)
A completed program accepted by this mask has no type error: the reference checker accepts it.

The contract is enforced twice, identically: by TypedJSONDAGCartridge.logits_processor (v0.4's
_stated mask, which greedy_masked also decodes with) and by this state, so the state alone is a
complete type-aware acceptor.
"""

from ttso.cartridges.base import as_list
from ttso.cartridges.typed_json_dag import (ARG_KEYS, CONVERT, TYPES, SchemaState, SchemaTrail,
                                           TypedJSONDAGCartridge, _program, json_end)

_TYPED_PROG = _program(typed=True)
_LIT_HEAD = {"U32": '"', "Nat": "n", "Float": ".", "USD": "u", "EUR": "e"}  # the character after the digits
_CONV = {f: (TYPES[s], TYPES[d]) for f, (s, d) in CONVERT.items()}
FALLBACKS = ("input_type", "out_type", "output")


class TypeAwareMask(SchemaState):
    """SchemaState with types. contract: {input id: type name}, or None. Extra state: types (the
    (id, type) pairs declared so far), ntype (the current node's out_type), ctype (the current
    input's written type), conv (the current converter), fb (fallback counts, FALLBACKS order)."""

    __slots__ = ("contract", "types", "ntype", "ctype", "conv", "fb")
    _prog = _TYPED_PROG

    def __init__(self, contract=None, pc=0, k=0, buf="", inputs=(), nodes=(), bound=(), cur=None, argk=None,
                 types=(), ntype=None, ctype=None, conv=None, fb=(0, 0, 0)):
        super().__init__(pc, k, buf, inputs, nodes, bound, cur, argk)
        self.contract = contract or {}
        self.types, self.ntype, self.ctype, self.conv, self.fb = types, ntype, ctype, conv, fb

    def _copy(self):
        return TypeAwareMask(self.contract, self.pc, self.k, self.buf, self.inputs, self.nodes, self.bound, self.cur,
                             self.argk, self.types, self.ntype, self.ctype, self.conv, self.fb)

    @property
    def fallbacks(self):
        return dict(zip(FALLBACKS, self.fb))

    def _of(self, T):
        return tuple(x for x, U in self.types if U == T)

    def _converters(self):
        return tuple(f for f, (s, d) in _CONV.items() if d == self.ntype and self._of(s))

    def _src(self, name):
        if name == "in_types":
            return (self.contract[self.cur],) if self.cur in self.contract else TYPES
        if name == "out_types":
            return TYPES
        if name == "arg_keys":
            ok = {"ref": bool(self._of(self.ntype)), "lit": True, "convert": bool(self._converters())}
            return tuple(k for k in ARG_KEYS if ok[k])
        if name == "ref_ids":
            return self._of(self.ntype)
        if name == "converters":
            return self._converters()
        if name == "convert_ids":
            return self._of(_CONV[self.conv][0])
        if name == "output_ids":
            return self.nodes
        return super()._src(name)

    def _count(self, which):
        j = FALLBACKS.index(which)
        self.fb = self.fb[:j] + (self.fb[j] + 1,) + self.fb[j + 1:]

    def _act(self, act, part=None):
        if act == "in_type":
            self.ctype = part
            if self.cur not in self.contract:
                self._count("input_type")
        elif act == "out_type":
            self.ntype = part
            self._count("out_type")
        elif act == "conv":
            self.conv = part
        elif act == "output":
            self._count("output")
        elif act == "push_input":
            self.types += ((self.cur, self.ctype),)
            super()._act(act, part)
        elif act == "push_node":
            self.types += ((self.cur, self.ntype),)
            super()._act(act, part)
        else:
            super()._act(act, part)

    def _want(self):
        """The literal's type: the node's out_type (arg) or the bound input's declared type (call)."""
        if self._prog[self.pc][1] == "arg":
            return self.ntype
        x = self.bound[-1]
        return next(U for y, U in self.types if y == x)

    def _literal(self, c):
        if self.k == 2 and not c.isdigit() and c != _LIT_HEAD[self._want()]:
            return False  # after the digits: only the wanted type's suffix (U32: the closing quote)
        return super()._literal(c)


class TypeAwareTrail(SchemaTrail):
    """SchemaTrail over TypeAwareMask states, for one task's contract."""

    def __init__(self, texts, contract=None):
        super().__init__(texts)
        self.states = [TypeAwareMask(contract)]


class TypeMaskedJSONDAGCartridge(TypedJSONDAGCartridge):
    """TypedJSONDAGCartridge whose forward mask (accept / candidates, decoding and repair edits) is
    TypeAwareMask. Always masked and incremental; pin_inputs as in v0.4 (True: the contract).
    bind_task rebuilds the trail for the task's contract. With audit = [] every program the
    reference checker sees (greedy, each repair round, each Best-of-N sample) is also audited:
    accepted by the type-aware mask at every prefix, failed symbolic-verifier checks, hard errors,
    fallback counts."""

    mask_name = "typed"

    def __init__(self, token_texts, encode=None, special_ids=(), n_slots=64, pin_inputs=True):
        super().__init__(token_texts, encode=encode, special_ids=special_ids, n_slots=n_slots,
                         pin_inputs=pin_inputs, apply_forward_mask=True, mask_impl="incremental")
        self._trail = TypeAwareTrail(token_texts)
        self.audit, self._task, self._sym = None, None, None

    def bind_task(self, task):
        super().bind_task(task)
        stated = {x: TYPES[k] for x, k in (self.contract or {}).items()}
        self._trail = TypeAwareTrail(self.texts, stated)
        self._task = task.get("i")

    def mask_state(self, tokens):
        """The type-aware state after `tokens` (None once a token left the mask)."""
        return self._trail.state(as_list(tokens))

    def check(self, tokens):
        v = super().check(tokens)
        if self.audit is not None:
            self.audit.append(self.audit_program(tokens, v))
        return v

    def audit_program(self, tokens, verdict):
        from ttso.core.symbolic import SymbolicProvenanceVerifier
        if self._sym is None:
            self._sym = SymbolicProvenanceVerifier.for_cartridge(self)
        toks = as_list(tokens)
        s = self.mask_state(toks)
        lx = self.lex(toks)
        failed = [(c.tag, c.t, c.expected, c.declared) for c in self._sym.run(toks, lx.ops) if not c.passed]
        return {"i": self._task, "tokens": len(toks), "typed_ok": s is not None, "done": bool(s is not None and s.done),
                "complete": json_end(self.text(toks)) is not None, "valid": verdict.valid, "error": verdict.error,
                "failed_checks": failed, "hard": [w for _, w in lx.hard],
                "fallbacks": s.fallbacks if s is not None else None}
