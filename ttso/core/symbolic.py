"""SymbolicProvenanceVerifier: the discrete reference for CompiledDeltaNet (v0.4).

A plain provenance table, in pure Python (no tensors, no autograd). It reads the lexer's ops in the
same order as the DeltaNet:

    DECLARE(s) at t          table[s] = (t, type_of[y_t], uses = 0)
    CHECK_VAR(s, u) at t     passes iff table[s] declares type u
    CHECK_CALL(s) at t       passes iff table[s] declares lit_of[y_t] (the literal's type)

and a passing check adds one to table[s]'s consistent uses. For the first failed check it returns
the provenance the routing rules need, with the same fields as routing.Diagnosis:

    t_decl   the declaration the check read, flagged when the check expects a type
             (not with LocalOnly, whose memory reads are detached)
    t_call   a failed CHECK_CALL's own literal token, flagged when its slot declares a type
    constructive[t]   which replacements at a flagged token would satisfy the check:
                      ("decl", k) tokens whose type is k, ("lit", k) literal suffixes of type k

Router.choose() then applies the same Smart (Cases A, B, C), Upstream, LocalOnly and Blind rules; a
"gradient" edit picks the model's most likely grammar-valid, non-banned constructive token.

Why this should match the DeltaNet: the DeltaNet's memory row for a slot holds the one-hot type of
the slot's latest declaration (a later DECLARE erases the row exactly), so a check's loss is 0 or 1
and its gradient reaches exactly the provenance above. At a declaration the gradient of the failed
check is -s_v (1[type(v) = k] - S_k), with s = softmax(z_t) and S_k the probability of the type-k
tokens, so the constructive tokens (-g_v > 0) are exactly the type-k tokens, ranked by
-g_v + 1e-5 z_v, which grows with z_v: the most likely one. The two can differ only where float32
rounds: a token whose probability underflows to 0, or S_k rounding to exactly 1.
tests/test_symbolic_equivalence.py measures how often they agree.
"""

from dataclasses import dataclass

from ttso.core.deltanet import CHECK_CALL, CHECK_VAR, DECLARE


@dataclass
class SymCheck:
    t: int
    kind: int
    slot: int
    tag: str
    expected: int | None  # the type the check wants (CHECK_VAR's u, or the literal's type)
    declared: int | None  # the type the slot's latest declaration holds
    decl_t: int | None  # that declaration's token
    uses_before: int
    passed: bool


class SymbolicProvenanceVerifier:
    def __init__(self, type_of, lit_of):
        """type_of / lit_of: per token id, the declared / literal type index, or -1 for none."""
        self.type_of = [int(k) for k in type_of]
        self.lit_of = [int(k) for k in lit_of]

    @classmethod
    def for_cartridge(cls, cart):
        return cls(cart.tcm.type_of.tolist(), cart.tcm.lit_of.tolist())

    def run(self, tokens, ops):
        """Every check, in the DeltaNet's order (by token, then op order)."""
        table, checks = {}, []
        for op in sorted(ops, key=lambda o: o.t):  # stable: ops at one token keep their order
            y = int(tokens[op.t])
            if op.kind == DECLARE:
                k = self.type_of[y]
                table[op.slot] = [op.t, k if k >= 0 else None, 0]
                continue
            entry = table.get(op.slot)
            declared = entry[1] if entry else None
            if op.kind == CHECK_VAR:
                expected = op.expected
            else:
                k = self.lit_of[y]
                expected = k if k >= 0 else None
            ok = declared is not None and declared == expected
            checks.append(SymCheck(op.t, op.kind, op.slot, op.tag, expected, declared,
                                   entry[0] if entry else None, entry[2] if entry else 0, ok))
            if ok:
                entry[2] += 1
        return checks

    def diagnose(self, cart, tokens, detach_state=False, lexed=None):
        """routing.Diagnosis of the first failed check (or hard error), with `constructive` set."""
        from ttso.core.routing import Diagnosis
        lx = lexed if lexed is not None else cart.lex(tokens)
        bad = [c for c in self.run(tokens, lx.ops) if not c.passed]
        c = min(bad, key=lambda x: x.t) if bad else None  # min keeps the first of equal t
        t_hard = min((t for t, _ in lx.hard), default=None)
        if c is None and t_hard is None:
            return Diagnosis("valid", lexed=lx)
        if c is None or (t_hard is not None and t_hard < c.t):
            return Diagnosis("hard", t_hard, min(lx.hard, key=lambda h: h[0])[1], lexed=lx)
        sites = {}
        if not detach_state and c.decl_t is not None and c.expected is not None:
            sites[c.decl_t] = ("decl", c.expected)
        if c.kind == CHECK_CALL and c.declared is not None:
            sites[c.t] = ("lit", c.declared)
        return Diagnosis("check", c.t, c.tag, c.slot, c.decl_t if c.decl_t in sites else None,
                         c.t if c.kind == CHECK_CALL and c.t in sites else None, sorted(sites), {},
                         c.uses_before, lx, constructive=sites)


__all__ = ["SymbolicProvenanceVerifier", "SymCheck", "CHECK_VAR", "CHECK_CALL", "DECLARE"]
