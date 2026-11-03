"""SynthLang: a token-level typed language for the exactness and direction gates.

Statements (one token each, space-separated in the vocabulary):
    let x<i> : T          DECLARE(slot i) at T
    use x<i> @ T          CHECK_VAR(slot i, type(T)) at T
    call x<i> <d> <s>     CHECK_CALL(slot i) at the literal suffix s
    pad | ;               NOOP filler
Type tokens come in four spellings (T, " T", t, " t"); only the first two name the type, the
lowercase ones are in the same equivalence class but declare nothing. Literal suffixes map to types
(n -> Nat, u -> U32, ...). `reference_check` is an independent string-level checker.
"""

import random

import torch

from ttso.cartridges.base import Cartridge, Lexed, Verdict, as_list
from ttso.core.deltanet import CHECK_CALL, CHECK_VAR, DECLARE, Op
from ttso.core.token_classes import TokenClassManager

ALL_TYPES = ("U32", "Nat", "Float", "String", "Bool", "I64", "Char", "Unit")
SUFFIX = {"u": "U32", "n": "Nat", "f": "Float", "s": "String", "b": "Bool", "i": "I64", "c": "Char", "v": "Unit"}


class SynthLang(Cartridge):
    def __init__(self, n_types=8, n_vars=64):
        self.type_names = list(ALL_TYPES[:n_types])
        self.n_slots = n_vars
        self.suffix = {s: T for s, T in SUFFIX.items() if T in self.type_names}
        v = ["<eos>", "let", "use", "call", ":", "@", ";", "pad"]
        v += [f"x{i}" for i in range(n_vars)]
        for T in self.type_names:
            v += [T, " " + T, T.lower(), " " + T.lower()]
        v += [str(d) for d in range(10)] + list(self.suffix)
        self.vocab = v
        self.id = {t: i for i, t in enumerate(v)}
        lit = lambda t: self.type_names.index(self.suffix[t]) if t in self.suffix else None
        self._tcm = TokenClassManager(v, self.type_names, lit_rule=lit, special_ids=[0])

    @property
    def tcm(self):
        return self._tcm

    def ids(self, words):
        return [self.id[w] for w in words]

    def lex(self, tokens):
        w = [self.vocab[i] for i in as_list(tokens)]
        lx, t = Lexed(), 0
        while t < len(w):
            if w[t] in ("let", "use", "call") and t + 3 < len(w) and w[t + 1].startswith("x"):
                s = int(w[t + 1][1:])
                if w[t] == "let":
                    lx.ops.append(Op(t + 3, DECLARE, s, tag="let"))
                    lx.sites[t + 3] = "decl"
                elif w[t] == "use":
                    k = int(self.tcm.type_of[self.id[w[t + 3]]])
                    if k < 0:
                        lx.hard.append((t + 3, "use of a non-type"))
                    else:
                        lx.ops.append(Op(t + 3, CHECK_VAR, s, k, tag="use"))
                else:
                    lx.ops.append(Op(t + 3, CHECK_CALL, s, tag="call"))
                    lx.sites[t + 3] = "lit"
                t += 4
            else:
                t += 1
        return lx

    def check(self, tokens):
        v = reference_check([self.vocab[i] for i in as_list(tokens)], self.type_names, self.suffix)
        bad = [t for t, r in enumerate(v["verdicts"]) if r is False]
        return Verdict(not bad, f"check failed at {bad[0]}" if bad else "")

    def random_program(self, rng, max_len=300, n_live=8, p_bad=0.3):
        """A random statement sequence of at most max_len tokens."""
        words, decl = [], {}
        live = rng.sample(range(self.n_slots), n_live)
        target = rng.randint(4, max_len)
        while len(words) < target - 4:
            r, x = rng.random(), rng.choice(live)
            if r < 0.25 or not decl:
                T = rng.choice(self.type_names)
                spell = rng.choice([T, " " + T, T, " " + T, T.lower(), " " + T.lower()])
                words += ["let", f"x{x}", ":", spell]
                decl[x] = T if spell.strip() == T else None
            elif r < 0.5:
                T = decl.get(x) if decl.get(x) and rng.random() > p_bad else rng.choice(self.type_names)
                words += ["use", f"x{x}", "@", rng.choice([T, " " + T])]
            elif r < 0.7:
                T = decl.get(x) if decl.get(x) and rng.random() > p_bad else rng.choice(self.type_names)
                s = {v: k for k, v in self.suffix.items()}[T]
                words += ["call", f"x{x}", str(rng.randrange(10)), s]
            else:
                words += [rng.choice(["pad", ";", "pad"])]
        return self.ids(words)


def reference_check(words, type_names, suffix):
    """Independent checker over token strings: per token None (no check) / True / False, and the
    symbol table (slot -> type name or None) after every token."""
    table, verdicts, tables = {}, [], []
    canon = lambda s: s.strip() if s.strip() in type_names else None
    i = 0
    pending = {}
    while i < len(words):
        w = words[i]
        if w in ("let", "use", "call") and i + 3 < len(words) and words[i + 1].startswith("x"):
            pending[i + 3] = (w, int(words[i + 1][1:]), words[i + 3])
            i += 4
        else:
            i += 1
    for t in range(len(words)):
        r = None
        if t in pending:
            kind, x, last = pending[t]
            if kind == "let":
                table[x] = canon(last)
            elif kind == "use":
                r = table.get(x) is not None and table.get(x) == canon(last)
            else:
                r = table.get(x) is not None and table.get(x) == suffix.get(last)
        verdicts.append(r)
        tables.append(dict(table))
    return {"verdicts": verdicts, "tables": tables}


def rng_for(seed):
    return random.Random(seed)


def table_matrix(table, type_names, n_slots):
    M = torch.zeros(n_slots, len(type_names))
    for x, T in table.items():
        if T is not None:
            M[x, type_names.index(T)] = 1.0
    return M
