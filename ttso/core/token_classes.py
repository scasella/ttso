"""Tokenizer-aware equivalence classes, type projections and grammar masks.

A cartridge names its K semantic types (e.g. "U32", "Nat") and a literal-suffix rule. Scanning the
vocabulary once gives:

* equivalence classes: tokens with the same text after stripping whitespace / quotes and casefolding
  (Class("Nat") = {"Nat", " Nat", "nat", " nat", "\"Nat", ...}). Banning a token bans its class, so a
  ban of " Nat" cannot be dodged by " nat" (the Blind-rewind failure).
* type tokens: a token is a type-k token when its core text (whitespace / quotes stripped, case
  kept) is the type's name or the first BPE piece of the name as the tokenizer writes it (" U" for
  U32 when U32 tokenizes as U,3,2; never " N" or " Na" for Nat, which lead to malformed types).
  W_decl[k, v] = 1 for those. Banning a type token also bans every token of its type class.
* W_lit: the literal-suffix projection (the token right after a literal's digits); the cartridge's
  rule maps a token text to a type, or None for the default type.
* G_type: a 0 / -inf mask over the type tokens, optionally restricted to the leading characters of
  a reference token (so a replacement keeps the slot's spacing and quoting).
"""

import torch

STRIP = " \t\r\n\"'`"


def core(text):
    return text.strip(STRIP)


def lead(text):
    return text[: len(text) - len(text.lstrip(STRIP))]


class TokenClassManager:
    def __init__(self, token_texts, type_names, lit_rule=None, lit_default=None, special_ids=(),
                 encode=None):
        self.texts = list(token_texts)
        self.V, self.K = len(self.texts), len(type_names)
        self.type_names = list(type_names)
        self.special = torch.zeros(self.V, dtype=torch.bool)
        for i in special_ids:
            if 0 <= i < self.V:
                self.special[i] = True
        key = [core(t).casefold() or t for t in self.texts]  # pure whitespace / quotes: exact text
        ids = {}
        for i, k in enumerate(key):
            ids.setdefault(k, len(ids))
        self.class_id = torch.tensor([ids[k] for k in key])
        self.heads = [type_heads(name, encode, self.texts) for name in type_names]
        for a in range(self.K):
            for b in range(a + 1, self.K):
                clash = self.heads[a] & self.heads[b]
                assert not clash, f"{type_names[a]} and {type_names[b]} share a first token {clash}"
        head_of = {h: k for k, hs in enumerate(self.heads) for h in hs}
        self.type_of = torch.tensor([-1 if self.special[i] else head_of.get(core(t), -1)
                                     for i, t in enumerate(self.texts)], dtype=torch.long)
        self.lit_of = torch.full((self.V,), -1 if lit_default is None else lit_default, dtype=torch.long)
        if lit_rule is not None:
            for i, t in enumerate(self.texts):
                k = lit_rule(t)
                if k is not None:
                    self.lit_of[i] = k
        lids = {}
        self.lead_id = torch.tensor([lids.setdefault(lead(t), len(lids)) for t in self.texts])

    def _proj(self, of):
        W = torch.zeros(self.K, self.V)
        m = of >= 0
        W[of[m], torch.nonzero(m).flatten()] = 1.0
        return W

    @property
    def W_decl(self):
        return self._proj(self.type_of)

    @property
    def W_lit(self):
        return self._proj(self.lit_of)

    def equivalence_class(self, tok):
        return self.class_id == self.class_id[tok]

    def type_class(self, k):
        return self.type_of == k

    def ban_mask(self, tok, site="other"):
        """Tokens banned together with `tok`: its equivalence class, plus its whole type class at a
        declaration site or its whole literal class at a literal site."""
        m = self.equivalence_class(tok)
        if site == "decl" and self.type_of[tok] >= 0:
            m = m | self.type_class(int(self.type_of[tok]))
        elif site == "lit":
            m = m | (self.lit_of == self.lit_of[tok])
        return m

    def type_mask(self, like=None, k=None):
        """Boolean mask of valid type-annotation tokens (of type k if given), keeping the leading
        characters of token `like` if given."""
        m = (self.type_of >= 0) if k is None else (self.type_of == k)
        if like is not None:
            m = m & (self.lead_id == self.lead_id[like])
        return m & ~self.special

    def G_type(self, like=None, k=None):
        """The additive grammar mask {0, -inf}^V for a type-annotation slot."""
        g = torch.full((self.V,), float("-inf"))
        g[self.type_mask(like, k)] = 0.0
        return g


def type_heads(name, encode=None, texts=None):
    """Cores of the tokens a type name starts with: the name itself, plus the first BPE piece of
    the name after each usual lead (nothing, a space, a quote) when an encoder is given."""
    heads = {name}
    if encode is not None:
        for lead_ in ("", " ", '"', ' "', ": "):
            ids = encode(lead_ + name)
            acc = ""
            for i in ids:
                acc += texts[i]
                if len(acc) > len(lead_):
                    c = core(texts[i])
                    if c and name.startswith(c):
                        heads.add(c)
                    break
    return heads

