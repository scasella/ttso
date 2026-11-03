"""Diagnosis of the first failed check and edit-site routing (Smart, Upstream, LocalOnly, Blind).

diagnose(): runs the cartridge's lexer and the verifier, backpropagates the first failed check into
the cached logits and returns the flagged tokens T_causal = {t <= t_error : ||dL/dz_t||_inf > eps},
the symbol s* of the check, its declaration token t_decl (the DECLARE the check read) and, for a
CHECK_CALL, the call-site literal token t_call. A hard error (parse failure, undeclared name, arity)
earlier than every failed check is reported as kind "hard" with no gradient.

Policies (one edit per round; `RepairState` carries what earlier rounds did):

  smart     Case A (t_decl and t_call flagged): consistent_uses[s*] > 0 -> the call site, else the
            declaration (or the latest flagged token once the declaration was edited).
            Case B (only t_decl flagged): the declaration, or the latest flagged token once edited.
            Gradient-directed replacement: argmax over grammar-valid, non-banned tokens with a
            constructive gradient (-g_v > 0) of -g_v + 1e-5 z_v. A site with no such token is
            exhausted and the next candidate site is tried (Case A: the other flagged site).
            Case C (nothing constructive): ban-and-regenerate at t_error.
  upstream  v1 Upstream: t* = min T_causal; once that declaration was tried, max T_causal.
  local     memory reads detached (only the check's own token can be flagged); Case C otherwise.
  blind     t* = min T_causal, the model's best other token there (no gradient direction).

Frozen declarations (Lexed.frozen, e.g. input types the prompt fixes) are never edited; with none
of the flagged tokens editable, Case C bans the token where the failing construct starts
(Lexed.local, e.g. the "ref" key of a JSON argument, so the model can write a conversion instead).

Banning: an edit bans the class of the token it replaces at t* (its equivalence class plus its
type / literal class at a typed site). Smart accumulates these bans per position across rounds, so
a declaration cannot oscillate U32 -> Nat -> U32 (the spec bans only the current class; B6 Patch 1
accumulated). Upstream and blind accumulate bans of single token ids, as in v1. Bans and the
edited set are dropped for positions after t*, which are regenerated. No rollback: every edit is
kept (Finding 3's control run).
"""

from dataclasses import dataclass, field

import torch

from ttso.core.deltanet import CHECK_CALL, DECLARE, first_failure_gradients

TIE = 1e-5


@dataclass
class Diagnosis:
    kind: str  # "check" | "hard" | "valid"
    t_error: int | None = None
    why: str = ""
    slot: int | None = None
    t_decl: int | None = None
    t_call: int | None = None
    flagged: list = field(default_factory=list)
    grads: dict = field(default_factory=dict)
    consistent_uses: int = 0
    lexed: object = None


@dataclass
class Edit:
    t: int
    old: int
    new: int
    case: str  # "A" | "B" | "C"
    route: str  # "call" | "decl" | "flagged" | "local"
    site: str  # "decl" | "lit" | "other"


@dataclass
class RepairState:
    taboo: dict = field(default_factory=dict)  # t -> bool mask (V,)
    edited: set = field(default_factory=set)
    tried_decl: set = field(default_factory=set)

    def after_edit(self, edit, ban):
        t = edit.t
        self.taboo = {s: m for s, m in self.taboo.items() if s <= t}
        self.taboo[t] = self.taboo.get(t, torch.zeros_like(ban)) | ban
        self.edited = {s for s in self.edited if s < t} | {t}
        self.tried_decl = {s for s in self.tried_decl if s <= t}


def diagnose(net, cart, tokens, logits, dtype=torch.float32, detach_state=False, eps=0.0):
    lx = cart.lex(tokens)
    G = first_failure_gradients(net, tokens, lx.ops, logits, dtype=dtype, detach_state=detach_state, eps=eps)
    j = G.check
    t_check = G.trace.checks[j].op.t if j is not None else None
    t_hard = min((t for t, _ in lx.hard), default=None)
    if t_check is None and t_hard is None:
        return Diagnosis("valid", lexed=lx)
    if t_check is None or (t_hard is not None and t_hard < t_check):
        why = min(lx.hard, key=lambda h: h[0])[1]
        return Diagnosis("hard", t_hard, why, lexed=lx)
    c = G.trace.checks[j]
    flagged = sorted(G.grads)
    t_decl = G.trace.decl_of.get(j)
    return Diagnosis("check", c.op.t, c.op.tag, c.op.slot,
                     t_decl if t_decl in G.grads else None,
                     c.op.t if c.op.kind == CHECK_CALL and c.op.t in G.grads else None,
                     flagged, G.grads, c.uses_before, lx)


def _best(s, C, accept):
    if not bool(C.any()):
        return None
    s = s.clone()
    s[~C] = float("-inf")
    if accept is None:
        return int(s.argmax())
    n = int(C.sum())
    for v in torch.argsort(s, descending=True)[:n].tolist():
        if accept(v):
            return v
    return None


def gradient_pick(z, g, allowed, banned, accept=None):
    """argmax over allowed, non-banned tokens with -g_v > 0 of -g_v + 1e-5 z_v (None if empty);
    with `accept`, the best such token the forward mask accepts."""
    score = -g.float()
    return _best(score + TIE * z.float(), allowed & ~banned & (score > 0), accept)


def best_pick(z, allowed, banned, accept=None):
    return _best(z.float(), allowed & ~banned, accept)


class Router:
    POLICIES = ("smart", "upstream", "local", "blind")

    def __init__(self, cart, policy="smart"):
        assert policy in self.POLICIES, policy
        self.cart, self.policy = cart, policy
        self.tcm = cart.tcm

    @property
    def detach_state(self):
        return self.policy == "local"

    def _ban(self, tokens, t, site):
        if self.policy in ("upstream", "blind"):  # v1: single token ids
            m = torch.zeros(self.tcm.V, dtype=torch.bool)
            m[int(tokens[t])] = True
            return m
        return self.tcm.ban_mask(int(tokens[t]), site)

    def _banned(self, state, tokens, t, site):
        return state.taboo.get(t, torch.zeros(self.tcm.V, dtype=torch.bool)) | self._ban(tokens, t, site)

    def _accept(self, tokens, t):
        if not getattr(self.cart, "apply_forward_mask", False):
            return None
        head = list(tokens[:t])
        return lambda v: self.cart.accept(head, v)

    def _allowed(self, tokens, t, site):
        allowed = self.cart.valid_mask(tokens, t, site)
        if getattr(self.cart, "apply_forward_mask", False):
            allowed = allowed & self.cart.candidates(list(tokens[:t]))
        return allowed

    def _gradient_edit(self, d, tokens, logits, state, t, case, route):
        site = d.lexed.sites.get(t, "other")
        allowed = self._allowed(tokens, t, site)
        new = gradient_pick(logits[t], d.grads[t], allowed, self._banned(state, tokens, t, site),
                            self._accept(tokens, t))
        return None if new is None else Edit(t, int(tokens[t]), new, case, route, site)

    def _local_edit(self, d, tokens, logits, state, t, case="C", route="local"):
        t = min(t, len(tokens) - 1)
        site = d.lexed.sites.get(t, "other")
        allowed = self._allowed(tokens, t, site)
        new = best_pick(logits[t], allowed, self._banned(state, tokens, t, site), self._accept(tokens, t))
        return None if new is None else Edit(t, int(tokens[t]), new, case, route, site)

    def choose(self, d, tokens, logits, state):
        """One edit for diagnosis d, or None when every candidate site is exhausted."""
        if d.kind == "valid":
            return None
        t_local = d.lexed.local.get(d.t_error, d.t_error)
        frozen = d.lexed.frozen
        if frozen:
            d.flagged = [t for t in d.flagged if t not in frozen]
            d.t_decl = None if d.t_decl in frozen else d.t_decl
        if d.kind == "hard" or not d.flagged:
            return self._local_edit(d, tokens, logits, state, t_local)
        if self.policy == "blind":
            t = d.flagged[0]
            return self._local_edit(d, tokens, logits, state, t, case="A", route="flagged")
        if self.policy == "upstream":
            t = d.flagged[0]
            if t in state.tried_decl:
                t = d.flagged[-1]
            state.tried_decl.add(d.flagged[0])
            e = self._gradient_edit(d, tokens, logits, state, t, "A", "decl" if t == d.t_decl else "flagged")
            return e or self._local_edit(d, tokens, logits, state, t_local)
        if self.policy == "local":
            for t in reversed(d.flagged):  # only the check's own token can be flagged here
                e = self._gradient_edit(d, tokens, logits, state, t, "A", "call")
                if e:
                    return e
            return self._local_edit(d, tokens, logits, state, t_local)
        # smart
        td, tc, latest = d.t_decl, d.t_call, d.flagged[-1]
        if td is not None and tc is not None:  # Case A
            if d.consistent_uses > 0:
                order = [(tc, "call"), (td, "decl")]
            elif td in state.edited:
                order = [(latest, "call" if latest == tc else "flagged"), (td, "decl")]
            else:
                order = [(td, "decl"), (tc, "call")]
            case = "A"
        elif td is not None:  # Case B
            order = [(td, "decl")] if td not in state.edited else [(latest, "flagged")]
            case = "B"
        else:
            order, case = [(t, "flagged") for t in reversed(d.flagged)], "A"
        seen = set()
        for t, route in order:
            if t in seen:
                continue
            seen.add(t)
            e = self._gradient_edit(d, tokens, logits, state, t, case, route)
            if e:
                return e
        return self._local_edit(d, tokens, logits, state, t_local)

    def ban_after(self, tokens, edit):
        """What to add to the taboo at edit.t once the edit is applied."""
        return self._ban(tokens, edit.t, edit.site)
