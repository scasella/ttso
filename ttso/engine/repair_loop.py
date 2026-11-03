"""The DiffRepair inference loop (up to K_max = 3 rounds) and its token-cost accounting.

repair(): greedy completion -> while the reference checker rejects it and rounds remain:
diagnose the first failed check, route one edit (ttso.core.routing), rewind the KV cache to
prompt_len + t*, feed y* and greedily regenerate the suffix. Every regenerated token, y* included,
counts toward `extra_tokens`. Edits are never rolled back.

best_of_n(): the resampling baseline. Draws whole completions at temperature T until it has spent a
given token budget (matched per problem to a repair run's extra tokens) or found a valid one.
"""

import time
from dataclasses import dataclass, field

import torch

from ttso.cartridges.base import as_list
from ttso.core.routing import RepairState, Router, diagnose


@dataclass
class RepairResult:
    tokens: list
    text: str
    verdict: object
    rounds: int = 0
    extra_tokens: int = 0
    greedy_tokens: int = 0
    history: list = field(default_factory=list)
    seconds: float = 0.0
    greedy_valid: bool = False


class RepairLoop:
    def __init__(self, backend, cartridge, policy="smart", k_max=3, max_new_tokens=256,
                 verifier_dtype=torch.float32, verifier_device="cpu", constrain=True):
        self.backend, self.cart = backend, cartridge
        self.router = Router(cartridge, policy)
        self.k_max, self.max_new = k_max, max_new_tokens
        self.dtype = verifier_dtype
        self.net = cartridge.compile(verifier_device)
        self.constrain = constrain

    def _hooks(self, head):
        head = list(head)
        stop = lambda new: self.cart.stop(head + new)
        proc = (lambda new, z: self.cart.logits_processor(head + new, z)) if self.constrain else None
        return stop, proc

    def greedy(self, prompt_ids):
        stop, proc = self._hooks([])
        toks, z, cache = self.backend.generate_from(prompt_ids, None, self.max_new, stop=stop, logits_processor=proc)
        return toks, z, cache

    def repair(self, prompt_ids, greedy=None):
        t0 = time.time()
        P = len(prompt_ids)
        toks, z, cache = greedy if greedy is not None else self.greedy(prompt_ids)
        toks = as_list(toks)
        res = RepairResult(toks, "", None, greedy_tokens=len(toks))
        state = RepairState()
        verdict = self.cart.check(toks)
        res.greedy_valid = verdict.valid
        for k in range(self.k_max):
            if verdict.valid:
                break
            d = diagnose(self.net, self.cart, toks, z, dtype=self.dtype, detach_state=self.router.detach_state)
            rec = {"round": k + 1, "kind": d.kind, "t_error": d.t_error, "why": d.why, "flagged": d.flagged,
                   "t_decl": d.t_decl, "t_call": d.t_call, "consistent_uses": d.consistent_uses}
            e = self.router.choose(d, toks, z, state) if d.kind != "valid" else None
            if e is None:
                rec["edit"] = None
                res.history.append(rec)
                break
            state.after_edit(e, self.router.ban_after(toks, e))
            texts = self.backend.token_texts()
            rec.update(t_star=e.t, old=texts[e.old], new=texts[e.new], case=e.case, route=e.route, site=e.site)
            cache = self.backend.rewind_kv_cache(cache, P + e.t)
            head = toks[:e.t] + [e.new]
            stop, proc = self._hooks(head)
            if self.cart.stop(head):
                new, z_new = [], torch.zeros(0, z.shape[1])
            else:
                new, z_new, cache = self.backend.generate_from([e.new], cache, max(self.max_new - e.t - 1, 0),
                                                               stop=stop, logits_processor=proc)
            toks = head + as_list(new)
            z = torch.cat([z[:e.t + 1], z_new.to(z.dtype)], 0)
            res.extra_tokens += 1 + len(new)
            res.rounds = k + 1
            verdict = self.cart.check(toks)
            rec["valid_after"] = verdict.valid
            res.history.append(rec)
        res.tokens, res.verdict = toks, verdict
        res.text = self.backend.decode(toks)
        res.seconds = time.time() - t0
        return res

    def best_of_n(self, prompt_ids, budget, temperature=0.7, seed=0):
        """Sample until `budget` extra tokens are spent; returns (best result, samples, tokens)."""
        g = torch.Generator().manual_seed(seed)
        spent, n, best = 0, 0, None
        stop, proc = self._hooks([])
        while spent < budget:
            toks, _, _ = self.backend.generate_from(prompt_ids, None, min(self.max_new, budget - spent),
                                                    stop=stop, logits_processor=proc,
                                                    temperature=temperature, generator=g)
            toks = as_list(toks)
            spent += max(len(toks), 1)
            n += 1
            v = self.cart.check(toks)
            if best is None or (v.valid and not best[1].valid):
                best = (toks, v)
            if v.valid:
                break
        return best, n, spent
