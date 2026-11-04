"""The DiffRepair inference loop (up to K_max = 3 rounds) and its token-cost accounting.

repair(): greedy completion -> while the reference checker rejects it (or the cartridge's task
contract, `contract_ok`) and rounds remain:
diagnose the first failed check, route one edit (ttso.core.routing), rewind the KV cache to
prompt_len + t*, feed y* and greedily regenerate the suffix. Every regenerated token, y* included,
counts toward `extra_tokens`. Edits are never rolled back.

best_of_n(): the resampling baseline. Draws completions at temperature T until it has spent a
given token budget (matched per problem to a repair run's extra tokens) or found a valid one. Each
sample may use at most the budget left, so the last one is often cut off before it finishes;
`self.last_samples` records every sample as "complete" (ended by EOS or the cartridge's stop),
"truncated" (cut off by the budget) or "capped" (cut off at max_new_tokens).
best_of_k(): K full samples (max_new_tokens each, no budget), stopping at the first valid one. With
the same seed its first sample is best_of_n's first sample, not cut off.
"""

import copy
import time
from dataclasses import dataclass, field

import torch

from ttso.cartridges.base import as_list
from ttso.core.routing import RepairState, Router, diagnose
from ttso.core.symbolic import SymbolicProvenanceVerifier


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


class ForwardMask:
    """The cartridge's forward structural mask as a decoding filter after a fixed head."""

    def __init__(self, cart, head):
        self.cart, self.head = cart, list(head)

    def __call__(self, new, v):
        return self.cart.accept(self.head + list(new), v)

    def candidates(self, new):
        return self.cart.candidates(self.head + list(new))


class RepairLoop:
    """verifier: "deltanet" (CompiledDeltaNet backward routing) or "symbolic"
    (SymbolicProvenanceVerifier); both feed the same Router. shadow=True also runs the other one at
    every round, from the same tokens, logits and a copy of the routing state, and records whether
    the two diagnoses and edits agree (rec["shadow"]); the loop follows `verifier`."""

    def __init__(self, backend, cartridge, policy="smart", k_max=3, max_new_tokens=256,
                 verifier_dtype=torch.float32, verifier_device="cpu", constrain=True, verifier="deltanet",
                 shadow=False):
        assert verifier in ("deltanet", "symbolic"), verifier
        self.backend, self.cart = backend, cartridge
        self.router = Router(cartridge, policy)
        self.k_max, self.max_new = k_max, max_new_tokens
        self.dtype = verifier_dtype
        self.net = cartridge.compile(verifier_device)
        self.sym = SymbolicProvenanceVerifier.for_cartridge(cartridge)
        self.verifier, self.shadow = verifier, shadow
        self.constrain = constrain

    def _diagnose(self, which, toks, z, lexed):
        if which == "symbolic":
            return self.sym.diagnose(self.cart, toks, detach_state=self.router.detach_state, lexed=lexed)
        return diagnose(self.net, self.cart, toks, z, dtype=self.dtype, detach_state=self.router.detach_state,
                        lexed=lexed)

    @staticmethod
    def _key(d):  # a diagnosis as routing sees it, before choose() filters frozen tokens out
        return (d.kind, d.t_error, d.slot, d.t_decl, d.t_call, tuple(d.flagged), d.consistent_uses)

    def _shadow(self, toks, z, state, lexed, key_main, e, t_main):
        other = "symbolic" if self.verifier == "deltanet" else "deltanet"
        t0 = time.perf_counter()
        d2 = self._diagnose(other, toks, z, lexed)
        t_other = time.perf_counter() - t0
        key_other = self._key(d2)
        e2 = self.router.choose(d2, toks, z, state) if d2.kind != "valid" else None
        edit = lambda x: None if x is None else (x.t, x.new, x.case, x.route)
        return {"same_diagnosis": key_main == key_other, "same_edit": edit(e) == edit(e2),
                self.verifier: edit(e), other: edit(e2), f"{self.verifier}_s": t_main, f"{other}_s": t_other}

    def _hooks(self, head):
        head = list(head)
        stop = lambda new: self.cart.stop(head + new)
        proc = (lambda new, z: self.cart.logits_processor(head + new, z)) if self.constrain else None
        acc = ForwardMask(self.cart, head) if getattr(self.cart, "apply_forward_mask", False) else None
        return stop, proc, acc

    def greedy(self, prompt_ids):
        stop, proc, acc = self._hooks([])
        toks, z, cache = self.backend.generate_from(prompt_ids, None, self.max_new, stop=stop, logits_processor=proc,
                                                    accept=acc)
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
            if self.done(toks, verdict):
                break
            lexed = self.cart.lex(toks)
            t0 = time.perf_counter()
            d = self._diagnose(self.verifier, toks, z, lexed)
            t_main = time.perf_counter() - t0
            rec = {"round": k + 1, "kind": d.kind, "t_error": d.t_error, "why": d.why, "flagged": list(d.flagged),
                   "t_decl": d.t_decl, "t_call": d.t_call, "consistent_uses": d.consistent_uses}
            state_before, key_main = (copy.deepcopy(state), self._key(d)) if self.shadow else (None, None)
            e = self.router.choose(d, toks, z, state) if d.kind != "valid" else None
            if self.shadow:
                rec["shadow"] = self._shadow(toks, z, state_before, lexed, key_main, e, t_main)
            if e is None:
                rec["edit"] = None
                res.history.append(rec)
                break
            state.after_edit(e, self.router.ban_after(toks, e))
            texts = self.backend.token_texts()
            rec.update(t_star=e.t, old=texts[e.old], new=texts[e.new], case=e.case, route=e.route, site=e.site)
            cache = self.backend.rewind_kv_cache(cache, P + e.t)
            head = toks[:e.t] + [e.new]
            stop, proc, acc = self._hooks(head)
            if self.cart.stop(head):
                new, z_new = [], torch.zeros(0, z.shape[1])
            else:
                new, z_new, cache = self.backend.generate_from([e.new], cache, max(self.max_new - e.t - 1, 0),
                                                               stop=stop, logits_processor=proc, accept=acc)
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

    def done(self, tokens, verdict):
        """The loop's stopping oracle: the reference checker accepts and the task contract holds."""
        return verdict.valid and self.cart.contract_ok(tokens)

    def _sample(self, prompt_ids, limit, temperature, g):
        stop, proc, acc = self._hooks([])
        toks, _, _ = self.backend.generate_from(prompt_ids, None, limit, stop=stop, logits_processor=proc,
                                                accept=acc, temperature=temperature, generator=g)
        toks = as_list(toks)
        ended = len(toks) < limit or self.cart.stop(toks)  # EOS, or the document is complete
        kind = "complete" if ended else ("truncated" if limit < self.max_new else "capped")
        return toks, self.cart.check(toks), kind

    def best_of_n(self, prompt_ids, budget, temperature=0.7, seed=0):
        """Sample until `budget` extra tokens are spent; returns (best result, samples, tokens)."""
        g = torch.Generator().manual_seed(seed)
        spent, n, best, self.last_samples = 0, 0, None, []
        while spent < budget:
            toks, v, kind = self._sample(prompt_ids, min(self.max_new, budget - spent), temperature, g)
            spent += max(len(toks), 1)
            n += 1
            self.last_samples.append(kind)
            if best is None or (self.done(toks, v) and not self.done(*best)):
                best = (toks, v)
            if self.done(toks, v):
                break
        return best, n, spent

    def best_of_k(self, prompt_ids, k=2, temperature=0.7, seed=0):
        """Up to k full samples (max_new_tokens each), stopping at the first that passes the loop's
        oracle; returns (best result, samples, tokens)."""
        g = torch.Generator().manual_seed(seed)
        spent, best, self.last_samples = 0, None, []
        for _ in range(k):
            toks, v, kind = self._sample(prompt_ids, self.max_new, temperature, g)
            spent += max(len(toks), 1)
            self.last_samples.append(kind)
            if best is None or (self.done(toks, v) and not self.done(*best)):
                best = (toks, v)
            if self.done(toks, v):
                break
        return best, len(self.last_samples), spent
