"""A deterministic test double: a language model defined by a weighted set of token sequences.

P(next | context) is proportional to the total weight of the scripts that continue `context` with
`next` (a script that ends exactly at `context` votes for EOS), plus a small floor for every other
token. Greedy decoding therefore follows the heaviest script consistent with the prefix, and an
edit that leaves every script stops at once (counted in `off_script`). The KV cache is the
context itself, so rewinding is truncation. Used by Gate 3 to replay recorded trajectories.
"""

import math

import torch

from ttso.backends.base import Backend, pick

FLOOR = 1e-6


class ScriptedBackend(Backend):
    def __init__(self, tokenizer, scripts, texts=None):
        """scripts: list of (weight, text or token ids) -- each a full context (prompt included)."""
        self.tok = tokenizer
        self.scripts = [(w, list(s) if not isinstance(s, str) else self.encode(s)) for w, s in scripts]
        self.eos = tokenizer.eos_token_id
        self.eos_ids = {self.eos}
        self._texts = texts
        self.off_script = 0

    @property
    def vocab_size(self):
        return len(self.tok)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids)

    def token_texts(self):
        if self._texts is None:
            self._texts = [self.tok.decode([i]) for i in range(self.vocab_size)]
        return self._texts

    def logits(self, ctx):
        mass = {}
        n = len(ctx)
        for w, s in self.scripts:
            if len(s) >= n and s[:n] == ctx:
                nxt = s[n] if len(s) > n else self.eos
                mass[nxt] = mass.get(nxt, 0.0) + w
        z = torch.full((self.vocab_size,), math.log(FLOOR))
        for v, m in mass.items():
            z[v] = math.log(m)
        return z, bool(mass)

    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None):
        ctx = list(kv_cache or []) + list(prefix_ids)
        out, rows = [], []
        for _ in range(max_new_tokens):
            z, on = self.logits(ctx)
            if not on:
                self.off_script += 1
                break
            zz = logits_processor(out, z) if logits_processor else z
            y = pick(zz, temperature, generator)
            if y in self.eos_ids:
                break
            out.append(y)
            rows.append(z)
            ctx.append(y)
            if stop and stop(out):
                break
        V = self.vocab_size
        return out, (torch.stack(rows) if rows else torch.zeros(0, V)), ctx[:-1] if out else ctx

    def rewind_kv_cache(self, kv_cache, target_seq_len):
        return list(kv_cache[:target_seq_len])

    def cache_len(self, kv_cache):
        return len(kv_cache or [])
