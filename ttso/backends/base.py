"""Backend interface: decoding with per-step logit caching and KV-cache rewind.

generate_from(prefix_ids, kv_cache, max_new_tokens) feeds `prefix_ids` on top of `kv_cache` (None:
an empty cache) and decodes up to max_new_tokens tokens. It returns (token_ids, cached_logits,
kv_cache): the new tokens, a (n, V) float32 cpu tensor whose row i holds the raw (unmasked)
pre-softmax logits token i was chosen from, and the cache, which holds every token fed so far
(the last generated token is not fed). rewind_kv_cache(kv_cache, n) keeps the first n positions.

Decoding is greedy unless temperature > 0. `stop(new_ids)` ends decoding after a token;
`logits_processor(new_ids, z)` constrains the choice (the cached row stays raw); `accept(new_ids, v)`
is a lazy token filter (the forward structural mask). EOS ends decoding and is not returned.
"""

from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    eos_ids: set = set()

    @property
    @abstractmethod
    def vocab_size(self):
        ...

    @abstractmethod
    def encode(self, text):
        ...

    @abstractmethod
    def decode(self, ids):
        ...

    def prompt_ids(self, text):
        """A prompt's token ids (a backend whose model needs a BOS token adds it here)."""
        return self.encode(text)

    @abstractmethod
    def token_texts(self):
        """Decoded text of every token id."""

    @abstractmethod
    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None, accept=None):
        ...

    @abstractmethod
    def rewind_kv_cache(self, kv_cache, target_seq_len):
        ...

    @abstractmethod
    def cache_len(self, kv_cache):
        ...

    def peak_memory(self):
        """Peak accelerator memory in bytes seen so far (None when not tracked)."""
        return None

    def fork_cache(self, kv_cache):
        """An independent copy of kv_cache, so one greedy decode can seed several repair runs."""
        raise NotImplementedError(f"{type(self).__name__} cannot copy its KV cache")


def bind(accept, out):
    """accept(new_ids, v) -> a one-argument filter over the current `out`, keeping `candidates`."""
    if accept is None:
        return None
    f = lambda v: accept(out, v)
    if hasattr(accept, "candidates"):
        f.candidates = lambda: accept.candidates(out)
    return f


def pick(z, temperature=0.0, generator=None, accept=None):
    """Greedy or temperature sampling; with `accept(token) -> bool`, only accepted tokens.

    The top token (or the first draw) is tried first. If it is rejected and the filter offers
    `candidates()` (a boolean superset of the accepted tokens), only those are considered: greedy
    takes the highest-scoring accepted one; sampling redraws without replacement until a draw is
    accepted, which samples exactly from the distribution restricted to accepted tokens."""
    if accept is None:
        if temperature and temperature > 0:
            p = torch.softmax(z.float() / temperature, -1)
            return int(torch.multinomial(p.cpu(), 1, generator=generator))
        return int(z.argmax())
    sampling = bool(temperature and temperature > 0)
    zf = z.float().cpu()
    if sampling:
        p = torch.softmax(zf / temperature, -1)
        v = int(torch.multinomial(p, 1, generator=generator))
    else:
        v = int(zf.argmax())
    if accept(v):
        return v
    cand = accept.candidates() if hasattr(accept, "candidates") else torch.ones_like(zf, dtype=torch.bool)
    cand = cand.clone()
    cand[v] = False
    if sampling:
        p = p * cand
        for _ in range(int(cand.sum())):
            if float(p.sum()) <= 0:  # every remaining candidate underflowed: fall back to greedy
                break
            v = int(torch.multinomial(p, 1, generator=generator))
            if accept(v):
                return v
            p[v] = 0.0
            cand[v] = False
    s = zf.masked_fill(~cand, float("-inf"))
    for v in torch.argsort(s, descending=True)[:int(cand.sum())].tolist():
        if accept(v):
            return v
    return int(zf.argmax())
