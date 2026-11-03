"""Backend interface: decoding with per-step logit caching and KV-cache rewind.

generate_from(prefix_ids, kv_cache, max_new_tokens) feeds `prefix_ids` on top of `kv_cache` (None:
an empty cache) and decodes up to max_new_tokens tokens. It returns (token_ids, cached_logits,
kv_cache): the new tokens, a (n, V) float32 cpu tensor whose row i holds the raw (unmasked)
pre-softmax logits token i was chosen from, and the cache, which holds every token fed so far
(the last generated token is not fed). rewind_kv_cache(kv_cache, n) keeps the first n positions.

Decoding is greedy unless temperature > 0. `stop(new_ids)` ends decoding after a token;
`logits_processor(new_ids, z)` constrains the choice (the cached row stays raw). EOS ends decoding
and is not returned.
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

    @abstractmethod
    def token_texts(self):
        """Decoded text of every token id."""

    @abstractmethod
    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None):
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


def pick(z, temperature=0.0, generator=None):
    if temperature and temperature > 0:
        p = torch.softmax(z.float() / temperature, -1)
        return int(torch.multinomial(p.cpu(), 1, generator=generator))
    return int(z.argmax())
