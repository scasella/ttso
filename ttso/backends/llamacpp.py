"""llama-cpp-python backend (Metal): KV rewind with llama_kv_cache_seq_rm.

The llama context owns the KV cache, so the `kv_cache` handle passed around is the number of
positions it holds. Rewinding removes positions >= n from sequence 0 and resets `n_tokens`.
`logits_all=True` keeps every evaluated position's logits in `Llama.scores` (n_ctx x V floats).
"""

import numpy as np
import torch

from ttso.backends.base import Backend, pick


class LlamaCppBackend(Backend):
    def __init__(self, model_path, n_ctx=1024, n_gpu_layers=-1, seed=0, verbose=False):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=True,
                         seed=seed, verbose=verbose)
        self.eos_ids = {self.llm.token_eos()}
        self._texts = None

    @property
    def vocab_size(self):
        return self.llm.n_vocab()

    def encode(self, text):
        return self.llm.tokenize(text.encode(), add_bos=False, special=False)

    def decode(self, ids):
        return self.llm.detokenize(list(ids)).decode("utf-8", errors="replace")

    def token_texts(self):
        if self._texts is None:
            self._texts = [self.llm.detokenize([i]).decode("utf-8", errors="replace") for i in range(self.vocab_size)]
        return self._texts

    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None):
        if kv_cache is None:
            self.llm.reset()
            self.llm._ctx.kv_cache_seq_rm(-1, 0, -1)
        else:
            assert kv_cache == self.llm.n_tokens, (kv_cache, self.llm.n_tokens)
        self.llm.eval(list(prefix_ids))
        out, rows = [], []
        for _ in range(max_new_tokens):
            z = torch.from_numpy(np.array(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float32))
            y = pick(logits_processor(out, z) if logits_processor else z, temperature, generator)
            if y in self.eos_ids:
                break
            out.append(y)
            rows.append(z)
            if stop and stop(out):
                break
            self.llm.eval([y])
        z_all = torch.stack(rows) if rows else torch.zeros(0, self.vocab_size)
        return out, z_all, self.llm.n_tokens

    def rewind_kv_cache(self, kv_cache, target_seq_len):
        if target_seq_len < self.llm.n_tokens:
            self.llm._ctx.kv_cache_seq_rm(0, target_seq_len, -1)
            self.llm.n_tokens = target_seq_len
        return self.llm.n_tokens

    def cache_len(self, kv_cache):
        return 0 if kv_cache is None else kv_cache
