"""llama-cpp-python backend (Metal): KV rewind with llama_kv_cache_seq_rm.

The llama context owns the KV cache, so the `kv_cache` handle passed around is the number of
positions it holds. Rewinding removes positions >= n from sequence 0 and resets `n_tokens`.
`logits_all=True` keeps every evaluated position's logits in `Llama.scores` (n_ctx x V floats).

Memory: llama.cpp allocates its buffers once, at load. `buffers` sums the sizes llama.cpp logs for
them ("... buffer size = X MiB"), split into Metal and CPU, plus the host-side `scores` array.
"""

import re

import numpy as np
import torch

from ttso.backends.base import Backend, bind, pick

_LOG = []
_BUF = re.compile(r"(\w+)(?:_Mapped)?\s+(?:model |KV |compute |output |)buffer size =\s+([\d.]+) MiB")
_CALLBACK = None  # module-level: llama.cpp logs while freeing a context, so the callback must outlive it


def _install_log_capture():
    global _CALLBACK
    if _CALLBACK is not None:
        return
    import ctypes

    import llama_cpp

    @llama_cpp.llama_log_callback
    def capture(level, text, user_data):
        _LOG.append(text.decode("utf-8", errors="replace"))

    _CALLBACK = capture
    llama_cpp.llama_log_set(_CALLBACK, ctypes.c_void_p(0))


class LlamaCppBackend(Backend):
    def __init__(self, model_path, n_ctx=2048, n_gpu_layers=-1, seed=0, verbose=False):
        from llama_cpp import Llama
        _install_log_capture()
        start = len(_LOG)
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, logits_all=True,
                         seed=seed, verbose=verbose)
        self.eos_ids = {self.llm.token_eos()}
        self._texts = None
        mib = {"metal": 0.0, "cpu": 0.0}
        for line in "".join(_LOG[start:]).splitlines():
            m = _BUF.search(line)
            if m:
                mib["metal" if "metal" in m.group(1).lower() or "mtl" in m.group(1).lower() else "cpu"] += float(m.group(2))
        self.buffers = {"metal_mib": mib["metal"], "cpu_mib": mib["cpu"],
                        "scores_mib": self.llm.scores.nbytes / 2**20}

    def peak_memory(self):
        """Metal buffers llama.cpp allocated at load (bytes); its allocation does not grow later."""
        return int(self.buffers["metal_mib"] * 2**20)

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
                      temperature=0.0, generator=None, accept=None):
        if kv_cache is None:
            self.llm.reset()
            self.llm._ctx.kv_cache_seq_rm(-1, 0, -1)
        else:
            assert kv_cache == self.llm.n_tokens, (kv_cache, self.llm.n_tokens)
        self.llm.eval(list(prefix_ids))
        out, rows = [], []
        for _ in range(max_new_tokens):
            z = torch.from_numpy(np.array(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float32))
            y = pick(logits_processor(out, z) if logits_processor else z, temperature, generator,
                     bind(accept, out))
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
