"""HuggingFace Transformers backend: float16 on Apple `mps`, DynamicCache rewind by crop."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ttso.backends.base import Backend, pick


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class HuggingFaceMPSBackend(Backend):
    def __init__(self, model_id="Qwen/Qwen3-0.6B", device=None, dtype=torch.float16):
        self.device = device or default_device()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)
        self._texts = None
        self._peak = 0
        self.forward_calls = 0

    @property
    def vocab_size(self):
        return self.model.config.vocab_size  # logits width (>= len(tokenizer))

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids)

    def token_texts(self):
        if self._texts is None:
            texts = self.tok.batch_decode([[i] for i in range(len(self.tok))])
            self._texts = texts + [""] * (self.vocab_size - len(texts))
        return self._texts

    def _track(self):
        if self.device == "mps":
            self._peak = max(self._peak, torch.mps.current_allocated_memory())
        elif str(self.device).startswith("cuda"):
            self._peak = max(self._peak, torch.cuda.max_memory_allocated())

    def peak_memory(self):
        return self._peak

    def reset_peak(self):
        self._peak = 0

    @torch.inference_mode()
    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None):
        cache = kv_cache if kv_cache is not None else DynamicCache()
        ids = torch.tensor([list(prefix_ids)], device=self.device)
        out, rows = [], []
        for _ in range(max_new_tokens):
            o = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            self.forward_calls += 1
            cache = o.past_key_values
            z = o.logits[0, -1, :].detach().float().cpu().clone()  # raw logits, cached as-is
            self._track()
            y = pick(logits_processor(out, z) if logits_processor else z, temperature, generator)
            if y in self.eos_ids:
                break
            out.append(y)
            rows.append(z)
            if stop and stop(out):
                break
            ids = torch.tensor([[y]], device=self.device)
        z_all = torch.stack(rows) if rows else torch.zeros(0, self.vocab_size)
        return out, z_all, cache

    def rewind_kv_cache(self, kv_cache, target_seq_len):
        cur = kv_cache.get_seq_length()
        if target_seq_len < cur:
            try:
                kv_cache.crop(target_seq_len - cur)  # transformers >= 5: number of tokens to remove (< 0)
            except ValueError:
                kv_cache.crop(target_seq_len)  # transformers 4.x: absolute length
        assert kv_cache.get_seq_length() == target_seq_len
        return kv_cache

    def cache_len(self, kv_cache):
        return 0 if kv_cache is None else kv_cache.get_seq_length()

    @torch.inference_mode()
    def score(self, ids):
        """Teacher-forced logits (len(ids), V) for every position (row i predicts ids[i + 1])."""
        o = self.model(input_ids=torch.tensor([list(ids)], device=self.device))
        return o.logits[0].float().cpu()
