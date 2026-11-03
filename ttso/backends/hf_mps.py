"""HuggingFace Transformers backend: float16 on Apple `mps`, KV-cache rewind.

Pure-attention models (Qwen3) rewind with `DynamicCache.crop`. Hybrid models whose linear-attention
layers carry a recurrent state (Qwen3.5's Gated DeltaNet layers) cannot be cropped: the state is
overwritten in place every step and keeps no history. For those the backend checkpoints every
linear layer's conv and recurrent state at the end of each multi-token feed (the prompt prefill)
and every `ckpt_every` positions. Rewinding to n restores the latest checkpoint c <= n, crops the
attention layers to c and re-feeds the tokens c..n-1 one at a time, the same single-token path
that produced them.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionCacheLayerMixin

from ttso.backends.base import Backend, bind, pick


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class HuggingFaceMPSBackend(Backend):
    def __init__(self, model_id="Qwen/Qwen3-0.6B", device=None, dtype=torch.float16, ckpt_every=16):
        self.device = device or default_device()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)
        text_cfg = self.model.config.get_text_config(decoder=True)
        self.hybrid = "linear_attention" in (getattr(text_cfg, "layer_types", None) or [])
        self.ckpt_every = ckpt_every
        self.ckpt_bytes_peak = 0
        self._texts = None
        self._peak = 0

    @property
    def vocab_size(self):
        return self.model.get_output_embeddings().weight.shape[0]  # logits width (>= len(tokenizer))

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

    # --- linear-attention state checkpoints (hybrid models) --------------------------------------------
    @staticmethod
    def _linear_layers(cache):
        return [(i, l) for i, l in enumerate(cache.layers) if isinstance(l, LinearAttentionCacheLayerMixin)]

    def _snapshot(self, cache):
        clone = lambda d: {k: (v.clone() if v is not None else None) for k, v in d.items()}
        cache._ttso_ck[len(cache._ttso_ids)] = {i: (clone(l.conv_states), clone(l.recurrent_states))
                                                for i, l in self._linear_layers(cache)}
        nbytes = sum(t.numel() * t.element_size() for snap in cache._ttso_ck.values() for pair in snap.values()
                     for d in pair for t in d.values() if t is not None)
        self.ckpt_bytes_peak = max(self.ckpt_bytes_peak, nbytes)

    def _restore(self, cache, pos):
        for i, layer in self._linear_layers(cache):
            conv, rec = cache._ttso_ck[pos][i]
            for k, v in conv.items():
                layer.conv_states[k] = v.clone() if v is not None else None
            for k, v in rec.items():
                if v is not None and layer.recurrent_states.get(k) is not None:
                    layer.recurrent_states[k].copy_(v)  # keep the tensor the cache was built around
                else:
                    layer.recurrent_states[k] = v.clone() if v is not None else None

    def _new_cache(self):
        cache = DynamicCache(config=self.model.config)
        if self.hybrid:
            cache._ttso_ids, cache._ttso_ck = [], {}
        return cache

    def _forward(self, cache, ids):
        o = self.model(input_ids=torch.tensor([list(ids)], device=self.device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        cache = o.past_key_values
        if self.hybrid:
            cache._ttso_ids.extend(int(x) for x in ids)
            if len(ids) > 1 or len(cache._ttso_ids) % self.ckpt_every == 0:
                self._snapshot(cache)
        return o, cache

    @torch.inference_mode()
    def generate_from(self, prefix_ids, kv_cache, max_new_tokens, stop=None, logits_processor=None,
                      temperature=0.0, generator=None, accept=None):
        cache = kv_cache if kv_cache is not None else self._new_cache()
        ids = list(prefix_ids)
        out, rows = [], []
        for _ in range(max_new_tokens):
            o, cache = self._forward(cache, ids)
            z = o.logits[0, -1, :].detach().float().cpu().clone()  # raw logits, cached as-is
            self._track()
            y = pick(logits_processor(out, z) if logits_processor else z, temperature, generator,
                     bind(accept, out))
            if y in self.eos_ids:
                break
            out.append(y)
            rows.append(z)
            if stop and stop(out):
                break
            ids = [y]
        z_all = torch.stack(rows) if rows else torch.zeros(0, self.vocab_size)
        return out, z_all, cache

    @torch.inference_mode()
    def rewind_kv_cache(self, kv_cache, target_seq_len):
        cur = kv_cache.get_seq_length()
        if target_seq_len > cur:
            raise ValueError(f"cannot rewind forward: cache holds {cur} positions, asked for {target_seq_len}")
        if target_seq_len == cur:
            return kv_cache
        if not self.hybrid:
            try:
                kv_cache.crop(target_seq_len - cur)  # transformers >= 5: number of tokens to remove (< 0)
            except ValueError:
                kv_cache.crop(target_seq_len)  # transformers 4.x: absolute length
        else:
            c = max(p for p in kv_cache._ttso_ck if p <= target_seq_len)
            self._restore(kv_cache, c)
            for layer in kv_cache.layers:  # attention KV, including the attention half of a hybrid layer
                if isinstance(layer, DynamicLayer) and DynamicLayer.get_seq_length(layer) > c:
                    DynamicLayer.crop(layer, c - DynamicLayer.get_seq_length(layer))
            replay = kv_cache._ttso_ids[c:target_seq_len]
            kv_cache._ttso_ids = kv_cache._ttso_ids[:c]
            kv_cache._ttso_ck = {p: s for p, s in kv_cache._ttso_ck.items() if p <= c}
            for y in replay:
                _, kv_cache = self._forward(kv_cache, [y])
        assert kv_cache.get_seq_length() == target_seq_len
        return kv_cache

    def cache_len(self, kv_cache):
        return 0 if kv_cache is None else kv_cache.get_seq_length()
