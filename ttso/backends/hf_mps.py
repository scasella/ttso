"""HuggingFace Transformers backend: float16 on Apple `mps`, KV-cache rewind.

Pure-attention models (Qwen3) rewind with `DynamicCache.crop`. Hybrid models whose linear-attention
layers carry a recurrent state (Qwen3.5's Gated DeltaNet layers) cannot be cropped: the state is
overwritten in place every step and keeps no history. For those the backend checkpoints every
linear layer's conv and recurrent state at the end of each multi-token feed (the prompt prefill)
and every `ckpt_every` positions. Rewinding to n restores the latest checkpoint c <= n, crops the
attention layers to c and re-feeds the tokens c..n-1 one at a time, the same single-token path
that produced them.

Sliding-window attention layers (Gemma 4's local layers) keep only the last `window - 1` positions,
so `crop` cannot rewind them once a sequence passes the window. The backend switches them to
recording mode when it builds a cache: they keep their full history and still hand attention the
same window, so decoding is unchanged and a rewind slices them back exactly, with no replay.

Multimodal checkpoints whose text decoder has its own causal-LM class (Gemma 4: the
`model.language_model.*` weights of Gemma4ForConditionalGeneration) are loaded text-only: the
vision and audio towers are never read from disk. Gemma 4's per-layer embedding table (2.35B of
E2B's 4.6B text parameters, a pure lookup) stays in a read-only memory map of the checkpoint on the
cpu (map_ple, when the checkpoint already has the requested dtype): a step reads only its tokens'
rows, the pages are file-backed (the OS drops them under pressure instead of swapping), and the
output is the same. Prompts start with the tokenizer's BOS token when
it has one (Gemma; Qwen's tokenizers have none); `encode` itself never adds special tokens.
"""

import copy
import json
import os
import struct

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer, LinearAttentionCacheLayerMixin

from ttso.backends.base import Backend, bind, pick


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


# multimodal wrapper model_type -> (text-only causal LM class name, checkpoint prefix of its decoder)
TEXT_ONLY = {"gemma4": ("Gemma4ForCausalLM", r"^model\.language_model\.")}


# checkpoint key of a per-layer embedding table to keep memory-mapped (map_ple)
PLE_KEY = {"gemma4": "model.language_model.embed_tokens_per_layer.weight"}
_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _checkpoint_file(model_id, key, revision=None):
    from huggingface_hub import hf_hub_download
    get = (lambda f: os.path.join(model_id, f)) if os.path.isdir(model_id) else \
        (lambda f: hf_hub_download(model_id, f, revision=revision))
    try:
        index = json.load(open(get("model.safetensors.index.json")))
        return get(index["weight_map"][key])
    except Exception:  # noqa: BLE001  (a single-file checkpoint)
        return get("model.safetensors")


def mapped_tensor(path, key):
    """A cpu tensor viewing `key` inside a safetensors file through a private, read-only-in-practice
    memory map: nothing is read until it is indexed."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(n))[key]
    a, b = info["data_offsets"]
    raw = torch.from_file(path, shared=False, size=os.path.getsize(path), dtype=torch.uint8)
    return raw[8 + n + a:8 + n + b].view(_ST_DTYPES[info["dtype"]]).view(info["shape"])


class MappedEmbedding(torch.nn.Module):
    """A scaled embedding (Gemma4TextScaledWordEmbedding's output) whose table is a memory-mapped cpu
    tensor. The table is held in a list, so it is neither a parameter nor a buffer and model.to()
    leaves it on the cpu; rows are gathered there and moved to the ids' device."""

    def __init__(self, table, scale):
        super().__init__()
        self.table, self.scale = [table], scale

    def forward(self, input_ids):
        t = self.table[0]
        out = torch.nn.functional.embedding(input_ids.to("cpu"), t) * torch.tensor(self.scale, dtype=t.dtype)
        return out.to(input_ids.device)


def load_causal_lm(model_id, dtype, revision=None, map_ple=True):
    """The model's causal LM; for a multimodal checkpoint listed in TEXT_ONLY, only the text decoder
    (its tied LM head included), with the vision / audio weights skipped at load. With map_ple, a
    per-layer embedding table stored in `dtype` is swapped for a MappedEmbedding."""
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    if cfg.model_type not in TEXT_ONLY:
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    import transformers
    cls_name, prefix = TEXT_ONLY[cfg.model_type]
    model = getattr(transformers, cls_name).from_pretrained(
        model_id, revision=revision, config=cfg.get_text_config(), dtype=dtype, key_mapping={prefix: "model."})
    key = PLE_KEY.get(cfg.model_type)
    emb = getattr(model.model, "embed_tokens_per_layer", None)
    if map_ple and key and emb is not None:
        table = mapped_tensor(_checkpoint_file(model_id, key, revision), key)
        if table.dtype == dtype and tuple(table.shape) == tuple(emb.weight.shape):
            model.model.embed_tokens_per_layer = MappedEmbedding(table, float(emb.embed_scale))
    return model


class HuggingFaceMPSBackend(Backend):
    def __init__(self, model_id="Qwen/Qwen3-0.6B", device=None, dtype=torch.float16, ckpt_every=16, revision=None,
                 map_ple=True):
        self.device = device or default_device()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = load_causal_lm(model_id, dtype, revision, map_ple).to(self.device).eval()
        # bytes of the weights on the device, and of tables left memory-mapped on the cpu (map_ple)
        self.param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.mapped_bytes = sum(m.table[0].numel() * m.table[0].element_size() for m in self.model.modules()
                                if isinstance(m, MappedEmbedding))
        self.bos_id = self.tok.bos_token_id  # prepended to prompts (prompt_ids); None for Qwen tokenizers
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)
        text_cfg = self.model.config.get_text_config(decoder=True)
        self.hybrid = "linear_attention" in (getattr(text_cfg, "layer_types", None) or [])
        self.ckpt_every = ckpt_every
        self.ckpt_bytes_peak = 0
        # rewind_kv_cache calls (including no-op rewinds to the current length) and tokens re-fed
        # after a checkpoint restore
        self.rewinds, self.replayed = 0, 0
        self._texts = None
        self._peak = 0

    @property
    def vocab_size(self):
        return self.model.get_output_embeddings().weight.shape[0]  # logits width (>= len(tokenizer))

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, text):
        """A prompt's ids: encode(text), led by the tokenizer's BOS token when it has one and the text
        (e.g. a rendered chat template) does not already start with it."""
        ids = self.encode(text)
        return [self.bos_id] + ids if self.bos_id is not None and ids[:1] != [self.bos_id] else ids

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
        """Peak PyTorch MPS tensor allocation seen while decoding (torch.mps.current_allocated_memory;
        a memory-mapped table on the cpu is not in it)."""
        return self._peak

    @staticmethod
    def peak_rss():
        """Peak resident set size of this process in bytes (ru_maxrss, bytes on macOS): cpu memory,
        including the touched pages of a memory-mapped table."""
        import resource
        import sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024

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
        if not self.hybrid:
            for layer in cache.layers:
                if isinstance(layer, DynamicSlidingWindowLayer):
                    layer.activate_past_recording()  # keep the full history, so a rewind is exact
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
        self.rewinds += 1
        if target_seq_len == cur:
            return kv_cache
        if self.hybrid:
            c = max(p for p in kv_cache._ttso_ck if p <= target_seq_len)
            self._restore(kv_cache, c)
            for layer in kv_cache.layers:  # attention KV, including the attention half of a hybrid layer
                if isinstance(layer, DynamicLayer) and DynamicLayer.get_seq_length(layer) > c:
                    DynamicLayer.crop(layer, c - DynamicLayer.get_seq_length(layer))
            replay = kv_cache._ttso_ids[c:target_seq_len]
            self.replayed += len(replay)
            kv_cache._ttso_ids = kv_cache._ttso_ids[:c]
            kv_cache._ttso_ck = {p: s for p, s in kv_cache._ttso_ck.items() if p <= c}
            for y in replay:
                _, kv_cache = self._forward(kv_cache, [y])
        elif any(isinstance(layer, DynamicSlidingWindowLayer) for layer in kv_cache.layers):
            for layer in kv_cache.layers:
                if isinstance(layer, DynamicSlidingWindowLayer):
                    # recording mode: the buffer holds every position, so slicing it is the rewind
                    assert layer.record_past and layer.keys.shape[-2] == layer.cumulative_length
                    layer.keys = layer.keys[:, :, :target_seq_len]
                    layer.values = layer.values[:, :, :target_seq_len]
                    layer.cumulative_length = target_seq_len
                else:
                    DynamicLayer.crop(layer, target_seq_len - cur)
        else:
            try:
                kv_cache.crop(target_seq_len - cur)  # transformers >= 5: number of tokens to remove (< 0)
            except ValueError:
                kv_cache.crop(target_seq_len)  # transformers 4.x: absolute length
        assert kv_cache.get_seq_length() == target_seq_len
        return kv_cache

    @torch.inference_mode()
    def fork_cache(self, kv_cache):
        return copy.deepcopy(kv_cache)  # includes the hybrid checkpoints (_ttso_ck, _ttso_ids)

    def cache_len(self, kv_cache):
        return 0 if kv_cache is None else kv_cache.get_seq_length()
