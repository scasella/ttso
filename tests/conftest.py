import os

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = os.environ.get("TTSO_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def qwen_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def qwen_texts(qwen_tokenizer):
    return qwen_tokenizer.batch_decode([[i] for i in range(len(qwen_tokenizer))])


@pytest.fixture(scope="session")
def hf_run():
    """Gate 4's HuggingFace run (tests/gate4_setup.py), shared by Gate 4 and the incremental-mask gate."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple mps")
    from tests.gate4_setup import run_hf
    return run_hf()
