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
