"""Rulebook (cartridge) interface: a lexer that compiles a token sequence into DeltaNet ops, a
pure-Python reference checker that defines validity, and the grammar of each edit site."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch

from ttso.core.deltanet import CompiledDeltaNet


@dataclass
class Lexed:
    ops: list = field(default_factory=list)  # deltanet.Op, sorted by token index
    sites: dict = field(default_factory=dict)  # token index -> "decl" | "lit"
    hard: list = field(default_factory=list)  # (token index, reason): errors with no gradient
    slots: dict = field(default_factory=dict)  # symbol name -> slot
    frozen: set = field(default_factory=set)  # declaration sites fixed by the prompt (never edited)
    local: dict = field(default_factory=dict)  # check token -> token where a local rewrite starts


@dataclass
class Verdict:
    valid: bool
    error: str = ""
    value: object = None  # what the program computes, when the cartridge can evaluate it


class Cartridge(ABC):
    type_names: list
    n_slots: int = 64

    @property
    @abstractmethod
    def tcm(self):
        """The TokenClassManager over this cartridge's vocabulary."""

    @abstractmethod
    def lex(self, tokens) -> Lexed:
        """Compile a completion (token ids, prompt excluded) into DeltaNet ops."""

    @abstractmethod
    def check(self, tokens) -> Verdict:
        """Reference checker: the ground truth for validity."""

    def valid_mask(self, tokens, t, site):
        """Tokens grammatical at position t of `tokens` for an edit at a site of kind `site`."""
        if site == "decl":
            return self.tcm.type_mask(like=int(tokens[t]))
        return ~self.tcm.special

    def stop(self, tokens):
        """True when generation should end after `tokens`."""
        return False

    def logits_processor(self, tokens, z):
        """Constrained-decoding hook applied before picking each token (identity by default)."""
        return z

    def compile(self, device="cpu"):
        return CompiledDeltaNet(self.tcm.W_decl, self.tcm.W_lit, self.n_slots).to(device)


class TextAligner:
    """Map character offsets of the decoded completion back to token indices."""

    def __init__(self, texts):
        self.texts = texts
        self.text = "".join(texts)
        self.starts = np.cumsum([0] + [len(t) for t in texts])

    def tok(self, c):
        return int(np.searchsorted(self.starts, c, side="right") - 1)


def as_list(tokens):
    return [int(x) for x in (tokens.tolist() if torch.is_tensor(tokens) else tokens)]
