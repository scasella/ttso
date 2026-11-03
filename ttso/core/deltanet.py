"""Zero-parameter Gated DeltaNet verifier with an exact straight-through token bridge.

Memory M_t in R^{S x K} (symbol slot x semantic type), M_0 = 0. A cartridge's lexer turns a token
sequence into ops; every token without an op is a NOOP:

    NOOP              M_t = M_{t-1}
    DECLARE(s)        M_t = M_{t-1} - k (k^T M_{t-1}) + k v^T,   k = e_s,  v = W_decl p_t,  beta = 1
    CHECK_VAR(s, u)   M_t = M_{t-1},  y^ = (k^T M_{t-1})^T,  L = 1 - <y^, u>          (u constant)
    CHECK_CALL(s)     M_t = M_{t-1},  y^ = (k^T M_{t-1})^T,  L = 1 - <y^, W_lit p_t>

Write and read addresses k are hard one-hot slots (a variable reference token never carries
gradient); the gradient of a check reaches the declaration through M, and a CHECK_CALL also
reaches its own literal token. `detach_state=True` cuts every memory read (the LocalOnly regime).

Straight-through token: p_t = onehot(y_t) + (softmax(z_t) - sg(softmax(z_t))). Bracketed this
way the forward value is the hard token exactly in any dtype (the difference is an exact zero;
(onehot + softmax) - sg(softmax) rounds before the subtraction in fp16). `_STProject` computes
W p_t without materialising a |V| vector per token: forward gathers W[:, y_t], backward returns
the softmax Jacobian applied to W^T g, i.e. s * (W^T g - <s, W^T g>), in float32.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

NOOP, DECLARE, CHECK_VAR, CHECK_CALL = range(4)
OP_NAMES = ("NOOP", "DECLARE", "CHECK_VAR", "CHECK_CALL")
PASS_TOL = 1e-4  # a check passes when L < PASS_TOL


@dataclass(frozen=True)
class Op:
    t: int  # token index inside the completion
    kind: int  # DECLARE | CHECK_VAR | CHECK_CALL
    slot: int
    expected: int = -1  # CHECK_VAR: index of the constant expected type u
    tag: str = ""  # human-readable rule name ("param", "group operand", "call arg", ...)


@dataclass
class CheckResult:
    op: Op
    loss: torch.Tensor  # scalar, differentiable when logits were given
    passed: bool
    uses_before: int  # consistent_uses[slot] when the check ran


@dataclass
class Trace:
    checks: list
    consistent_uses: dict
    decl_of: dict  # check index -> token index of the DECLARE the check read (None: undeclared)
    states: torch.Tensor | None = None  # (T, S, K) memory after each token (record_states=True)
    final: torch.Tensor | None = None

    @property
    def first_failure(self):
        bad = [i for i, c in enumerate(self.checks) if not c.passed]
        return min(bad, key=lambda i: self.checks[i].op.t) if bad else None


class _STProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, y, W, out_dtype):
        ctx.save_for_backward(z, W)
        return W[:, y].to(out_dtype)

    @staticmethod
    def backward(ctx, g):
        z, W = ctx.saved_tensors
        s = torch.softmax(z.float(), dim=-1)
        q = g.float().to(W.device) @ W.float()  # W^T g, (V,)
        q = q.to(s.device)
        gz = s * (q - (s * q).sum())
        return gz.to(z.dtype), None, None, None


def st_project(z, y, W, dtype):
    """W p_t for the straight-through token p_t (forward exact, backward through softmax(z))."""
    if z is None or not z.requires_grad:
        return W[:, y].to(dtype)
    return _STProject.apply(z, y, W, dtype)


class CompiledDeltaNet(nn.Module):
    """The verifier. W_decl, W_lit: (K, V) 0/1 projections (buffers; nothing here is learned)."""

    def __init__(self, W_decl, W_lit, n_slots=64):
        super().__init__()
        assert W_decl.shape == W_lit.shape
        self.K, self.V = W_decl.shape
        self.S = n_slots
        self.register_buffer("W_decl", W_decl.float())
        self.register_buffer("W_lit", W_lit.float())
        self.requires_grad_(False)

    def forward(self, tokens, ops, logits=None, dtype=torch.float32, detach_state=False,
                record_states=False):
        """tokens: sequence of T token ids; ops: list of Op; logits: None, a (T, V) tensor or a
        {t: (V,) tensor} mapping of cached pre-softmax logits (may require grad)."""
        dev = self.W_decl.device
        T = len(tokens)
        Wd, Wl = self.W_decl.to(dtype), self.W_lit.to(dtype)
        eye_s = torch.eye(self.S, device=dev, dtype=dtype)
        eye_k = torch.eye(self.K, device=dev, dtype=dtype)
        M = torch.zeros(self.S, self.K, device=dev, dtype=dtype)
        by_t = {}
        for op in ops:
            assert 0 <= op.t < T and 0 <= op.slot < self.S, op
            by_t.setdefault(op.t, []).append(op)
        uses, decl_t, checks, decl_of, states = {}, {}, [], {}, []
        row = (lambda t: None) if logits is None else (lambda t: logits[t])
        for t in range(T):
            for op in by_t.get(t, ()):  # tokens without an op: M_t = M_{t-1}
                k = eye_s[op.slot]
                Mr = M.detach() if detach_state else M
                if op.kind == DECLARE:
                    v = st_project(row(t), int(tokens[t]), Wd, dtype)
                    M = M - torch.outer(k, k @ M) + torch.outer(k, v)
                    uses[op.slot], decl_t[op.slot] = 0, t
                else:
                    y_hat = k @ Mr
                    u = eye_k[op.expected] if op.kind == CHECK_VAR else st_project(row(t), int(tokens[t]), Wl, dtype)
                    loss = 1.0 - torch.dot(y_hat, u)
                    ok = bool(loss.item() < PASS_TOL)
                    decl_of[len(checks)] = decl_t.get(op.slot)
                    checks.append(CheckResult(op, loss, ok, uses.get(op.slot, 0)))
                    if ok:
                        uses[op.slot] = uses.get(op.slot, 0) + 1
            if record_states:
                states.append(M)
        return Trace(checks, uses, decl_of, torch.stack(states).detach() if record_states else None, M)


@dataclass
class Gradients:
    trace: Trace
    check: int | None  # index of the first failed check
    grads: dict = field(default_factory=dict)  # t -> dL/dz_t (V,) float32 on cpu, nonzero rows only


def first_failure_gradients(net, tokens, ops, logits, dtype=torch.float32, detach_state=False, eps=0.0):
    """Backpropagate the first failed check into the cached logits of every token that could carry
    gradient (DECLARE and CHECK_CALL sites); returns the rows with ||g||_inf > eps."""
    rows = sorted({op.t for op in ops if op.kind in (DECLARE, CHECK_CALL)})
    leaves = {t: logits[t].detach().float().to(net.W_decl.device).clone().requires_grad_(True) for t in rows}
    tr = net(tokens, ops, leaves, dtype=dtype, detach_state=detach_state)
    j = tr.first_failure
    out = Gradients(tr, j)
    if j is None:
        return out
    loss = tr.checks[j].loss
    if not loss.requires_grad:
        return out
    t_err = tr.checks[j].op.t
    live = [t for t in rows if t <= t_err]
    gs = torch.autograd.grad(loss, [leaves[t] for t in live], allow_unused=True)
    for t, g in zip(live, gs):
        if g is not None and float(g.abs().max()) > eps:
            out.grads[t] = g.detach().float().cpu()
    return out
