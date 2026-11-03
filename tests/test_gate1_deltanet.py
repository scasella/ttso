"""Gate 1: exactness and non-fading memory of the compiled DeltaNet.

* 200 random SynthLang programs (<= 300 tokens): per-token check verdicts and the memory after every
  token agree with the string-level reference checker, in float32 and float16, on cpu and mps.
* The gradient at a declaration is identical 8 and 256 tokens before its use (ratio 1.000000), and
  every token strictly between them receives exactly 0.0.
* The straight-through bridge's custom backward equals autograd through the spec's literal formula
  p_t = onehot(y_t) + softmax(z_t) - sg(softmax(z_t)).
"""

import time

import pytest
import torch

from ttso.cartridges.synthetic import SynthLang, reference_check, rng_for, table_matrix
from ttso.core.deltanet import DECLARE, CHECK_VAR, first_failure_gradients

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
CONFIGS = [(d, dt) for d in DEVICES for dt in (torch.float32, torch.float16)]
N_PROGRAMS = 200


@pytest.fixture(scope="module")
def lang():
    return SynthLang(n_types=8, n_vars=64)


@pytest.fixture(scope="module")
def programs(lang):
    rng = rng_for(1)
    return [lang.random_program(rng, max_len=300) for _ in range(N_PROGRAMS)]


@pytest.mark.parametrize("device,dtype", CONFIGS, ids=[f"{d}-{str(t)[6:]}" for d, t in CONFIGS])
def test_forward_agrees_with_reference(lang, programs, device, dtype):
    net = lang.compile(device)
    n_tok = n_chk = n_fail = 0
    t0 = time.time()
    for prog in programs:
        words = [lang.vocab[i] for i in prog]
        ref = reference_check(words, lang.type_names, lang.suffix)
        lx = lang.lex(prog)
        tr = net(prog, lx.ops, dtype=dtype, record_states=True)
        mine = [None] * len(prog)
        for c in tr.checks:
            mine[c.op.t] = c.passed
        assert mine == ref["verdicts"]
        states = tr.states.float().cpu()
        for t, table in enumerate(ref["tables"]):
            assert torch.equal(states[t], table_matrix(table, lang.type_names, lang.n_slots)), t
        n_tok += len(prog)
        n_chk += len(tr.checks)
        n_fail += sum(not c.passed for c in tr.checks)
    print(f"\n[gate1] {device} {dtype}: {N_PROGRAMS} programs, {n_tok} tokens, {n_chk} checks "
          f"({n_fail} failing): 100% token-for-token agreement ({time.time() - t0:.1f}s)")


def _distance_program(lang, d):
    # the use's type token sits exactly d tokens after the declaration's type token
    words =["let", "x3", ":", "Nat"] + ["pad"] * (d - 4) + ["use", "x3", "@", "U32"]
    return lang.ids(words)


@pytest.mark.parametrize("device,dtype", CONFIGS, ids=[f"{d}-{str(t)[6:]}" for d, t in CONFIGS])
def test_gradient_does_not_fade(lang, device, dtype):
    net = lang.compile(device)
    torch.manual_seed(0)
    z_decl = torch.randn(len(lang.vocab)) * 2
    norms = {}
    for d in (8, 256):
        prog = _distance_program(lang, d)
        lx = lang.lex(prog)
        t_decl = next(o.t for o in lx.ops if o.kind == DECLARE)
        t_use = next(o.t for o in lx.ops if o.kind == CHECK_VAR)
        assert t_use - t_decl == d
        z = torch.randn(len(prog), len(lang.vocab)) * 2
        z[t_decl] = z_decl
        z.requires_grad_(True)
        tr = net(prog, lx.ops, logits=z.to(device), dtype=dtype)
        j = tr.first_failure
        assert j is not None and tr.checks[j].op.t == t_use
        (g,) = torch.autograd.grad(tr.checks[j].loss, z)
        assert torch.count_nonzero(g[t_decl + 1:t_use]) == 0, "gradient leaked into intermediate tokens"
        assert torch.count_nonzero(g[t_use]) == 0, "the use's own token carries no gradient"
        norms[d] = g[t_decl].norm().item()
        assert norms[d] > 0
    ratio = norms[256] / norms[8]
    print(f"\n[gate1] {device} {dtype}: |grad| at t_decl, d=8: {norms[8]:.6e}, d=256: {norms[256]:.6e}, "
          f"ratio {ratio:.6f}")
    assert abs(ratio - 1.0) <= 1e-6


def test_st_bridge_matches_spec_formula(lang):
    """Custom backward == autograd through onehot + softmax - sg(softmax), in float64."""
    torch.manual_seed(1)
    rng = rng_for(7)
    net = lang.compile("cpu")
    W_decl = net.W_decl.double()
    W_lit = net.W_lit.double()
    n_cmp = 0
    for _ in range(40):
        prog = lang.random_program(rng, max_len=120, p_bad=0.6)
        lx = lang.lex(prog)
        z = torch.randn(len(prog), len(lang.vocab), dtype=torch.float64)
        G = first_failure_gradients(net, prog, lx.ops, z, dtype=torch.float64)
        if G.check is None:
            continue
        zz = z.clone().requires_grad_(True)
        p = torch.nn.functional.one_hot(torch.tensor(prog), len(lang.vocab)).double()
        s = torch.softmax(zz, -1)
        p = p + (s - s.detach())
        M = {}
        loss = None
        for op in sorted(lx.ops, key=lambda o: o.t):
            if op.kind == DECLARE:
                M[op.slot] = W_decl @ p[op.t]
            else:
                y_hat = M.get(op.slot, torch.zeros(len(lang.type_names), dtype=torch.float64))
                u = torch.eye(len(lang.type_names), dtype=torch.float64)[op.expected] if op.kind == CHECK_VAR else W_lit @ p[op.t]
                L = 1 - y_hat @ u
                if op.t == G.trace.checks[G.check].op.t:
                    loss = L
                    break
        if not loss.requires_grad:  # e.g. an undeclared slot: nothing carries gradient
            assert not G.grads
            continue
        (g_ref,) = torch.autograd.grad(loss, zz, allow_unused=True)
        g_ref = torch.zeros_like(z) if g_ref is None else g_ref
        n_cmp += 1
        for t in range(len(prog)):
            mine = G.grads.get(t, torch.zeros(len(lang.vocab))).double()
            assert torch.allclose(mine, g_ref[t], atol=1e-6), t
    assert n_cmp >= 10
    print(f"\n[gate1] straight-through bridge == spec formula on {n_cmp} failing programs")
