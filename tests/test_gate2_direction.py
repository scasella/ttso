"""Gate 2: one gradient-directed edit picks the right type at a faulty declaration (K = 5).

200 SynthLang programs, each declaring x : T_a and later using x @ T_b (b != a) as its first failed
check, among other well-typed statements. Cached logits are random, LLM-like (the emitted token
dominant, every other token below it). One round: diagnose -> Smart routing (Case B, the
declaration) -> argmax over the declaration slot's grammar of -grad + 1e-5 z, with the original
token's whole type class banned. Pass: the targeted check passes in 200 / 200.
For contrast the test also reports Blind (the model's own best other type token at the same site).
"""

import torch

from ttso.cartridges.synthetic import SynthLang, reference_check, rng_for
from ttso.core.routing import RepairState, Router, best_pick, diagnose

N = 200


def make_case(lang, rng):
    names = lang.type_names
    a, b = rng.sample(names, 2)
    live = rng.sample(range(lang.n_slots), 6)
    x, others = live[0], live[1:]
    words = []
    decl = {}
    for o in others[:3]:  # well-typed noise before the faulty declaration
        T = rng.choice(names)
        words += ["let", f"x{o}", ":", rng.choice([T, " " + T])] + ["pad"] * rng.randrange(3)
        decl[o] = T
    words += ["let", f"x{x}", ":", rng.choice([a, " " + a])]
    t_decl = len(words) - 1
    for _ in range(rng.randrange(1, 6)):
        o = rng.choice(list(decl))
        words += ["use", f"x{o}", "@", decl[o]] + [";"] * rng.randrange(20)
    words += ["use", f"x{x}", "@", b]
    t_use = len(words) - 1
    words += ["pad"] * rng.randrange(5)
    return lang.ids(words), t_decl, t_use, b


def llm_like_logits(lang, prog, gen):
    V = len(lang.vocab)
    z = torch.randn(len(prog), V, generator=gen) * 2.0
    z[torch.arange(len(prog)), torch.tensor(prog)] += 10.0  # the emitted token dominates
    return z


def test_single_edit_fixes_declaration():
    lang = SynthLang(n_types=5, n_vars=64)
    net = lang.compile("cpu")
    router = Router(lang, "smart")
    rng, gen = rng_for(2), torch.Generator().manual_seed(2)
    fixed = blind_fixed = 0
    for _ in range(N):
        prog, t_decl, t_use, b = make_case(lang, rng)
        z = llm_like_logits(lang, prog, gen)
        d = diagnose(net, lang, prog, z)
        assert d.kind == "check" and d.t_error == t_use and d.t_decl == t_decl
        assert d.flagged == [t_decl] and d.t_call is None  # Case B: only the declaration
        e = router.choose(d, prog, z, RepairState())
        assert e is not None and e.t == t_decl and e.case == "B" and e.route == "decl"
        new = list(prog)
        new[e.t] = e.new
        ref = reference_check([lang.vocab[i] for i in new], lang.type_names, lang.suffix)
        if ref["verdicts"][t_use] is True:
            fixed += 1
        # Blind: same site, the model's own best other type token (no gradient direction)
        allowed = lang.valid_mask(prog, t_decl, "decl")
        alt = best_pick(z[t_decl], allowed, lang.tcm.ban_mask(prog[t_decl], "decl"))
        nb = list(prog)
        nb[t_decl] = alt
        blind_fixed += reference_check([lang.vocab[i] for i in nb], lang.type_names, lang.suffix)["verdicts"][t_use] is True
    print(f"\n[gate2] K=5: gradient-directed single edit fixed {fixed}/{N} targeted declarations "
          f"(Blind next-best type token: {blind_fixed}/{N})")
    assert fixed == N
