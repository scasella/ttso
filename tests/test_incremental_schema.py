"""Incremental forward mask (v0.3): SchemaState / SchemaTrail must accept exactly what the v0.2
full-reparse acceptor (SchemaPrefix) accepts.

1. Differential fuzz (CPU): every prefix of random schema documents, half with a one-character
   mutation. Status (more / done / bad), the next-character set and chunked advances must match
   SchemaPrefix.
2. Cartridge level (CPU, Qwen3 vocabulary): accept() and candidates() of mask_impl="incremental"
   and "reparse" agree at every prefix of tokenized documents, queried in random order so the trail
   is rewound and re-extended.
3. Model gate (mps): the 20 seed-0 development tasks with masked greedy, masked ttso repair and
   masked Best-of-N, reparse vs incremental: token-for-token equal greedy completions, repair
   trajectories, final completions and Best-of-N samples. Logs both acceptors' mask-evaluation time
   and per-problem latency, and a CPU replay of the mask over the logged completions.
"""

import random
import time

import pytest
import torch

from ttso.cartridges.typed_json_dag import CONVERT, OPS, TYPES, SchemaState, TypedJSONDAGCartridge, schema_state


def _doc(rng):
    ws = lambda: rng.choice(["", "", " ", "\n", "  ", " \n "])
    lit = lambda: f'"{rng.randint(0, 999)}{rng.choice(["", "n", ".5", ".25", "usd", "eur"])}"'

    def name(pool=None):
        if pool and rng.random() < 0.9:
            return rng.choice(pool)
        return rng.choice(["a", "b", "stock", "x1", "_q", "level", "total", "lv"]) + rng.choice(["", "2", "_b"])

    kv = lambda k, v: ws() + f'"{k}"' + ws() + ":" + ws() + v
    ins, nodes, parts = [], [], []
    for _ in range(rng.randint(1, 3)):
        x = name()
        ins.append(x)
        parts.append(ws() + "{" + kv("id", f'"{x}"') + ws() + "," + kv("type", f'"{rng.choice(TYPES)}"') + ws() + "}")
    node_parts = []
    for _ in range(rng.randint(1, 3)):
        x, args = name(), []
        for _ in range(rng.randint(1, 3)):
            k = rng.choice(["ref", "lit", "convert"])
            v = (f'"{name(ins + nodes)}"' if k == "ref" else lit() if k == "lit"
                 else f'"{rng.choice(list(CONVERT))}({name(ins + nodes)})"')
            args.append(ws() + "{" + kv(k, v) + ws() + "}")
        node_parts.append(ws() + "{" + kv("id", f'"{x}"') + ws() + "," + kv("op", f'"{rng.choice(OPS)}"') + ws() + ","
                          + kv("out_type", f'"{rng.choice(TYPES)}"') + ws() + "," + kv("args", "[" + ",".join(args) + ws() + "]")
                          + ws() + "}")
        nodes.append(x)
    order = ins[:] if rng.random() < 0.8 else ins + [name(ins)]
    rng.shuffle(order)
    call = "{" + ",".join(ws() + f'"{x}"' + ws() + ":" + ws() + lit() for x in order) + ws() + "}"
    s = (ws() + "{" + kv("inputs", "[" + ",".join(parts) + ws() + "]") + ws() + "," + kv("nodes", "[" + ",".join(node_parts) + ws() + "]")
         + ws() + "," + kv("output", f'"{name(nodes)}"') + ws() + "," + kv("call", call) + ws() + "}")
    s += rng.choice(["", " ", "\n", "\t\r", " x", " ", "}"])
    if rng.random() < 0.5:
        s = list(s)
        i, r = rng.randrange(len(s)), rng.random()
        if r < 0.33:
            del s[i]
        elif r < 0.66:
            s.insert(i, rng.choice(list(' \n\t"{}[]:,()._0129nusdeurabxU²')))
        else:
            s[i] = rng.choice(list(' \n"{}[]:,()._09nuexU'))
        s = "".join(s)
    return s


def test_schema_state_matches_reparse():
    rng = random.Random(0)
    n = complete = 0
    for _ in range(150):
        d = _doc(rng)
        st = SchemaState()
        for i in range(len(d) + 1):
            want = schema_state(d[:i])
            assert ("bad" if st is None else st.status) == want, (d[:i], want)
            n += 1
            if st is not None and i % 11 == 0:
                legacy = {chr(c) for c in range(128) if schema_state(d[:i] + chr(c)) != "bad"}
                assert set(st.next_chars()) == legacy, d[:i]
            if i < len(d):
                st = None if st is None else st.advance(d[i])
        complete += want == "done"
        cuts = sorted(rng.sample(range(len(d) + 1), min(len(d), 12))) + [len(d)]
        s2, prev = SchemaState(), 0
        for c in cuts:
            s2, prev = None if s2 is None else s2.advance(d[prev:c]), c
        assert ("bad" if s2 is None else s2.status) == schema_state(d)
    print(f"\n[incremental] {n} prefixes of 150 documents ({complete} complete): SchemaState == SchemaPrefix")
    assert complete > 10


def test_cartridge_accept_and_candidates_match(qwen_tokenizer, qwen_texts):
    from ttso.engine import benchmark
    import json
    enc = lambda s: qwen_tokenizer(s, add_special_tokens=False)["input_ids"]
    specials = qwen_tokenizer.all_special_ids
    mk = lambda impl: TypedJSONDAGCartridge(qwen_texts, encode=enc, special_ids=specials, apply_forward_mask=True,
                                            mask_impl=impl)
    inc, leg = mk("incremental"), mk("reparse")
    rng = random.Random(1)
    docs = [json.dumps(o) for _, o in benchmark.SHOTS] + [_doc(rng) for _ in range(3)]
    n = 0
    for d in docs:
        toks = enc(d)
        ts = list(range(len(toks) + 1))
        rng.shuffle(ts)  # out-of-order queries rewind and re-extend the trail
        for t in ts:
            head = toks[:t]
            probe = ([toks[t]] if t < len(toks) else []) + rng.sample(range(len(qwen_texts)), 8) + specials[:2]
            for v in probe:
                assert inc.accept(head, v) == leg.accept(head, v), (d, t, v)
                n += 1
            if t % 4 == 0:
                assert torch.equal(inc.candidates(head), leg.candidates(head)), (d, t)
    print(f"\n[incremental] {n} accept() queries and the candidates() masks agree on {len(docs)} tokenized documents")


def _replay(cart, completions):
    """CPU mask work of a masked decode replayed over fixed completions: accept() of each token and
    candidates() at each position, timed separately."""
    t_acc = t_cand = 0.0
    for toks in completions:
        for t in range(len(toks)):
            t0 = time.perf_counter()
            cart.accept(toks[:t], toks[t])
            t1 = time.perf_counter()
            cart.candidates(toks[:t])
            t_acc, t_cand = t_acc + t1 - t0, t_cand + time.perf_counter() - t1
    return t_acc, t_cand


def test_incremental_mask_model_gate(hf_run):
    from tests.gate4_setup import _regimes, _trajectory
    from ttso.engine import benchmark

    rows, s, run_dir, be = hf_run
    k, g, b = "ttso_pinned_masked", "greedy_masked", "best_of_n_masked"
    leg_rows, leg_s = benchmark.run(be, _regimes(be, (k,), mask_impl="reparse"), benchmark.make_tasks(20, seed=0),
                                    bon={b: k}, log=lambda *a: None)
    diff = []
    for a, c in zip(rows, leg_rows):
        same = (a[g]["text"] == c[g]["text"] and _trajectory(a, k) == _trajectory(c, k) and a[k]["text"] == c[k]["text"]
                and a[b].get("text") == c[b].get("text") and a[b]["samples"] == c[b]["samples"]
                and a[b]["tokens"] == c[b]["tokens"])
        if not same:
            diff.append(a["i"])
    mask = lambda rs, arm: sum(r[arm].get("mask_s", 0.0) for r in rs)
    n = len(rows)
    print(f"\n[incremental] {n} seed-0 tasks, masked greedy / masked ttso repair / masked Best-of-N: "
          f"{n - len(diff)}/{n} token-for-token identical (reparse vs incremental)")
    print(f"  {'':34s} {'reparse':>9s} {'incremental':>11s} {'speedup':>7s}")
    for label, arm in (("mask s/problem, greedy", g), ("mask s/problem, greedy + repair", k), ("mask s/problem, Best-of-N", b)):
        x, y = mask(leg_rows, arm) / n, mask(rows, arm) / n
        print(f"  {label:34s} {x:9.3f} {y:11.3f} {x / max(y, 1e-9):6.1f}x")
    for label, arm in (("latency s/problem, greedy", g), ("latency s/problem, greedy + repair", k)):
        x, y = leg_s[arm]["latency_mean_s"], s[arm]["latency_mean_s"]
        print(f"  {label:34s} {x:9.2f} {y:11.2f} {x / y:6.2f}x")
    comps = [be.encode(r[k]["text"]) for r in rows]
    ntok = sum(len(c) for c in comps)
    leg_c, inc_c = _regimes(be, (k,), mask_impl="reparse")[k][0], _regimes(be, (k,))[k][0]
    (la, lc), (ia, ic) = _replay(leg_c, comps), _replay(inc_c, comps)
    print(f"  CPU replay over the {n} repaired completions ({ntok} tokens): accept() {la * 1e6 / ntok:.0f} vs "
          f"{ia * 1e6 / ntok:.0f} us/token ({la / ia:.1f}x), candidates() {lc * 1e3 / ntok:.2f} vs "
          f"{ic * 1e3 / ntok:.2f} ms/token ({lc / ic:.1f}x)")
    assert not diff, diff
    assert mask(rows, k) < mask(leg_rows, k)
