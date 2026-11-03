"""v0.2 forward-mask development on the seed-0 dev tasks only."""
import json, sys, time, collections
import torch
from ttso.backends.hf_mps import HuggingFaceMPSBackend
from ttso.cartridges.typed_json_dag import TypedJSONDAGCartridge
from ttso.engine import benchmark
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B-Base"
out = sys.argv[2]
be = HuggingFaceMPSBackend(model)
mk = lambda **kw: TypedJSONDAGCartridge(be.token_texts(), encode=be.encode, special_ids=be.tok.all_special_ids, **kw)
regimes = {"ttso_pinned": (mk(), "smart"), "ttso_pinned_masked": (mk(apply_forward_mask=True), "smart")}
bon = {"best_of_n": "ttso_pinned", "best_of_n_masked": "ttso_pinned_masked"}
t0 = time.time()
rows, s = benchmark.run(be, regimes, benchmark.make_tasks(20, seed=0), bon=bon)
json.dump({"rows": rows, "summary": s}, open(out, "w"), indent=1)
for k in s["arms"]:
    print(k, {f: s[k][f] for f in ("valid", "correct", "faithful")}, s[k].get("latency_mean_s"))
print("greedy errors:", collections.Counter(r["greedy"]["error"].split(":")[0] for r in rows if not r["greedy"]["valid"]))
print("masked greedy errors:", collections.Counter(r["greedy_masked"]["error"] for r in rows if not r["greedy_masked"]["valid"]))
print("total", time.time() - t0)
