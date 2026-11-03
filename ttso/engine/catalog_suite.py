"""BendLite catalog synthesis suite (v0.3): seeded prompts inside the grammar BendLiteCatalogCartridge
parses, with the DiffRepair study's prompt format and task check.

A task lists 2-4 typed catalog items (U32 values 1-60, Nat counts 1-12, at least one of each) and
asks for their sum, sometimes with one U32 item counted twice. The model writes one def that takes
every item as a typed parameter and is called from `main`:

    import Base

    def quote(stock: Nat, level: U32, qty: U32) -> U32:
      (U32.from_nat(stock) + qty + level : U32)

    def main() -> U32:
      quote(6n, 38, 47)

The study's own 50 held-out catalog prompts (the first 50 of its seed-777 split, in the diffspec
repository) also use Bool flags with `match` and `*` products, which this cartridge does not parse,
so they are not used here; this generator keeps the study's item names, function names, system
prompt (without the Bool, `match` and `*` rules) and user instruction.

Outcomes:
  valid     the cartridge's reference checker accepts the program (the repair loop's oracle);
  correct   valid, the expected value, and the study's task check (interface_ok): every catalog
            item is a parameter of a non-main def and is used in a non-main body, non-main bodies
            hold no number literals, and main passes every catalog value;
  faithful  correct, and every catalog item's parameter keeps the type the catalog states.
"""

import random
from collections import Counter

from ttso.cartridges.bendlite import parse

NAMES = ["price", "qty", "fee", "tax", "bonus", "rate", "count", "units", "base", "extra",
         "days", "hours", "limit", "stock", "refund", "tip", "level", "boost", "shift", "weight",
         "size", "cost", "pay", "load"]
FN = ["quote", "total", "calc", "score", "amount", "bill", "plan", "value"]

SYSTEM = (
    "You write programs in BendLite, a subset of the Bend2 language. Rules: start with `import Base`; "
    "define each def before it is used; `def main() -> U32:` returns the answer. Types: U32 numbers "
    "(42) and Nat counters (5n). Arithmetic goes in `( ... : U32)` with spaces around `+`. "
    "U32.from_nat(n) turns a Nat into a U32. Use `+name: Type` for parameters used more than once. "
    "Two-space indentation. Reply with the program only.\n\n"
    "Example task: Catalog: width = 7 (U32 value), rows = 4 (Nat count). Task: return width + rows + width.\n"
    "Example program:\n"
    "import Base\n\n"
    "def area(+width: U32, rows: Nat) -> U32:\n"
    "  (width + U32.from_nat(rows) + width : U32)\n\n"
    "def main() -> U32:\n"
    "  area(7, 4n)\n")
INSTRUCTION = (" Write one def that takes every catalog item as a typed parameter (use the item "
               "names), compute the result from those parameters only (no number literals in the "
               "def), and call it from main with the catalog values.")


def catalog_task(rng):
    k = rng.randint(2, 4)
    n_nat = rng.randint(1, min(2, k - 1))
    names = rng.sample(NAMES, k)
    nats, us = names[:n_nat], names[n_nat:]
    vals = {x: rng.randint(1, 12) if x in nats else rng.randint(1, 60) for x in names}
    terms = names[:]
    rng.shuffle(terms)
    reuse = rng.choice(us) if rng.random() < 0.3 else None
    if reuse:
        terms.append(reuse)
    order = names[:]
    rng.shuffle(order)
    fn = rng.choice(FN)
    types = {x: "Nat" if x in nats else "U32" for x in names}
    cat = ", ".join(f"{x} = {vals[x]} ({'Nat count' if x in nats else 'U32 value'})" for x in order)
    text = f"Catalog: {cat}. Task: return {' + '.join(terms)}."
    plus = lambda x: "+" if terms.count(x) > 1 else ""
    body = " + ".join(f"U32.from_nat({x})" if x in nats else x for x in terms)
    ref = (f"import Base\n\ndef {fn}({', '.join(f'{plus(x)}{x}: {types[x]}' for x in order)}) -> U32:\n"
           f"  ({body} : U32)\n\ndef main() -> U32:\n"
           f"  {fn}({', '.join(f'{vals[x]}n' if x in nats else str(vals[x]) for x in order)})\n")
    return {"text": text, "items": order, "types": [types[x] for x in order],
            "values": [vals[x] for x in order], "expected": sum(vals[x] for x in terms) % 2**32,
            "reference": ref}


def make_tasks(n, seed, exclude=()):
    rng, out, seen = random.Random(seed), [], {t["text"] for t in exclude}
    while len(out) < n:
        t = catalog_task(rng)
        if t["text"] in seen:
            continue
        seen.add(t["text"])
        out.append(dict(t, i=len(out)))
    return out


def dev_tasks():
    return make_tasks(20, seed=0)


def heldout_tasks():
    return make_tasks(50, seed=42, exclude=dev_tasks())


def messages(task):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Write a BendLite program. {task['text']}{INSTRUCTION}"}]


def _operands(body):
    if body["kind"] == "group":
        return body["operands"]
    if body["kind"] == "call":
        return body["args"]
    return [body["operand"]]


def interface_ok(task, text):
    """The study's task check, on this grammar (see the module docstring)."""
    fns, err = parse(text)
    if err is not None:
        return False
    defs = [f for f in fns if f.name != "main"]
    main = [f for f in fns if f.name == "main"]
    params = {p for f in defs for p, _, _ in f.params}
    ops = [o for f in defs for o in _operands(f.body)]
    used = {o["name"] for o in ops if o["kind"] in ("var", "from_nat")}
    if not set(task["items"]) <= params or not set(task["items"]) <= used or any(o["kind"] == "lit" for o in ops):
        return False
    if not main or main[0].body["kind"] != "call":
        return False
    have = Counter(a["value"] for a in main[0].body["args"] if a["kind"] == "lit")
    return all(have[v] >= c for v, c in Counter(task["values"]).items())


def faithful_types(task, text):
    fns, _ = parse(text)
    declared = {}
    for f in fns:
        if f.name != "main":
            for p, T, _ in f.params:
                declared.setdefault(p, set()).add(T)
    return all(declared.get(x) == {T} for x, T in zip(task["items"], task["types"]))


class CatalogSuite:
    """Chat prompts (thinking off, as in the study) and the catalog outcomes for benchmark.run.
    The prompt uses the backend's HuggingFace tokenizer (backend.tok) for the chat template."""

    def prompt_ids(self, backend, task):
        text = backend.tok.apply_chat_template(messages(task), add_generation_prompt=True, tokenize=False,
                                               enable_thinking=False)
        return backend.encode(text)

    def outcome(self, task, verdict, text):
        ok = verdict.valid and verdict.value == task["expected"] and interface_ok(task, text)
        return {"valid": verdict.valid, "correct": ok, "faithful": ok and faithful_types(task, text),
                "error": verdict.error}
