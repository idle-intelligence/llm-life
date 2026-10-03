"""Per-cell prompt text and outer-totalistic rules for the bert-text task.

The prompt format mirrors `crates/llm-life/src/variant_a.rs::cell_prompt`
and `norules_prefix` exactly (same strings, read not reimplemented from a
guess): "Neighbors: n0 n1 ... n7 / Self: s / Next: ", trailing space kept so
a from-scratch tokenizer sees the same token boundaries variant A's LLM
prompt does. `norules_prefix` is the `a-norules` adapter's prefix (generic
answer-format instruction, no rule text) - this is "the norules format"
named in docs/runs/2026-09-20-eval-a-base-norules.md and friends.

Case indexing (`case_from_index`, `label`) mirrors
`crates/llm-life/src/bert/data.rs`: bits 0..7 are the 8 neighbours in
`Grid::neighbor_indices` order, bit 8 is the cell's own state.
"""

from __future__ import annotations

from dataclasses import dataclass


def norules_prefix() -> str:
    return "For each cell, answer with one digit: its next state.\n"


def cell_prompt(neighbors: list[int], self_state: int) -> str:
    s = "Neighbors:"
    for n in neighbors:
        s += " " + ("1" if n else "0")
    s += " / Self: " + ("1" if self_state else "0") + " / Next: "
    return s


def case_from_index(k: int) -> tuple[list[int], int]:
    """Returns (neighbors[8], self_state) for case index 0..511."""
    neighbors = [(k >> b) & 1 for b in range(8)]
    self_state = (k >> 8) & 1
    return neighbors, self_state


@dataclass(frozen=True)
class Rule:
    birth: frozenset[int]
    survive: frozenset[int]

    @staticmethod
    def parse(s: str) -> "Rule":
        b, sv = s.split("/")
        assert b[0] in "Bb" and sv[0] in "Ss", s
        birth = frozenset(int(c) for c in b[1:])
        survive = frozenset(int(c) for c in sv[1:])
        return Rule(birth=birth, survive=survive)

    def to_rulestring(self) -> str:
        b = "".join(str(n) for n in range(9) if n in self.birth)
        sv = "".join(str(n) for n in range(9) if n in self.survive)
        return f"B{b}/S{sv}"

    def next(self, alive: bool, n: int) -> int:
        return int(n in (self.survive if alive else self.birth))

    def label(self, neighbors: list[int], self_state: int) -> int:
        n = sum(neighbors)
        return self.next(bool(self_state), n)


CONWAY = Rule.parse("B3/S23")

NAMED_TRAIN_RULES = {
    "conway": Rule.parse("B3/S23"),
    "highlife": Rule.parse("B36/S23"),
    "seeds": Rule.parse("B2/S"),
    "daynight": Rule.parse("B3678/S34678"),
    "life_without_death": Rule.parse("B3/S012345678"),
}

NAMED_HELDOUT_RULES = {
    "2x2": Rule.parse("B36/S125"),
    "maze": Rule.parse("B3/S12345"),
}


def random_rule(rng) -> Rule:
    birth = frozenset(n for n in range(9) if rng.random() < 0.4)
    survive = frozenset(n for n in range(9) if rng.random() < 0.4)
    return Rule(birth=birth, survive=survive)


def random_rules(seed: int, count: int, exclude: set[str] = frozenset()) -> dict[str, Rule]:
    import random as _random

    rng = _random.Random(seed)
    out: dict[str, Rule] = {}
    while len(out) < count:
        r = random_rule(rng)
        name = f"rand_{r.to_rulestring()}"
        if name in exclude or name in out:
            continue
        out[name] = r
    return out


def all_512_cases() -> list[tuple[list[int], int]]:
    return [case_from_index(k) for k in range(512)]


def v1_examples() -> tuple[list[str], list[int]]:
    """Variant 1: norules prefix + cell prompt, all 512 cases, Conway labels."""
    prefix = norules_prefix()
    prompts, labels = [], []
    for k in range(512):
        neighbors, self_state = case_from_index(k)
        prompts.append(prefix + cell_prompt(neighbors, self_state))
        labels.append(CONWAY.label(neighbors, self_state))
    return prompts, labels


def v2_examples(rules: dict[str, Rule]) -> tuple[list[str], list[int], list[str]]:
    """Variant 2: "Rule: <B/S>\\n" + cell prompt, 512 cases per rule.
    Returns (prompts, labels, rule_name_per_example)."""
    prompts, labels, names = [], [], []
    for name, rule in rules.items():
        prefix = f"Rule: {rule.to_rulestring()}\n"
        for k in range(512):
            neighbors, self_state = case_from_index(k)
            prompts.append(prefix + cell_prompt(neighbors, self_state))
            labels.append(rule.label(neighbors, self_state))
            names.append(name)
    return prompts, labels, names
