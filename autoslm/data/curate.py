"""Quality controls (paper Section 2.3).

5 constraints:
1. 2-for-1 rule
2. label balancing (no label > 3x another)
3. context-length matching
4. entity diversification (NER)
5. CoT annotation (handled in hard_negatives / teacher pipeline)

Plus surface-pattern diversity: 3-5 patterns per label.
"""
from __future__ import annotations
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class Example:
    input: str
    output: str
    label: Optional[str] = None
    is_hard_negative: bool = False
    is_replay: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    total: int
    by_label: dict[str, int]
    rejected: int
    reasons: dict[str, int]
    length_match_violations: int
    surface_pattern_violations: dict[str, int]
    final_size: int


def _len_distribution(items: list[str], buckets: int = 5) -> list[int]:
    if not items:
        return [0] * buckets
    lens = [len(x) for x in items]
    lo, hi = min(lens), max(lens)
    if hi == lo:
        return [len(items)] + [0] * (buckets - 1)
    edges = [lo + (hi - lo) * i / buckets for i in range(buckets + 1)]
    counts = [0] * buckets
    for l in lens:
        b = min(buckets - 1, max(0, int((l - lo) / (hi - lo) * buckets)))
        counts[b] += 1
    return counts


def _surface_pattern_count(label_examples: list[Example]) -> int:
    """Heuristic: count distinct lemmatized first-3-words signatures per label."""
    sigs = set()
    for ex in label_examples:
        toks = re.findall(r"\w+", ex.input.lower())
        sigs.add(" ".join(toks[:3]))
    return len(sigs)


def label_balance(examples: list[Example], max_ratio: float = 3.0) -> list[Example]:
    by_lab: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_lab[ex.label or "_unlabeled"].append(ex)
    if len(by_lab) < 2:
        return examples
    smallest = min(len(v) for v in by_lab.values())
    cap = max(1, int(smallest * max_ratio))
    out: list[Example] = []
    for lab, items in by_lab.items():
        if len(items) > cap:
            random.Random(42).shuffle(items)
            items = items[:cap]
        out.extend(items)
    return out


def length_match(
    examples: list[Example],
    target_inputs: list[str],
    tolerance_ratio: float = 0.25,
) -> tuple[list[Example], int]:
    """Drop examples whose length puts them in a bucket >tolerance off target distribution."""
    if not target_inputs:
        return examples, 0
    target = _len_distribution(target_inputs)
    target_total = sum(target) or 1
    target_pct = [c / target_total for c in target]
    lo, hi = min(len(t) for t in target_inputs), max(len(t) for t in target_inputs)
    span = max(1, hi - lo)
    by_bucket: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        b = min(len(target) - 1, max(0, int((len(ex.input) - lo) / span * len(target))))
        by_bucket[b].append(ex)
    out: list[Example] = []
    violations = 0
    n = len(examples) or 1
    for b, bucket in by_bucket.items():
        target_n = max(1, int(target_pct[b] * n))
        cap = int(target_n * (1 + tolerance_ratio))
        if len(bucket) > cap:
            random.Random(42).shuffle(bucket)
            violations += len(bucket) - cap
            bucket = bucket[:cap]
        out.extend(bucket)
    return out, violations


def enforce_two_for_one(examples: list[Example]) -> list[Example]:
    """For each gold, ensure at least one hard-negative paired by metadata['pair_key']."""
    pairs = defaultdict(list)
    others = []
    for ex in examples:
        k = ex.metadata.get("pair_key")
        if k is not None:
            pairs[k].append(ex)
        else:
            others.append(ex)
    out = list(others)
    for k, group in pairs.items():
        has_gold = any(not ex.is_hard_negative for ex in group)
        has_neg = any(ex.is_hard_negative for ex in group)
        if has_gold and has_neg:
            out.extend(group)
        elif has_gold:
            # gold without partner: keep but mark
            out.extend(g for g in group if not g.is_hard_negative)
    return out


def diversify_entities(examples: list[Example], max_repeats: int = 3) -> list[Example]:
    """NER: cap each entity surface form to <= max_repeats appearances."""
    counts: Counter = Counter()
    out: list[Example] = []
    ent_re = re.compile(r"\[([^\[\]]{1,80})\]\(([A-Z_]+)\)")  # simple bracketed annotations
    for ex in examples:
        ents = ent_re.findall(ex.output) or ent_re.findall(ex.input)
        if not ents:
            out.append(ex)
            continue
        if any(counts[(v.lower(), t)] >= max_repeats for v, t in ents):
            continue
        for v, t in ents:
            counts[(v.lower(), t)] += 1
        out.append(ex)
    return out


def curate_dataset(
    gold: list[Example],
    hard_negs: list[Example],
    replay: Optional[list[Example]] = None,
    target_inputs: Optional[list[str]] = None,
    max_examples: Optional[int] = None,
    label_max_ratio: float = 3.0,
    twofor_one: bool = True,
    entity_diversify: bool = False,
    surface_min_patterns: int = 3,
) -> tuple[list[Example], QualityReport]:
    reasons: Counter = Counter()
    initial = len(gold) + len(hard_negs) + (len(replay) if replay else 0)

    examples: list[Example] = []
    examples.extend(gold)
    examples.extend(hard_negs)
    if replay:
        examples.extend(replay)

    if twofor_one:
        before = len(examples)
        examples = enforce_two_for_one(examples)
        reasons["twofor_one_drop"] += before - len(examples)

    if entity_diversify:
        before = len(examples)
        examples = diversify_entities(examples)
        reasons["entity_repeats_drop"] += before - len(examples)

    before = len(examples)
    examples = label_balance(examples, max_ratio=label_max_ratio)
    reasons["label_imbalance_drop"] += before - len(examples)

    length_violations = 0
    if target_inputs:
        examples, length_violations = length_match(examples, target_inputs)
        reasons["length_mismatch_drop"] += length_violations

    # surface-pattern diversity audit
    by_lab: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_lab[ex.label or "_unlabeled"].append(ex)
    surface_violations: dict[str, int] = {}
    for lab, items in by_lab.items():
        n = _surface_pattern_count(items)
        if n < surface_min_patterns:
            surface_violations[lab] = n

    if max_examples and len(examples) > max_examples:
        random.Random(42).shuffle(examples)
        examples = examples[:max_examples]

    report = QualityReport(
        total=initial,
        by_label={k: len(v) for k, v in by_lab.items()},
        rejected=initial - len(examples),
        reasons=dict(reasons),
        length_match_violations=length_violations,
        surface_pattern_violations=surface_violations,
        final_size=len(examples),
    )
    return examples, report
