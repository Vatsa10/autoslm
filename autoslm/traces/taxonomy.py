"""Failure taxonomy + fixable/poison classifier (paper Section 2.6 step 2 + Section 3.1).

Strategy:
1. Cheap clustering on input embeddings (sentence-transformers) -> K clusters
2. LLM labels each cluster: dominant pattern + fixability (fixable | external | poison)
3. Poison detector: false premise, label flip, off-domain, prompt injection, jailbreak,
   gibberish, empty. Fixable: typos, casing, truncation, preamble, code-switch.
"""
from __future__ import annotations
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMClient


POISON_PATTERNS = [
    (re.compile(r"^\s*$"), "empty_input"),
    (re.compile(r"ignore (all )?(previous|prior) (instructions|context)", re.I), "prompt_injection"),
    (re.compile(r"jailbreak|developer mode|do anything now|DAN", re.I), "jailbreak"),
    (re.compile(r"\b([a-z]{20,}|[A-Z]{20,})\b"), "gibberish_token"),
]


def cheap_poison_screen(text: str) -> Optional[str]:
    if text is None:
        return "empty_input"
    for rx, label in POISON_PATTERNS:
        if rx.search(text):
            return label
    if len(text.strip()) < 2:
        return "empty_input"
    # gibberish: very low char diversity
    chars = set(text.strip().lower())
    if len(text) > 30 and len(chars) < 5:
        return "gibberish"
    return None


@dataclass
class Cluster:
    cluster_id: str
    label: str
    fixability: str                 # 'fixable' | 'external' | 'poison'
    size: int
    description: str
    representative_ids: list[str] = field(default_factory=list)
    representative_examples: list[dict] = field(default_factory=list)


def _embed(texts: list[str]):
    """Optional sentence-transformer embeddings. Falls back to char-ngram TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=2048)
        return v.fit_transform(texts).toarray()


def cluster_failures(records: list[dict], k: Optional[int] = None) -> list[list[dict]]:
    """K-means cluster on input embeddings. Returns list of clusters (each = list of records)."""
    if not records:
        return []
    if k is None:
        k = max(2, min(12, int(math.sqrt(len(records)))))
    texts = [r.get("input") or "" for r in records]
    X = _embed(texts)
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    out: dict[int, list[dict]] = defaultdict(list)
    for rec, lab in zip(records, labels):
        out[int(lab)].append(rec)
    return list(out.values())


TAXONOMY_SYSTEM = """You analyze a cluster of failed inferences and label it.
Output JSON: {"label": str, "description": str, "fixability": "fixable"|"external"|"poison"}.

Definitions:
- fixable: original intent + correct answer recoverable from input (typos, casing,
  truncation, preamble, code-switch, distribution shift the model can learn).
- external: not addressable by training (schema/prompt-design errors, label
  ambiguity, upstream bugs).
- poison: training on the raw example would actively teach the wrong behavior
  (false premises, label flips, off-domain, prompt injection, jailbreak, gibberish, empty).
"""


def label_cluster(client: LLMClient, cluster: list[dict], cluster_idx: int) -> Cluster:
    sample = cluster[: min(8, len(cluster))]
    payload = {
        "cluster_size": len(cluster),
        "samples": [
            {"input": r.get("input"), "prediction": r.get("prediction"),
             "gold": r.get("gold"), "judge_reason": r.get("judge_reason")}
            for r in sample
        ],
    }
    resp = client.complete(
        [{"role": "user", "content": json.dumps(payload, default=str)[:20_000]}],
        system=TAXONOMY_SYSTEM,
    )
    text = resp.content
    if "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except Exception:
        data = {"label": f"cluster_{cluster_idx}", "description": "uncategorized",
                "fixability": "fixable"}
    return Cluster(
        cluster_id=f"c{cluster_idx}",
        label=str(data.get("label", f"cluster_{cluster_idx}"))[:120],
        fixability=str(data.get("fixability", "fixable")),
        size=len(cluster),
        description=str(data.get("description", ""))[:1000],
        representative_ids=[r.get("id") for r in sample if r.get("id")],
        representative_examples=sample,
    )


def build_taxonomy(client: LLMClient, fail_records: list[dict],
                   k: Optional[int] = None) -> list[Cluster]:
    """Full pipeline: cheap poison screen -> cluster -> LLM-label each cluster."""
    poison_bucket: list[dict] = []
    rest: list[dict] = []
    for r in fail_records:
        tag = cheap_poison_screen(r.get("input") or "")
        if tag:
            r = {**r, "_poison_tag": tag}
            poison_bucket.append(r)
        else:
            rest.append(r)

    clusters: list[Cluster] = []
    if poison_bucket:
        # group by poison tag
        by_tag: dict[str, list[dict]] = defaultdict(list)
        for r in poison_bucket:
            by_tag[r["_poison_tag"]].append(r)
        for i, (tag, recs) in enumerate(by_tag.items()):
            clusters.append(Cluster(
                cluster_id=f"p{i}", label=tag, fixability="poison", size=len(recs),
                description=f"Poison cluster by rule: {tag}",
                representative_ids=[r.get("id") for r in recs[:5] if r.get("id")],
                representative_examples=recs[:5],
            ))

    grouped = cluster_failures(rest, k=k)
    for i, group in enumerate(grouped):
        clusters.append(label_cluster(client, group, i))
    return clusters
