"""Task-appropriate metrics: exact match, token F1, ROUGE, pass@k."""
from __future__ import annotations
import re
import string
from collections import Counter


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p_toks)
    recall = n / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(gold, pred)["rougeL"].fmeasure
    except Exception:
        return token_f1(pred, gold)


def rouge_2(pred: str, gold: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        return scorer.score(gold, pred)["rouge2"].fmeasure
    except Exception:
        return 0.0


def code_pass_at_1(pred_code: str, test_code: str, timeout: int = 5) -> float:
    """Run pred function + test_code in subprocess. Returns 1.0 on pass."""
    import subprocess
    import textwrap
    src = textwrap.dedent(pred_code) + "\n\n" + textwrap.dedent(test_code)
    try:
        proc = subprocess.run(
            ["python", "-c", src],
            capture_output=True, text=True, timeout=timeout,
        )
        return 1.0 if proc.returncode == 0 else 0.0
    except Exception:
        return 0.0
