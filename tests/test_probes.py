"""Live confirmation probes (paper §2.6 step 3)."""
from autoslm.traces.probes import confirm_weakness
from autoslm.traces.taxonomy import Cluster
from autoslm.eval.metrics import exact_match
from autoslm.eval.harness import EvalExample


def _cluster(cid="c1", n=10):
    return Cluster(cluster_id=cid, label="confusion_AB", fixability="fixable",
                  size=n, description="A vs B confusion",
                  representative_ids=[],
                  representative_examples=[{"input": "x", "gold": "A", "prediction": "B"}])


def test_confirm_when_pass_rate_low_returns_confirmed_true():
    probes = [EvalExample(input=f"p{i}", gold="A", slice="boundary") for i in range(10)]
    # model wrong on 8/10
    answers = ["A", "A"] + ["B"] * 8

    def predict(_): return answers

    r = confirm_weakness(_cluster(), probes, predict, exact_match)
    assert r.confirmed
    assert r.pass_rate <= 0.7
    assert len(r.failing_probes) == 8


def test_confirm_demoted_when_pass_rate_high():
    probes = [EvalExample(input=f"p{i}", gold="A", slice="boundary") for i in range(10)]
    # model right on 9/10
    answers = ["A"] * 9 + ["B"]

    def predict(_): return answers

    r = confirm_weakness(_cluster(), probes, predict, exact_match)
    assert not r.confirmed
    assert r.pass_rate >= 0.7


def test_no_probes_returns_unconfirmed():
    r = confirm_weakness(_cluster(), [], lambda x: [], exact_match)
    assert not r.confirmed
    assert r.pass_rate == 1.0
