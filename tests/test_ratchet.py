"""Ratchet (paper §2.7): regression on prior eval set blocks accept."""
from autoslm.eval.harness import EvalSet, EvalExample, score_pipeline
from autoslm.search.pipeline import LearningStrategy


def _ex(inp, gold, slc="pos"):
    return EvalExample(input=inp, gold=gold, slice=slc)


def test_prior_set_increases_regression_count():
    cur = EvalSet(pos=[_ex("q1", "a"), _ex("q2", "b")])
    cur.prior = EvalSet(pos=[_ex("q3", "c"), _ex("q4", "d"), _ex("q5", "e")])
    # current model gets cur right but blows prior set entirely
    answers = {"q1": "a", "q2": "b", "q3": "X", "q4": "X", "q5": "X"}

    def predict(prompts):
        return [answers.get(p, "X") for p in prompts]

    # regression set = small previously-passing slice (3 items, all wrong)
    reg = [_ex("r1", "z"), _ex("r2", "z")]   # 2 regressions on R alone
    a, r, _ = score_pipeline(cur, reg, predict,
                             LearningStrategy(eval_method="exact_match"))
    assert a == 1.0   # current set fully correct
    # r should reflect the worse of (R=2, prior=3) -> 3
    assert r >= 3, f"expected >=3 regressions due to prior set, got {r}"


def test_prior_absent_falls_back_to_regression_only():
    cur = EvalSet(pos=[_ex("q1", "a")])
    cur.prior = None
    reg = [_ex("r1", "z")]    # 1 regression

    def predict(prompts):
        return ["wrong" for _ in prompts]

    _, r, _ = score_pipeline(cur, reg, predict,
                             LearningStrategy(eval_method="exact_match"))
    assert r == 1
