# autoslm — data curation log

- **run_id:** `coldstart-299d62`
- **mode:** `cold-start`
- **started:** 2026-05-05T19:08:11Z

---

## Task spec
_2026-05-05T19:08:11Z_

- spec: synthetic classification
- base_model: `test-model`
- dataset_hint: `synthetic`

---

## Task classification
_2026-05-05T19:08:11Z_

- **task_type**: `classification`
- **eval_method**: `exact_match`
- **supervision**: `direct`
- **model_family**: `decoder`
- **canonical_dataset**: `None`
- **labels**: `None`
- **rationale**: `test`

---

## Dataset acquired
_2026-05-05T19:08:11Z_

- n: 50
- source: `synthetic`

---

## Baseline survey
_2026-05-05T19:08:11Z_

- published_sota: 0.8
- target_threshold (τ): 0.9
- notes: test

---

## Held-out E (paper Eq. 7)
_2026-05-05T19:08:11Z_

- E_pos: 0
- E_neg: 0
- E_boundary: 0

---

## Curriculum
_2026-05-05T19:08:12Z_

- total: 0
- by_label: `{}`
- rejected: 0 ({'twofor_one_drop': 0, 'label_imbalance_drop': 0, 'length_mismatch_drop': 0})

---

