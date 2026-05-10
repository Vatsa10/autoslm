# autoslm

Open-source re-implementation of **Pioneer Agent** ([arXiv:2604.09791](https://arxiv.org/abs/2604.09791) — Atreja et al., Fastino Labs, 2026): a closed-loop autonomous system for continually improving small language models in production.

> Given a deployed SLM and judged inference failures, an LLM agent diagnoses error patterns, synthesizes a corrective training curriculum, retrains under explicit regression constraints, and verifies improvements — without breaking what already works.

This repo is a from-scratch implementation of the architecture described in the paper, swapping the proprietary Tinker SDK for HuggingFace `peft` + `trl` and exposing a multi-provider orchestrator (Claude / Gemini / OpenAI / DeepSeek / local) via LiteLLM.

---

## What is implemented

| Component | Paper § | Status |
|---|---|---|
| Pipeline space `π = (D, H, S)` | 2.2 | done |
| MCGS over training pipelines (UCT, fuse, prune, rollback) | 2.2 | done |
| Failure taxonomy + fixable/poison classifier | 2.6, 3.1 | done |
| Trace store with `query_traces` SQL+bash semantics (Listing 1) | 2.6 | done |
| Quality controls: 2-for-1, label balance, length-match, entity diversification, surface patterns | 2.3 | done |
| Hard-negative generation (LLM teacher + NN-label-swap fallback) | 2.3, 2.6 | done |
| Replay buffer (Eq. 15) | 2.6 | done |
| CoT annotation by teacher model | 2.3 | done |
| LoRA SFT trainer (peft + trl), 4/8-bit quant, hardware tier presets | 2.1 | done |
| LLM-as-judge + token-F1 / EM / ROUGE / pass@1 metrics | 2.2 | done |
| Eval set (E_pos ∪ E_neg ∪ E_boundary) + regression set R | 2.5 (Eq. 7), 2.6 | done |
| Production-mode closed loop with cross-checkpoint regression gate (Section 2.7 ratchet) | 2.6, 2.7 | done |
| Iteration policy: <0.80 rework, 0.80–0.95 tune H, >0.95 surgical, regress → rollback | 2.4 | done |
| Orchestrator with agent tools (`query_traces`, `bash`, `read/edit_file`, `delegate_task`, `web_search`, `run_search`) | 2.5, 2.6 | done |
| AdaptFT-Bench: stage-based perturbation pipeline (15% → 25% → 40% poison) | 3 | done |
| Live confirmation probes (cluster demotion + probe-failure harvesting) | 2.6 step 3 | done |
| Cross-checkpoint regression gate (ratchet on prior eval set) | 2.7 | done |
| Persistent `data-curation.md` audit log | 2.1 | done |
| Cold-start mode (5-stage workflow, unconstrained MCGS) | 2.5 | done |
| Trace Analyzer sub-agent (own LLM context, ~100K out, disk-backed) | 2.1 | done |
| GLiNER2 encoder path (NER + classification, full FT or LoRA) | 2.1 | done |
| Modal sandbox runner (`@modal.function(gpu="A10G")`) | 2.1 | done |
| Confidence calibration + TF-IDF correction propagation | 2.7 | done |
| FSDP for `big` tier; DPO/KTO objectives | 2.1, 6.3 | done |
| Distributed MCGS — parallel branches per iteration | 2.2 (Eq. 3) | done |
| Cost telemetry (tokens + GPU hours per run) | 6.1 | done |
| Paper benchmark replication suite (CLINC150, ARC, GSM8K, HumanEval, CoNLL-2003) | 4 | done |
| End-to-end smoke harness (`autoslm smoke-e2e`) | n/a | done |

---

## Architecture (high level)

```
                       ┌─────────────────────────┐
deployed model ───────▶│  Trace Store (DuckDB)   │◀──── inference logs
+ judged failures      └────────────┬────────────┘      (LLM-as-judge or human)
                                    │
                       ┌────────────▼────────────┐
                       │ Failure Diagnosis        │  (taxonomy + fixable/poison
                       │  - cluster + label       │   classifier; live confirmation)
                       │  - parent lineage        │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ Curriculum Synthesis     │  D = D_gold ∪ D_hard ∪ D_replay
                       │  - 2-for-1 hard negs     │  + label balance + length match
                       │  - replay buffer         │  + (optional) CoT annotation
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐         ┌──────────────────┐
                       │ MCGS over π = (D,H,S)    │◀───────│ Orchestrator LLM │
                       │  - UCT-guided expand     │ EXPAND │ (Claude/Gemini/  │
                       │  - fuse top-K branches   │ FUSE   │  GPT-5/DeepSeek) │
                       │  - rollback on regress   │        └──────────────────┘
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ Train (LoRA SFT)         │  peft + trl, 4/8-bit,
                       │  + Eval (held-out E + R) │  hardware tier presets
                       │  + Regression gate ε=2   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                            new checkpoint
                       (only deployed if a(π) ≥ τ
                        AND r(π) ≤ ε on E AND prior E)
```

---

## Install

Python ≥ 3.10. Recommended: 3.11 or 3.12 (PyTorch wheels for 3.13 are still patchy).

```bash
git clone <this-repo> slm-forge
cd slm-forge
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Core (orchestrator + bench + traces; no GPU/training):
pip install -e .

# Full ML stack (training + inference; needs CUDA for non-CPU work):
pip install -e .[ml]

# Optional speedups:
pip install -e .[unsloth]    # 2× faster LoRA on Qwen/Llama
pip install -e .[modal]      # Modal cloud sandbox runner
```

If `pip install -e .[ml]` errors on Windows + CUDA, install PyTorch first from <https://pytorch.org/get-started/locally/>, then retry the extras install.

---

## Quickstart

### 1. Set provider keys

```bash
# pick whichever you have:
export ANTHROPIC_API_KEY=...        # default orchestrator
export OPENAI_API_KEY=...            # default judge
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Switch the orchestrator model:

```bash
export AUTOSLM_ORCH_MODEL=anthropic/claude-sonnet-4-6     # default, paper match
# or:
export AUTOSLM_ORCH_MODEL=gemini/gemini-2.5-pro
export AUTOSLM_ORCH_MODEL=openai/gpt-5
export AUTOSLM_ORCH_MODEL=deepseek/deepseek-reasoner
export AUTOSLM_JUDGE_MODEL=openai/gpt-4o-mini
```

### 2. Run the smoke-test demo

Spawns a tiny synthetic intent corpus, builds an AdaptFT-Bench scenario, ingests traces, runs production-mode MCGS:

```bash
python examples/run_production.py --tier edge --iters 3
```

### 3. Real production run

```bash
# ingest existing trace JSONL into the store
autoslm ingest-traces ./my_traces.jsonl --db-path .autoslm/traces.duckdb

# run closed-loop adaptation
autoslm production deploy-model-uuid \
    --base-model meta-llama/Llama-3.2-3B \
    --tier mid \
    --iters 10 \
    --eval-method exact_match \
    --task classification
```

Each `autoslm production` call writes:

- `runs/<model_id>/taxonomy.json` — failure clusters with fixability labels
- `runs/<model_id>/train.jsonl` — curated curriculum (gold + hard-negs + replay)
- `runs/<model_id>/iter_*/` — per-iteration LoRA checkpoint + eval
- `runs/<model_id>/graph/graph.json` — full MCGS lineage

### 4. Free-form orchestrator

```bash
autoslm orchestrate "Improve deploy-uuid-123 — focus on label confusion between intents." \
    --deployed-model-id deploy-uuid-123 \
    --base-model meta-llama/Llama-3.2-3B
```

The orchestrator LLM drives the loop with `query_traces`, `bash`, `delegate_task`, `run_search` tools.

---

## Hardware tiers

Selected via `--tier` CLI flag or `AutoSLMConfig(hardware_tier=...)`.

| Tier | Quant | LoRA r | Max seq | Grad ckpt | Suggested base models |
|------|-------|--------|---------|-----------|------------------------|
| `edge` | 4-bit (NF4) | 8 | 1024 | yes | SmolLM2-360M, Qwen3-0.5B, Llama-3.2-1B |
| `mid` | 8-bit | 32 | 2048 | yes | Llama-3.2-3B, Qwen3-3B/8B |
| `big` | bf16 | 64 | 4096 | no | Qwen3-8B, Llama-3.1-8B (full FT optional) |

`autoslm tiers` prints the active presets.

---

## Repo layout

```
autoslm/
  config.py             # AutoSLMConfig + hardware tier presets
  orchestrator.py       # top-level agent loop with tool use
  cli.py                # `autoslm` Typer CLI
  llm/                  # LiteLLM-based provider-agnostic client
  search/
    pipeline.py         # π = (D, H, S) dataclasses
    mcgs.py             # graph, UCT, expand/fuse/prune, rollback
    expand.py           # LLM-driven EXPAND + FUSE operators
  data/
    curate.py           # 2-for-1, label balance, length match, surface patterns
    hard_negatives.py   # teacher-LLM + NN-label-swap fallback
    replay.py           # D_replay sampling from D_parent
    cot_annotate.py     # teacher CoT supervision
  train/
    lora_sft.py         # peft + trl SFT, 4/8-bit, tier-aware
    inference.py        # batch generate on adapter
  eval/
    metrics.py          # EM, token-F1, ROUGE, pass@1
    llm_judge.py        # judge-as-scorer
    harness.py          # E + R gating; score_pipeline()
  traces/
    schema.sql          # inferences / model_lineage / failure_clusters
    ingest.py           # TraceStore + query_traces (SQL + bash pipe)
    taxonomy.py         # cluster + LLM-label fixability
  modes/
    production.py       # paper §2.6 closed loop
    cold_start.py       # paper §2.5 (full 5-stage workflow)
  tools/
    registry.py         # OpenAI/Anthropic tool specs + handlers

bench/adaptft/
  perturbations.py      # FIXABLE + POISON kinds (paper §3.1)
  build_scenario.py     # stage-based protocol (15% → 25% → 40%)

examples/
  run_production.py     # end-to-end demo
```

---

## How it differs from the paper

| Paper | This repo |
|---|---|
| Tinker SDK (LoRA + instant inference) | `peft` + `trl` SFT, optional `unsloth` |
| Modal sandboxes (16 GB, 24 h) | local Docker / direct execution; Modal optional |
| Claude Sonnet 4.6 + 32 K thinking | configurable orchestrator (default Claude Sonnet 4.6); 1 M-context caching when supported |
| Proprietary Context Manager | LangGraph-style state machine + simple older-turn summarization |
| Persistent `data-curation.md` | `runs/<model_id>/` with `taxonomy.json`, `train.jsonl`, `graph/graph.json` |
| Frontend GLiNER2 (encoder NER + classification) | decoder-only path first (Llama-3.2-3B, Qwen3-8B); GLiNER2 path scheduled for v0.2 |

---

## AdaptFT-Bench

```bash
# scenario from a JSONL of {"input":..., "gold":..., "metadata":{...}} rows
autoslm build-scenario my_intent ./examples_intent.jsonl \
    --base-model meta-llama/Llama-3.2-3B \
    --task classification \
    --out-dir ./bench_out \
    --stage-size 500
```

Produces `stage_{0..3}_train.jsonl`, `stage_{0..3}_test.jsonl`, and `held_out_test.jsonl`. Stages apply increasing poison rates per paper Section 3.1.

Perturbation kinds:

- **Fixable** (intent + answer recoverable): `typo`, `misspelling`, `grammatical_corruption`, `casing`, `truncation`, `preamble_injection`, `code_switching`
- **Poison** (training on raw teaches wrong behavior): `false_premise`, `label_flip`, `off_domain`, `prompt_injection`, `jailbreak`, `gibberish`, `empty`

The agent's `failure_diagnosis` step labels each cluster fixable/poison; only fixable clusters drive curriculum synthesis.

---

## Configuration

Most runtime options come from `AutoSLMConfig` (see [`autoslm/config.py`](autoslm/config.py)) or env vars:

| Env var | Default | Purpose |
|---|---|---|
| `AUTOSLM_ORCH_MODEL` | `anthropic/claude-sonnet-4-6` | Orchestrator + EXPAND/FUSE LLM |
| `AUTOSLM_JUDGE_MODEL` | `openai/gpt-4o-mini` | LLM-as-judge |
| `AUTOSLM_TEACHER_REASONING` | `deepseek/deepseek-reasoner` | CoT for math/reasoning |
| `AUTOSLM_TEACHER_GENERAL` | `openai/gpt-4.1` | CoT for code/general |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` | — | provider auth |

Pipeline-level knobs (`AutoSLMConfig`):

- `score_threshold` (default `0.96`, paper τ)
- `regression_epsilon` (default `2`, paper ε)
- `parallel_branches` (default `2`)
- `explore_coef_init` / `explore_coef_final` (UCT decay)

---

## Roadmap

- **v0.1** (done) — Production mode + MCGS + AdaptFT-Bench + multi-provider orchestrator + LoRA SFT
- **v0.2** (done) — Live confirmation probes, cross-checkpoint ratchet, audit log, Trace Analyzer sub-agent, GLiNER2 encoder path, cold-start mode (`autoslm cold-start`)
- **v0.3** (done) — Confidence calibration + TF-IDF correction propagation, FSDP for `big` tier, DPO/KTO objectives
- **v0.4** (done) — Distributed MCGS (parallel branch eval), Modal sandbox runner, paper benchmark replication suite (`autoslm repro <scenario>`), cost telemetry
- **Real-time validation** — `autoslm smoke-e2e` exercises the full closed loop on a tiny model in <10 min

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{atreja2026pioneer,
  title  = {Pioneer Agent: Continual Improvement of Small Language Models in Production},
  author = {Atreja, Dhruv and White, Julia and Nayak, Nikhil and Zhang, Kelton
            and Princis, Henrijs and Hurn-Maloney, George and Lewis, Ash
            and Zaratiana, Urchade},
  journal = {arXiv preprint arXiv:2604.09791},
  year   = {2026}
}
```

---

## License

MIT (this implementation). The original Pioneer Agent system from Fastino Labs is **not** open-source — this is an independent re-implementation following the published architecture only.
