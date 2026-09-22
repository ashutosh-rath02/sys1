# sys1

A small, self-hosted "System One" model: fast typed judgments (Choice / Score / Noul)
instead of free-text generation, inspired by TypeSafe's Jev. Code owns the workflow;
the model supplies calibrated probabilities over a fixed set of answers.

This is being built in public — progress documented on Twitter as each stage lands.

## Roadmap

- **v0 (here now):** zero-shot logit scoring. Wrap any small open-weight instruct
  model (default: Qwen2.5-0.5B-Instruct, CPU-friendly) and derive calibrated
  probabilities directly from next-token logits over a constrained answer set
  (A/B/C/... or Yes/No), instead of asking the model to "say" a probability.
  No training required — this runs today, on CPU, with no GPU.
- **v1: LoRA fine-tune (done).** `training/train_lora.ipynb` fine-tunes the
  base model with plain next-token cross-entropy (via `SFTTrainer`) on
  judgment-shaped data pulled from real public datasets (`training/build_from_hf.py`).
  Kept as the benchmark baseline. Trains on a free Colab GPU; this machine
  only prepares data and runs eval.
- **v2: calibrated-decision training (current).** Plain SFT optimizes "say
  the right letter," not "have a calibrated probability for it." TypeSafe's
  docs describe Jev's own training as RLCD — Reinforcement Learning for
  Calibrated Decisions. Our decision (softmax over a handful of candidate
  letters) is a single differentiable step, so full RL machinery (reward
  model, PPO) isn't needed to get the same property: `training/train_calibrated.ipynb`
  trains directly on the exact candidate-restricted distribution
  `src/sys1/engine.py` reads at inference, via plain gradient descent on
  that proper scoring rule. `src/sys1/promptutil.py` is the single shared
  prompt-building module both training and inference use, so the two can't
  drift into different formats the way v1's Score prompts once did.
- **v3: scale up.** Bigger base model, more/real training data, a served
  API in front of it.

## Why logits instead of asking the model for a probability?

LLMs are bad at introspecting and reporting their own confidence as text
("I'd say 73% confident..."). But the next-token logits over a small set of
candidate answers already encode a real distribution. Constraining the
answer to a short fixed set (single letters, Yes/No) and reading off the
softmax over just those candidates gives an actually calibratable signal —
this is the same idea behind standard multiple-choice-via-logprobs evaluation.

## Primitives

- **Choice** — pick one of N options; returns the full probability distribution.
- **Noul** — is a condition true; returns a probability of "yes".
- **Score** — a degree along an ordered set of levels; returns a distribution
  over levels plus the probability-weighted expected level.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python examples/demo.py
```

First run downloads the base model from Hugging Face (~1GB for the 0.5B default).

## Layout

```
src/sys1/       core engine + primitives (the actual "model wrapper")
examples/       runnable demos
training/       LoRA fine-tuning pipeline (data prep + Colab notebook) — v1
data/           seed/training examples in judgment-shaped JSONL
eval/           calibration evaluation (Brier score, accuracy vs. confidence)
```
