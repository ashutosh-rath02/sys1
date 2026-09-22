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
- **v1: LoRA fine-tune.** Use `training/` to fine-tune the base model on
  judgment-shaped data (state + instructions + criteria -> answer) so it gets
  better at the task and better calibrated. Trains on a free Colab/Kaggle GPU;
  this machine only prepares data and runs eval.
- **v2: scale up.** Bigger base model, more/real training data, dedicated
  eval/calibration suite, maybe a served API in front of it.

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
