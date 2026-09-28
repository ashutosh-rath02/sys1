# sys1

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A small, open, self-hosted **"System One" model**: fast typed judgments
(`Choice` / `Score` / `Noul`) with real calibrated probabilities, instead of
generated text. Code owns the workflow and the rules it already knows; the
model supplies the one judgment that actually needs semantic understanding.

Inspired by TypeSafe's Jev and Convai's Laya. Built in public, including the
parts that didn't work — see [Progress](#progress-the-honest-version) below.

```python
from sys1 import Choice, FastEngine

engine = FastEngine("models/sys1-fast-encoder-v2")

result = Choice(["billing", "technical support", "sales", "spam"]).ask(
    "Classify what this support ticket is about.",
    {"ticket": "My card was charged twice for the same order, please refund me."},
    engine,
)
# {"choice": "billing", "probabilities": {"billing": 0.999, "technical support": 0.0001, ...}}
```

## Why not just ask an LLM to say a probability?

LLMs are bad at introspecting and reporting their own confidence as text
("I'd say 73% confident..."). `sys1` instead reads a real probability
distribution directly off the model — either the softmax over a small set of
candidate-answer logits (the v1/v2 causal-LM engine), or a small encoder's
own decision head scored per option (the v3 fast engine, the current
default). Either way, the number you get back is a property of the model's
computation, not a sentence it composed.

## Primitives

- **`Choice`** — pick one of N options; returns the full probability distribution.
- **`Noul`** — is a condition true; returns a probability of "yes".
- **`Score`** — a degree along an ordered set of levels; returns a
  distribution over levels plus the probability-weighted expected level.

Each is a single call: `Primitive(...).ask(instructions, state, engine)`,
against either `Engine` (a causal chat model, read via constrained logits) or
`FastEngine` (a small non-autoregressive encoder + decision head) — same
interface, swap freely.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; `source .venv/bin/activate` elsewhere
pip install -e .[serve,ui]
python examples/demo.py         # runs the primitives against a real model
```

First run downloads a base model from Hugging Face (tens of MB for the
default fast encoder; ~1GB for the fallback 0.5B causal model).

### Run the API server

```bash
sys1-serve                      # http://localhost:8000, docs at /docs
curl -X POST localhost:8000/choice -H "Content-Type: application/json" -d '{
  "instructions": "Classify what this support ticket is about.",
  "options": ["billing", "technical support", "sales", "spam"],
  "state": {"ticket": "My card was charged twice, please refund me."}
}'
```

### Run the interactive playground

```bash
streamlit run ui/app.py         # judgment playground + a live Snake demo
```

## Architecture

- **v3 (default): fast encoder.** A small non-autoregressive encoder
  (`sentence-transformers/all-MiniLM-L6-v2`, 22M params) with a one-neuron
  decision head, scored per candidate option in one batched forward pass —
  no chat template, no generation. `src/sys1/fast_engine.py`.
- **v1/v2: causal LM.** Any small HF instruct model (default
  Qwen2.5-0.5B/1.5B-Instruct), read via softmax over constrained
  candidate-letter logits. Kept as the benchmark baseline.
  `src/sys1/engine.py`.

Measured on one CPU laptop, no GPU:

| | v1/v2 (causal LM) | v3 (fast encoder) |
|---|---|---|
| Latency per decision | ~2.3s | **34.6ms** |
| Accuracy (own held-out eval) | 85.9% | **87.1%** → **91.3%** after domain training |
| Brier score | 0.21 | 0.18 → 0.12 |

## Progress: the honest version

Every stage below actually happened, in order, including the ones that made
things worse:

1. **v0** — zero-shot logit scoring, no training at all. 23.1% accuracy.
2. **v1** — LoRA SFT fine-tune on real public data. 59–72% accuracy
   depending on data volume and base model size.
3. **v2** — switched from plain next-token cross-entropy to training
   directly on the exact candidate-restricted decision distribution
   (closer to what Jev's own training, RLCD, actually optimizes for).
   85.9% accuracy, same model — the training *objective* mattered more than
   scale.
4. **v3** — swapped the causal-LM-plus-logits architecture for a small
   non-autoregressive encoder, matching Laya's actual approach instead of
   retrofitting a chat model. 130x faster and more accurate, on the same CPU.
5. **The reality check** — benchmarked against the same public datasets used
   to compare Jev and Laya ([`Luni/laya-jev-benchmark`](https://huggingface.co/datasets/Luni/laya-jev-benchmark)),
   not just our own eval set. Result: 37.4% on an agent-trace risk-assessment
   task completely unlike our training data. Trained on that benchmark's own
   train split (never touching its test data) and closed most of the gap —
   but calibration got *worse* even as accuracy improved. That overconfidence
   problem is the current open issue, not a new dataset.

See `eval/benchmark_typed_decisions.py` and `eval/benchmark_phishing.py` to
reproduce the public-benchmark numbers yourself.

## Training your own

```bash
python training/build_from_hf.py                        # real labeled data, no synthetic generation
python training/build_benchmark_domain_data.py           # + public benchmark train splits
python training/train_fast_encoder.py data/*.jsonl --out-dir models/sys1-fast-encoder-v2
```

The causal-LM path (`training/train_lora.ipynb`, `training/train_calibrated.ipynb`)
runs on a free Colab GPU; the fast-encoder path trains locally on CPU in
minutes, since the model is tiny.

## Layout

```
src/sys1/       engine.py (causal LM) + fast_engine.py (v3 encoder) + primitives + server.py (FastAPI)
src/sys1/games/ Snake, used as a live decision-making demo (see ui/app.py)
training/       data builders + fine-tuning scripts (Colab notebooks + local)
eval/           calibration + public-benchmark evaluation harnesses
examples/       runnable demos
ui/             Streamlit playground (judgment primitives + live Snake)
data/           judgment-shaped training/eval data (schema in data/schema.md)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
