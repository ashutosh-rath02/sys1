# Closing the gap

Where we are, where the funded models are, and the specific steps between.

## Baseline (measured, same held-out data each time)

| | own domain | typed-decisions | phishing |
|---|---|---|---|
| v2 (cross-encoder, 22.7M) | 91.3% / Brier 0.119 | 64.4% / 0.263 | 56.5% / 0.370, recall 14.7% |
| v4 (dual-encoder, 22.9M) | 96.3% / 0.064 | 62.6% / 0.223 | 58.0% / 0.416, recall 17.6% |

Targets (published, different eval conditions — see the comparison card):
Laya typed-decisions 76.6%, Laya phishing 98.0%, Jev phishing 95.0%
(5-question decomposition), Laya ECE 0.081 after temperature fitting.

Laya is 421M params against our 22.9M. We do not close that by scaling —
we have no GPU budget for an 18x model. We close it with technique
(decomposition, calibration), data alignment, and a modestly bigger
backbone. Some of the gap will not close, and that's a fine answer.

---

## Step 1 — One-command scoreboard + ECE

We can't iterate fast while every benchmark is a separate manual run, and
we can't compare against Laya at all without ECE (we only measure Brier).

- `eval/run_all.py`: all three benchmarks, one command, one table.
- Add ECE alongside accuracy / Brier / latency.
- Record v2 and v4 baselines through it.

**Gate:** one command prints a full scoreboard. Baselines recorded.

## Step 2 — Temperature scaling

One scalar per primitive, fit on held-out data. Laya reports 0.466 → 0.081
ECE from exactly this. Costs no training and cannot change accuracy
(temperature never moves the argmax), so it is pure calibration gain.

**Gate:** ECE < 0.12 on all three benchmarks, accuracy unchanged.

## Step 3 — Question decomposition

The single biggest available win, and it needs no training. Jev goes
62.6% → 95.0% on phishing purely by splitting one judgment into five
narrow sub-questions and combining them with fitted weights. Our whole
premise is that code composes narrow judgments, so this is on-thesis.

- Decomposition layer: N sub-questions → combine via weights fit on
  held-out labels.
- Phishing first (clearest target, worst current number).

**Gate:** phishing 56.5% → 80%+, recall above 50% (currently 14.7% — the
real failure is that it almost always answers "legitimate").

## Step 4 — Fix the phishing data mismatch

Current phishing training is 76% phishing examples, yet the model answers
"legitimate" 96% of the time at eval with precision 1.0. That is not class
imbalance — the core split's phishing (subtle: a friendly work email with
a GitHub Pages link) looks nothing like our training phishing
(`real_phishing_validation`, likely blatant).

- Split PhishNChips `core` 50/50, train on one half, eval on the other.
- If the hypothesis holds, this confirms the number was never measuring
  model capacity.

**Gate:** mismatch confirmed or killed, with numbers either way.

## Step 5 — Training recipe

v3 added 16k examples and got *worse* everywhere — same 6 epochs spread
over 4.5x more data means far fewer effective passes over what we're
measured on.

- Domain-weighted sampling so `bev-decision` stops drowning out the
  target domains.
- More epochs, now that Colab GPU training works.

**Gate:** typed-decisions 64.4% → 72%+ without regressing own domain.

## Step 6 — Bigger backbone (only if 1–5 land)

MiniLM-L12 or ModernBERT-base (~150M) — still 3x smaller than Laya, but
6x ours, and trainable on a free Colab GPU.

**Gate:** typed-decisions 76%+ (Laya parity on the metric we can compare).

---

## Rules

- All training on Colab GPU. Local CPU training is abandoned — it hung
  twice and segfaulted twice, and the fixes didn't hold.
- All eval local, through `eval/run_all.py`, same held-out data every time.
- Every step reports its real number, including the ones that don't work.
  v3 made things worse and that's recorded here rather than buried.
