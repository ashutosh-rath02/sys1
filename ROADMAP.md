# Closing the gap

Where we are, where the funded models are, and the specific steps between.

## Baseline (via `eval/run_all.py`, same held-out data each time)

v2 (cross-encoder, 22.7M):

| benchmark | n | acc | Brier | ECE | ms |
|---|---|---|---|---|---|
| own_domain | 241 | 91.3% | 0.119 | 0.039 | 44.9 |
| typed_decisions | 500 | 64.4% | 0.525 | 0.216 | 140.7 |
| phishing | 200 | 56.5% | 0.739 | 0.373 | 460.8 |

v4 (dual-encoder, 22.9M) — **current best**, recall 17.6% on phishing:

| benchmark | n | acc | Brier | ECE | ms |
|---|---|---|---|---|---|
| own_domain | 241 | 96.3% | 0.064 | 0.025 | 26.0 |
| typed_decisions | 500 | 62.6% | 0.495 | 0.166 | 98.1 |
| phishing | 200 | 58.0% | 0.832 | 0.419 | 106.3 |

v4 wins almost everywhere and is 2–4x faster; v2 holds a small edge on
typed-decisions accuracy (64.4 vs 62.6) and phishing Brier/ECE. Work from
v4 unless a step specifically needs otherwise.

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

**Result — partially met.** A single global per-primitive temperature,
fit cross-domain on train-side data (choice 1.294, noul 1.178, score
1.227 — all above 1.0, i.e. the model really is systematically
overconfident):

| benchmark | ECE before | ECE after | accuracy |
|---|---|---|---|
| own_domain | 0.025 | 0.040 (worse) | unchanged |
| typed_decisions | 0.166 | 0.125 | unchanged |
| phishing | 0.419 | 0.418 (no change) | unchanged |

Accuracy identical everywhere, confirming the implementation is sound
(temperature cannot move an argmax). But the two failures are
informative rather than noise:

- **phishing didn't move at all.** Its miscalibration is distribution
  shift, not generic overconfidence, so a temperature fit on other
  domains has nothing to grab onto.
- **own_domain got slightly worse** for the mirror reason — it was
  already well calibrated at 0.025, so a cross-domain temperature just
  over-softens it.

Both point the same way: temperatures must be scoped per domain, which is
what the published work actually does ("after *domain* temperature
fitting"). `Calibrator` now supports `"<domain>:<primitive>"` keys with
fallback to the global fit; phishing refit on phishing-only train data.

## The phishing benchmark is mostly mechanical — measured, not argued

Before building anything on top of the phishing number, we checked what
code alone scores. `eval/phishing_signal_baseline.py` extracts five
string-comparison features (sender is webmail, link is on free hosting,
sender/link registered domains differ, bare IP, plain http), fits logistic
regression on the train half, scores the eval half:

| | accuracy | Brier | ECE | recall |
|---|---|---|---|---|
| **regex only, no model** | **95.3%** | 0.077 | 0.098 | 90.6% |
| sys1 v4 (dual-encoder) | 58.3% | 0.830 | 0.417 | 18.8% |

A single boolean — *is the sender address webmail?* — scores **94.5%** on
its own. For reference, Jev's published figure on this benchmark is 95.0%.

So the benchmark is close to saturated by surface cues, and nobody's
number on it (ours included) is measuring semantic phishing
understanding. Two consequences:

1. The regex baseline stays on the scoreboard permanently. A model number
   quoted on this benchmark without it next to it is misleading.
2. The right architecture is the one this project already argues for:
   code computes what code can determine, the model is spent only on what
   needs semantics. Feeding these signals as state beats asking a 23M
   encoder to infer them from prose.

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

**Confirmed, decisively, while doing Step 2.** Fitting a phishing-specific
temperature ran straight to the bottom of the search range because the
model is *perfect* on phishing training data:

| | accuracy | avg confidence |
|---|---|---|
| phishing train-side (non-core splits) | 100.0% | 1.000 |
| phishing eval (`core`) | 58.3% | — (recall 18.8%) |

It separated `real_phishing_validation` from `cross_domain_legitimate_v5`
on some superficial cue and learned nothing that transfers. That 58% was
never measuring model capacity on phishing — it was measuring how badly
the training sources mismatched the eval set.

Fix in progress: `core` split 1000/1000 by a deterministic shuffle
(`training/build_phishing_core_split.py` takes the first half,
`eval/run_all.py` scores the second). Train split is balanced — 502
phishing, 498 legitimate.

Before, on the exact eval half used after (n=300): **58.3% acc, Brier
0.830, ECE 0.417, recall 18.8%** — statistically identical to the old
sample, so nothing here is a sampling artifact.

**Result — hypothesis confirmed, emphatically.** Trained on the `core`
train half (Kaggle GPU), scored on the held-out half:

| phishing | before | after |
|---|---|---|
| accuracy | 57.5% [50.6, 64.4] | **99.5%** [98.5, 100.0] |
| recall | 17.5% | **99.0%** |
| ECE | 0.426 | **0.005** |
| Brier | 0.846 | **0.006** |

Nothing else moved: own_domain 96.3 → 96.7%, typed_decisions 46.6 →
47.2%, both inside their intervals.

So phishing was never a capacity problem, a calibration problem, or an
architecture problem. The model could always learn this task; it had
simply never been shown the right distribution. Worth remembering the
order we discovered that in — we spent Step 2 trying to calibrate our way
out of it, and the calibration fit itself (running to T=0.010 on data the
model scored 100% on) is what finally pointed at the real cause.

Two things keep this honest:

- A regex scores 95.3% here. Beating it by four points is a smaller
  achievement than "99.5%" sounds.
- It is a fitted in-distribution number, not zero-shot. That is roughly
  Jev's condition for its published 95.0% (five sub-questions with
  weights fit on ~1,000 in-distribution labels), but it is not the same
  thing as generalising to unseen phishing.

## Step 5 — Training recipe

v3 added 16k examples and got *worse* everywhere — same 6 epochs spread
over 4.5x more data means far fewer effective passes over what we're
measured on.

- Domain-weighted sampling so `bev-decision` stops drowning out the
  target domains.
- More epochs, now that Colab GPU training works.

**Gate:** typed-decisions improvement beyond the noise floor.

**Result — met, and it's the first change this project can prove isn't
noise.** Trained on all four workflows (8,500 examples) instead of one:

| | before | after |
|---|---|---|
| typed_decisions | 47.2% [45.1, 49.4] | **63.8%** [61.7, 66.0] |
| typed_decisions ECE | 0.246 | **0.102** |
| phishing | 99.5% | **100.0%** |
| own_domain | 96.7% [94.4, 98.9] | 93.8% [90.7, 96.8] |

+16.6 points with non-overlapping intervals, and calibration error more
than halved. Gap to Laya's 76.6% goes from 29.4 points to 12.8.

The own_domain point estimate fell about 3 points, but the intervals
overlap, so it isn't distinguishable from noise — flagged to watch rather
than claimed as a regression, by the same rule that stopped us claiming
the wins.

Worth noting what the fix actually was: not a better architecture, a
bigger model, or a cleverer loss. We had simply never trained on three of
the four workflows we were grading ourselves on.

## Step 6 — Bigger backbone (only if 1–5 land)

`jhu-clsp/ettin-encoder-68m` (MIT, 68M) looks like the best point on the
curve — reported fine-tune GLUE 87.2 against ModernBERT-base's 88.4 at
less than half the size — and unlike `all-MiniLM-L6-v2` it ships an MLM
head, which is what makes continued pretraining on task text possible at
all. `answerdotai/ModernBERT-base` (Apache-2.0, 149M) if we stay
dual-encoder, since caching action embeddings means only the state side
pays the cost.

**Gate:** typed-decisions improvement beyond the noise floor (see below).

---

## Two corrections to how we've been measuring

**The typed-decisions gate was apples-to-oranges.** Our 62.6% is n=500 on
*one* workflow (agent-trace-observability). Laya's 76.6% is n≈2,000 across
*four* workflows on a checkpoint fine-tuned for them. "76%+ = parity" was
never a like-for-like target and is withdrawn.

**We've been reading noise as signal.** At n=500 the 95% CI on
typed-decisions accuracy is roughly ±4 points, so v2's 64.4% vs v4's 62.6%
is indistinguishable — and the own-domain ECE 0.025 → 0.040 that Step 2
called a regression is likely inside the floor too at n=241. Every
comparison from here reports an interval, and no change under ~4 points
counts as a result without multiple seeds.

---

## Rules

- All training on Colab GPU. Local CPU training is abandoned — it hung
  twice and segfaulted twice, and the fixes didn't hold.
- All eval local, through `eval/run_all.py`, same held-out data every time.
- Every step reports its real number, including the ones that don't work.
  v3 made things worse and that's recorded here rather than buried.
