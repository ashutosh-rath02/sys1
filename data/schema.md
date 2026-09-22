# Training data schema

Each line in a training JSONL file is one judgment example:

```json
{
  "primitive": "choice",
  "instructions": "Classify what this support ticket is about.",
  "state": {"ticket": "My card was charged twice for the same order."},
  "options": ["billing", "technical support", "sales", "spam"],
  "answer": "billing"
}
```

- `primitive`: `"choice"`, `"noul"`, or `"score"`.
- `instructions`: the judgment being asked, in plain language.
- `state`: arbitrary JSON context the judgment is about. Optional.
- `options` / `levels`: required for `choice` and `score`; omitted for `noul`
  (always a yes/no question, so no reason to interpret them as calibrated
  ground-truth probabilities).
- `answer`: the correct label — one of `options`/`levels`, or `"yes"`/`"no"`
  for `noul`. Single ground-truth label, not a probability: at training time
  we don't invent a fake distribution, we just teach the model to prefer the
  correct letter, and let logit-based scoring (see `src/sys1/engine.py`)
  produce the actual probability distribution over the fixed answer set at
  inference time.

See `data/seed_examples.jsonl` for a small hand-written starter set.
Expand it before running `training/prepare_dataset.py`.
