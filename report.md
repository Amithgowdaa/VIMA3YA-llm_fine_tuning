# NimbusPay ticket triage: report

**Base model:** `Qwen/Qwen2.5-3B-Instruct` (3.40B parameters), QLoRA in 4-bit NF4, rank 16, 29.9M trainable parameters.

## Hardware: departure from the brief
The brief requires a free Colab T4. I could not get a working GPU session: Colab's T4 was unavailable, and Kaggle sessions would not start on a different laptop, network or account. The final model, test predictions and experiments were therefore run on my laptop (RTX 5070, 8 GB). `notebook.ipynb` targets a T4, but it **has not been executed end to end on a T4**, so its run time there is unconfirmed. Peak memory and minutes below are laptop figures.

To fit 8 GB I set `max_len` to 1,500 tokens, which skipped 32 of the 1,338 clean rows (the longest email threads). Time limits also reduced the runs, as described under Experiments.

## Part A: audit
Every check went from hypothesis, to code that measures it, to reading real examples, to a decision. Details are in `01_data_audit.ipynb`; every change is in `cleaning_log.csv`.

| Problem | How found | Rows | Action |
|---|---|---|---|
| Truncated label JSON | `json.loads` fails | 25 | fix: category kept, rest rebuilt from the ticket |
| Old schema (`prio`, high/medium/low) | key-set counts | 70 | fix using SCHEMA v2 rules |
| Dates as DD/MM/YYYY | format regex | 30 | fix to ISO |
| Transaction id not normalized | regex `NP\d{10}` | 20 | fix |
| Priority breaks the SCHEMA rules | rule checker (0 hits on dev) | 60 | fix |
| Amount in rupees, not paise | label × 100 equals rule-based amount | 25 | fix |
| Exact duplicates | group by text | 60 | drop extra copies |
| Junk text with a full label | shortest bodies | 10 | drop |
| Tickets also in dev (leakage) | body overlap | 37 | drop |
| Non-whole-rupee amounts | checked against tickets | 158 | keep: these are correct |

**Result:** 1,445 rows became 1,338 (107 dropped, 200 fixed).

**Limitations.** A rule-based labeller rebuilds each label except category, and rules cannot detect a wrong category, so category noise may remain.

**Disclosure about dev and test.** While building that labeller I made two mistakes. I counted phrasing patterns across train, dev and test, and added some test-only phrasings to the rules. I also debugged the rules against dev labels and at first called that validation. I removed the test-derived patterns; `train_clean.jsonl` came out byte-identical (0 of 1,338 labels changed), and the augmentation uses only `train_clean.jsonl` and `SCHEMA.md`. The dev debugging remains and is noted in `triage_rules.py`, so the labeller's agreement with dev is not independent evidence. No dev or test row was used for training.

## Part C: pipeline
- **Chat template:** the tokenizer's own template with the fixed system and user messages, and the generation prompt added.
- **Loss:** prompt tokens are labelled -100, so loss covers only the JSON answer and `<|im_end|>`.
- **Sequence length:** token counts show a median of about 171 with long email threads up to about 2,214. I would use 2,300 on a T4; 1,500 was forced by 8 GB. Rows over the limit are skipped, never truncated, because truncation would remove the answer.
- **Settings:** rank 16 / alpha 32 on all attention and MLP linear layers (about 30M trainable, under the 50M limit), lr 2e-4 with linear decay, up to 8 rows per batch with a token cap to bound memory, gradient checkpointing on.
- **Final model:** all usable clean rows, 2 epochs. This configuration was fixed before the experiment results, because of time.
- **Inference:** chat template, plain `model.generate(do_sample=False, max_new_tokens=200)`, whitespace stripping only, one ticket at a time, on the adapter reloaded from disk. The base model's `generation_config` applies `repetition_penalty=1.05`, and I left that default in place.

## Parts B and D: results
Experiments use one change at a time from `exp_base`: 200 sampled training rows, 2 epochs, batched scoring on the first 80 dev rows. **These are small-sample comparisons, so differences of a few points are not meaningful.** Batched scoring can differ slightly from one-at-a-time scoring.

| Run | What changed | Dev exact | Dev field acc | Peak GPU GB | Train min |
|---|---|---|---|---|---|
TBD_TABLE

Conclusions: TBD after results.

## Part E: error analysis (final model, dev)
Dev exact TBD, mean field accuracy TBD (batched scoring on the saved adapter).

TBD: weakest fields and categories, and at least 10 failures with explanations.

## With one more week
TBD after results. Candidates: run the full set of experiments on a real T4 at full size; audit category labels by hand; train on the longest threads again with `max_len` 2,300; measure the augmentation's effect by transform type.

## AI tool use
I used Claude Code (Anthropic) throughout: it helped write and debug the audit notebook, `triage_rules.py`, `clean_data.py`, `augment.py`, `pipeline.py` and the notebooks, and it drafted this report from my results. The audit decisions on keeping or fixing truncated rows were mine. No AI produced labels or predictions: the predictions come only from the adapter.
