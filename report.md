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
Experiments use one change at a time from `exp_base`: 200 sampled training rows, 2 epochs, batched scoring on the first 80 dev rows. **These are small-sample comparisons, so differences of a few points are not meaningful.** Memory limits differ between some runs, as noted below. Batched scoring can differ slightly from one-at-a-time scoring.

| Run | What changed | Dev exact | Dev field acc | Peak GPU GB | Train min |
|---|---|---|---|---|---|
| baseline_base | no fine-tuning, fixed prompt | 0.0% | 0.0% | - | - |
| baseline_rules | no fine-tuning, rules in prompt | 2.5% | 20.8% | - | - |
| exp_base | reference: clean, 4-bit, rank 16 | 41.2% | 86.9% | 5.46 | 1.43 |
| exp_raw | 1. raw data instead of clean | 41.2% | 90.8% | 6.29 | 1.74 |
| exp_4bit_500 | 2a. 4-bit with max_len 500, batch 2 (match for 2b) | 51.2% | 89.7% | 4.42 | 1.69 |
| exp_16bit | 2b. 16-bit with max_len 500, batch 2, to fit 8 GB | 77.5% | 95.0% | 7.02 | 1.65 |
| exp_rank8 | 3. rank 8 instead of 16 | 32.5% | 84.7% | 6.28 | 1.43 |
| exp_aug | 4. own idea: SCHEMA-based augmentation | did not complete | | | |
| final | final: all clean rows, 2 epochs, all 200 dev | 93.0% | 98.8% | 6.45 | 10.29 |

**Part D is incomplete, and I want that to be clear.** `exp_aug` ran out of GPU memory and time ran out before it could be re-run, so my own idea has no result. The raw, 16-bit and rank comparisons are measured.

**Conclusions, with caution given 200 training rows and 80 dev tickets:**
- **Baselines:** the untrained model scored 0.0% exact, and 2.5% with the rules in the prompt. Writing the rules into the prompt did not produce the required format.
- **Raw vs clean:** this is a negative result for cleaning at this scale. Raw data matched the reference on exact match (41.2% each) and was higher on field accuracy (90.8% against 86.9%). With 200 different sampled rows and 80 dev tickets, it does not show that cleaning hurts, but it also gives no evidence that cleaning helped here.
- **16-bit vs 4-bit (2a and 2b, matched at `max_len` 500 and batch 2):** 16-bit reached 77.5% exact against 51.2% for 4-bit, at 7.02 GB against 4.42 GB. This needs confirming at full size. Scoring batch size differed between these two runs.
- **Rank 8 vs 16:** 32.5% against 41.2%. Rank 8 was run with `token_budget` 1500 and the reference with 1000, after the reference ran out of memory at 1500, so this is not a strict one-change comparison.
- **Amount of data matters most:** the final model on all usable rows reached 93.0%, far above any 200-row run.
- The final model is 4-bit because that configuration was fixed before these results. The 16-bit result is the first thing I would test at full size.

## Part E: error analysis (final model, dev)
All 200 dev tickets: **93.0% exact match, 98.8% mean field accuracy, 100% valid JSON** (batched scoring of the saved adapter). This configuration was fixed in advance, but dev was also used to debug the labeller, so read the score as somewhat optimistic.

**Weakest fields:** priority 95.5%, needs_human 97.0%, txn_date 98.5%, category 99.0%. Amount, transaction id, channel and language were 100%.
**Weakest categories (exact match):** fraud 86.7% (n=15), account_access 89.5% (n=19), payment_failed 90.6% (n=53), refund 90.9% (n=55). KYC, offers and other were 100%.

**Main finding: the model reads amounts correctly but compares them poorly.** `amount_paise` was right on all 200 tickets, yet 7 of the 9 priority errors are mistakes about the ₹5,000 or ₹50,000 thresholds. Five of the 6 `needs_human` errors follow directly from a wrong P1 decision, since `needs_human` depends on priority. The model is pattern-matching on size, not doing arithmetic.

All 14 failures:

| Ticket | Wrong | What happened | Likely cause |
|---|---|---|---|
| dv-00034 | priority, needs_human | ₹36,737 marked P1 | Below ₹50,000; threshold error |
| dv-00056 | priority, needs_human | ₹44,206.75 marked P1 | Threshold error |
| dv-00098 | priority, needs_human | 41.5k marked P1 | Threshold error on `k` notation, with a balance also present |
| dv-00129 | priority, needs_human | ₹49,657 marked P1 | ₹343 below the boundary |
| dv-00037 | priority, needs_human | ₹2,84,669 marked P2 | Well over ₹50,000; threshold error in the other direction |
| dv-00055 | priority | ₹7,576 refund marked P3 | Over ₹5,000 for a refund, so P2 |
| dv-00180 | priority | ₹3,130 refund marked P2 | Under ₹5,000; a balance of ₹2,41,670 is a possible distractor |
| dv-00083 | priority | Account access, no amount, marked P3 | The rule gives P2 for this category regardless of amount; absence of an amount seems to suggest P3 |
| dv-00157 | category, priority | OTP theft read as account access, so P3 | Category error cascades into priority: fraud is always P1 |
| dv-00177 | category | "Red flag on my top up" read as fraud | Genuinely ambiguous wording |
| dv-00113 | needs_human | "Pehle bhi complaint kar chuka hoon" missed | Repeat-contact phrase in Hinglish, on a ticket with few other signals |
| dv-00006 | txn_date | "Day before yesterday" gave one day ago | Off-by-one in relative date resolution |
| dv-00171 | txn_date | 24-05-26 became 2026-05-26 | Two-digit year in DD-MM-YY misread as the day |
| dv-00114 | txn_date | "Dated Aug 30" returned null | Date phrase after several distractors (ticket number, legal threat) |

## With one more week
1. **Thresholds first:** this is the largest error group. Add training rows spread across the ₹5,000 and ₹50,000 boundaries, then test this change on its own at full size.
2. Complete the augmentation experiment, then run every experiment on a real T4 with the full data and all 200 dev rows, with several seeds, since the current comparisons are small-sample.
3. Audit category labels by hand, because the rules could not check them.
4. Restore `max_len` 2,300 so the longest email threads are used in training, and add examples of two-digit years and Hinglish repeat-contact phrases.

## AI tool use
I used Claude Code (Anthropic) throughout: it helped write and debug the audit notebook, `triage_rules.py`, `clean_data.py`, `augment.py`, `pipeline.py` and the notebooks, and it drafted this report from my results. The audit decisions on keeping or fixing truncated rows were mine. No AI produced labels or predictions: the predictions come only from the adapter.
