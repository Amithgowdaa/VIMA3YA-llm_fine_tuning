# NimbusPay ticket triage: LoRA fine-tune

- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (Qwen license; ungated on Hugging Face)
- **Loaded in 4-bit:** yes (NF4, double quantization, float16 compute)
- **Best result:** 93.0% exact match and 98.8% mean field accuracy on all 200 dev tickets (batched scoring of the saved adapter)
- **Hardware:** trained on a local RTX 5070 (8 GB), because T4 sessions were unavailable on Colab and Kaggle. See `report.md`.

## Reproducing predictions
Load the base model in 4-bit with the settings above, apply `adapter/` with PEFT, and generate one ticket at a time with:

`model.generate(**inputs, do_sample=False, max_new_tokens=200)`

Use the chat template with the system and user messages only, and strip whitespace from the decoded output. The base model's default `generation_config` (`repetition_penalty=1.05`) was left on, so setting it to 1.0 can change the outputs.

## Files
| File | Purpose |
|---|---|
| `adapter/` | Final PEFT adapter |
| `predictions_test.jsonl` | Test predictions |
| `notebook.ipynb` | Pipeline, targeting a T4 |
| `01_data_audit.ipynb` | Part A audit with outputs |
| `report.md` | Findings, experiments, errors, AI use |
| `cleaning_log.csv` | Every dropped or fixed row |
| `pipeline.py`, `clean_data.py`, `augment.py`, `triage_rules.py` | Code the notebook runs |
