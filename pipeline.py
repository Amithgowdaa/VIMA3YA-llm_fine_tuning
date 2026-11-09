"""Train and evaluate a LoRA / QLoRA adapter for NimbusPay ticket triage.

    python pipeline.py --name base_clean                       # default config
    python pipeline.py --name raw --data raw                   # change one thing at a time
    python pipeline.py --name baseline_rules --mode rules      # no training, rules in the prompt

Inference is exactly the assignment's: system + user through the chat template, then a plain
model.generate(do_sample=False, max_new_tokens=200), with whitespace stripping only.
"""
import argparse
import gc
import json
import math
import os
import random
import re
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

import augment as A
import clean_data
from score import score as score_rows

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_PARAMS, MAX_TRAINABLE = 4_000_000_000, 50_000_000


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def training_rows(kind, limit=0, seed=42):
    """clean | raw | aug. 'raw' only removes the rows that also appear in dev (never train on dev).
    limit > 0 takes a fixed random sample; for 'aug', only the sampled rows are augmented."""
    def pick(rows):
        return random.Random(seed).sample(rows, limit) if 0 < limit < len(rows) else rows
    clean = load("data/train_clean.jsonl")
    if kind == "clean":
        return pick(clean)
    if kind == "aug":
        sample = pick(clean)
        return sample + A.augment(sample)
    dev_bodies = {clean_data.norm_body(r) for r in load("data/dev.jsonl")}
    return pick([r for r in load("data/train.jsonl") if clean_data.norm_body(r) not in dev_bodies])


def encode(tok, row, max_len):
    """Prompt tokens get label -100, so loss is computed only on the JSON answer and its end token."""
    prompt = tok.apply_chat_template(row["messages"][:2], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(row["messages"][2]["content"] + tok.eos_token, add_special_tokens=False).input_ids
    ids = p + a
    return None if len(ids) > max_len else {"input_ids": ids, "labels": [-100] * len(p) + a}


def batches(examples, batch_size, token_budget, rng):
    """Group similar lengths; cap batch_size * longest so long email threads cannot exhaust memory."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    out = []
    for i in range(0, len(idx), 64):
        chunk = sorted(idx[i:i + 64], key=lambda j: len(examples[j]["input_ids"]))
        cur = []
        for j in chunk:
            n = len(examples[j]["input_ids"])
            if cur and (len(cur) >= batch_size or n * (len(cur) + 1) > token_budget):
                out.append(cur); cur = []
            cur.append(j)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def collate(examples, idx, pad, device):
    m = max(len(examples[i]["input_ids"]) for i in idx)
    ids, lab, att = [], [], []
    for i in idx:
        e = examples[i]; k = m - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad] * k); lab.append(e["labels"] + [-100] * k); att.append([1] * len(e["input_ids"]) + [0] * k)
    return {"input_ids": torch.tensor(ids, device=device), "labels": torch.tensor(lab, device=device), "attention_mask": torch.tensor(att, device=device)}


def load_base(model_id, bits):
    tok = AutoTokenizer.from_pretrained(model_id)
    assert tok.eos_token == "<|im_end|>", tok.eos_token
    if bits == 4:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map={"": 0})
    # 4-bit parameters report a packed size, so count on an empty, unquantized copy of the architecture
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id))
    n = sum(p.numel() for p in skeleton.parameters())
    assert n <= MAX_PARAMS, n
    return tok, model, n


@torch.no_grad()
def predict(model, tok, rows, system_suffix="", **gen_overrides):
    """One ticket at a time and no autocast: the plain call the reviewers will use to re-run the adapter."""
    model.eval()
    out = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"][:2]]
        msgs[0]["content"] += system_suffix
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**inp, do_sample=False, max_new_tokens=200, pad_token_id=tok.pad_token_id, **gen_overrides)
        out.append({"id": r["id"], "output": tok.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()})
    return out


@torch.no_grad()
def predict_batched(model, tok, rows, system_suffix, batch_size, token_budget):
    """Faster scoring, used only to compare experiments. Final dev scores and test predictions use predict()."""
    model.eval()
    texts = []
    for r in rows:
        msgs = [dict(m) for m in r["messages"][:2]]
        msgs[0]["content"] += system_suffix
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    groups, cur = [], []
    for j in sorted(range(len(rows)), key=lens.__getitem__):
        if cur and (len(cur) >= batch_size or (lens[j] + 200) * (len(cur) + 1) > token_budget):
            groups.append(cur); cur = []
        cur.append(j)
    if cur:
        groups.append(cur)
    tok.padding_side = "left"      # generation continues from the end of each prompt
    out = [None] * len(rows)
    for g in groups:
        inp = tok([texts[j] for j in g], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**inp, do_sample=False, max_new_tokens=200, pad_token_id=tok.pad_token_id)
        for j, seq in zip(g, gen):
            out[j] = {"id": rows[j]["id"], "output": tok.decode(seq[inp.input_ids.shape[1]:], skip_special_tokens=True).strip()}
    return out


def dev_metrics(preds):
    ids = {p["id"] for p in preds}      # with --limit, score only the dev rows that were predicted
    return score_rows([g for g in load("data/dev.jsonl") if g["id"] in ids], preds)


def rules_text():
    s = open("SCHEMA.md", encoding="utf-8").read()
    return "\n\nFollow this specification exactly. Output only the JSON object.\n\n" + s[s.index("## Output"):]


def train(a, tok, model, n_params):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) if a.bits == 4 else model
    if a.bits != 4:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS, bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable <= MAX_TRAINABLE, trainable
    print(f"base params {n_params/1e9:.2f}B | trainable {trainable/1e6:.1f}M", flush=True)

    rows = training_rows(a.data, a.train_limit, a.seed)
    enc =[encode(tok, r, a.max_len) if _usable(r) else None for r in rows]
    examples = [e for e in enc if e]
    print(f"rows {len(rows)} | usable {len(examples)} | skipped {len(rows) - len(examples)}", flush=True)

    rng = random.Random(a.seed)
    per_epoch = len(batches(examples, a.batch_size, a.token_budget, random.Random(a.seed)))
    total = per_epoch * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0, step = time.time(), 0
    model.train()
    for epoch in range(a.epochs):
        running, seen = 0.0, 0
        for b in batches(examples, a.batch_size, a.token_budget, rng):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**collate(examples, b, tok.pad_token_id, model.device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item(); seen += 1; step += 1
            if step % 50 == 0:
                print(f"epoch {epoch+1} step {step}/{total} loss {running/seen:.4f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
        print(f"== epoch {epoch+1} mean loss {running/seen:.4f}", flush=True)
    minutes = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1024**3
    model.config.use_cache = True
    return model, minutes, peak, trainable, len(examples)


def _usable(r):
    try:
        return len(r["messages"]) == 3 and bool(r["messages"][2]["content"])
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--mode", default="train", choices=["train", "base", "rules", "reload"])
    p.add_argument("--data", default="clean", choices=["clean", "raw", "aug"])
    p.add_argument("--bits", type=int, default=4, choices=[4, 16]); p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--epochs", type=int, default=3); p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8); p.add_argument("--token_budget", type=int, default=2400)
    p.add_argument("--test_only", action="store_true", help="reload mode: write test predictions for one shard, then exit")
    p.add_argument("--test_shard", default="0/1", help="i/n: predict every n-th test row starting at i")
    p.add_argument("--max_len", type=int, default=2300); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predict_test", action="store_true"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--eval_batch", type=int, default=1, help="above 1: batched dev scoring, for comparing experiments only")
    a = p.parse_args()
    set_seed(a.seed)
    out = os.path.join("runs", a.name); os.makedirs(out, exist_ok=True)
    tok, model, n = load_base(a.model, a.bits)
    info = {"name": a.name, "config": vars(a), "model": a.model, "base_params": n}
    suffix = ""
    if a.mode == "train":
        model, minutes, peak, trainable, n_rows = train(a, tok, model, n)
        model.save_pretrained(os.path.join(out, "adapter"))
        info.update(train_minutes=round(minutes, 2), peak_gpu_gb=round(peak, 2), trainable=trainable, train_rows=n_rows)
        # Evaluate the adapter as saved. Training changes some layer precisions in memory, so scoring
        # the in-memory model would not match what someone gets by loading the adapter.
        del model
        gc.collect(); torch.cuda.empty_cache()
        tok, model, _ = load_base(a.model, a.bits)
        model = PeftModel.from_pretrained(model, os.path.join(out, "adapter"))
    elif a.mode == "reload":
        model = PeftModel.from_pretrained(model, os.path.join(out, "adapter"))
        if a.test_only:
            i, n = map(int, a.test_shard.split("/"))
            shard = predict(model, tok, load("data/test_inputs.jsonl")[i::n])
            with open(os.path.join(out, f"predictions_test.part{i}.jsonl"), "w", encoding="utf-8") as f:
                for x in shard:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print("DONE shard", a.test_shard, flush=True)
            return
    elif a.mode == "rules":
        suffix = rules_text()
    dev = load("data/dev.jsonl")
    if a.limit:
        dev = dev[:a.limit]
    t0 = time.time()
    preds = predict(model, tok, dev, suffix) if a.eval_batch <= 1 else predict_batched(model, tok, dev, suffix, a.eval_batch, a.token_budget * 2)
    info["eval_minutes"] = round((time.time() - t0) / 60, 2)
    tag = "_reload" if a.mode == "reload" else ""     # a reload must not overwrite the training results
    with open(os.path.join(out, f"dev_predictions{tag}.jsonl"), "w", encoding="utf-8") as f:
        for x in preds:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    if a.mode == "reload":
        # Is the answer sensitive to the repetition_penalty stored in the model's generation_config?
        rp1 = predict(model, tok, dev, suffix, repetition_penalty=1.0)
        info["outputs_changed_with_repetition_penalty_1"] = sum(x["output"] != y["output"] for x, y in zip(preds, rp1))
        info["generation_config_repetition_penalty"] = getattr(model.generation_config, "repetition_penalty", None)
        if not a.limit:
            info["exact_match_repetition_penalty_1"] = dev_metrics(rp1)["all"]["exact_match"]
    if a.mode != "reload" or not a.limit:
        res = dev_metrics(preds)["all"]
        info.update(exact_match=res["exact_match"], mean_field_acc=res["mean_field_acc"], json_valid=res["json_valid"], fields=res["fields"])
        print(json.dumps({k: info[k] for k in ("exact_match", "mean_field_acc", "json_valid", "fields")}, indent=1), flush=True)
    if a.predict_test:
        test_preds = predict(model, tok, load("data/test_inputs.jsonl"), suffix)
        with open(os.path.join(out, "predictions_test.jsonl"), "w", encoding="utf-8") as f:
            for x in test_preds:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(out, f"metrics{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    print("DONE", a.name, flush=True)


if __name__ == "__main__":
    main()
