"""Clean train.jsonl -> data/train_clean.jsonl + cleaning_log.csv

Decisions (see report):
  DROP  junk text            : body has no real content ("test", "asdf asdf", empty) but carries a full label
  DROP  dev overlap          : same ticket body as a dev row -> required by 'never train on dev'
  DROP  duplicate            : exact copy of an earlier train row (labels never conflict) -> keep first copy
  FIX   truncated JSON       : keep category, rebuild the rest of the label from the ticket with triage_rules
  FIX   v1 schema            : 'prio' key with high/medium/low -> 'priority' P1/P2/P3 by v2 rules; DD/MM/YYYY -> ISO
  FIX   field disagreements  : value differs from the SCHEMA.md rule-based reference

Category is never changed: rules cannot verify it, so the original category is kept.
"""
import csv, json, re, collections as C
from triage_rules import KEYS, get, body, reference_label

PAIR = re.compile(r'"(\w+)":\s*("[^"]*"|-?\d+|true|false|null)\s*[,}]')

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def norm_body(row):
    return re.sub(r"\s+", " ", body(get(row, "user"))).strip().lower()

def is_junk(row):
    b = norm_body(row)
    return len(b) < 15 or bool(re.fullmatch(r"(test|asdf|hello|hi|\W|\d)+( (test|asdf|\d+))*", b))

def main():
    train, dev = load("data/train.jsonl"), load("data/dev.jsonl")
    dev_bodies = {norm_body(r) for r in dev}
    seen_text = {}
    log, clean = [], []

    for r in train:
        rid, ticket, raw = r["id"], get(r, "user"), get(r, "assistant")

        # ---------- drops ----------
        if is_junk(r):
            log.append((rid, "drop", "junk ticket text with no content but a full label")); continue
        if norm_body(r) in dev_bodies:
            log.append((rid, "drop", "ticket also appears in dev (leakage)")); continue
        if ticket in seen_text:
            log.append((rid, "drop", f"exact duplicate of {seen_text[ticket]}")); continue
        seen_text[ticket] = rid

        # ---------- parse label ----------
        reasons = []
        try:
            old = json.loads(raw)
        except json.JSONDecodeError:
            old = {k: json.loads(v) for k, v in PAIR.findall(raw)}
            missing = [k for k in KEYS if k not in old]
            reasons.append(f"truncated label JSON (missing {','.join(missing)}): recomputed from ticket")
        if "prio" in old:
            reasons.append(f"v1 schema: key 'prio'='{old['prio']}' -> 'priority' by v2 rules")
            old["priority"] = old.pop("prio")

        new = reference_label(ticket, old["category"])

        # ---------- field-level fixes ----------
        for k in KEYS:
            if k in old and old[k] != new[k]:
                if k == "priority" and reasons and reasons[-1].startswith("v1"):
                    continue                                    # already explained
                if k == "txn_date" and isinstance(old[k], str) and re.fullmatch(r"\d\d/\d\d/\d{4}", old[k]):
                    reasons.append(f"txn_date in DD/MM/YYYY format {old[k]} -> {new[k]}")
                elif k == "txn_id" and old[k] and new[k] and re.sub(r"\W", "", old[k]).upper() == new[k]:
                    reasons.append(f"txn_id not normalized '{old[k]}' -> {new[k]}")
                elif k == "amount_paise" and old[k] and new[k] and old[k] == new[k] // 100:   # rupees (decimals cut off)
                    reasons.append(f"amount in rupees not paise {old[k]} -> {new[k]}")
                else:
                    reasons.append(f"{k} {json.dumps(old[k])} -> {json.dumps(new[k])} (violates SCHEMA.md rule)")

        if reasons:
            log.append((rid, "fix", "; ".join(reasons)))
        r2 = {"id": rid, "messages": [r["messages"][0], r["messages"][1],
                                      {"role": "assistant", "content": json.dumps(new, ensure_ascii=False)}]}
        clean.append(r2)

    with open("data/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("cleaning_log.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "action", "reason"]); w.writerows(log)

    acts = C.Counter(a for _, a, _ in log)
    print(f"train {len(train)} -> clean {len(clean)} | {dict(acts)}")
    kinds = C.Counter()
    for _, a, why in log:
        for part in why.split("; "):
            kinds[(a, re.sub(r"[\"'].*|\d.*| -> .*|\(.*", "", part).strip())] += 1
    for k, v in kinds.most_common():
        print(f"{v:5}  {k}")

if __name__ == "__main__":
    main()
