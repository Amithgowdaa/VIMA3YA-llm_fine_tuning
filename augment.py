"""Part D "own idea": rule-driven augmentation built ONLY from train_clean.jsonl and SCHEMA.md.

Each new row is a clean training ticket with ONE surface change that SCHEMA.md describes:
  order_no   add an order number (SCHEMA: order numbers are not txn ids)
  fee        add a fee (SCHEMA: ignore other amounts such as fees)
  quoted     add a quoted earlier email with a date (SCHEMA: '>' lines are not the txn date)
  id_form    rewrite the id with spaces / a hyphen / lowercase (SCHEMA: normalize)
  rel_date   rewrite an absolute date as yesterday / N days ago / kal / parso (SCHEMA: resolve)
  amount     replace the amount with a value near the Rs 5,000 / Rs 50,000 thresholds, in
             rupees, k or lakh (SCHEMA: priority thresholds, k, lakh)

The label is recomputed with triage_rules and the row is kept only if the rules read the new
ticket back as intended. No text from dev or test is used.
"""
import datetime as D
import json
import random
import re

import triage_rules as T

AMOUNT_RE = re.compile(r"(?:₹\s?|Rs\.?\s?|INR\s)\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s(?:rupees|rupaye)|\b\d+(?:\.\d+)?k\b|\b\d+(?:\.\d+)?\slakhs?\b")
DATE_RES = [re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
            re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + T.MON_RE + r"(?:,?\s+20\d\d\b)?", re.I),
            re.compile(r"\b" + T.MON_RE + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s+20\d\d\b)?", re.I)]
THRESHOLDS = [(4_999, "Rs 4,999"), (5_000, "Rs 5,000"), (5_001, "₹5,001"), (4_900, "4.9k"), (5_000, "5k"), (5_500, "5.5k"),
              (49_999, "Rs 49,999"), (50_000, "INR 50,000"), (50_001, "₹50,001"), (49_500, "49.5k"), (50_000, "50k"),
              (51_000, "51k"), (150_000, "1.5 lakh"), (100_000, "1 lakh"), (250_000, "2.5 lakh"), (1_300, "1.3k")]


def split_ticket(ticket):
    header, body = ticket.split("\n\n", 1)
    return header, body


def insert_sentence(ticket, sentence):
    """Insert after the first complete sentence of the customer's message."""
    header, body = split_ticket(ticket)
    m = re.search(r"[.?!]\s+(?=[A-Z@₹\d])", body)
    if not m:
        return None
    return header + "\n\n" + body[:m.end()] + sentence + " " + body[m.end():]


def order_no(ticket, label, rng):
    n = rng.randint(10000, 99999)
    return insert_sentence(ticket, f"Merchant ka order number {n} hai." if label["language"] == "hi-en" else f"Merchant order number: {n}.")


def fee(ticket, label, rng):
    v = rng.choice([10, 15, 20, 25, 49, 99])
    return insert_sentence(ticket, f"Convenience fee Rs {v} alag se lagi." if label["language"] == "hi-en" else f"A convenience fee of Rs {v} was also charged.")


def quoted(ticket, label, rng):
    if "Source: email" not in ticket:
        return None
    d = T.received(ticket) - D.timedelta(rng.randint(1, 30))
    return (ticket.rstrip() + f"\n\n> On {d:%d %b %Y}, NimbusPay Support wrote:\n"
            f"> Thank you for contacting NimbusPay. Your request has been registered with ticket #{rng.randint(10000, 99999)}.")


def id_form(ticket, label, rng):
    i = label["txn_id"]
    if not i or i not in ticket:
        return None
    digits = i[2:]
    return ticket.replace(i, rng.choice([f"np {digits[:5]} {digits[5:]}", f"NP-{digits}", f"np{digits}", f"NP {digits}"]))


def rel_date(ticket, label, rng):
    if not label["txn_date"]:
        return None
    days = (T.received(ticket) - D.date.fromisoformat(label["txn_date"])).days
    if not 1 <= days <= 9:
        return None
    header, body = split_ticket(ticket)
    hi = label["language"] == "hi-en"
    word = {1: "kal" if hi else "yesterday", 2: "parso" if hi else "day before yesterday"}.get(days, f"{days} din pehle" if hi else f"{days} days ago")
    for rx in DATE_RES:
        m = rx.search(body)
        if m:
            start = m.start() - 3 if body[:m.start()].endswith(" on") and body[m.start() - 3:m.start()] == " on" else m.start()
            sep = " " if start != m.start() else ""
            return header + "\n\n" + body[:start] + sep + word + body[m.end():]
    return None


def amount(ticket, label, rng):
    if label["amount_paise"] is None:
        return None
    header, body = split_ticket(ticket)
    hits = [m for m in AMOUNT_RE.finditer(body) if not re.search(r"balance|limit|fee|abhi", body[max(0, m.start() - 40):m.end() + 15], re.I)]
    if len(hits) != 1:
        return None
    _, text = rng.choice(THRESHOLDS)
    return header + "\n\n" + body[:hits[0].start()] + text + body[hits[0].end():]


TRANSFORMS = {"order_no": order_no, "fee": fee, "quoted": quoted, "id_form": id_form, "rel_date": rel_date, "amount": amount}


def augment(rows, fraction=0.6, seed=13):
    rng = random.Random(seed)
    out = []
    for r in rows:
        if rng.random() > fraction:
            continue
        label = json.loads(T.get(r, "assistant"))
        ticket = T.get(r, "user")
        names = list(TRANSFORMS)
        rng.shuffle(names)
        for name in names:
            new = TRANSFORMS[name](ticket, label, rng)
            if new is None or new == ticket:
                continue
            ref = T.reference_label(new, label["category"])
            ok = ref == label if name != "amount" else {k: v for k, v in ref.items() if k not in ("amount_paise", "priority", "needs_human")} == {k: v for k, v in label.items() if k not in ("amount_paise", "priority", "needs_human")} and ref["amount_paise"] is not None
            if ok:
                out.append({"id": f"{r['id']}-aug-{name}", "messages": [r["messages"][0], {"role": "user", "content": new},
                                                                       {"role": "assistant", "content": json.dumps(ref, ensure_ascii=False)}]})
                break
    return out


if __name__ == "__main__":
    import collections as C
    rows = [json.loads(l) for l in open("data/train_clean.jsonl", encoding="utf-8")]
    aug = augment(rows)
    print(len(rows), "clean rows ->", len(aug), "augmented rows", dict(C.Counter(r["id"].rsplit("-", 1)[1] for r in aug)))
    for r in aug[:4]:
        print("-" * 60, r["id"]); print(T.body(T.get(r, "user"))[:300]); print("->", T.get(r, "assistant"))
