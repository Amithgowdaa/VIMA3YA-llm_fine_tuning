"""Rule-based reference labeller written from SCHEMA.md.

Used ONLY for auditing / repairing train.jsonl. Never used at inference time.

Disclosure: these rules were written from SCHEMA.md and train, but they were also
debugged against dev labels, so the dev agreement is NOT a held-out score. Phrasings
found only in test_inputs.jsonl were removed; that changed none of the cleaned labels.
"""
import re, json, datetime as D

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
TXN_FIELDS = ("amount_paise", "txn_date", "txn_id", "channel")
NO_TXN_CATS = ("account_access", "kyc", "other")   # schema: txn fields always null for these


# ---------- helpers ----------
def get(row, role):
    """Return the content of the message with this role."""
    return next(m["content"] for m in row["messages"] if m["role"] == role)

def received(ticket):
    """The 'Received: YYYY-MM-DD' header date."""
    return D.date.fromisoformat(re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", ticket).group(1))

def body(ticket):
    """Customer message only: drop the header and quoted earlier emails (lines starting with '>')."""
    text = ticket.split("\n\n", 1)[1]
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith(">"))

NP_ID = r"\bNP[\s\-#:]*(?:\d[\s\-]*){10}"          # NP + 10 digits, spaces/hyphens allowed
NOISE_NUMBERS = r"#\d+|bank ref(?: no)?\.?\s*\d+|order number\s*\d+|version\s*\d+(?:\.\d+)+|\d+(?:\.\d+){2,}"


# ---------- txn_id ----------
def extract_txn_id(ticket):
    m = re.search(r"\bNP[\s\-#:]*((?:\d[\s\-]*){10})(?!\d)", body(ticket), re.I)
    return "NP" + re.sub(r"\D", "", m.group(1)) if m else None


# ---------- channel (only when explicitly stated) ----------
CHANNELS = [("netbanking", r"net\s*banking|internet banking"),
            ("upi",        r"\bupi\b"),
            ("wallet",     r"wallet"),
            ("card",       r"\bcard\b|visa|mastercard")]

def extract_channel(ticket):
    b = re.sub(r"scratch\s*card|srcatch card", "", body(ticket).lower())
    for name, pat in CHANNELS:
        if re.search(pat, b):
            return name
    return None


# ---------- language ----------
HINDI_WORDS = (r"\b(hai|hain|nahi|kar|ho|gaya|gaye|mera|meri|mere|kya|kyun|kyu|aur|bhi|hua|tha|paise|rupaye|"
               r"baar|kal|parso|raha|rahi|chahiye|wapas|lekin|abhi|ke|ka|ki|ko|se|mein|bhai|karo|dino|pehle|"
               r"yeh|jaunga|dunga|kat|kaise|karun|karna|kitna)\b")

def extract_language(ticket):
    """hi-en if at least 2 common Hindi (Latin script) words appear."""
    return "hi-en" if len(re.findall(HINDI_WORDS, body(ticket).lower())) >= 2 else "en"


# ---------- flags used by priority / needs_human ----------
REPEAT = (r"third time|again and again|baar baar|about this before|pehle bhi|written \d+ times|teesri baar|"
          r"second complaint|times about")
LEGAL = r"\brbi\b|ombudsman|consumer court|lawyer|legal notice|legal action|court"

def says_repeat(ticket): return bool(re.search(REPEAT, body(ticket).lower()))
def says_legal(ticket):  return bool(re.search(LEGAL, body(ticket).lower()))


# ---------- txn_date ----------
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MON_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"

def _most_recent(rec, month, day):
    """Year missing -> most recent such date on or before Received."""
    d = D.date(rec.year, month, day)
    return d if d <= rec else D.date(rec.year - 1, month, day)

def extract_txn_date(ticket):
    rec, b = received(ticket), body(ticket)
    bl = b.lower()
    clean = re.sub(NOISE_NUMBERS, "", re.sub(NP_ID, "", b, flags=re.I), flags=re.I)  # numbers that are not dates

    # relative dates (order matters: 'day before yesterday' contains 'yesterday')
    if re.search(r"day before yesterday|\bparso\b", bl):  return str(rec - D.timedelta(2))
    if re.search(r"yesterday|\bkal\b", bl):    return str(rec - D.timedelta(1))
    if re.search(r"\btoday\b|\baaj\b", bl):               return str(rec)
    m = re.search(r"(\d+)\s*(?:days ago|din pehle)", bl)
    if m: return str(rec - D.timedelta(int(m.group(1))))

    # absolute dates: DD/MM/YYYY, DD-MM-YY
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", clean)
    if m:
        d, mo, y = map(int, m.groups())
        return str(D.date(y + 2000 if y < 100 else y, mo, d))
    # 13th June 2026 / 5 Jul
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + MON_RE + r"(?:,?\s+(20\d\d)\b)?", clean, re.I)
    if m:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()], m.group(3)
        return str(D.date(int(y), mo, d)) if y else str(_most_recent(rec, mo, d))
    # Jul 28 / Sep 12, 2026
    m = re.search(r"\b" + MON_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(20\d\d)\b)?", clean, re.I)
    if m:
        mo, d, y = MONTHS[m.group(1).lower()], int(m.group(2)), m.group(3)
        return str(D.date(int(y), mo, d)) if y else str(_most_recent(rec, mo, d))
    return None


# ---------- amount_paise ----------
NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|k\b)?"
# amounts the ticket is NOT about: "my account balance is Rs 432", "daily limit is 27.7k", "fee of ₹20", "Account mein abhi Rs 500 pade hain"
DISTRACTOR_CLAUSE = (r"(?:account balance|balance|daily limit|limit|fee|charge of|account mein abhi)"
                     r"[^.\d₹]{0,25}(?:₹|rs\.?|inr)?\s*\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|k|rupees|rupaye)?(?:\s*pade hain)?")

def to_paise(num, unit):
    v = float(num.replace(",", ""))
    if unit and unit.startswith("lakh"): v *= 100_000
    elif unit == "k":                    v *= 1_000
    return int(round(v * 100))

def extract_amount(ticket):
    b = re.sub(NP_ID, "", body(ticket), flags=re.I)
    b = re.sub(NOISE_NUMBERS, "", b, flags=re.I)
    b = re.sub(DISTRACTOR_CLAUSE, "", b, flags=re.I)
    for sent in re.split(r"(?<![Rr]s\.)(?<=[.!?])\s+(?=[A-Z@\d₹])|\n", b):
        s = sent.lower()
        m = (re.search(r"(?:₹|rs\.?|inr)\s*" + NUM, s)
             or re.search(NUM + r"\s*(?:rupees|rupaye|rs\b)", s)
             or re.search(r"(?:amount|total|value|rakam|was for|is for)\D{0,15}?" + NUM, s)
             or re.search(NUM + r"\s*(?:is the amount|was the value|in total|kat gaye|ka transaction|ka tha|tha\b)", s))
        if m:
            return to_paise(m.group(1), m.group(2))
    return None


# ---------- derived fields (pure schema rules) ----------
def rule_priority(category, amount_paise, legal):
    if category == "fraud" or (amount_paise or 0) >= 50_000_00 or legal:
        return "P1"
    if (category in ("payment_failed", "refund") and (amount_paise or 0) >= 5_000_00) or category == "account_access":
        return "P2"
    return "P3"

def rule_needs_human(priority, repeat):
    return priority == "P1" or repeat


def reference_label(ticket, category):
    """Full label from the ticket + a known category (category itself is NOT predicted by rules)."""
    has_txn = category not in NO_TXN_CATS
    amt = extract_amount(ticket) if has_txn else None
    pr = rule_priority(category, amt, says_legal(ticket))
    return {"category": category, "priority": pr,
            "amount_paise": amt,
            "txn_date": extract_txn_date(ticket) if has_txn else None,
            "txn_id": extract_txn_id(ticket) if has_txn else None,
            "channel": extract_channel(ticket) if has_txn else None,
            "language": extract_language(ticket),
            "needs_human": rule_needs_human(pr, says_repeat(ticket))}
