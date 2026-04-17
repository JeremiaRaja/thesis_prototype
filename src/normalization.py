"""
normalization.py
────────────────
Indonesian-specific text normalization for tweets.
Handles slang, abbreviations, and informal writing common in PPKM-era Twitter.
"""

import re

# ─── Slang / abbreviation dictionary (Indonesian informal → formal) ────────────
SLANG_DICT = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "tdk": "tidak",
    "sdh": "sudah", "blm": "belum", "msh": "masih", "krn": "karena",
    "gmn": "bagaimana", "gimana": "bagaimana", "gak": "tidak", "ga": "tidak",
    "ngga": "tidak", "nggak": "tidak", "kalo": "kalau", "aja": "saja",
    "nih": "", "sih": "", "deh": "", "dong": "", "lah": "",
    "bgt": "banget", "banget": "sangat", "banyak": "banyak",
    "lg": "lagi", "lagi": "lagi", "jg": "juga", "juga": "juga",
    "tp": "tapi", "tapi": "tapi", "kyk": "seperti", "kayak": "seperti",
    "udah": "sudah", "udh": "sudah", "emg": "memang", "emang": "memang",
    "gue": "saya", "gw": "saya", "lo": "kamu", "lu": "kamu",
    "km": "kamu", "sy": "saya", "mrk": "mereka", "kt": "kita",
    "byk": "banyak", "skrg": "sekarang", "mkn": "mungkin",
    "pd": "pada", "dr": "dari", "dlm": "dalam", "dg": "dengan",
    "jln": "jalan", "kpd": "kepada", "ttg": "tentang",
    "hrs": "harus", "hrs": "harus", "tsb": "tersebut",
    "spy": "supaya", "krna": "karena", "karna": "karena",
    "lbh": "lebih", "jd": "jadi", "trs": "terus", "terus": "terus",
    "sampe": "sampai", "ampe": "sampai", "brg": "barang",
    "org": "orang", "blg": "bilang", "bilang": "bilang",
    "nnton": "nonton", "ntn": "nonton", "bs": "bisa", "bisa": "bisa",
    "iya": "ya", "yep": "ya", "ok": "oke", "oke": "oke",
    "info": "informasi", "pake": "pakai", "pakai": "pakai",
    "wkwk": "", "haha": "", "hehe": "", "xixi": "",
    "ppkm": "ppkm", "covid": "covid", "vaksin": "vaksin",
}


def normalize_slang(text: str) -> str:
    """Replace informal/slang words with their standard equivalents."""
    tokens = text.split()
    return " ".join(SLANG_DICT.get(t.lower(), t) for t in tokens).strip()


def normalize_repeated_chars(text: str) -> str:
    """Collapse 3+ repeated characters to 2, e.g. 'tiadaaaa' → 'tiadaa'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """
    Full normalization pipeline:
    1. Lowercase
    2. Slang replacement
    3. Repeated-char collapsing
    4. Whitespace cleanup
    """
    text = text.lower()
    text = normalize_slang(text)
    text = normalize_repeated_chars(text)
    text = normalize_whitespace(text)
    return text


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Yg gak ikut PPKM bakal kena sanksi bgt!!",
        "Udah vaksin kok msh takut jugaaaa",
        "Gue ga setuju sama kebijakan ini deh 😤",
    ]
    for s in samples:
        print(f"  IN : {s}")
        print(f"  OUT: {normalize(s)}")
        print()
