"""
normalization.py — Indonesian text normalization (same as before)
"""
import re

SLANG_DICT = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "tdk": "tidak",
    "sdh": "sudah", "blm": "belum", "msh": "masih", "krn": "karena",
    "gmn": "bagaimana", "gimana": "bagaimana", "gak": "tidak", "ga": "tidak",
    "ngga": "tidak", "nggak": "tidak", "kalo": "kalau", "aja": "saja",
    "nih": "", "sih": "", "deh": "", "dong": "", "lah": "",
    "bgt": "banget", "banget": "sangat", "lg": "lagi", "jg": "juga",
    "tp": "tapi", "kyk": "seperti", "kayak": "seperti",
    "udah": "sudah", "udh": "sudah", "emg": "memang", "emang": "memang",
    "gue": "saya", "gw": "saya", "lo": "kamu", "lu": "kamu",
    "km": "kamu", "sy": "saya", "mrk": "mereka", "kt": "kita",
    "byk": "banyak", "skrg": "sekarang", "mkn": "mungkin",
    "pd": "pada", "dr": "dari", "dlm": "dalam", "dg": "dengan",
    "hrs": "harus", "tsb": "tersebut", "spy": "supaya",
    "krna": "karena", "karna": "karena", "lbh": "lebih",
    "jd": "jadi", "trs": "terus", "sampe": "sampai",
    "org": "orang", "bs": "bisa", "iya": "ya",
    "wkwk": "", "haha": "", "hehe": "",
}

def normalize_slang(text):
    tokens = text.split()
    return " ".join(SLANG_DICT.get(t.lower(), t) for t in tokens).strip()

def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def normalize(text):
    text = text.lower()
    text = normalize_slang(text)
    text = normalize_repeated_chars(text)
    text = normalize_whitespace(text)
    return text
