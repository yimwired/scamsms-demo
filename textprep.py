"""Text normalisation + Thai tokenisation shared by train.py and app.py.

Kept in its own module so the pickled pipeline can find `tokenize` at load time.
"""
import re

from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as th_normalize

# Order matters: money before phone before plain numbers.
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|co|cc|xyz|top|info|me|link|vip|th|io|ly|shop|site|club"
    r"|to|fun|biz|bond|ce|app|pro|my|ee|cn)\b(?:/\S*)?)",
    re.I,
)
LINE_RE = re.compile(r"(line\s*(?:id)?\s*[:：]?\s*@?[\w.-]+|@[\w.-]{3,})", re.I)
MONEY_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?\s*(?:บาท|฿|thb)|฿\s*\d[\d,]*)", re.I)
PHONE_RE = re.compile(r"(?:\+?66|0)[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{3,4}")
NUM_RE = re.compile(r"\d+")
PUNCT_ONLY = re.compile(r"[\W_]+")

# Placeholder tokens. Replacing the literal link/number means the model learns
# "this message contains a link" instead of memorising one scam domain.
PLACEHOLDERS = {"xurl", "xlineid", "xmoney", "xphone", "xnum"}


def mask_phone(text: str) -> str:
    """Privacy mask used before saving collected messages to disk."""
    return PHONE_RE.sub(" xphone ", str(text))


def clean(text: str, lower: bool = True) -> str:
    t = th_normalize(str(text))
    t = URL_RE.sub(" xurl ", t)
    t = LINE_RE.sub(" xlineid ", t)
    t = MONEY_RE.sub(" xmoney ", t)
    t = PHONE_RE.sub(" xphone ", t)
    t = NUM_RE.sub(" xnum ", t)
    return t.lower() if lower else t


def tokenize(text: str, lower: bool = True) -> list[str]:
    # lower=False is for display only (show "KTB" as the sender wrote it); the model always uses lower case.
    toks = word_tokenize(clean(text, lower), engine="newmm", keep_whitespace=False)
    return [t for t in toks if t.strip() and not PUNCT_ONLY.fullmatch(t)]
