"""Keyword search (BM25)."""
import math
import re
from collections import Counter, defaultdict

from config import BM25_B, BM25_K1, SEARCH_TOP_K

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?|\$")
_STOP = set("""a an and are as at be by can do does for from has have how i in is it its me my of on
or our please the this to us was what when where which who will with you your dupaco s tell
would like know want need give provide get let could should may might im i'd i'm about any""".split())


# Member wording -> site wording, applied to queries and documents alike.
SYNONYMS = {
    "mortgage": ["home", "loan"], "mortgages": ["home", "loan"],
    "cd": ["certificate"], "cds": ["certificate"],
    "heloc": ["home", "equity"],
    "atm": ["atm", "branch"],
    "thru": ["up"],                  # "drive-thru" -> the site's "drive-up"
}


def _stem(t: str) -> str:
    """Fold plurals so 'rates' matches 'Rate' and 'loans' matches 'Loan'."""
    if len(t) > 3 and t.isalpha():
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text.lower().replace("’", "'")):
        if t in _STOP:
            continue
        toks += SYNONYMS.get(t, [])
        t = _stem(t)
        toks.append(t)
        if "%" in t:                  # "4.10%" also matches "4.10"
            toks.append(t.rstrip("%"))
        if "," in t and t.replace(",", "").replace(".", "").isdigit():
            toks.append(t.replace(",", ""))
        m = re.fullmatch(r"(\d+)(month|year|yr|mo)s?", t)   # "17month" -> "17", "month"
        if m:
            toks += [m.group(1), m.group(2)]
    return toks


class BM25Index:
    def __init__(self):
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.avg_len = 1.0

    def build(self, records):
        self.postings.clear()
        self.doc_len = []
        for rec in records:
            tf = Counter(tokenize(rec.embed_text))
            self.doc_len.append(sum(tf.values()))
            for tok, n in tf.items():
                self.postings[tok].append((rec.content_id, n))
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))

    def search(self, query: str, allowed=None, k: int = SEARCH_TOP_K) -> list[tuple[int, float]]:
        """allowed(content_id) -> bool restricts results to one crawl."""
        n_docs = len(self.doc_len)
        scores: dict[int, float] = defaultdict(float)
        for tok in set(tokenize(query)):
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for cid, tf in plist:
                dl = self.doc_len[cid]
                scores[cid] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_len))
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if allowed is not None:
            ranked = [(c, s) for c, s in ranked if allowed(c)]
        return ranked[:k]
