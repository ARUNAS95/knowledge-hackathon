"""URL normalisation, so the same page under different URLs is recognised as one page."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that never change page content.
_DROP_PARAMS = {"hslang", "scrollto", "_hsenc", "_hsmi", "hsctatracking"}


def normalize_url(url: str) -> str:
    """https://www.dupaco.com/personal/loans/loan-protection?hsLang=en#x/ -> .../loan-protection"""
    p = urlsplit(url.strip())
    host = p.netloc.lower()
    path = p.path.rstrip("/") or ""
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in _DROP_PARAMS and not k.lower().startswith("utm_")]
    return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))


def is_junk_url(url: str) -> bool:
    """Pop-up (modal) captures are copies of a real page and add nothing."""
    q = urlsplit(url).query.lower()
    return "containertype=modal" in q


def clean_url(url: str) -> str:
    """Remove tracking parameters from a link shown to the member."""
    p = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in _DROP_PARAMS and not k.lower().startswith("utm_")]
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(query), p.fragment))
