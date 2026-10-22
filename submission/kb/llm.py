"""OpenAI-compatible chat client.

Set LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (optional LLM_REASONING_EFFORT) in the environment
or a .env file. With no LLM configured, `available` is False and answers use reduced mode.
"""
import concurrent.futures as cf
import http.client
import io
import json
import logging
import os
import re
import ssl
import threading
import time
import urllib.error
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger("kb.llm")
_ROOT = Path(__file__).resolve().parents[1]


def load_env():
    """Read KEY=VALUE lines from .env; real environment variables win."""
    for p in (_ROOT / ".env", _ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Longest total wait for rate-limit retries in one call (an answer must finish in 60 s).
RATE_LIMIT_BUDGET_S = 25.0

# If no reply after this many seconds, send the same request again; the first reply wins.
# "off" disables it; 0 always sends two.
_HEDGE = os.getenv("LLM_HEDGE_AFTER_S", "1.4").strip().lower()
HEDGE_AFTER_S = None if _HEDGE in ("", "off", "none") else float(_HEDGE)


def _retry_after(headers) -> float:
    """Seconds to wait, from 'Retry-After: 7' or 'x-ratelimit-reset-tokens: 1m2.5s'."""
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = headers.get(name) if headers else None
        if not v:
            continue
        try:
            return max(0.5, float(v))
        except ValueError:
            m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)(ms|s)", v.strip())
            if m:
                secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
                return max(0.5, int(m.group(1) or 0) * 60 + secs)
    return 2.0


def _ssl_context():
    """Use certifi's CA bundle when present (some macOS Python builds ship without one)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, timeout: float = 30.0):
        load_env()
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "")
        self.timeout = timeout
        self.ssl = _ssl_context()
        # One kept-alive connection per thread, to skip a new TLS handshake per call.
        self._local = threading.local()
        self._hedge_pool = cf.ThreadPoolExecutor(max_workers=8)
        self.last_usage = None
        self.available = bool(self.base_url and self.model and self.api_key)
        if not self.available:
            print("[dupaco] No LLM configured (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY); "
                  "answering in reduced mode. See submission/NOTES.md.", flush=True)

    def chat_json(self, system: str, user: str, max_tokens: int = 400) -> dict:
        """One JSON-mode call at temperature 0, with retries; raises LLMError on failure."""
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens, "seed": 7,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if "openrouter.ai" in self.base_url:
            # Snapshot text must not be redistributed: use only providers that keep no prompt data.
            body["provider"] = {"data_collection": "deny", "zdr": True}
        last = None
        deadline = time.time() + RATE_LIMIT_BUDGET_S
        network_failures = 0
        while True:
            try:
                t = time.time()
                data = self._post_hedged(json.dumps(body).encode())
                content = data["choices"][0]["message"]["content"] or ""
                self.last_usage = data.get("usage")
                log.debug("llm %.2fs %s", time.time() - t, data.get("usage"))
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429 or e.code >= 500:
                    # Rate limited or server error: wait as told, within the budget.
                    wait = _retry_after(e.headers) if e.code == 429 else 1.0
                    if time.time() + wait > deadline:
                        log.warning("llm %s; retry would exceed the %ss budget", e.code, RATE_LIMIT_BUDGET_S)
                        break
                    log.warning("llm %s; waiting %.1fs before retrying", e.code, wait)
                    time.sleep(wait)
                    continue
                log.warning("llm call failed: %s %s", e.code, e.read()[:200])
                break                           # 400/401/403/404: bad request / key / model
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, KeyError, ValueError, OSError) as e:
                last = e
                network_failures += 1
                log.warning("llm call failed (attempt %d): %s", network_failures, e)
                if network_failures >= 2:
                    break
                time.sleep(0.3)
                continue
            try:
                return _parse_json(content)
            except ValueError as e:
                # At temperature 0 a retry would return the same broken reply.
                _log_bad_reply(content, e)
                raise LLMError(f"unparseable reply: {e}")
        raise LLMError(str(last))


    def _post(self, payload: bytes) -> dict:
        """POST /chat/completions on this thread's connection; HTTP errors raise HTTPError."""
        u = urlsplit(self.base_url)
        for attempt in range(2):
            conn = getattr(self._local, "conn", None)
            reused = conn is not None
            if conn is None:
                cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                kw = {"context": self.ssl} if u.scheme == "https" else {}
                conn = self._local.conn = cls(u.netloc, timeout=self.timeout, **kw)
            try:
                conn.request("POST", f"{u.path}/chat/completions", body=payload,
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
                r = conn.getresponse()
                raw = r.read()
                break
            except (http.client.HTTPException, OSError) as e:
                conn.close()                   # server closed an idle connection: reconnect once
                self._local.conn = None
                if not reused or attempt or isinstance(e, TimeoutError):
                    raise
            except Exception:
                conn.close()                   # broken connection: the next call opens a new one
                self._local.conn = None
                raise
        if r.status >= 400:
            raise urllib.error.HTTPError(self.base_url, r.status, r.reason, r.headers, io.BytesIO(raw))
        return json.loads(raw)

    def _post_hedged(self, payload: bytes) -> dict:
        """Send a second identical request if the first is slow; first success wins."""
        if HEDGE_AFTER_S is None:
            return self._post(payload)
        first = self._hedge_pool.submit(self._post, payload)
        futures = [first]
        if HEDGE_AFTER_S > 0:
            try:
                return first.result(timeout=HEDGE_AFTER_S)
            except cf.TimeoutError:
                pass
        futures.append(self._hedge_pool.submit(self._post, payload))
        error = None
        for f in cf.as_completed(futures):
            try:
                return f.result()
            except Exception as e:             # one copy failed: wait for the other
                error = error or e
        raise error


def _parse_json(content: str) -> dict:
    content = content.strip()
    if "</think>" in content:                   # models that still emit a thinking block
        content = content.split("</think>", 1)[1]
    start, end = content.find("{"), content.rfind("}")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {content[:120]!r}")
    try:
        return json.loads(content[start:end + 1])
    except ValueError:
        salvaged = _salvage(content[start:])
        if salvaged:
            log.info("salvaged truncated/malformed JSON reply")
            return salvaged
        raise


def _log_bad_reply(content: str, err: Exception):
    path = _ROOT / ".cache" / "llm_errors.log"
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {err}\n{content}\n")
    log.warning("unparseable LLM reply (%s); raw reply saved to %s", err, path)


def _salvage(text: str) -> dict:
    """Recover the complete sentences of a reply cut off by the token limit."""
    lst = re.search(r'"answer"\s*:\s*\[(.*?)\]', text, re.S)
    if lst:                                   # answer given as a list of strings
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', lst.group(1))
        ids = re.search(r'"used_ids"\s*:\s*\[([\d,\s]*)', text)
        return {"found": bool(items), "answer": " ".join(i.strip().rstrip(".") + "." for i in items),
                "used_ids": [int(x) for x in re.findall(r"\d+", ids.group(1))] if ids else []}
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.S)
    if not m:
        return {}
    raw = m.group(1).rstrip("\\")
    try:
        answer = json.loads(f'"{raw}"')                  # decodes \n, \" and keeps "–", "’"
    except ValueError:
        answer = raw.replace('\\"', '"').replace("\\n", "\n")
    if not text[m.end():].lstrip().startswith('"'):          # string never closed: truncated
        cut = max(answer.rfind(". "), answer.rfind(".\n"))
        answer = answer[: cut + 1] if cut > 0 else ""
    found = re.search(r'"found"\s*:\s*(true|false)', text)
    ids = re.search(r'"used_ids"\s*:\s*\[([\d,\s]*)', text)
    return {"found": bool(answer) and (found is None or found.group(1) == "true"), "answer": answer.strip(),
            "used_ids": [int(x) for x in re.findall(r"\d+", ids.group(1))] if ids else []}
