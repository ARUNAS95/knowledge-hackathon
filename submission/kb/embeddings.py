"""Local bge-small embeddings (fastembed) with an on-disk cache, so only changed text is re-embedded."""
import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from config import EMBED_BATCH, EMBED_MODEL, NORMALIZATION_VERSION

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
# bge was trained with this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding
        self.model_name = model_name
        self.model = TextEmbedding(model_name)
        self.tokenizer = self.model.model.tokenizer
        CACHE_DIR.mkdir(exist_ok=True)
        self.db = sqlite3.connect(CACHE_DIR / "embeddings.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}|v{NORMALIZATION_VERSION}|{text}".encode()).hexdigest()

    def count_tokens(self, text: str) -> int:
        """Token count including special tokens, as the model sees it."""
        self.tokenizer.no_truncation()
        n = len(self.tokenizer.encode(text).ids)
        self.tokenizer.enable_truncation(512)
        return n

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found = {}
        for i in range(0, len(keys), 900):
            batch = keys[i:i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in self.db.execute(q, batch)})
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in found}.items(), key=lambda kt: len(kt[1]))
        if todo:
            print(f"  embedding {len(todo)} new chunks ({len(found)} cached)", flush=True)
        for i in range(0, len(todo), 512):          # commit as we go: an interrupted run keeps its work
            part = todo[i:i + 512]
            rows = []
            for (k, _), v in zip(part, self.model.embed([t for _, t in part], batch_size=EMBED_BATCH)):
                v = np.asarray(v, dtype=np.float32)
                v /= np.linalg.norm(v) or 1.0
                found[k] = v
                rows.append((k, v.tobytes()))
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.dim), np.float32)

    @property
    def dim(self) -> int:
        return 384 if "small" in self.model_name else 768

    def embed_query(self, text: str) -> np.ndarray:
        v = np.asarray(next(iter(self.model.embed([QUERY_PREFIX + text]))), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)
