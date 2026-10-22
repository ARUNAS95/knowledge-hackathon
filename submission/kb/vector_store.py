"""Vector search on embedded LanceDB. Results are filtered to one crawl before ranking."""
import atexit
import shutil
import tempfile

import numpy as np

from .embeddings import CACHE_DIR


class VectorStore:
    def __init__(self):
        import lancedb
        # Own folder per process, so parallel runs don't delete each other's table.
        (CACHE_DIR / "lancedb").mkdir(parents=True, exist_ok=True)
        path = tempfile.mkdtemp(prefix="run-", dir=CACHE_DIR / "lancedb")
        atexit.register(shutil.rmtree, path, True)
        self.db = lancedb.connect(path)
        self.table = None

    def rebuild(self, ids: list[int], vectors: np.ndarray, visible_in: list[list[str]]):
        rows = [{"content_id": i, "vector": v, "visible_in": vis} for i, v, vis in zip(ids, vectors, visible_in)]
        self.table = self.db.create_table("chunks", data=rows, mode="overwrite")

    def search(self, query_vec: np.ndarray, snapshot_id: str, k: int) -> list[tuple[int, float]]:
        """Returns (content_id, cosine similarity), best first."""
        if self.table is None:
            return []
        res = (self.table.search(query_vec).metric("cosine")
               .where(f"array_has(visible_in, '{snapshot_id}')", prefilter=True)
               .limit(k).to_list())
        return [(r["content_id"], 1.0 - r["_distance"]) for r in res]
