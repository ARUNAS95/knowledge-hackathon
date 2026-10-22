"""In-memory chunk store. Each unique chunk is stored once; every place it appeared
(crawl, url, position, metadata) is an occurrence.
"""
import hashlib
from dataclasses import dataclass, field

from config import NORMALIZATION_VERSION


@dataclass
class Occurrence:
    snapshot_id: str
    taken_on: str
    fetched_at: str
    url: str
    url_norm: str
    title: str
    page_type: str
    chunk_index: int
    chunk_type: str
    heading_path: list
    label: str
    table_id: str
    row_index: int
    text: str
    meta: dict
    carried_forward: bool = False


@dataclass
class ChunkRecord:
    content_id: int
    content_hash: str
    chunk_type: str
    embed_text: str
    text: str
    occurrences: list = field(default_factory=list)
    present_in: set = field(default_factory=set)
    # Crawls where the page was missing, so the latest earlier copy is used at lower trust.
    carried_into: set = field(default_factory=set)

    @property
    def visible_in(self) -> set:
        return self.present_in | self.carried_into


class ChunkStore:
    def __init__(self):
        self.records: list[ChunkRecord] = []
        self._by_hash: dict[str, ChunkRecord] = {}
        self.new_since_index = 0
        # (snapshot_id, table_id) -> the whole-table record, to widen a row to its table
        self.table_index: dict[tuple, ChunkRecord] = {}
        # (snapshot_id, url_norm) -> records on that page, for same-page expansion
        self.page_index: dict[tuple, list] = {}

    def add(self, embed_text: str, occ: Occurrence) -> ChunkRecord:
        h = hashlib.sha1(f"v{NORMALIZATION_VERSION}|{occ.chunk_type}|{embed_text}".encode()).hexdigest()
        rec = self._by_hash.get(h)
        if rec is None:
            rec = ChunkRecord(len(self.records), h, occ.chunk_type, embed_text, occ.text)
            self.records.append(rec)
            self._by_hash[h] = rec
            self.new_since_index += 1
        rec.occurrences.append(occ)
        rec.present_in.add(occ.snapshot_id)
        if occ.chunk_type == "table":
            self.table_index.setdefault((occ.snapshot_id, occ.table_id), rec)
        page = self.page_index.setdefault((occ.snapshot_id, occ.url_norm), [])
        if not page or page[-1] is not rec:
            page.append(rec)
        return rec

    def occurrences_in(self, rec: ChunkRecord, snapshot_id: str) -> list[Occurrence]:
        """Occurrences live in this crawl; for a carried-forward chunk, its newest earlier ones."""
        live = [o for o in rec.occurrences if o.snapshot_id == snapshot_id]
        if live or snapshot_id not in rec.carried_into:
            return live
        older = [o for o in rec.occurrences if o.snapshot_id < snapshot_id]
        newest = max(o.snapshot_id for o in older)
        return [o for o in older if o.snapshot_id == newest]
