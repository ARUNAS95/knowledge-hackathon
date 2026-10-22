"""Split a cleaned page into retrieval chunks.

Chunk types: table (whole table with label and notes), row (one table row with its header),
branch (one branch/ATM block), footer (dated "About Dupaco" facts on news posts), section (prose).
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from config import CHUNK_MAX_WORDS, CHUNK_TARGET_WORDS

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW = re.compile(r"^[\s|:-]+$")
_ADDRESS = re.compile(r"\b\d+[^,\n]{2,60},\s*([A-Z][A-Za-z .'-]+),\s*(IA|IL|WI),?\s*\d{5}")
_LINK_ONLY = re.compile(r"^(\[[^\]]*\]\([^)]*\)\s*)+$")
_NOTE_WORDS = re.compile(r"effective|rate|apy|apr|dividend|minimum|fee|payment|loan|subject to change|penalty|\$", re.I)
_ROUTINE_DATE = re.compile(
    r"^(Rates are effective as of [\d/]+.*|(January|February|March|April|May|June|July|August|"
    r"September|October|November|December) \d{1,2}, \d{4})$", re.M)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")


@dataclass
class Chunk:
    chunk_type: str
    text: str                      # verbatim from the cleaned page
    heading_path: list[str]
    label: str = ""                # product/table label or branch name
    table_id: Optional[str] = None
    table_header: list[str] = field(default_factory=list)
    row_index: Optional[int] = None
    notes: str = ""                # table notes (effective date, footnotes) kept for row chunks
    chunk_index: int = 0

    def context(self, title: str) -> str:
        # URL words are added because some page titles don't name the product.
        parts = [title] + [h for h in self.heading_path if h and h != title]
        if self.label and self.label not in parts:
            parts.append(self.label)
        return " > ".join(parts)

    def embed_text(self, title: str) -> str:
        body = _ROUTINE_DATE.sub("", self.text)
        if self.chunk_type == "row" and self.notes:
            body += "\n" + _ROUTINE_DATE.sub("", self.notes)
        return f"{self.context(title)}\n{re.sub(chr(10) + '{2,}', chr(10), body).strip()}"


def _plain(s: str) -> str:
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return re.sub(r"[*_`#]+", "", s).strip()


def _words(s: str) -> int:
    return len(s.split())


def _is_table(b: str) -> bool:
    lines = b.split("\n")
    return sum(1 for l in lines if _TABLE_ROW.match(l)) >= max(2, len(lines) // 2)


def _is_label(b: str) -> bool:
    """'Share Savings', 'Special Certificates': short plain lines that name the table below."""
    return "\n" not in b and _words(b) <= 6 and not _HEADING.match(b) and "](" not in b \
        and not b.rstrip().endswith((".", ":", "?", "!"))


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _split_long(text: str) -> list[str]:
    """Split an oversized block by sentences (lists by line) to stay under CHUNK_MAX_WORDS."""
    units = text.split("\n") if text.count("\n") >= 3 else _SENT_SPLIT.split(text)
    out, cur = [], []
    for u in units:
        if cur and _words(" ".join(cur + [u])) > CHUNK_TARGET_WORDS:
            out.append(("\n" if "\n" in text else " ").join(cur))
            cur = []
        cur.append(u)
    if cur:
        out.append(("\n" if "\n" in text else " ").join(cur))
    return out


def enforce_token_limit(chunks: list[Chunk], title: str, count_tokens, limit: int) -> list[Chunk]:
    """Split chunks whose full embedding input exceeds the model's token limit."""
    out = []
    for c in chunks:
        if count_tokens(c.embed_text(title)) <= limit:
            out.append(c)
            continue
        lines = c.text.split("\n")
        head = lines[:2] if c.chunk_type in ("table", "row") and len(lines) > 1 else []
        body = lines[len(head):]
        units = body if len(body) > 1 else _SENT_SPLIT.split(c.text)
        pieces, cur = [], []
        for u in units:
            trial = Chunk(c.chunk_type, "\n".join(head + cur + [u]), c.heading_path, c.label, c.table_id,
                          c.table_header, c.row_index, c.notes)
            if cur and count_tokens(trial.embed_text(title)) > limit:
                pieces.append(cur)
                cur = []
            cur.append(u)
        pieces.append(cur)
        for p in pieces:
            text = "\n".join(head + p)
            while count_tokens(Chunk(c.chunk_type, text, c.heading_path, c.label).embed_text(title)) > limit:
                text = text[: int(len(text) * 0.85)]       # one giant unbreakable line: trim
            out.append(Chunk(c.chunk_type, text, c.heading_path, c.label, c.table_id, c.table_header,
                             c.row_index, c.notes))
    for i, c in enumerate(out):
        c.chunk_index = i
    return out


def _sections(clean: str):
    """Yield (heading_path, [blocks]) in page order."""
    stack: list[tuple[int, str]] = []
    blocks: list[str] = []
    for b in re.split(r"\n\s*\n", clean):
        b = b.strip()
        if not b:
            continue
        h = _HEADING.match(b.split("\n", 1)[0])
        if h and "\n" not in b:
            if blocks:
                yield [t for _, t in stack], blocks
                blocks = []
            level, text = len(h.group(1)), _plain(h.group(2))
            stack = [(l, t) for l, t in stack if l < level]
            if text:
                stack.append((level, text))
            continue
        blocks.append(b)
    if blocks:
        yield [t for _, t in stack], blocks


def chunk_page(clean: str, page_key: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()       # pages repeat whole sections (desktop + mobile tabs)

    def add(c: Chunk):
        # Rows of only buttons carry no information.
        if c.chunk_type == "section" and _words(re.sub(r"\[([^\]]*)\]\([^)]*\)", "", c.text)) < 6 \
                and not re.search(r"\d", c.text):
            return
        key = hashlib.sha1(f"{c.chunk_type}|{c.label}|{c.text}".encode()).hexdigest()
        if key in seen:
            return
        seen.add(key)
        c.chunk_index = len(chunks)
        chunks.append(c)

    table_no = 0
    for path, blocks in _sections(clean):
        joined = "\n\n".join(blocks)
        addr = _ADDRESS.search(joined)
        if addr and _words(joined) <= 80 and not any(_is_table(b) for b in blocks):
            city = addr.group(1).strip()
            name = path[-1] if path else ""
            label = f"{city} branch" if "branch" in joined.lower() and "ATM @" not in name else f"{name} ({city})"
            add(Chunk("branch", joined, path, label=label))
            continue

        text_buf: list[str] = []

        def flush():
            if not text_buf:
                return
            body = "\n\n".join(text_buf)
            text_buf.clear()
            pieces = [body] if _words(body) <= CHUNK_MAX_WORDS else _split_long(body)
            for p in pieces:
                add(Chunk("section", p, path))

        i = 0
        while i < len(blocks):
            b = blocks[i]
            if b.startswith("**ABOUT DUPACO"):
                flush()
                body = b + ("\n\n" + blocks[i + 1] if i + 1 < len(blocks) else "")
                add(Chunk("footer", body, path, label="About Dupaco Community Credit Union"))
                i += 2
                continue
            if _is_table(b):
                # Labels are the short lines right before the table.
                labels = []
                while text_buf and _is_label(text_buf[-1]) and len(labels) < 2:
                    labels.insert(0, text_buf.pop())
                flush()
                notes = []
                j = i + 1
                while j < len(blocks) and len(notes) < 4:
                    nb = blocks[j]
                    if _is_table(nb) or _is_label(nb) or _LINK_ONLY.match(nb) or not _NOTE_WORDS.search(nb):
                        break
                    notes.append(nb)
                    j += 1
                table_no += 1
                tid = f"{page_key}#t{table_no}"
                label = " > ".join(labels)
                lines = [l for l in b.split("\n") if _TABLE_ROW.match(l)]
                header = _cells(lines[0]) if lines else []
                data = [l for l in lines[1:] if not _SEP_ROW.match(l)]
                note_text = "\n\n".join(notes)
                head = "\n".join(lines[:2]) if len(lines) > 1 and _SEP_ROW.match(lines[1]) else lines[0]
                # Long tables are split into parts that each repeat the header.
                parts, cur = [], []
                for row in data:
                    if cur and _words(" ".join(cur + [row])) > CHUNK_MAX_WORDS:
                        parts.append(cur)
                        cur = []
                    cur.append(row)
                parts.append(cur)
                for part in parts:
                    body = head + "\n" + "\n".join(part)
                    if note_text and part is parts[-1]:
                        body += "\n\n" + note_text
                    add(Chunk("table", body, path, label=label, table_id=tid, table_header=header,
                              notes=note_text))
                if len(data) > 1:
                    for r, row in enumerate(data):
                        add(Chunk("row", head + "\n" + row, path, label=label, table_id=tid,
                                  table_header=header, row_index=r, notes=note_text))
                i = j
                continue
            text_buf.append(b)
            if not _is_label(b) and _words("\n\n".join(text_buf)) >= CHUNK_TARGET_WORDS:
                flush()
            i += 1
        flush()
    return chunks
