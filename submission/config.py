"""Tunable constants in one place."""

# Bump when cleaning or chunking changes, so old cached embeddings are not reused.
NORMALIZATION_VERSION = 1

# Chunk size in words (about 1.3 tokens per word).
CHUNK_TARGET_WORDS = 270
CHUNK_MAX_WORDS = 380

# Small model so the first ingest stays well inside the time limit.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 8          # small batches of similar-length text are fastest on CPU
MAX_EMBED_TOKENS = 512

# Keyword search
BM25_K1 = 1.4
BM25_B = 0.6
SEARCH_TOP_K = 40

# Merges keyword and vector rankings.
RRF_K = 60
# Keep a hit if its meaning match OR its keyword match reaches these. Loose on purpose:
# the writer makes the real "I don't know" call; this only drops noise.
MIN_COSINE = 0.60
MIN_BM25 = 6.0
# Extra relevance for rate-table chunks with a % or $ value, so prose can't bury the rate sheet.
RATE_TABLE_BOOST = 0.3
# Extra relevance for chunks holding a link that matches a link question.
LINK_BOOST = 0.3
# If this many of the top 10 hits come from one page, give the writer that whole page.
DIGEST_MIN_HITS = 6

# ---- answer path -------------------------------------------------------------------------
import os as _os

# Optional LLM intent parsing; off by default because it adds a round trip to every answer.
USE_INTENT_LLM = _os.getenv("USE_INTENT_LLM", "0") == "1"

# Trim each evidence item to this many characters.
EVIDENCE_MAX_CHARS = 1400
# Evidence items given to the writer (list questions get at least 8).
EVIDENCE_ITEMS = int(_os.getenv("EVIDENCE_ITEMS", "5"))
# Writer system prompt version (v1, v2 or v3 in kb/generate.py).
WRITER_PROMPT = _os.getenv("WRITER_PROMPT", "v3")
WRITER_MAX_TOKENS = 450      # room for long list answers

# Development only: reuse answers across eval runs when code, LLM and question are unchanged.
# Never set for the graded run.
DEV_ANSWER_CACHE = _os.getenv("DEV_ANSWER_CACHE", "0") == "1"
