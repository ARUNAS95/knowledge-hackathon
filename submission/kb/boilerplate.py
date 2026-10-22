"""Remove site clutter from page markdown.

Source wording is never rewritten: whole blocks are dropped and decorative images removed.
"""
import re

# ---- whole blocks to drop -------------------------------------------------------------------
_DROP_EXACT = {
    "✕",            # pop-up close button
    "Submit", "Load More", "Next", "* * *", "###", "\\# # #", "🇺🇸", "Disclosure",
    "#### Filter",   # filter widget
}
_DROP_RE = [
    re.compile(r"^\d+ min read$"),                                   # reading-time badge
    re.compile(r"^Sort By:"),                                        # listing sort widget
    re.compile(r"^\[Continue\]\([^)]*\)\s*Cancel$"),                 # leaving-site modal buttons
    re.compile(r"^\*\*Heads up! This link leads to a different website\.\*\*$"),
    re.compile(r"^We only do this when it's helpful for you\. But we must inform you"),
    re.compile(r"^We’ll only use your information to offer Dupaco products"),   # form footer
    re.compile(r"^!\[Organization logo\]"),                           # calculator vendor logo
    re.compile(r"^\(Required\)\\?\*?$"),
    re.compile(r"^\d{1,3}%$"),                                       # form progress bar
    # dealer / branch locator widget
    re.compile(r"^(Go|With \d+ miles|Loading locations\.\.\.|Can't find what you're looking for\?)$"),
    re.compile(r"^Sorry, we couldn't find any results within|^Try searching again using a different city"),
]
# Form field labels such as "First Name\*"
_FORM_FIELD = re.compile(r"^[^\n]{1,80}\\\*$")
# A block that is only tag links
_TAG_LINKS = re.compile(r"^(\[[^\]]+\]\([^)]*/(?:tag|category)/[^)]*\)\s*)+$")
# Country picker in phone fields
_FLAG_LINE = re.compile(r"^- [\U0001F1E6-\U0001F1FF]{2}")
_US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}

# Card sections that copy the opening of other articles; left in, posts would match unrelated questions.
_CARD_SECTION = re.compile(r"^#{1,3} (Handpicked for You|Recent News|Related Articles|You May Also Like)\s*$", re.I)
_HEADING = re.compile(r"^(#{1,6}) ")

# ---- images ---------------------------------------------------------------------------------
_IMG = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINKED_IMG = re.compile(r"\[!\[([^\]]*)\]\([^)]*\)\]\([^)]*\)")
_DECORATIVE_ALT = re.compile(
    r"^(orange graphic of|graphic of an? |icon|logo|recaptcha|map|photo of|image of)|logo$|icon$", re.I)


def _keep_alt(alt: str) -> bool:
    """Keep alt text that carries information; drop icons, logos and file-name alts."""
    alt = alt.strip()
    if alt.lower().startswith("infographic"):
        return True
    if not alt or " " not in alt or len(alt.split()) < 8:
        return False
    return not _DECORATIVE_ALT.search(alt)


def _strip_images(block: str) -> str:
    def repl(m):
        return f"[Image: {m.group(1).strip()}]" if _keep_alt(m.group(1)) else ""
    block = _LINKED_IMG.sub(repl, block)
    block = _IMG.sub(repl, block)
    # Card buttons keep their text; drop the "\\" line breaks.
    block = re.sub(r"\\\\\n", "\n", block)
    block = re.sub(r"\n{2,}", "\n", block)
    return block.strip()


# ---- tables ---------------------------------------------------------------------------------
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PLACEHOLDER_CELL = re.compile(r"^(Col\d+|\s*)$")      # PDF/XLS converters emit "Col2|Col3|…"


def _clean_table(block: str) -> str:
    """Drop empty rows and 'ColN' placeholder headers; return '' if almost nothing is left."""
    out, filled, total = [], 0, 0
    for line in block.split("\n"):
        if not _TABLE_ROW.match(line):
            out.append(line)
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?|", c) for c in cells):     # separator row
            out.append(line)
            continue
        cells = ["" if _PLACEHOLDER_CELL.match(c) else c for c in cells]
        total += len(cells)
        nonempty = sum(1 for c in cells if c)
        filled += nonempty
        if nonempty:
            out.append("| " + " | ".join(cells) + " |")
    if total and filled / total < 0.15:   # mostly empty spreadsheet grid
        return ""
    rows = [l for l in out if _TABLE_ROW.match(l) and not re.fullmatch(r"[\s|:-]+", l)]
    return "\n".join(out).strip() if rows else ""


def _is_table(block: str) -> bool:
    lines = block.split("\n")
    return sum(1 for l in lines if _TABLE_ROW.match(l)) >= max(2, len(lines) // 2)


# ---- main -----------------------------------------------------------------------------------
def split_header(md: str) -> tuple[str, str]:
    """Split off the byline before the first '# ' title, so the post date and author can be read."""
    m = re.search(r"^# ", md, re.M)
    if not m or m.start() == 0:
        return "", md
    return md[: m.start()], md[m.start():]


def clean_markdown(md: str) -> str:
    md = md.replace("\\r\\n", "\n").replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md)]
    out = []
    in_cards = False      # inside a "Handpicked for You" section
    in_form = False       # just saw a form field; short answer options that follow are clutter
    for b in blocks:
        if not b:
            continue
        first = b.split("\n", 1)[0]
        h = _HEADING.match(first)
        if _CARD_SECTION.match(first):
            in_cards = True
            continue
        if in_cards:
            if h and len(h.group(1)) <= 2:     # cards use '####'; a real section ends them
                in_cards = False
            else:
                continue
        if b in _DROP_EXACT or any(r.search(b) for r in _DROP_RE) or _TAG_LINKS.match(b):
            continue
        lines = b.split("\n")
        if sum(1 for l in lines if _FLAG_LINE.match(l)) >= 3:
            continue
        items = [l[2:].strip() for l in lines if l.startswith("- ")]
        if len(items) >= 10 and sum(1 for i in items if i in _US_STATES) >= 0.8 * len(items):
            continue
        if _FORM_FIELD.match(b):
            in_form = True
            continue
        if in_form:
            if h or len(b) > 60 or "](" in b:
                in_form = False
            else:
                continue          # short answer options of a form
        if _is_table(b):
            b = _clean_table(_IMG.sub(lambda m: f"[Image: {m.group(1).strip()}]" if _keep_alt(m.group(1)) else "", b))
        else:
            b = _strip_images(b)
        if b:
            out.append(b)
    return "\n\n".join(out)
