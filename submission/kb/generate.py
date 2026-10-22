"""Write the member-facing answer from the resolver's evidence with one LLM call.

The writer returns {found, answer, used_ids}; found=false becomes "I don't know".
"""
import re

from config import EVIDENCE_MAX_CHARS, WRITER_MAX_TOKENS, WRITER_PROMPT

from .llm import LLMError
from .resolver import YEAR_QUESTION, unnamed_savings
from .urls import clean_url

# Lines that never answer a question: testimonials, carousel controls, captions, bare link lists.
_NOISE_LINE = re.compile(r"^(>.*|Dupaco member|Previous|Next|- \d+|\[Image: .*\]|- \[[^\]]+\]\([^)]*#\))\s*$")
# Markdown links become "text (url)" so the writer can copy a clean address.
_MD_LINK = re.compile(r"(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)")

SYSTEM = """You are the member assistant for Dupaco Community Credit Union. Answer the member's question using ONLY the numbered evidence taken from the Dupaco website.

Rules:
- Use only facts stated in the evidence. Never add outside knowledge, typical industry values or guesses.
- Evidence [1] is the most authoritative and current source. If items disagree, follow the lower-numbered item unless a higher-numbered item is an official rate table with a later effective date.
- Give the current value for exactly the tier the member asked about (term, balance, loan amount, model year, jumbo or regular, purchase or refinance). Do not mention other tiers' values unless the member asked for all of them or named no tier.
- If the member named no tier, or several rows still match what they asked (e.g. a 30-year mortgage at four loan amounts), give each matching row with its value, starting with the smallest amount or lowest tier.
- For "what do you offer" or "list" questions, or a question about a whole category ("the credit cards", "certificate rates"), cover every product of that category found in any item, both personal and business, with its value.
- Copy numbers exactly as written, with their %, $, APR or APY label. Drop footnote markers (write "4.10% APY", not "4.10%2").
- If details such as hours differ by branch, give the typical schedule and say that hours vary by branch.
- If the member asks whether something is the same everywhere (all branches, all accounts), answer yes or no, then give each different version with its times or values (for example each distinct Saturday schedule).
- Write 1 to 4 short, plain sentences. Never mention "evidence", "facts", "context", "documents", "sources" or "snapshots". Never ask the member a question.
- If the member asks for a specific value (a rate, fee, APR, hours, address or number) and no evidence item states that value for exactly the product or place asked about, return found=false. Saying the product exists is not an answer when a value was asked for, and an adjustment cap or limit ("adjusts up to 2% per year", "never more than 6% above the original rate") is not the rate itself.
- A general question (not about one named branch), such as "what are your lobby hours", is answered by the schedule given for any Dupaco branch; say that hours can vary by branch.
- If the member asks for a link, website or URL, give the full address: either a link written in the evidence text (such as https://join.dupaco.com/...) or the page URL in an item's header (the address in parentheses), whichever fits the request best.
- Information from a news article or older post still answers the question when nothing newer contradicts it; use it rather than returning found=false.
- If the evidence does not answer the question, return found=false and an empty answer.

Return only JSON: {"found": true or false, "answer": "...", "used_ids": [numbers of the items you used]}"""

# v2: rewrites the v1 rules that made the writer drop facts it had.
SYSTEM_V2 = (SYSTEM
    .replace("- Evidence [1] is the most authoritative and current source. If items disagree,",
             "- Items are ordered by trust: [1] is the most authoritative and current. Use every item that adds a fact the member asked about (a later item often names the places, counts or dates). If items disagree,")
    .replace("- If details such as hours differ by branch, give the typical schedule and say that hours vary by branch.\n",
             "- If hours differ by branch, give each different schedule found, with its times, and say that hours vary by branch.\n")
    .replace("- Write 1 to 4 short, plain sentences.",
             "- Be brief (at most 5 plain sentences), but keep every specific value, count or ordinal (\"the fifth time\"), name, place, date and link from the evidence that answers the question; never replace one with a vague word such as \"multiple\" or \"several\".")
    .replace("- A general question (not about one named branch), such as \"what are your lobby hours\", is answered by the schedule given for any Dupaco branch; say that hours can vary by branch.\n",
             "- A general hours question (not about one named branch), such as \"what are your lobby hours\", is answered from the schedules given for Dupaco branches.\n"))
assert SYSTEM_V2.count("\n- ") == SYSTEM.count("\n- "), "a v2 replacement did not apply"

# v3: v2 plus rules against listing only some rows, and to connect a defined term to its places.
SYSTEM_V3 = (SYSTEM_V2
    .replace("give each matching row with its value, starting with the smallest amount or lowest tier.",
             "give each matching row with its value, starting with the smallest amount or lowest tier. Do not pick a subset or one representative row.")
    .replace("- If hours differ by branch, give each different schedule found, with its times, and say that hours vary by branch.\n",
             "- If hours differ by branch, give every distinct schedule found, with its times, and say that hours vary by branch. Do not pick a subset, representative examples, or merge schedules together.\n")
    .replace("- Copy numbers exactly as written,",
             "- If one item uses a defined term, zone or qualifying rule (such as \"charter area\" or \"service area\") and another item names the places or entities it covers (such as the counties or states Dupaco serves), connect them and name those places instead of only repeating the label.\n- Copy numbers exactly as written,")
    .replace("- If the evidence does not answer the question, return found=false and an empty answer.",
             "- If the member asks what will happen (future rates, whether something will change, a prediction), return found=false: the website does not forecast, and market commentary in an article is not a forecast.\n"
             "- If the member states a rate or value that the evidence does not show, do not repeat it; give the value the evidence shows.\n"
             "- If the evidence does not answer the question, return found=false and an empty answer."))
assert SYSTEM_V3.count("\n- ") == SYSTEM_V2.count("\n- ") + 3, "a v3 replacement did not apply"


def format_evidence(items) -> str:
    blocks = []
    for n, e in enumerate(items, 1):
        o = e.occ
        when = []
        d = o.meta["dates"]
        if d.get("effective"):
            when.append(f"rates effective {d['effective']['date']}")
        if d.get("post"):
            when.append(f"posted {d['post']['date']}")
        head = f"[{n}] {o.title} ({o.page_type} page, {o.url})" + (f", {', '.join(when)}" if when else "")
        path = " > ".join(p for p in o.heading_path + ([o.label] if o.label else []) if p)
        blocks.append(f"{head}\n{('Section: ' + path + chr(10)) if path else ''}{trim(e.text, 2 if o.chunk_type in ('table', 'digest', 'merged', 'hours') else 1)}")
    return "\n\n".join(blocks)


def trim(text: str, factor: int = 1) -> str:
    """Drop noise lines and cap the length."""
    limit = EVIDENCE_MAX_CHARS * factor
    lines = [l for l in text.strip().split("\n") if not _NOISE_LINE.match(l.strip())]
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    out = _MD_LINK.sub(_readable_link, out)
    if len(out) > limit:
        cut = out.rfind("\n", 0, limit)
        out = out[: cut if cut > limit // 2 else limit].rstrip() + " …"
    return out


def _readable_link(m: re.Match) -> str:
    text, url = m.group(1).strip(), m.group(2)
    if not url.startswith("http"):              # in-page anchors ("#rates") carry no address
        return text
    url = clean_url(url)
    if not text or clean_url(text) == url or text.rstrip("/") == url.rstrip("/"):
        return url                              # don't print the same address twice
    return f"{text} ({url})"


FORMAT_REMINDER = ('Reply with exactly this JSON and nothing else: {"found": true or false, '
                   '"answer": "plain sentences for the member", "used_ids": [item numbers]}. '
                   'The answer must be a single text string, not a table or nested object.')

def notes_for(question: str, items, taken_on: str = "") -> list[str]:
    """Extra notes for the writer on questions that need a specific reading."""
    notes = []
    if unnamed_savings(question) and any("| Share Savings |" in e.text for e in items):
        notes.append("The member named no specific savings account. Answer for Share Savings, Dupaco's basic "
                     "savings account, first; then name any other accounts that match the question, with their values.")
    if taken_on and YEAR_QUESTION.search(question):
        notes.append(f"The website was downloaded on {taken_on}; work out which year the member means from that date. "
                     "If an item gives a figure for that period, give only that figure. If none does, say the site has "
                     "no figure for that period, then give each year-labelled figure the items show, newest first, "
                     "each with its year.")
    return notes


def user_message(question: str, items, taken_on: str = "") -> str:
    notes = "".join(f"\nNote: {n}" for n in notes_for(question, items, taken_on))
    return f"Evidence:\n\n{format_evidence(items)}\n\nMember question: {question}\n{notes}\n{FORMAT_REMINDER}"


def system_prompt() -> str:
    return {"v2": SYSTEM_V2, "v3": SYSTEM_V3}.get(WRITER_PROMPT, SYSTEM)


def write_answer(llm, question: str, items, taken_on: str = "") -> dict:
    """Returns {"found", "answer", "used_ids"}; retries once with a correction if the reply is malformed."""
    system = system_prompt()
    user = user_message(question, items, taken_on)
    out = None
    for attempt in range(2):
        try:
            out = llm.chat_json(system, user, max_tokens=WRITER_MAX_TOKENS)
        except LLMError:
            if attempt:
                raise
            out = None
        answer = out.get("answer") if isinstance(out, dict) else None
        if isinstance(answer, list):
            answer = " ".join(str(a).strip().rstrip(".") + "." for a in answer if str(a).strip())
        if isinstance(answer, str) and "found" in out:
            break
        if attempt:
            raise LLMError("writer reply not in the expected format")
        user += ("\n\nYour previous reply did not follow the format. Write the answer as plain "
                 "sentences inside the \"answer\" string.")
    ids = [int(i) for i in out.get("used_ids") or [] if str(i).isdigit() and 1 <= int(i) <= len(items)]
    return {"found": bool(out.get("found")) and bool(answer.strip()), "answer": answer.strip(), "used_ids": ids}
