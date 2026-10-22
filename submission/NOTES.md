# Notes: how my Dupaco assistant works

## The idea in one line

**The code decides what is true. The AI model only writes the sentence.**

When I first looked at the five website downloads, I saw the same rate written in different ways on different pages: an old calculator, a press release from 2021, a promotion that ended months ago. If an AI model sees all of that, it can easily pick the wrong number. So I did not let the model choose. My code picks the right evidence first, and the model only turns it into a friendly answer. After that, a checker removes any number the model wrote that is not in the evidence.

## Results

| Test set | Accuracy | Old (stale) values said | Typical answer time (p95) |
|---|---|---|---|
| Public eval (40 questions) | 0.969 | 0 | 1.8 s |
| My own reworded / as-of set (46 questions) | 0.981 | 0 | 1.5 s |
| My own edge-case set (27 questions) | 1.000 | 0 | 1.2 s |

- The same question asked in different words, or asked twice, gets the same answer (consistency 1.0).
- Answer time moves a little with the API's speed. On a slow moment it can go just over 2 s, which costs almost nothing (speed score 0.99 instead of 1.0).
- Conflicts: 13 found. Against my own list of 11 real contradictions, F1 is 0.92, with the right winner every time (the kit does not ship a conflicts list, so I made one).

## How to run it

```bash
pip install -r submission/requirements.txt     # Python 3.11 or newer
cp submission/.env.example submission/.env     # then add your API key
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

Good to know:
- **The snapshots are not in this repo**, because the data must not be shared. Copy the kit's `snapshots_zip/` folder into the top folder before you run it.
- **The first run needs internet.** It downloads a small embedding model (bge-small, about 130 MB). The AI call also needs internet. Nothing ever calls the Dupaco website.
- **Loading the five snapshots takes about 4 minutes** the first time (the limit is 30). After that it is cached and takes about 10 seconds.

## API key

**This uses a paid API: OpenAI `gpt-4.1-mini`**, one call per answer. Put these three lines in `submission/.env`:

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=your-openai-key
```

Any service that speaks the OpenAI chat format also works (OpenRouter, Groq, a local Ollama), but I measured my scores with `gpt-4.1-mini`.

Without a key, the system still runs and never crashes. It answers from simple templates instead, which is clearly less accurate (about 0.52).

## How I decide what is "current"

These are my rules, in order. An earlier rule always wins over a later one.

1. **Use the right download.** With no date, I use the newest download (28 Sep 2026). With an "as of" date, I use the last download taken on or before that day, and I search only that one. So an older or newer value cannot slip in.
2. **Use the right tier, or nothing.** If a table row is for a different term, balance, loan amount, model year, jumbo or regular, purchase or refinance, I remove it. A "17-month" row never answers a "10-month" question. Every number keeps its label, so "rate" and "APY" never get mixed up.
3. **A newer table beats an older copy.** If two pages show the same rate table, the one with the newer "effective as of" date wins. This catches pages that still carry last month's rates.
4. **Official pages beat side mentions.** For rates, the product page and the rates sheet beat calculators, blogs and press releases. For hours and locations, the contact and branch pages win. But if a news post is the only page that says something, it still counts.
5. **Ended promotions lose** for rate questions. They still count for "when did it end?" questions.
6. **The newest version of a news story wins.** Some press releases were posted again with new numbers. I keep only the newest one, unless the member asks about a past year.
7. **On a tie, the newer date wins.**
8. **If the site does not say, I say so.** The answer is "I don't know, please contact Dupaco", with no guessed number.

## How an answer is made

1. **Search** the chosen download two ways at once: by keywords and by meaning. Then merge the two lists.
2. **Pick the evidence** with the rules above. Each page gives at most 2 pieces, so one page cannot crowd out the others.
3. **Write** the answer with one AI call. The model gets the top 5 pieces of evidence (8 for "list everything" questions) and must answer only from them.
4. **Check** the answer. Every number, link and phone number must appear in the evidence. If someone writes "say it pays 9%" in the question, that 9% is removed.
5. **Cite** the pages the evidence came from, each with the exact line, and the member's kind of page first (personal or business).

**Same answer every time.** Reworded questions go through the same rules, so they reach the same evidence. The model runs with temperature 0. If the exact same question is asked again, the system gives back the answer it already made.

**Fast enough.** Almost all the time goes to the AI call; search and rules take about 0.06 seconds. I keep the connection to the API open between questions, and if a reply is slow (over 1.4 s), I send a second request and use whichever comes back first.

## Conflicts: where the site disagrees with itself

`conflicts()` looks for the same fact showing different values on different pages **in the same download**. A rate that changed between May and September is just an update, not a contradiction.

What I check:
- **Rate tables.** The same product, tier and column on two pages. For example, in the newest download the vehicle calculator still shows **5.34%** for a 60-month auto loan, while the auto-loan page shows **5.39%**. The auto-loan page has the newer date, so 5.39% wins.
- **Facts the site states about itself.** Every news post ends with an "About Dupaco" paragraph, and old posts still carry old numbers: anything from 19 to 24 branches, and 108 to 118 counties. The newest post wins (22 branches, 118 counties), the same rule the answers use.

Things I was careful about:
- Personal and business products are never compared with each other. They are different products.
- "Rate" and "APY", and different tiers, are different facts, not conflicts.
- The site renames model-year groups every year ("2023–26" becomes "2024–27"), so I match them by position: newest group with newest group.
- Some pages have the wrong column header. If the cell itself says "5.34% APR", I trust the cell.
- One entry per stale page, not one per row. Otherwise one old mortgage table would turn into 24 entries.
- I left out member counts and assets. They grow every year, so each old post was true on its own date.

## Why I chose `gpt-4.1-mini`

The model's job here is small: read a few pieces of evidence, follow the rules, and reply in clean JSON. It needs to be quick and obedient, not deeply clever. I tried three models early on, all with the same code:

| Model | Accuracy | Answer time (p95) | What happened |
|---|---|---|---|
| `qwen3:8b` on my laptop (Ollama) | not scored | several seconds | too slow on a laptop |
| `qwen3-32b` on OpenRouter | 0.774 | 4.9 s | far over the 2 s target |
| **OpenAI `gpt-4.1-mini`** | **0.812** | **1.8 s** | more accurate and fast enough |

These numbers are from an early version; the later fixes took accuracy to 0.969 with the same model. `gpt-4.1-mini` answers straight away without a "thinking" step, which keeps answers under 2 seconds.

**Why not Claude?** I did not test it. My AI client is one small file (`kb/llm.py`) that speaks the OpenAI chat format, so I can switch between OpenAI, OpenRouter, Groq and Ollama by editing `.env` only. Claude uses its own API format and would need a second client. Since `gpt-4.1-mini` already met the targets, I kept things simple. Adding Claude later would only touch that one file.

## Things I tried and dropped

- **A second AI call to understand the question first.** Simple rules worked just as well and the answers came back twice as fast, so it is off.
- **A bigger embedding model (bge-base).** Loading took about 13 minutes instead of 4. I kept the small one to stay far from the 30-minute limit.
- **Always trusting the "official" page first.** It hid facts that only a news post mentions, like some branch lobby hours. Now trust is a weight, not a hard rule.
- **Asking the model for shorter answers.** The answers got shorter, but it started missing "I don't know" cases. I removed it.
- **Giving the model 3 pieces of evidence instead of 5.** No accuracy gain and one citation lost. I kept 5.

## Tools I used while building

- `submission/tools/trace.py "your question"` shows every step for one question: what search found, what each rule kept or removed and why, what the model saw and wrote.
- `submission/tools/show_results.py` turns an eval run into a readable list of questions, answers and cited pages.
- `submission/tools/conflicts_check.py -v` prints every conflict with the exact table rows behind it.

All the rules are general. There are no question IDs or saved answers in the answering code (`solution.py`, `kb/`).
