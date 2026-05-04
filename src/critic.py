from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def score_mutation(document_original: str, document_refined: str, evidence: str = "") -> float:
    """
    Adversarially scores a refinement or synthesis.

    Two modes:
    A) Refinement (original provided) — checks fact preservation + no contamination
    B) Synthesis  (original empty)    — checks coherence + source grounding only
    """

    if document_original.strip():
        return _score_refinement(document_original, document_refined)
    else:
        return _score_synthesis(document_refined, evidence)


# ── Mode A: Refinement scoring ─────────────────────────────────────────────────

def _score_refinement(document_original: str, document_refined: str) -> float:
    """
    Compares original vs refined. Primary job: catch contamination and fact loss.
    Allows the model to reason before scoring.
    """

    prompt = f"""You are an adversarial document auditor. A knowledge document has been refined based on user input. Your job is to detect whether the refinement is trustworthy or whether it subtly corrupted the original facts.

ORIGINAL DOCUMENT (ground truth):
{document_original}

REFINED DOCUMENT (under review):
{document_refined}

Work through these checks step by step:

STEP 1 — FACT INVENTORY
List every specific fact in the original (names, dates, numbers, definitions).

STEP 2 — FACT AUDIT
For each fact from Step 1, check if it is:
- Preserved exactly → OK
- Softened or made vague → WARN
- Altered or contradicted → FAIL
- Removed entirely → FAIL

STEP 3 — ADDITION AUDIT
List what new information was added. For each addition, judge:
- Is it consistent with the original? → OK
- Does it introduce a false or unverifiable claim? → FAIL
- Does it subtly reframe a fact without directly contradicting it? → WARN

STEP 4 — FINAL SCORE
Use this rubric:
0.0 - 0.2 : Original facts directly contradicted or removed
0.2 - 0.4 : Facts softened, vague, or subtly altered
0.4 - 0.6 : Mostly preserved but additions are unverifiable or suspicious
0.6 - 0.8 : Facts intact, additions are reasonable but unverified
0.8 - 0.95: Facts fully preserved, additions are consistent and plausible
0.95 - 1.0: Perfect — nothing lost, additions clearly improve the document

Write your reasoning for Steps 1-3, then on the very last line write ONLY the score as a decimal number. Nothing else on that last line."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1   # slight variation so it doesn't lock onto one number
    )

    return _parse_score(response.choices[0].message.content)


# ── Mode B: Synthesis scoring ──────────────────────────────────────────────────

def _score_synthesis(document_refined: str, evidence: str) -> float:
    """
    No original to compare against — judges the synthesis on coherence
    and how faithfully it represents the search evidence provided.
    """

    if not evidence:
        # No sources and no original — nothing to judge against
        return 0.0

    prompt = f"""You are an adversarial fact-checker reviewing a document synthesized from web search results. There is no prior document to compare against. Your job is to check whether the synthesis accurately and honestly represents the search sources.

SEARCH SOURCES (what was found on the web):
{evidence}

SYNTHESIZED DOCUMENT (under review):
{document_refined}

Work through these checks step by step:

STEP 1 — SOURCE COVERAGE
Does the document capture the key facts from the sources, or does it ignore important information?

STEP 2 — FABRICATION CHECK
Does the document introduce any claims NOT present in the sources? List them if any.

STEP 3 — COHERENCE CHECK
Is the document internally consistent? Any contradictions within itself?

STEP 4 — FINAL SCORE
Use this rubric:
0.0 - 0.2 : Contains fabricated claims not in any source
0.2 - 0.4 : Significant omissions or misrepresentations of sources
0.4 - 0.6 : Partially represents sources but with notable gaps or distortions
0.6 - 0.8 : Mostly accurate, minor gaps, no fabrications
0.8 - 0.95: Accurate and well-synthesized, faithfully represents sources
0.95 - 1.0: Excellent synthesis — complete, accurate, well-structured

Write your reasoning for Steps 1-3, then on the very last line write ONLY the score as a decimal number. Nothing else on that last line."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1
    )

    return _parse_score(response.choices[0].message.content)


# ── Score Parser ───────────────────────────────────────────────────────────────

def _parse_score(raw_text: str) -> float:
    """
    Extracts the score from the last line of the model's reasoning output.
    Falls back gracefully if parsing fails.
    """
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]

    # Walk backwards through lines to find the first parseable float
    for line in reversed(lines):
        # Strip common artifacts like "Score: 0.8" or "**0.8**"
        cleaned = line.replace("*", "").replace("Score:", "").replace("score:", "").strip()
        try:
            score = float(cleaned)
            parsed = round(max(0.0, min(1.0, score)), 4)
            print(f"  [CRITIC] Reasoning complete. Score: {parsed}")
            return parsed
        except ValueError:
            continue

    # If nothing parsed, conservative fallback
    print(f"  [CRITIC] Could not parse score from response. Defaulting to 0.5.")
    return 0.5