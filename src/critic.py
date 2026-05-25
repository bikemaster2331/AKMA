from groq import Groq
from config import GROQ_API_KEY
from prompts import SCORE_REFINEMENT_PROMPT, SCORE_SYNTHESIS_PROMPT

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

    prompt = SCORE_REFINEMENT_PROMPT.format(
        document_original=document_original,
        document_refined=document_refined
    )

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

    prompt = SCORE_SYNTHESIS_PROMPT.format(
        evidence=evidence,
        document_refined=document_refined
    )

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
    Only accepts scores in the valid 0.0–1.0 range.
    Falls back gracefully if parsing fails.
    """
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]

    # Walk backwards through lines to find the first parseable float
    for line in reversed(lines):
        # Strip common artifacts like "Score: 0.8" or "**0.8**"
        cleaned = line.replace("*", "").replace("Score:", "").replace("score:", "").strip()
        try:
            score = float(cleaned)
            if 0.0 <= score <= 1.0:
                parsed = round(score, 4)
                print(f"  [CRITIC] Reasoning complete. Score: {parsed}")
                return parsed
            else:
                print(f"  [CRITIC] Ignoring out-of-range score: {score}")
                continue
        except ValueError:
            continue

    # If nothing parsed, conservative fallback
    print(f"  [CRITIC] Could not parse score from response. Defaulting to 0.5.")
    return 0.5