from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def score_mutation(document_original: str, document_refined: str, evidence: str = "") -> float:
    """
    Adversarially scores the quality of a refinement.
    If evidence is provided (from web search), the Critic uses it to fact-check.
    If no evidence, falls back to LLM-only judgment.

    Returns a float between 0.0 and 1.0.
    """

    if evidence:
        grounded_section = f"""
You also have the following EXTERNAL SEARCH EVIDENCE for the new claims in the refined document.
Use this to verify whether the new information is factually supported by real sources.
Heavily penalize any claim marked UNVERIFIED — it means no external source confirmed it.

SEARCH EVIDENCE:
{evidence}
"""
    else:
        grounded_section = "\nNo external evidence available. Score based on logic and consistency only.\n"

    prompt = f"""You are an adversarial fact-checker. Your job is to find flaws in a document refinement.

ORIGINAL DOCUMENT:
{document_original}

REFINED DOCUMENT:
{document_refined}
{grounded_section}

Evaluate the refinement on these criteria:
1. Fact Preservation — are all original facts kept intact and not contradicted?
2. Logic Consistency — is the refined document internally coherent?
3. Source Grounding — are the NEW claims supported by the search evidence above?
   - VERIFIED claims with strong sources → reward
   - UNVERIFIED claims with no sources → penalize heavily
   - If no evidence was provided → score on logic alone

Be critical. Actively look for:
- Subtle contradictions with the original
- Removed or altered facts
- New claims that have no external support
- Logical inconsistencies

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation. No other text.
0.0 = completely wrong, contradicts facts, or full of unverified claims
1.0 = perfect refinement, all facts preserved, all new claims verified"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return round(max(0.0, min(1.0, score)), 4)
    except ValueError:
        return 0.0