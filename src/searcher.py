from tavily import TavilyClient
from groq import Groq
from config import (
    TAVILY_API_KEY,
    GROQ_API_KEY,
    BLOCKED_DOMAINS,
    TRUSTED_DOMAINS,
    MAX_CLAIMS_TO_CHECK,
    MIN_SEARCH_RESULTS,
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)
groq   = Groq(api_key=GROQ_API_KEY)


def extract_claims(document: str) -> list[str]:
    """
    Uses the LLM to pull out specific, searchable factual claims
    from the refined document. Returns a list of plain string claims.
    """

    prompt = f"""Read this document and extract the {MAX_CLAIMS_TO_CHECK} most specific, 
verifiable factual claims in it. These should be concrete facts that can be 
confirmed or denied by searching the web — not vague statements.

DOCUMENT:
{document}

Return ONLY a numbered list of claims. No preamble, no explanation.
Example format:
1. Python was created by Guido van Rossum
2. Python was first released in 1991
3. Python supports object-oriented programming"""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    claims = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Strip the number prefix (e.g. "1. " or "1) ")
            parts = line.split(".", 1) if "." in line else line.split(")", 1)
            if len(parts) == 2:
                claim = parts[1].strip()
                if claim:
                    claims.append(claim)

    return claims[:MAX_CLAIMS_TO_CHECK]


def search_claim(claim: str) -> dict:
    """
    Searches Tavily for a single claim.
    Filters out blocked domains and flags whether enough sources were found.

    Returns:
        {
            "claim": str,
            "sources": [{ "url": str, "content": str }],
            "grounded": bool,   # True if enough trusted sources found
            "summary": str      # Combined content from sources
        }
    """

    try:
        results = tavily.search(
            query=claim,
            search_depth="basic",
            max_results=5,
            exclude_domains=BLOCKED_DOMAINS,
            include_domains=TRUSTED_DOMAINS if TRUSTED_DOMAINS else None,
        )
    except Exception as e:
        print(f"  [SEARCH] Error searching for claim: {e}")
        return {
            "claim": claim,
            "sources": [],
            "grounded": False,
            "summary": "Search failed — could not verify this claim."
        }

    # Filter out any blocked domains that slipped through
    clean_results = [
        r for r in results.get("results", [])
        if not any(blocked in r.get("url", "") for blocked in BLOCKED_DOMAINS)
    ]

    sources = [
        {
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500]  # cap per-source content
        }
        for r in clean_results
    ]

    grounded = len(sources) >= MIN_SEARCH_RESULTS
    summary  = "\n".join(
        f"Source: {s['url']}\n{s['content']}" for s in sources
    ) if sources else "No supporting sources found."

    return {
        "claim": claim,
        "sources": sources,
        "grounded": grounded,
        "summary": summary
    }


def ground_document(document: str) -> dict:
    """
    Full grounding pipeline for a refined document:
    1. Extract key claims
    2. Search each claim
    3. Return structured evidence for the Critic to use

    Returns:
        {
            "claims_checked": int,
            "claims_grounded": int,
            "claims_unverified": list[str],   # claims with no search support
            "evidence": str                    # full text to pass to the Critic
        }
    """

    print(f"  [SEARCH] Extracting claims from refined document...")
    claims = extract_claims(document)

    if not claims:
        print(f"  [SEARCH] No extractable claims found.")
        return {
            "claims_checked": 0,
            "claims_grounded": 0,
            "claims_unverified": [],
            "evidence": "No specific claims could be extracted for verification."
        }

    print(f"  [SEARCH] Found {len(claims)} claim(s) to verify:")
    for c in claims:
        print(f"           → {c}")

    results       = []
    unverified    = []
    grounded_count = 0

    for claim in claims:
        print(f"  [SEARCH] Searching: \"{claim}\"")
        result = search_claim(claim)
        results.append(result)

        if result["grounded"]:
            grounded_count += 1
            print(f"           ✓ Grounded ({len(result['sources'])} source(s))")
        else:
            unverified.append(claim)
            print(f"           ✗ Unverified — no supporting sources found")

    # Build the evidence block the Critic will read
    evidence_parts = []
    for r in results:
        status = "VERIFIED" if r["grounded"] else "UNVERIFIED"
        evidence_parts.append(
            f"CLAIM [{status}]: {r['claim']}\n"
            f"SEARCH EVIDENCE:\n{r['summary']}\n"
        )

    evidence = "\n---\n".join(evidence_parts)

    return {
        "claims_checked": len(claims),
        "claims_grounded": grounded_count,
        "claims_unverified": unverified,
        "evidence": evidence
    }


def search_web(query: str) -> list[dict]:
    """
    Searches Tavily for a general query.
    Returns a list of source dictionaries: [{"url": str, "content": str}]
    """

    try:
        results = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            exclude_domains=BLOCKED_DOMAINS,
            include_domains=TRUSTED_DOMAINS if TRUSTED_DOMAINS else None,
        )
    except Exception as e:
        print(f"  [SEARCH] Error searching web: {e}")
        return []

    # Filter out any blocked domains that slipped through
    clean_results = [
        r for r in results.get("results", [])
        if not any(blocked in r.get("url", "") for blocked in BLOCKED_DOMAINS)
    ]

    sources = [
        {
            "url": r.get("url", ""),
            "content": r.get("content", "")[:1000]  # cap per-source content for synthesis
        }
        for r in clean_results
    ]

    return sources


def synthesize_from_search(user_query: str, search_results: list[dict]) -> str:
    """
    Uses the LLM to synthesize a coherent document from web search results.
    """

    if not search_results:
        return ""

    # Combine all sources into a single text block
    sources_text = "\n\n".join(
        f"Source: {s['url']}\n{s['content']}" for s in search_results
    )

    prompt = f"""Based on the following web search results, create a comprehensive, factual document answering the query: "{user_query}"

Search Results:
{sources_text}

Write a well-structured document that synthesizes the information from these sources. Focus on accurate facts, avoid speculation, and cite sources where possible. If the sources conflict, note the discrepancies.

Document:"""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0
    )

    return response.choices[0].message.content.strip()