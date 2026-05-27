from tavily import TavilyClient
from groq import Groq
from telemetry import record_groq_usage, log_telemetry
from config import (
    TAVILY_API_KEY,
    GROQ_API_KEY,
    BLOCKED_DOMAINS,
    TRUSTED_DOMAINS,
    MAX_CLAIMS_TO_CHECK,
    MIN_SEARCH_RESULTS,
)
from prompts import EXTRACT_CLAIMS_PROMPT, SYNTHESIZE_FROM_SEARCH_PROMPT, EXTRACT_DELTA_CLAIMS_PROMPT, VERIFY_CLAIM_PROMPT

tavily = TavilyClient(api_key=TAVILY_API_KEY)
groq   = Groq(api_key=GROQ_API_KEY)


def extract_claims(document: str) -> list[str]:
    """
    Uses the LLM to pull out specific, searchable factual claims
    from the refined document. Returns a list of plain string claims.
    """

    prompt = EXTRACT_CLAIMS_PROMPT.format(
        max_claims_to_check=MAX_CLAIMS_TO_CHECK,
        document=document
    )

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    # Record token usage if available
    try:
        record_groq_usage(response, "extract_claims")
    except Exception:
        pass

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



def extract_delta_claims(original: str, refined: str) -> dict:
    """
    Uses the LLM to extract ONLY the new factual claims that were added
    in the refinement — ignoring facts already present in the original.

    Returns a dict with:
      - "claims": up to MAX_CLAIMS_TO_CHECK claim strings (to verify)
      - "total": total number of extracted claims (may be > cap)
    """

    prompt = EXTRACT_DELTA_CLAIMS_PROMPT.format(
        document_original=original,
        document_refined=refined
    )

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0
    )

    # Record token usage if available
    try:
        record_groq_usage(response, "extract_delta_claims")
    except Exception:
        pass

    raw = response.choices[0].message.content.strip()

    if "NONE" in raw.upper():
        return {"claims": [], "total": 0}

    all_claims = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            parts = line.split(".", 1) if "." in line else line.split(")", 1)
            if len(parts) == 2:
                claim = parts[1].strip()
                if claim:
                    all_claims.append(claim)
        else:
            # Fallback: treat any non-empty line as a potential claim
            all_claims.append(line)

    total = len(all_claims)
    claims = all_claims[:MAX_CLAIMS_TO_CHECK]
    return {"claims": claims, "total": total}


def ground_delta(original: str, refined: str) -> dict:
    """
    Lazy delta verification: extract only NEW claims from a refinement
    and verify each one against web sources.

    Returns:
        {
            "claims_checked": int,
            "claims_grounded": int,
            "claims_unverified": list[str],
            "passed": bool
        }
    """

    print(f"  [VERIFY] Extracting delta claims (new facts only)...")
    res = extract_delta_claims(original, refined)
    claims = res.get("claims", [])
    total_claims = int(res.get("total", len(claims)))

    if total_claims == 0:
        print(f"  [VERIFY] No new verifiable claims found in delta. Passing.")
        return {
            "claims_checked": 0,
            "claims_grounded": 0,
            "claims_unverified": [],
            "total_claims": 0,
            "passed": True
        }

    print(f"  [VERIFY] Extracted {total_claims} total delta claim(s); verifying up to {len(claims)}:")
    for c in claims:
        print(f"           → {c}")

    unverified = []
    grounded_count = 0

    for claim in claims:
        print(f"  [VERIFY] Searching: \"{claim}\"")
        result = search_claim(claim)
        if result["grounded"]:
            # Sources found — but do they actually CONFIRM the claim?
            print(f"           Found {len(result['sources'])} source(s). Checking factual agreement...")
            try:
                judge_prompt = VERIFY_CLAIM_PROMPT.format(
                    claim=claim,
                    evidence=result["summary"][:2000]
                )
                judge_response = groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=10,
                    temperature=0
                )
                # Record token usage for judgement
                try:
                    record_groq_usage(judge_response, "ground_delta_judge")
                except Exception:
                    pass
                verdict = judge_response.choices[0].message.content.strip().upper()
                if "CONFIRMED" in verdict:
                    grounded_count += 1
                    print(f"           ✓ CONFIRMED by web sources")
                else:
                    unverified.append(claim)
                    print(f"           ✗ DENIED by web sources — claim contradicts evidence")
            except Exception as e:
                print(f"           ⚠ Judge error: {e}. Treating as unverified.")
                unverified.append(claim)
        else:
            unverified.append(claim)
            print(f"           ✗ Unverified — insufficient web evidence")

    # If we parsed more claims than we verified, treat the delta as unverified
    truncated = total_claims > len(claims)
    if truncated:
        unverified.append(f"{total_claims - len(claims)} additional claim(s) were detected but not checked (MAX_CLAIMS_TO_CHECK={MAX_CLAIMS_TO_CHECK}).")

    passed = (len(unverified) == 0) and (not truncated)

    return {
        "claims_checked": len(claims),
        "claims_grounded": grounded_count,
        "claims_unverified": unverified,
        "total_claims": total_claims,
        "passed": passed
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
            # No include_domains here — cast a wide net for discovery.
            # include_domains is an allowlist, not a boost, so using it
            # blocks all non-technical topics from returning results.
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

    # Log number of search results (best-effort)
    try:
        log_telemetry("tavily_search", {"query": query, "results": len(sources)})
    except Exception:
        pass

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

    prompt = SYNTHESIZE_FROM_SEARCH_PROMPT.format(
        user_query=user_query,
        sources_text=sources_text
    )

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0
    )

    # Record token usage if available
    try:
        record_groq_usage(response, "synthesize_from_search")
    except Exception:
        pass

    return response.choices[0].message.content.strip()