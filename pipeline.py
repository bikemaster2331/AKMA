import uuid
import json
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi
from groq import Groq

from database import active_collection, candidate_collection, get_embedding
from refiner import refine_document
from critic import score_mutation
from searcher import search_claim, search_web, synthesize_from_search
from config import (
    GROQ_API_KEY,
    REJECTION_THRESHOLD,
    PROMOTION_SCORE_THRESHOLD,
    PROMOTION_SIMILARITY_THRESHOLD,
    PROMOTION_COUNT_THRESHOLD,
    PROMOTION_SESSION_THRESHOLD,
    CANDIDATE_MATCH_THRESHOLD,
    BM25_QUERY_THRESHOLD,
    OVERWRITE_THRESHOLD,
    RELEVANCE_THRESHOLD,
)

groq_client = Groq(api_key=GROQ_API_KEY)


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine_similarity(vec1: list, vec2: list) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def bm25_query_similarity(stored_query: str, incoming_query: str) -> float:
    """Normalized BM25 similarity between two queries. Returns 0-1.
    Better than cosine at catching keyword differences (dates, names, numbers)."""
    stored_tokens  = stored_query.lower().split()
    incoming_tokens = incoming_query.lower().split()

    if not stored_tokens or not incoming_tokens:
        return 0.0

    bm25 = BM25Okapi([stored_tokens])
    score     = bm25.get_scores(incoming_tokens)[0]
    max_score = bm25.get_scores(stored_tokens)[0]

    if max_score == 0:
        return 0.0

    return min(score / max_score, 1.0)


# ── Main Entry Point ───────────────────────────────────────────────────────────

def run_akm(user_query: str, session_id: str) -> tuple:
    print(f"\n{'='*50}")
    print(f"[AKM] Query   : {user_query}")
    print(f"[AKM] Session : {session_id}")
    print(f"{'='*50}")

    # ── Step 1: Retrieve closest active document ───────────────────────────────
    results = active_collection.query(
        query_texts=[user_query],
        n_results=1,
        where={"status": "active"},
        include=["documents", "metadatas", "embeddings", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        print("[AKM] No active documents found. Seed the database first.")
        msg = "No knowledge base entries found. Type 'seed' to add starter data."
        return msg, msg

    doc_original   = results["documents"][0][0]
    doc_id         = results["ids"][0][0]
    original_embed = results["embeddings"][0][0]

    query_embed    = get_embedding(user_query)
    doc_similarity = cosine_similarity(query_embed, original_embed)

    print(f"[AKM] Retrieved     : {doc_id[:8]}...")
    print(f"[AKM] Query-to-doc  : {doc_similarity:.4f} (threshold: {RELEVANCE_THRESHOLD})")

    # ── Step 2: Route based on relevance ──────────────────────────────────────
    if doc_similarity < RELEVANCE_THRESHOLD:
        print(f"[AKM] No relevant document found — routing to Web Search Path")
        full_doc = _web_search_path(user_query, session_id)
    else:
        print(f"[AKM] Relevant document found — routing to Refinement Path")
        full_doc = _refinement_path(user_query, session_id, doc_original, doc_id, original_embed)

    # ── Step 3: Summarize into a short conversational answer ──────────────────
    if not full_doc or len(full_doc) < 200:
        return full_doc, full_doc

    short_answer = summarize_for_query(user_query, full_doc)
    return short_answer, full_doc


# ── Path A: Refinement (known topic) ──────────────────────────────────────────

def _refinement_path(
    user_query: str,
    session_id: str,
    doc_original: str,
    doc_id: str,
    original_embed: list
) -> str:

    # Step A1: Refine
    print("[AKM] Refining document...")
    refined       = refine_document(user_query, doc_original)
    doc_refined   = refined["refined_text"]
    mutation_type = refined["mutation_type"]
    print(f"[AKM] Mutation type : {mutation_type}")

    # Step A2: Critic scores the refinement
    # No web grounding here — the original document IS the ground truth
    # The Critic's job is to check the refinement didn't break anything
    print("[AKM] Scoring with Critic...")
    mutation_score = score_mutation(doc_original, doc_refined)
    print(f"[AKM] Mutation Score  : {mutation_score}")

    # Step A3: Semantic similarity check
    refined_embed    = get_embedding(doc_refined)
    similarity_score = cosine_similarity(original_embed, refined_embed)
    print(f"[AKM] Similarity Score: {similarity_score:.4f}")

    # Step A4: Decision gate
    if mutation_score < REJECTION_THRESHOLD:
        print(f"[AKM] REJECTED — score {mutation_score} below {REJECTION_THRESHOLD}")
        return doc_original

    if similarity_score > 0.99:
        print(f"[AKM] Refinement added nothing new — returning original")
        return doc_original

    print(f"[AKM] Gate passed — storing as candidate...")

    # Step A5: Candidate matching
    matched_id   = None
    matched_meta = None
    matched_doc  = None

    try:
        existing = candidate_collection.query(
            query_texts=[doc_refined],
            n_results=5,
            where={"original_id": doc_id},
            include=["documents", "metadatas", "embeddings"]
        )
        if existing["documents"] and existing["documents"][0]:
            for i, cand_doc in enumerate(existing["documents"][0]):
                cand_embed = existing["embeddings"][0][i]
                sim = cosine_similarity(refined_embed, cand_embed)
                if sim >= CANDIDATE_MATCH_THRESHOLD:
                    matched_id   = existing["ids"][0][i]
                    matched_meta = existing["metadatas"][0][i]
                    matched_doc  = cand_doc
                    print(f"[AKM] Matched candidate {matched_id[:8]}... (sim: {sim:.4f})")
                    break
    except Exception as e:
        print(f"[AKM] Candidate search error: {e}")

    if matched_id:
        return _increment_and_check(
            matched_id, matched_meta, matched_doc,
            session_id, mutation_score, similarity_score, doc_original
        )
    else:
        return _insert_candidate(
            doc_refined, doc_id, mutation_type,
            mutation_score, similarity_score, session_id, doc_original
        )


# ── Path B: Web Search (unknown topic) ────────────────────────────────────────

def _web_search_path(user_query: str, session_id: str) -> str:

    # Step B0: Check for an existing web-search candidate matching this query
    try:
        existing = candidate_collection.query(
            query_texts=[user_query],
            n_results=5,
            where={"source": "web_search"},
            include=["documents", "metadatas", "embeddings"]
        )
        if existing["documents"] and existing["documents"][0]:
            for i, cand_doc in enumerate(existing["documents"][0]):
                cand_meta  = existing["metadatas"][0][i]
                stored_query = cand_meta.get("query", "")

                # BM25 query-to-query check — catches keyword diffs cosine misses
                bm25_sim = bm25_query_similarity(stored_query, user_query)
                print(f"[AKM] Candidate {existing['ids'][0][i][:8]}... BM25 query sim: {bm25_sim:.4f}")

                if bm25_sim >= BM25_QUERY_THRESHOLD:
                    matched_id   = existing["ids"][0][i]
                    matched_meta = cand_meta
                    print(f"[AKM] ✓ Query match confirmed (BM25: {bm25_sim:.4f})")
                    return _increment_web_candidate(matched_id, matched_meta, cand_doc, session_id, user_query)
                else:
                    print(f"[AKM] ✗ Query mismatch — skipping candidate (BM25: {bm25_sim:.4f} < {BM25_QUERY_THRESHOLD})")
    except Exception as e:
        print(f"[AKM] Web candidate check error: {e}")

    # Step B1: Search the web for the query directly
    print("[AKM] Searching web for query...")
    search_results = search_web(user_query)

    if not search_results:
        print("[AKM] No web results found — cannot answer this query")
        return "I couldn't find reliable information on this topic."

    print(f"[AKM] Got {len(search_results)} source(s) from web")

    # Step B2: Synthesize search results into a clean document
    print("[AKM] Synthesizing document from web sources...")
    new_document = synthesize_from_search(user_query, search_results)

    if not new_document:
        print("[AKM] Synthesis failed")
        return "I couldn't find reliable information on this topic."

    # Step B3: Critic judges the synthesized document
    print("[AKM] Critic scoring synthesized document...")
    evidence = "\n".join(
        f"Source: {r['url']}\n{r['content']}" for r in search_results
    )
    score = score_mutation(
        document_original="",
        document_refined=new_document,
        evidence=evidence
    )
    print(f"[AKM] Synthesis Score : {score}")

    if score < REJECTION_THRESHOLD:
        print(f"[AKM] Synthesis rejected — sources insufficient or incoherent")
        return "I couldn't find reliable information on this topic."

    # Step B4: Store as CANDIDATE — needs cross-session confirmation to go active
    candidate_id = str(uuid.uuid4())
    candidate_collection.add(
        ids=[candidate_id],
        documents=[new_document],
        metadatas=[{
            "source"          : "web_search",
            "query"           : user_query,
            "score"           : score,
            "occurrence_count": 1,
            "source_sessions" : json.dumps([session_id]),
            "timestamps"      : json.dumps([datetime.utcnow().isoformat()]),
            "status"          : "candidate",
        }]
    )
    print(f"[AKM] ✓ Web document stored as candidate: {candidate_id[:8]}...")
    print(f"[AKM]   Needs 1 more confirmation from a different session to go active.")

    return new_document


# ── Candidate Helpers ──────────────────────────────────────────────────────────

def _increment_and_check(
    matched_id, matched_meta, matched_doc,
    session_id, mutation_score, similarity_score, doc_original
) -> str:

    source_sessions  = json.loads(matched_meta.get("source_sessions", "[]"))
    timestamps       = json.loads(matched_meta.get("timestamps", "[]"))
    occurrence_count = int(matched_meta.get("occurrence_count", 1)) + 1

    source_sessions.append(session_id)
    timestamps.append(datetime.utcnow().isoformat())
    best_score = max(float(matched_meta.get("mutation_score", 0)), mutation_score)

    candidate_collection.update(
        ids=[matched_id],
        metadatas=[{
            **matched_meta,
            "occurrence_count": occurrence_count,
            "mutation_score"  : best_score,
            "source_sessions" : json.dumps(source_sessions),
            "timestamps"      : json.dumps(timestamps),
        }]
    )
    print(f"[AKM] Candidate count now: {occurrence_count}")

    distinct_sessions = len(set(source_sessions))
    stored_similarity = float(matched_meta.get("similarity_score", similarity_score))

    print(f"[AKM] Promotion check:")
    print(f"       count      {occurrence_count} >= {PROMOTION_COUNT_THRESHOLD} : {occurrence_count >= PROMOTION_COUNT_THRESHOLD}")
    print(f"       score      {best_score} >= {PROMOTION_SCORE_THRESHOLD} : {best_score >= PROMOTION_SCORE_THRESHOLD}")
    print(f"       similarity {stored_similarity:.4f} >= {PROMOTION_SIMILARITY_THRESHOLD} : {stored_similarity >= PROMOTION_SIMILARITY_THRESHOLD}")
    print(f"       sessions   {distinct_sessions} >= {PROMOTION_SESSION_THRESHOLD} : {distinct_sessions >= PROMOTION_SESSION_THRESHOLD}")

    if (occurrence_count  >= PROMOTION_COUNT_THRESHOLD and
        best_score        >= PROMOTION_SCORE_THRESHOLD and
        stored_similarity >= PROMOTION_SIMILARITY_THRESHOLD and
        distinct_sessions >= PROMOTION_SESSION_THRESHOLD):

        print(f"[AKM] ALL CONDITIONS MET — Promoting...")
        original_id = matched_meta.get("original_id")
        return _promote(matched_id, original_id, matched_doc, stored_similarity)

    print(f"[AKM] Not ready to promote yet.")
    return doc_original


def _insert_candidate(
    doc_refined, doc_id, mutation_type,
    mutation_score, similarity_score, session_id, doc_original
) -> str:

    candidate_id = str(uuid.uuid4())
    candidate_collection.add(
        ids=[candidate_id],
        documents=[doc_refined],
        metadatas=[{
            "original_id"     : doc_id,
            "mutation_type"   : mutation_type,
            "mutation_score"  : mutation_score,
            "similarity_score": similarity_score,
            "occurrence_count": 1,
            "source_sessions" : json.dumps([session_id]),
            "timestamps"      : json.dumps([datetime.utcnow().isoformat()]),
            "status"          : "candidate",
        }]
    )
    print(f"[AKM] New candidate: {candidate_id[:8]}...")
    print(f"[AKM] Needs {PROMOTION_COUNT_THRESHOLD - 1} more confirmation(s) from a different session.")
    return doc_original


# ── Promotion ──────────────────────────────────────────────────────────────────

def _promote(candidate_id: str, original_id: str, refined_text: str, similarity_score: float) -> str:

    if similarity_score >= OVERWRITE_THRESHOLD:
        print("[AKM] Cosmetic change — overwriting in place...")
        active_collection.update(
            ids=[original_id],
            documents=[refined_text]
        )
    else:
        print("[AKM] Meaningful change — archiving original, activating refined...")
        original_meta = active_collection.get(ids=[original_id])["metadatas"][0]
        active_collection.update(
            ids=[original_id],
            metadatas=[{**original_meta, "status": "archived"}]
        )
        new_id = str(uuid.uuid4())
        active_collection.add(
            ids=[new_id],
            documents=[refined_text],
            metadatas=[{
                "status"                 : "active",
                "parent_id"              : original_id,
                "promoted_from_candidate": candidate_id,
                "promoted_at"            : datetime.utcnow().isoformat(),
            }]
        )

    candidate_collection.delete(ids=[candidate_id])
    print("[AKM] ✓ Promotion complete.")
    return refined_text


# ── Web Candidate Helper ───────────────────────────────────────────────────────

def _increment_web_candidate(
    matched_id: str, matched_meta: dict, matched_doc: str,
    session_id: str, user_query: str
) -> str:
    source_sessions  = json.loads(matched_meta.get("source_sessions", "[]"))
    timestamps       = json.loads(matched_meta.get("timestamps", "[]"))
    occurrence_count = int(matched_meta.get("occurrence_count", 1)) + 1
    best_score       = float(matched_meta.get("score", 0))

    source_sessions.append(session_id)
    timestamps.append(datetime.utcnow().isoformat())

    candidate_collection.update(
        ids=[matched_id],
        metadatas=[{
            **matched_meta,
            "occurrence_count": occurrence_count,
            "source_sessions" : json.dumps(source_sessions),
            "timestamps"      : json.dumps(timestamps),
        }]
    )
    print(f"[AKM] Candidate count now: {occurrence_count}")

    distinct_sessions = len(set(source_sessions))
    print(f"[AKM] Promotion check:")
    print(f"       count    {occurrence_count} >= {PROMOTION_COUNT_THRESHOLD} : {occurrence_count >= PROMOTION_COUNT_THRESHOLD}")
    print(f"       score    {best_score} >= {PROMOTION_SCORE_THRESHOLD} : {best_score >= PROMOTION_SCORE_THRESHOLD}")
    print(f"       sessions {distinct_sessions} >= {PROMOTION_SESSION_THRESHOLD} : {distinct_sessions >= PROMOTION_SESSION_THRESHOLD}")

    if (occurrence_count  >= PROMOTION_COUNT_THRESHOLD and
        best_score        >= PROMOTION_SCORE_THRESHOLD and
        distinct_sessions >= PROMOTION_SESSION_THRESHOLD):

        print(f"[AKM] ALL CONDITIONS MET — Refining before promotion...")

        # Refine the candidate using the confirming session's query
        refined      = refine_document(user_query, matched_doc)
        doc_to_store = refined["refined_text"] or matched_doc
        refine_score = score_mutation(matched_doc, doc_to_store)
        print(f"[AKM] Refinement score : {refine_score}")

        # Fall back to raw candidate if refinement degrades quality
        if refine_score < REJECTION_THRESHOLD or not doc_to_store.strip():
            print(f"[AKM] Refinement below threshold — promoting raw candidate")
            doc_to_store = matched_doc

        new_id = str(uuid.uuid4())
        active_collection.add(
            ids=[new_id],
            documents=[doc_to_store],
            metadatas=[{
                "status"                 : "active",
                "source"                 : "web_search",
                "query"                  : matched_meta.get("query", ""),
                "promoted_from_candidate": matched_id,
                "promoted_at"            : datetime.utcnow().isoformat(),
                "score"                  : best_score,
            }]
        )
        candidate_collection.delete(ids=[matched_id])
        print(f"[AKM] ✓ Promotion complete.")
        return doc_to_store
    else:
        blockers = []
        if occurrence_count < PROMOTION_COUNT_THRESHOLD:
            blockers.append(f"{PROMOTION_COUNT_THRESHOLD - occurrence_count} more confirmation(s)")
        if distinct_sessions < PROMOTION_SESSION_THRESHOLD:
            blockers.append(f"{PROMOTION_SESSION_THRESHOLD - distinct_sessions} more distinct session(s)")
        print(f"[AKM] Not yet active. Needs {' and '.join(blockers)}.")

    return matched_doc


# ── Summary Layer ──────────────────────────────────────────────────────────────

def summarize_for_query(user_query: str, document: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": (
                f"Answer the following question in 2-3 sentences using only "
                f"the document below. Be direct and conversational. "
                f"Do not mention the document or sources explicitly.\n\n"
                f"QUESTION: {user_query}\n\nDOCUMENT:\n{document}"
            )
        }],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()