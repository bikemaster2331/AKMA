import uuid
import json
import numpy as np
from datetime import datetime
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
    WEB_CANDIDATE_MATCH_THRESHOLD,
    DELTA_MATCH_THRESHOLD,
    IDENTICAL_REFINEMENT_THRESHOLD,
    RELEVANCE_THRESHOLD,
)
from prompts import CONFIRM_AND_REFINE_PROMPT, SUMMARIZE_FOR_QUERY_PROMPT

groq_client = Groq(api_key=GROQ_API_KEY)


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine_similarity(vec1: list, vec2: list) -> float:
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))




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

    doc_meta = results["metadatas"][0][0] if results["metadatas"] and results["metadatas"][0] else {}
    doc_topic = doc_meta.get("topic", "unknown")
    print(f"[AKM] Retrieved     : {doc_id[:8]}... (Topic: {doc_topic})")
    print(f"[AKM] Doc Snippet   : {doc_original[:80].strip()}...")
    print(f"[AKM] Query-to-doc  : {doc_similarity:.4f} (threshold: {RELEVANCE_THRESHOLD})")

    # ── Step 2: Route based on relevance ──────────────────────────────────────
    if doc_similarity < RELEVANCE_THRESHOLD:
        print(f"[AKM] No relevant document found — routing to PATH B (Web Search)")
        full_doc = _web_search_path(user_query, session_id)
    else:
        print(f"[AKM] Relevant document found — evaluating for PATH A (Refinement)")
        full_doc = _refinement_path(user_query, session_id, doc_original, doc_id, original_embed)

    # ── Step 3: Summarize into a short conversational answer ──────────────────
    if not full_doc or len(full_doc) < 200:
        return full_doc, full_doc

    short_answer = summarize_for_query(user_query, full_doc)
    return short_answer, full_doc

def _confirm_and_refine(user_query: str, doc_original: str) -> dict:
    """
    Single LLM call that does two jobs:
    1. Confirms whether the query actually belongs to this document
    2. If yes, produces the refinement
    Saves one LLM call vs doing these separately.
    """
    prompt = CONFIRM_AND_REFINE_PROMPT.format(
        doc_original=doc_original,
        user_query=user_query
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )

    text = response.choices[0].message.content.strip()

    routing       = "STATIC_MATCH"
    mutation_type = "expansion"
    refined_text  = ""
    in_refined    = False

    for line in text.split("\n"):
        if line.startswith("ROUTING:"):
            routing = line.split(":", 1)[1].strip().upper()
        elif line.startswith("MUTATION_TYPE:"):
            raw = line.split(":", 1)[1].strip().lower()
            mutation_type = raw if raw in ["correction", "expansion", "none"] else "expansion"
        elif line.startswith("REFINED_DOCUMENT:"):
            in_refined = True
        elif line.startswith("CHANGES_MADE:"):
            in_refined = False
        elif in_refined:
            refined_text += line + "\n"

    return {
        "routing"      : routing,
        "refined_text" : refined_text.strip(),
        "mutation_type": mutation_type
    }


# ── Path A: Refinement (known topic) ──────────────────────────────────────────

def _refinement_path(
    user_query: str,
    session_id: str,
    doc_original: str,
    doc_id: str,
    original_embed: list
) -> str:

    # Step A1: Confirm relevance + Refine (single call)
    print("[AKM] Confirming relevance and refining...")
    refined       = _confirm_and_refine(user_query, doc_original)
    routing       = refined["routing"]
    mutation_type = refined["mutation_type"]
    
    if routing != "STATIC_MATCH":
        print(f"[AKM] Smart Router classified query as {routing} — rerouting to PATH B (Web Search)")
        pass_parent = doc_id if routing in ["VOLATILE", "CONFLICT", "INSUFFICIENT"] else None
        return _web_search_path(user_query, session_id, parent_id=pass_parent, routing=routing)
    
    doc_refined = refined["refined_text"]
    if not doc_refined:
        print("[AKM] No refinement produced — returning original")
        return doc_original

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

    if similarity_score > IDENTICAL_REFINEMENT_THRESHOLD:
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
            orig_arr = np.array(original_embed)
            delta_new = np.array(refined_embed) - orig_arr

            for i, cand_doc in enumerate(existing["documents"][0]):
                cand_embed = existing["embeddings"][0][i]

                # Gate 1: full doc similarity (are the overall docs close?)
                doc_sim = cosine_similarity(refined_embed, cand_embed)

                # Gate 2: delta similarity (did they change in the same direction?)
                delta_cand = np.array(cand_embed) - orig_arr
                delta_sim  = cosine_similarity(delta_new.tolist(), delta_cand.tolist())

                cand_id = existing['ids'][0][i][:8]
                print(f"[AKM] Candidate {cand_id}... doc: {doc_sim:.4f}, delta: {delta_sim:.4f}")

                if doc_sim >= CANDIDATE_MATCH_THRESHOLD and delta_sim >= DELTA_MATCH_THRESHOLD:
                    matched_id   = existing["ids"][0][i]
                    matched_meta = existing["metadatas"][0][i]
                    matched_doc  = cand_doc
                    print(f"[AKM] ✓ Matched — same change direction")
                    break
                elif doc_sim >= CANDIDATE_MATCH_THRESHOLD:
                    print(f"[AKM] ✗ Doc similar but different change (delta: {delta_sim:.4f} < {DELTA_MATCH_THRESHOLD})")
                else:
                    print(f"[AKM] ✗ Different document (doc: {doc_sim:.4f} < {CANDIDATE_MATCH_THRESHOLD})")
    except Exception as e:
        print(f"[AKM] Candidate search error: {e}")

    if matched_id:
        _increment_and_check(
            matched_id, matched_meta, matched_doc,
            session_id, mutation_score, similarity_score, doc_original
        )
    else:
        _insert_candidate(
            doc_refined, doc_id, mutation_type,
            mutation_score, similarity_score, session_id, doc_original,
            user_query=user_query
        )
        
    return doc_refined


# ── Path B: Web Search (unknown topic) ────────────────────────────────────────

def _web_search_path(user_query: str, session_id: str, parent_id: str = None, routing: str = None) -> str:

    is_conflict = (routing == "CONFLICT" and parent_id is not None)

    if is_conflict:
        print("[AKM] ⚠ CONFLICT MODE — disputed fact detected, searching web for ground truth...")

    # Step B1: Search the web for the query directly
    print("[AKM] Searching web for query...")
    search_results = search_web(user_query)

    if not search_results:
        print("[AKM] No web results found — cannot answer this query")
        if is_conflict:
            _quarantine_document(parent_id, user_query, session_id, reason="no_web_evidence")
        return "I couldn't find reliable information on this topic."

    print(f"[AKM] Got {len(search_results)} source(s) from web")

    # Step B2: Synthesize search results into a clean document
    print("[AKM] Synthesizing document from web sources...")
    new_document = synthesize_from_search(user_query, search_results)

    if not new_document:
        print("[AKM] Synthesis failed")
        if is_conflict:
            _quarantine_document(parent_id, user_query, session_id, reason="synthesis_failed")
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
        if is_conflict:
            _quarantine_document(parent_id, user_query, session_id, reason="synthesis_rejected")
        return "I couldn't find reliable information on this topic."

    # ── CONFLICT FAST-PATH: Immediate replacement ─────────────────────────────
    if is_conflict:
        print("[AKM] ⚠ CONFLICT — Web evidence verified. Replacing poisoned document immediately.")
        return _conflict_replace(parent_id, new_document, score, user_query, session_id)

    # Step B3.5: Semantic Candidate Deduplication
    print("[AKM] Checking if this knowledge already exists in Candidates...")
    new_doc_embed = get_embedding(new_document)

    try:
        existing = candidate_collection.query(
            query_embeddings=[new_doc_embed],
            n_results=1,
            where={"source": "web_search"},
            include=["documents", "metadatas", "embeddings"]
        )
        if existing["documents"] and existing["documents"][0]:
            cand_embed = existing["embeddings"][0][0]
            doc_sim = cosine_similarity(new_doc_embed, cand_embed)
            # Relax threshold because synthesized documents vary in exact wording
            if doc_sim >= WEB_CANDIDATE_MATCH_THRESHOLD:
                matched_id   = existing["ids"][0][0]
                matched_meta = existing["metadatas"][0][0]
                matched_doc  = existing["documents"][0][0]
                print(f"[AKM] ✓ Candidate match confirmed (Doc Sim: {doc_sim:.4f} >= {WEB_CANDIDATE_MATCH_THRESHOLD})")
                return _increment_web_candidate(matched_id, matched_meta, matched_doc, session_id, user_query)
            else:
                print(f"[AKM] ✗ Unique knowledge — proceeding to store as new candidate. (Doc Sim: {doc_sim:.4f} < {WEB_CANDIDATE_MATCH_THRESHOLD})")
    except Exception as e:
        print(f"[AKM] Web candidate check error: {e}")

    # Step B4: Store as CANDIDATE — needs cross-session confirmation to go active
    candidate_id = str(uuid.uuid4())
    meta = {
        "source"          : "web_search",
        "query"           : user_query,
        "score"           : score,
        "occurrence_count": 1,
        "source_sessions" : json.dumps([session_id]),
        "timestamps"      : json.dumps([datetime.utcnow().isoformat()]),
        "status"          : "candidate",
    }
    if parent_id:
        meta["parent_id"] = parent_id

    candidate_collection.add(
        ids=[candidate_id],
        documents=[new_document],
        metadatas=[meta]
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
    mutation_score, similarity_score, session_id, doc_original,
    user_query=""
) -> str:

    candidate_id = str(uuid.uuid4())
    candidate_collection.add(
        ids=[candidate_id],
        documents=[doc_refined],
        metadatas=[{
            "original_id"     : doc_id,
            "query"           : user_query,
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
            "topic"                  : original_meta.get("topic", "unknown"),
            "parent_id"              : original_id,
            "promoted_from_candidate": candidate_id,
            "promoted_at"            : datetime.utcnow().isoformat(),
        }]
    )

    candidate_collection.delete(ids=[candidate_id])
    print("[AKM] ✓ Promotion complete.")
    return refined_text


# ── Conflict & Quarantine Handlers ─────────────────────────────────────────────

def _conflict_replace(parent_id: str, new_document: str, score: float, user_query: str, session_id: str) -> str:
    """
    Immediately replaces a poisoned active document with a web-verified correction.
    The old document is archived with status 'poisoned' for forensic tracing.
    """
    try:
        parent_data = active_collection.get(ids=[parent_id], include=["metadatas", "documents"])
        if parent_data["metadatas"] and parent_data["metadatas"][0]:
            original_meta = parent_data["metadatas"][0]
            original_doc  = parent_data["documents"][0] if parent_data["documents"] else ""

            # Archive the poisoned document with full forensic trail
            active_collection.update(
                ids=[parent_id],
                metadatas=[{
                    **original_meta,
                    "status"           : "poisoned",
                    "poisoned_at"      : datetime.utcnow().isoformat(),
                    "disputed_by"      : session_id,
                    "dispute_query"    : user_query,
                    "original_content" : original_doc[:500],  # snapshot for forensics
                }]
            )
            print(f"[AKM] ☠ Poisoned document archived: {parent_id[:8]}...")

            # Insert the corrected document as a new active entry
            new_id = str(uuid.uuid4())
            active_collection.add(
                ids=[new_id],
                documents=[new_document],
                metadatas=[{
                    "status"       : "active",
                    "source"       : "conflict_correction",
                    "topic"        : original_meta.get("topic", "unknown"),
                    "query"        : user_query,
                    "parent_id"    : parent_id,
                    "score"        : score,
                    "corrected_at" : datetime.utcnow().isoformat(),
                    "corrected_by" : session_id,
                }]
            )
            print(f"[AKM] ✓ Corrected document now active: {new_id[:8]}...")
            return new_document
    except Exception as e:
        print(f"[AKM] Conflict replacement error: {e}")

    return new_document


def _quarantine_document(doc_id: str, user_query: str, session_id: str, reason: str = "unknown") -> None:
    """
    Quarantines a disputed active document so it is no longer served to users.
    Triggered when a CONFLICT is detected but the web cannot provide a verified correction.
    """
    try:
        doc_data = active_collection.get(ids=[doc_id], include=["metadatas"])
        if doc_data["metadatas"] and doc_data["metadatas"][0]:
            original_meta = doc_data["metadatas"][0]

            # Only quarantine if it's still active (avoid double-quarantine)
            if original_meta.get("status") != "active":
                print(f"[AKM] Document {doc_id[:8]}... already not active (status: {original_meta.get('status')}) — skipping quarantine.")
                return

            active_collection.update(
                ids=[doc_id],
                metadatas=[{
                    **original_meta,
                    "status"            : "disputed",
                    "disputed_at"       : datetime.utcnow().isoformat(),
                    "disputed_by"       : session_id,
                    "dispute_query"     : user_query,
                    "quarantine_reason" : reason,
                }]
            )
            print(f"[AKM] 🔒 Document quarantined: {doc_id[:8]}... (reason: {reason})")
            print(f"[AKM]   This document will no longer be served to users.")
    except Exception as e:
        print(f"[AKM] Quarantine error: {e}")


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

        meta = {
            "status"                 : "active",
            "source"                 : "web_search",
            "topic"                  : matched_meta.get("query", "unknown").replace(" ", "_"),
            "query"                  : matched_meta.get("query", ""),
            "promoted_from_candidate": matched_id,
            "promoted_at"            : datetime.utcnow().isoformat(),
            "score"                  : best_score,
        }
        
        parent_id = matched_meta.get("parent_id")
        if parent_id:
            meta["parent_id"] = parent_id
            print(f"[AKM] Archiving parent document {parent_id}...")
            try:
                original_meta = active_collection.get(ids=[parent_id])["metadatas"][0]
                active_collection.update(
                    ids=[parent_id],
                    metadatas=[{**original_meta, "status": "archived"}]
                )
            except Exception as e:
                print(f"[AKM] Could not archive parent document: {e}")

        new_id = str(uuid.uuid4())
        active_collection.add(
            ids=[new_id],
            documents=[doc_to_store],
            metadatas=[meta]
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
            "content": SUMMARIZE_FOR_QUERY_PROMPT.format(user_query=user_query, document=document)
        }],
        max_tokens=150,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()