import uuid
from database import active_collection, candidate_collection, get_embedding
from pipeline import run_akm


SEED_DOCUMENTS = [
    {
        "text": "Python is a high-level interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
        "metadata": {"status": "active", "topic": "python", "source": "seed"}
    },
    {
        "text": "Machine learning is a subset of artificial intelligence where systems learn patterns from data. Common approaches include supervised, unsupervised, and reinforcement learning.",
        "metadata": {"status": "active", "topic": "machine_learning", "source": "seed"}
    },
    {
        "text": "The human immune system defends the body against pathogens. It consists of innate immunity, which responds immediately, and adaptive immunity, which targets specific threats.",
        "metadata": {"status": "active", "topic": "immune_system", "source": "seed"}
    },
    {
        "text": "The Roman Empire was one of the largest empires in ancient history, lasting from 27 BC to 476 AD in the west. It significantly shaped European law, language, and governance.",
        "metadata": {"status": "active", "topic": "roman_empire", "source": "seed"}
    },
    {
        "text": "Photosynthesis is the process by which plants convert sunlight into energy. It occurs in chloroplasts and produces glucose and oxygen from carbon dioxide and water.",
        "metadata": {"status": "active", "topic": "photosynthesis", "source": "seed"}
    },
    {
        "text": "Bitcoin is a decentralized digital cryptocurrency created by an anonymous person or group using the pseudonym Satoshi Nakamoto. It was introduced in a 2008 whitepaper and launched in 2009.",
        "metadata": {"status": "active", "topic": "bitcoin", "source": "seed"}
    },
    {
        "text": "The water cycle describes how water moves through Earth's systems. Key stages include evaporation, condensation, precipitation, and collection in bodies of water.",
        "metadata": {"status": "active", "topic": "water_cycle", "source": "seed"}
    },
    {
        "text": "World War 2 lasted from 1939 to 1945 and involved most of the world's nations. It resulted in an estimated 70-85 million deaths, making it the deadliest conflict in human history.",
        "metadata": {"status": "active", "topic": "world_war_2", "source": "seed"}
    },
    {
        "text": "DNA is a molecule that carries genetic information in living organisms. It consists of two strands forming a double helix, composed of nucleotide base pairs.",
        "metadata": {"status": "active", "topic": "dna", "source": "seed"}
    },
    {
        "text": "The internet is a global network of interconnected computers that communicate using standardized protocols. It evolved from ARPANET, a US military research project in the 1960s.",
        "metadata": {"status": "active", "topic": "internet", "source": "seed"}
    },
]


def seed_database():
    existing_count = active_collection.count()
    if existing_count > 0:
        print(f"[SEED] Database already has {existing_count} active document(s). Skipping seed.")
        return

    ids  = [str(uuid.uuid4()) for _ in SEED_DOCUMENTS]
    docs = [d["text"] for d in SEED_DOCUMENTS]
    meta = [d["metadata"] for d in SEED_DOCUMENTS]

    active_collection.add(ids=ids, documents=docs, metadatas=meta)
    print(f"[SEED] Added {len(SEED_DOCUMENTS)} documents to the knowledge base.")


def show_status():
    active_count    = active_collection.count()
    candidate_count = candidate_collection.count()
    print(f"\n[STATUS] Active nodes    : {active_count}")
    print(f"[STATUS] Candidate nodes : {candidate_count}\n")


def show_active_docs():
    results = active_collection.get(where={"status": "active"}, include=["documents", "metadatas"])
    if not results["ids"]:
        print("[DOCS] No active documents found.")
        return
    print(f"\n[DOCS] {len(results['ids'])} active document(s):\n")
    for i, (doc_id, doc, meta) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
        print(f"  [{i+1}] ID     : {doc_id[:8]}...")
        print(f"       Topic  : {meta.get('topic', 'unknown')}")
        print(f"       Text   : {doc[:120]}...")
        print()


def show_candidates():
    results = candidate_collection.get(include=["documents", "metadatas"])
    if not results["ids"]:
        print("[CANDIDATES] No candidate documents found.")
        return
    print(f"\n[CANDIDATES] {len(results['ids'])} candidate(s) awaiting confirmation:\n")
    for i, (doc_id, doc, meta) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
        import json as _json
        sessions = _json.loads(meta.get("source_sessions", "[]"))
        distinct  = len(set(sessions))
        count     = int(meta.get("occurrence_count", 1))
        score     = meta.get("score", meta.get("mutation_score", "N/A"))
        query     = meta.get("query", "unknown")
        source    = meta.get("source", "unknown")
        parent    = meta.get("original_id", meta.get("parent_id", "none"))

        print(f"  [{i+1}] ID        : {doc_id[:8]}...")
        print(f"       Query     : {query}")
        print(f"       Source    : {source}")
        print(f"       Score     : {score}")
        print(f"       Count     : {count} / 2")
        print(f"       Sessions  : {distinct} distinct")
        if parent and parent != "none":
            print(f"       Parent    : {parent[:8]}...")
        else:
            print(f"       Parent    : none")
        print(f"       Text      : {doc[:120]}...")
        print()


def print_help():
    print("""
Commands:
  seed        — Load starter documents into the knowledge base
  status      — Show active and candidate document counts
  docs        — List all active documents
  candidates  — List all candidate documents with confirmation progress
  forensics   — View all poisoned and disputed documents
  admin       — Enter admin mode (database surgery)
  full        — View the full document from the last query
  help        — Show this menu
  quit        — Exit

Anything else is treated as a query to the AKM pipeline.
""")


def show_forensics():
    """Show all poisoned and disputed documents for forensic inspection."""
    print(f"\n{'='*60}")
    print("[FORENSICS] Scanning for poisoned & disputed documents...")
    print(f"{'='*60}")

    found = 0

    for status_label in ["poisoned", "disputed"]:
        try:
            results = active_collection.get(
                where={"status": status_label},
                include=["documents", "metadatas"]
            )
            if results["ids"]:
                for i, (doc_id, doc, meta) in enumerate(zip(
                    results["ids"], results["documents"], results["metadatas"]
                )):
                    found += 1
                    print(f"\n  [{found}] Status  : {status_label.upper()}")
                    print(f"       ID      : {doc_id[:8]}...")
                    print(f"       Topic   : {meta.get('topic', 'unknown')}")
                    print(f"       Text    : {doc[:150]}...")
                    print(f"       Poisoned at    : {meta.get('poisoned_at', 'N/A')}" if status_label == "poisoned" else f"       Disputed at    : {meta.get('disputed_at', 'N/A')}")
                    print(f"       Disputed by    : {meta.get('disputed_by', 'N/A')[:8]}...")
                    print(f"       Dispute query  : {meta.get('dispute_query', 'N/A')}")
                    if status_label == "poisoned":
                        print(f"       Original snap  : {meta.get('original_content', 'N/A')[:120]}...")
                    else:
                        print(f"       Reason         : {meta.get('quarantine_reason', 'N/A')}")
        except Exception as e:
            print(f"  [FORENSICS] Error querying {status_label}: {e}")

    # Also scan active documents for unverified disputes to display them in forensics
    try:
        import json
        active_results = active_collection.get(
            where={"status": "active"},
            include=["documents", "metadatas"]
        )
        if active_results["ids"]:
            for doc_id, doc, meta in zip(
                active_results["ids"], active_results["documents"], active_results["metadatas"]
            ):
                if meta and "unverified_disputes" in meta:
                    disputes = json.loads(meta.get("unverified_disputes", "[]"))
                    for disp in disputes:
                        found += 1
                        print(f"\n  [{found}] Status  : ACTIVE (Dispute Logged)")
                        print(f"       ID      : {doc_id[:8]}...")
                        print(f"       Topic   : {meta.get('topic', 'unknown')}")
                        print(f"       Text    : {doc[:150]}...")
                        print(f"       Disputed at    : {disp.get('disputed_at', 'N/A')}")
                        print(f"       Disputed by    : {disp.get('disputed_by', 'N/A')[:8]}...")
                        print(f"       Dispute query  : {disp.get('dispute_query', 'N/A')}")
                        print(f"       Reason         : {disp.get('reason', 'N/A')}")
    except Exception as e:
        print(f"  [FORENSICS] Error scanning active disputes: {e}")

    if found == 0:
        print("\n  [FORENSICS] No poisoned, disputed, or active dispute logs found. Database is clean.")

    print(f"\n{'='*60}\n")


# ── Admin Mode ─────────────────────────────────────────────────────────────────

def admin_mode():
    """Interactive admin panel for direct database surgery."""
    print(f"\n{'='*60}")
    print("[ADMIN] Entering Admin Mode")
    print("[ADMIN] Search any query to find documents. Type 'done' to leave.")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input("Admin Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[ADMIN] Exiting admin mode.")
            return

        if not query:
            continue
        if query.lower() in ("exit", "done", "back"):
            print("[ADMIN] Exiting admin mode.\n")
            return

        # Search both collections, all statuses
        results_list = []

        # Search active collection (all statuses)
        try:
            active_results = active_collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas"]
            )
            if active_results["ids"] and active_results["ids"][0]:
                for i in range(len(active_results["ids"][0])):
                    results_list.append({
                        "collection": "active",
                        "id": active_results["ids"][0][i],
                        "doc": active_results["documents"][0][i],
                        "meta": active_results["metadatas"][0][i],
                    })
        except Exception as e:
            print(f"[ADMIN] Active search error: {e}")

        # Search candidate collection
        try:
            cand_results = candidate_collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas"]
            )
            if cand_results["ids"] and cand_results["ids"][0]:
                for i in range(len(cand_results["ids"][0])):
                    results_list.append({
                        "collection": "candidate",
                        "id": cand_results["ids"][0][i],
                        "doc": cand_results["documents"][0][i],
                        "meta": cand_results["metadatas"][0][i],
                    })
        except Exception as e:
            print(f"[ADMIN] Candidate search error: {e}")

        if not results_list:
            print("[ADMIN] No documents found.\n")
            continue

        # Display results
        print(f"\n[ADMIN] Found {len(results_list)} result(s):\n")
        for i, r in enumerate(results_list):
            status = r["meta"].get("status", "unknown")
            topic  = r["meta"].get("topic", "unknown")
            coll   = r["collection"].upper()
            print(f"  [{i+1}] [{coll}] [{status.upper()}] Topic: {topic}")
            print(f"       ID  : {r['id'][:12]}...")
            print(f"       Text: {r['doc'][:120]}...")
            print()

        # Pick a document
        try:
            pick = input("Pick a document number (or 'done' to exit/'back' to search again): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[ADMIN] Exiting admin mode.")
            return

        if pick.lower() in ("done", "exit"):
            print("[ADMIN] Exiting admin mode.\n")
            return
        if pick.lower() == "back" or not pick:
            continue

        try:
            idx = int(pick) - 1
            if idx < 0 or idx >= len(results_list):
                print("[ADMIN] Invalid number.\n")
                continue
        except ValueError:
            print("[ADMIN] Enter a number.\n")
            continue

        selected = results_list[idx]
        _admin_document_view(selected)


def _admin_document_view(selected: dict):
    """Display full document details and offer surgery options."""
    coll_name = selected["collection"]
    collection = active_collection if coll_name == "active" else candidate_collection
    doc_id = selected["id"]
    doc    = selected["doc"]
    meta   = selected["meta"]

    while True:
        print(f"\n{'─'*60}")
        print(f"[ADMIN] Document Surgery")
        print(f"{'─'*60}")
        print(f"  Collection : {coll_name.upper()}")
        print(f"  ID         : {doc_id}")
        print(f"  Status     : {meta.get('status', 'unknown')}")
        print(f"  Topic      : {meta.get('topic', 'unknown')}")
        print(f"  Source     : {meta.get('source', 'unknown')}")
        print(f"{'─'*60}")
        print(f"  FULL TEXT:")
        print(f"  {doc}")
        print(f"{'─'*60}")
        print(f"  METADATA:")
        for k, v in meta.items():
            print(f"    {k}: {v}")
        print(f"{'─'*60}")
        print(f"\n  Surgery Options:")
        print(f"    edit    — Rewrite the entire document")
        print(f"    replace — Find and replace specific text")
        print(f"    delete  — Permanently remove this document")
        print(f"    back    — Return to search\n")

        try:
            action = input("  Action: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[ADMIN] Returning to search.")
            return

        if action == "back" or not action:
            return

        elif action == "edit":
            print("\n  Type your new document below. Type 'END' on a new line when done.")
            lines = []
            while True:
                try:
                    line = input("  > ")
                except (KeyboardInterrupt, EOFError):
                    print("\n[ADMIN] Edit cancelled.")
                    break
                if line.strip() == "END":
                    break
                lines.append(line)

            if lines:
                new_doc = "\n".join(lines)
                try:
                    # Re-embed and update the document
                    new_embed = get_embedding(new_doc)
                    collection.update(
                        ids=[doc_id],
                        documents=[new_doc],
                        embeddings=[new_embed]
                    )
                    doc = new_doc  # update local copy
                    print(f"\n  [ADMIN] ✓ Document updated and re-embedded.")
                except Exception as e:
                    print(f"\n  [ADMIN] Error updating document: {e}")

        elif action == "replace":
            try:
                old_text = input("  Find text    : ").strip()
                new_text = input("  Replace with : ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[ADMIN] Replace cancelled.")
                continue

            if not old_text:
                print("  [ADMIN] No search text provided.")
                continue

            if old_text not in doc:
                print(f"  [ADMIN] ✗ Text '{old_text}' not found in document.")
                continue

            new_doc = doc.replace(old_text, new_text)
            count = doc.count(old_text)
            try:
                new_embed = get_embedding(new_doc)
                collection.update(
                    ids=[doc_id],
                    documents=[new_doc],
                    embeddings=[new_embed]
                )
                doc = new_doc
                print(f"\n  [ADMIN] ✓ Replaced {count} occurrence(s) and re-embedded.")
            except Exception as e:
                print(f"\n  [ADMIN] Error replacing: {e}")

        elif action == "delete":
            try:
                confirm = input(f"  Are you sure you want to DELETE {doc_id[:12]}...? (yes/no): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n[ADMIN] Delete cancelled.")
                continue

            if confirm == "yes":
                try:
                    collection.delete(ids=[doc_id])
                    print(f"  [ADMIN] ✓ Document {doc_id[:12]}... permanently deleted.")
                    return  # go back to search since the doc no longer exists
                except Exception as e:
                    print(f"  [ADMIN] Error deleting: {e}")
            else:
                print("  [ADMIN] Delete cancelled.")

        else:
            print("  [ADMIN] Unknown action. Use: edit, replace, delete, or back.")


def main():
    print("\n=== Autonomous Knowledge Mutation — MVP ===")
    print("Type 'help' for available commands.\n")

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id[:8]}...\n")

    show_status()

    last_full_doc = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[AKM] Exiting.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            print("[AKM] Goodbye.")
            break
        elif cmd == "seed":
            seed_database()
        elif cmd == "status":
            show_status()
        elif cmd == "docs":
            show_active_docs()
        elif cmd == "candidates":
            show_candidates()
        elif cmd == "forensics":
            show_forensics()
        elif cmd == "admin":
            admin_mode()
        elif cmd == "help":
            print_help()
        elif cmd == "full":
            if last_full_doc:
                print(f"\n[FULL DOCUMENT]\n{'─'*50}\n{last_full_doc}\n{'─'*50}\n")
            else:
                print("[AKM] No document stored yet. Ask a question first.\n")
        else:
            short_answer, full_doc = run_akm(user_input, session_id)
            last_full_doc = full_doc
            print(f"\nResponse: {short_answer}")
            print(f"[Full document stored — type 'full' to view it.]\n")


if __name__ == "__main__":
    main()