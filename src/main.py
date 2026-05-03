import uuid
from database import active_collection, candidate_collection
from pipeline import run_akm


SEED_DOCUMENTS = [
    {
        "text": (
            "Python is a high-level, interpreted programming language known for its "
            "simple syntax and readability. It supports multiple programming paradigms "
            "including procedural, object-oriented, and functional programming. "
            "Python was created by Guido van Rossum and first released in 1991."
        ),
        "metadata": {"status": "active", "topic": "python"}
    },
    {
        "text": (
            "ChromaDB is an open-source vector database designed for AI applications. "
            "It stores text alongside vector embeddings and supports fast similarity search. "
            "ChromaDB can run locally without any external services, making it ideal for "
            "development and prototyping."
        ),
        "metadata": {"status": "active", "topic": "chromadb"}
    },
    {
        "text": (
            "Machine learning is a subset of artificial intelligence where systems learn "
            "from data to improve their performance on tasks without being explicitly programmed. "
            "Common approaches include supervised learning, unsupervised learning, and "
            "reinforcement learning."
        ),
        "metadata": {"status": "active", "topic": "machine_learning"}
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
    results = active_collection.get(where={"status": "active"}, include=["documents", "metadatas", "ids"])
    if not results["ids"]:
        print("[DOCS] No active documents found.")
        return
    print(f"\n[DOCS] {len(results['ids'])} active document(s):\n")
    for i, (doc_id, doc, meta) in enumerate(zip(results["ids"], results["documents"], results["metadatas"])):
        print(f"  [{i+1}] ID     : {doc_id[:8]}...")
        print(f"       Topic  : {meta.get('topic', 'unknown')}")
        print(f"       Text   : {doc[:120]}...")
        print()


def print_help():
    print("""
Commands:
  seed    — Load starter documents into the knowledge base
  status  — Show active and candidate document counts
  docs    — List all active documents
  full    — View the full document from the last query
  help    — Show this menu
  quit    — Exit

Anything else is treated as a query to the AKM pipeline.
""")


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