import uuid
from database import active_collection, candidate_collection
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
        "text": "Bitcoin is a decentralized digital currency created in 2009 by an anonymous entity known as Satoshi Nakamoto. It operates on a blockchain, a distributed public ledger.",
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