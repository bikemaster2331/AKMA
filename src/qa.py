from database import active_collection


QA_PAIRS = [

    # ── Python ────────────────────────────────────────────────────────────────
    {
        "topic": "python",
        "questions": [
            "Who created Python?",
            "When was Python first released?",
            "Who invented the Python programming language?",
            "What year did Python come out?",
            "What is Python used for?",
            "What kind of programming language is Python?",
            "What makes Python different from other languages?",
            "Does Python support object-oriented programming?",
            "What programming paradigms does Python support?",
            "Is Python a high-level language?",
        ]
    },

    # ── Machine Learning ──────────────────────────────────────────────────────
    {
        "topic": "machine_learning",
        "questions": [
            "What is machine learning?",
            "How does machine learning work?",
            "What is the difference between supervised and unsupervised learning?",
            "What are the types of machine learning?",
            "Is machine learning part of artificial intelligence?",
            "What is reinforcement learning?",
            "How do machines learn from data?",
            "What problems can machine learning solve?",
            "What is a machine learning model?",
            "How is machine learning different from traditional programming?",
        ]
    },

    # ── Immune System ─────────────────────────────────────────────────────────
    {
        "topic": "immune_system",
        "questions": [
            "How does the immune system work?",
            "What does the immune system do?",
            "What is innate immunity?",
            "What is adaptive immunity?",
            "How does the body fight off infections?",
            "What is the difference between innate and adaptive immunity?",
            "How does the body defend itself against pathogens?",
            "What protects the human body from disease?",
            "What are the two types of immune response?",
            "How does immunity work in the human body?",
        ]
    },

    # ── Roman Empire ──────────────────────────────────────────────────────────
    {
        "topic": "roman_empire",
        "questions": [
            "What was the Roman Empire?",
            "When did the Roman Empire fall?",
            "How long did the Roman Empire last?",
            "When did the Roman Empire begin?",
            "What did the Roman Empire contribute to modern civilization?",
            "How did Rome influence European law?",
            "What caused the fall of the Roman Empire?",
            "When was the Western Roman Empire destroyed?",
            "What languages did the Romans influence?",
            "How large was the Roman Empire?",
        ]
    },

    # ── Photosynthesis ────────────────────────────────────────────────────────
    {
        "topic": "photosynthesis",
        "questions": [
            "What is photosynthesis?",
            "How do plants make food?",
            "How do plants convert sunlight into energy?",
            "What is produced during photosynthesis?",
            "Where does photosynthesis take place?",
            "What are the inputs and outputs of photosynthesis?",
            "What role do chloroplasts play in photosynthesis?",
            "Why is photosynthesis important?",
            "What do plants need to perform photosynthesis?",
            "How does photosynthesis produce oxygen?",
        ]
    },

    # ── Bitcoin ───────────────────────────────────────────────────────────────
    {
        "topic": "bitcoin",
        "questions": [
            "What is Bitcoin?",
            "Who created Bitcoin?",
            "When was Bitcoin created?",
            "How does Bitcoin work?",
            "What is a blockchain?",
            "What is Satoshi Nakamoto?",
            "Is Bitcoin decentralized?",
            "What makes Bitcoin different from regular currency?",
            "How is Bitcoin stored and transferred?",
            "What is a distributed public ledger?",
        ]
    },

    # ── Water Cycle ───────────────────────────────────────────────────────────
    {
        "topic": "water_cycle",
        "questions": [
            "What is the water cycle?",
            "How does the water cycle work?",
            "What are the stages of the water cycle?",
            "What is evaporation in the water cycle?",
            "What is condensation?",
            "What is precipitation?",
            "How does water move through the environment?",
            "Why is the water cycle important?",
            "What causes rain?",
            "How does water get from the ocean to the clouds?",
        ]
    },

    # ── World War 2 ───────────────────────────────────────────────────────────
    {
        "topic": "world_war_2",
        "questions": [
            "When did World War 2 start?",
            "When did World War 2 end?",
            "How long did World War 2 last?",
            "How many people died in World War 2?",
            "What countries were involved in World War 2?",
            "What caused World War 2?",
            "What was the deadliest war in history?",
            "How many casualties were there in WW2?",
            "What happened during World War 2?",
            "What was the scale of World War 2?",
        ]
    },

    # ── DNA ───────────────────────────────────────────────────────────────────
    {
        "topic": "dna",
        "questions": [
            "What is DNA?",
            "What does DNA do?",
            "What is the structure of DNA?",
            "What is a double helix?",
            "How is genetic information stored?",
            "What are nucleotides?",
            "What are base pairs in DNA?",
            "Where is DNA found in the body?",
            "How does DNA carry genetic information?",
            "What makes up a DNA molecule?",
        ]
    },

    # ── Internet ──────────────────────────────────────────────────────────────
    {
        "topic": "internet",
        "questions": [
            "What is the internet?",
            "How does the internet work?",
            "Who invented the internet?",
            "Where did the internet come from?",
            "What is ARPANET?",
            "When was the internet created?",
            "What protocols does the internet use?",
            "How are computers connected on the internet?",
            "What is the history of the internet?",
            "What was the internet originally designed for?",
        ]
    },

]


def get_qa_pairs_for_topic(topic: str) -> list[str]:
    """Returns all questions for a given topic."""
    for entry in QA_PAIRS:
        if entry["topic"] == topic:
            return entry["questions"]
    return []


def get_all_questions_with_topics() -> list[dict]:
    """
    Returns a flat list of {question, topic} dicts.
    Used when building the QA collection in ChromaDB.
    """
    flat = []
    for entry in QA_PAIRS:
        for q in entry["questions"]:
            flat.append({
                "question": q,
                "topic": entry["topic"]
            })
    return flat