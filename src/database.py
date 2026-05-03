import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Persistent storage — creates an akm_db/ folder in your project
client = chromadb.PersistentClient(path="./akm_db")

# Shared embedding function used across both collections and for manual embedding
embed_fn = DefaultEmbeddingFunction()

# Active knowledge base — the source of truth
# Only documents with status = 'active' are used in retrieval
active_collection = client.get_or_create_collection(
    name="active_nodes",
    embedding_function=embed_fn
)

# Candidate pool — unproven refinements waiting for enough confirmation
candidate_collection = client.get_or_create_collection(
    name="candidate_nodes",
    embedding_function=embed_fn
)


def get_embedding(text: str) -> list:
    """Get a raw embedding vector for a piece of text."""
    return embed_fn([text])[0]