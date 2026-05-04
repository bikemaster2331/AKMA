import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Decision Gate ---
REJECTION_THRESHOLD = 0.70
RELEVANCE_THRESHOLD  = 0.50

# --- Promotion Rules ---
PROMOTION_SCORE_THRESHOLD      = 0.80
PROMOTION_SIMILARITY_THRESHOLD = 0.85
PROMOTION_COUNT_THRESHOLD      = 2
PROMOTION_SESSION_THRESHOLD    = 2

# --- Candidate Matching ---
CANDIDATE_MATCH_THRESHOLD = 0.90
DELTA_MATCH_THRESHOLD     = 0.70
BM25_QUERY_THRESHOLD      = 0.85

# --- Persistence ---
OVERWRITE_THRESHOLD = 0.90

# --- Search Grounding ---
BLOCKED_DOMAINS = [
    "wikipedia.org",
    "reddit.com",
    "quora.com",
    "medium.com",
    "twitter.com",
    "x.com",
    "facebook.com",
]

TRUSTED_DOMAINS = [
    "docs.python.org",
    "arxiv.org",
    "github.com",
    "stackoverflow.com",
    "developer.mozilla.org",
    "docs.microsoft.com",
    "ieee.org",
    "nature.com",
    "sciencedirect.com",
]

MAX_CLAIMS_TO_CHECK = 3
MIN_SEARCH_RESULTS  = 2