import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Load Thresholds ---
THRESHOLDS_FILE = os.path.join(os.path.dirname(__file__), "thresholds.json")
with open(THRESHOLDS_FILE, "r") as f:
    _t = json.load(f)

# --- Decision Gate ---
REJECTION_THRESHOLD = _t["REJECTION_THRESHOLD"]
SELECTION_THRESHOLD = _t["SELECTION_THRESHOLD"]
TOP_K_ACTIVE        = _t["TOP_K_ACTIVE"]

# --- Promotion Rules ---
PROMOTION_SCORE_THRESHOLD      = _t["PROMOTION_SCORE_THRESHOLD"]
PROMOTION_SIMILARITY_THRESHOLD = _t["PROMOTION_SIMILARITY_THRESHOLD"]
PROMOTION_COUNT_THRESHOLD      = _t["PROMOTION_COUNT_THRESHOLD"]
PROMOTION_SESSION_THRESHOLD    = _t["PROMOTION_SESSION_THRESHOLD"]

# --- Candidate Matching ---
CANDIDATE_MATCH_THRESHOLD      = _t["CANDIDATE_MATCH_THRESHOLD"]
WEB_CANDIDATE_MATCH_THRESHOLD  = _t["WEB_CANDIDATE_MATCH_THRESHOLD"]
DELTA_MATCH_THRESHOLD          = _t["DELTA_MATCH_THRESHOLD"]
IDENTICAL_REFINEMENT_THRESHOLD = _t["IDENTICAL_REFINEMENT_THRESHOLD"]

# --- Search Grounding & Stale Candidate Cleanup ---
MAX_CLAIMS_TO_CHECK    = _t["MAX_CLAIMS_TO_CHECK"]
MIN_SEARCH_RESULTS     = _t["MIN_SEARCH_RESULTS"]
CANDIDATE_MAX_AGE_DAYS = _t["CANDIDATE_MAX_AGE_DAYS"]

# --- Volatile Cache ---
VOLATILE_CACHE_TTL_SECONDS = _t["VOLATILE_CACHE_TTL_SECONDS"]

# --- Persistence ---
# OVERWRITE_THRESHOLD has been removed. Active documents are always archived on promotion.

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