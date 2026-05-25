# Autonomous Knowledge Mutation (AKM) — Architecture & Changelog

> A self-growing, self-correcting knowledge base that learns from user queries, validates new knowledge against the web, and evolves its stored facts through a multi-session consensus mechanism.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Structure](#2-file-structure)
3. [Core Concepts](#3-core-concepts)
4. [Full Pipeline Flow](#4-full-pipeline-flow)
   - [Step 1: Retrieval & Routing](#step-1-retrieval--routing)
   - [Path A: Refinement](#path-a-refinement-known-topic)
   - [Path B: Web Search](#path-b-web-search-unknown-topic)
   - [The Conflict Fast-Path](#the-conflict-fast-path)
5. [Candidate Consensus System](#5-candidate-consensus-system)
6. [Promotion to Active](#6-promotion-to-active)
7. [Document Status Lifecycle](#7-document-status-lifecycle)
8. [Thresholds Reference](#8-thresholds-reference)
9. [CLI Commands Reference](#9-cli-commands-reference)
10. [Admin Mode](#10-admin-mode)
11. [Forensics System](#11-forensics-system)
12. [Changelog — Session History](#12-changelog--session-history)

---

## 1. System Overview

AKM is a locally-running, autonomous knowledge pipeline. It does not simply query an LLM and return a response. Instead, it:

- Maintains a **persistent vector database** (ChromaDB) of verified knowledge nodes
- Routes every query through a **relevance check** before touching the LLM
- Requires **multi-session consensus** before promoting any new knowledge to its active store
- Uses a **web search firewall** to ensure no user-fabricated claim can corrupt the database
- Detects and **immediately neutralises poisoned entries** when a conflict is raised
- Provides a full **admin surgery panel** for direct database inspection and repair

**Technology Stack:**
- **Vector DB:** ChromaDB (persistent, local)
- **LLM:** Groq API — `llama-3.3-70b-versatile`
- **Embeddings:** ChromaDB `DefaultEmbeddingFunction` (sentence-transformers)
- **Web Search:** Tavily API
- **Language:** Python 3.12

---

## 2. File Structure

```
akm/
├── src/
│   ├── main.py          # CLI entry point, all commands, admin mode
│   ├── pipeline.py      # Core AKM routing logic, all paths, conflict & dispute handlers
│   ├── searcher.py      # Web search (Tavily), claim extraction, document synthesis
│   ├── critic.py        # LLM-based document scoring (synthesis & refinement)
│   ├── refiner.py       # Document refinement via LLM
│   ├── database.py      # ChromaDB client, collection definitions, embedding function
│   ├── config.py        # Threshold loading, API keys, blocked/trusted domains
│   ├── prompts.py       # All LLM prompt templates
│   └── thresholds.json  # Tunable decision-gate values
├── akm_db/              # Persistent ChromaDB storage (auto-created)
├── ARCHITECTURE.md      # This file
├── README.md
└── setup.md
```

---

## 3. Core Concepts

### Active Collection (`active_nodes`)
The source of truth. Only documents with `status: "active"` are ever retrieved and served to users. Documents with any other status are completely invisible to the normal query flow.

### Candidate Collection (`candidate_nodes`)
A quarantine zone for unproven knowledge. Every new piece of web-sourced or refined knowledge lands here first. A candidate must accumulate confirmations from **at least 2 distinct sessions** before it is eligible for promotion to active.

### Session ID
Every time the app starts, a new UUID is generated as the session ID. This is the system's mechanism for verifying that knowledge confirmations come from genuinely independent users, not the same user refreshing the app.

### Embeddings
Every document is converted into a high-dimensional vector (embedding) that captures its semantic meaning. The system uses **cosine similarity** between these vectors to determine relevance and detect duplicate knowledge — even when worded differently.

---

## 4. Full Pipeline Flow

### Step 1: Retrieval & Routing

Every user query triggers this sequence:

1. The query is embedded into a vector.
2. ChromaDB performs a nearest-neighbour search across all **active** documents.
3. Cosine similarity is computed between the query embedding and the retrieved document's embedding.
4. The score is compared against `RELEVANCE_THRESHOLD` (0.50):

```
Query similarity >=  0.50  →  PATH A (Refinement)
Query similarity <   0.50  →  PATH B (Web Search)
```

---

### Path A: Refinement (Known Topic)

Triggered when the query is deemed relevant to an existing active document.

**Step A1 — Smart Router (Intent Classifier)**

A single LLM call (`_confirm_and_refine`) performs two jobs simultaneously:
1. Classifies the user's intent into one of **five buckets**
2. If `STATIC_MATCH`, produces the refined document in the same response

| Classification | Meaning | Action |
|---|---|---|
| `STATIC_MATCH` | Same topic, stable facts, and document has enough info to answer | Continue refinement in Path A |
| `VOLATILE` | Same topic, time-sensitive facts (prices, current events) | Reroute to Path B with `parent_id` |
| `CONFLICT` | User is explicitly disputing a stored fact | Reroute to **Conflict Fast-Path** with `parent_id` |
| `INSUFFICIENT` | Topic is relevant, but document lacks the details to answer | Reroute to Path B with `parent_id` for detail enrichment |
| `DIFFERENT` | Vector DB false positive — actually a different topic | Reroute to Path B as a fresh query |

**Step A2 — Critic Scoring (Refinement Mode)**

The Critic LLM (`critic.py → _score_refinement()`) reviews the refined document against the original using the `SCORE_REFINEMENT_PROMPT`. It performs a structured 4-step audit:
1. **Fact Inventory** — Lists every specific fact in the original (names, dates, numbers).
2. **Preservation Audit** — Checks each original fact: preserved exactly (OK), softened (WARN), altered/removed (FAIL).
3. **Structural Audit** — Reviews additions for coherence and internal consistency. Crucially, the Critic is **forbidden from using its own training data** to judge factual accuracy — it only checks structural integrity.
4. **Final Score** — A 0.0–1.0 rubric score.

Score must meet `REJECTION_THRESHOLD` (0.70) to proceed.

**Step A3 — Semantic Similarity Check**

Computes cosine similarity between the original document embedding and the refined document embedding. If they are nearly identical (`>= IDENTICAL_REFINEMENT_THRESHOLD: 0.99`), the refinement added nothing new and is discarded.

**Step A4 — Candidate Gate**

The refined document is compared against existing candidates for the same original document using a **dual-gate system**:

- **Gate 1 — Doc Similarity:** Are the refined documents overall semantically close? (`>= CANDIDATE_MATCH_THRESHOLD: 0.70`)
- **Gate 2 — Delta Similarity:** Did they change the original document in the same *direction*? (`>= DELTA_MATCH_THRESHOLD: 0.70`)

Both gates must pass to group the refinement with an existing candidate. Otherwise, a new candidate is inserted.

---

### Path B: Web Search (Unknown Topic)

Triggered when no relevant active document exists for the query.

**Step B1 — Web Search**

Tavily API is queried with the user's raw question. Returns up to 5 source URLs and their content.

Search results are filtered through two domain lists defined in `config.py`:
- **Blocked Domains** (excluded): `wikipedia.org`, `reddit.com`, `quora.com`, `medium.com`, `twitter.com`, `x.com`, `facebook.com`
- **Trusted Domains** (prioritised): `docs.python.org`, `arxiv.org`, `github.com`, `stackoverflow.com`, `developer.mozilla.org`, `docs.microsoft.com`, `ieee.org`, `nature.com`, `sciencedirect.com`

Per-source content is capped at 1,000 characters to prevent context overflow.

**Step B2 — Synthesis**

The LLM reads the web sources and synthesises a clean, structured document answering the query using `SYNTHESIZE_FROM_SEARCH_PROMPT`. The synthesis is grounded strictly in the returned sources. If sources conflict, the LLM is instructed to note the discrepancies rather than silently choosing one version.

**Step B3 — Critic Scoring (Synthesis Mode)**

The Critic scores the synthesised document in a different mode than Path A. Instead of comparing against an original document, `critic.py → _score_synthesis()` uses the `SCORE_SYNTHESIS_PROMPT` to evaluate the synthesis against the raw web evidence:
1. **Source Coverage** — Does the document capture key facts from the sources?
2. **Fabrication Check** — Does it introduce any claims NOT present in the sources?
3. **Coherence Check** — Is the document internally consistent?
4. **Final Score** — A 0.0–1.0 rubric score.

Score must meet `REJECTION_THRESHOLD` (0.70) to proceed.

**Step B3.4 — Active Pool Deduplication**

Before checking the candidate pool, the system first checks if the synthesized document already exists in the **active** collection — but **only for fresh queries** (no `parent_id`). When Path B is handling a rerouted `VOLATILE` or `INSUFFICIENT` query, the synthesized document will naturally be semantically similar to its parent active document. Running active pool deduplication in this case would incorrectly discard the enrichment as a "duplicate." The skip condition is:

```python
if not parent_id:  # Only dedup for genuinely fresh queries
    # ... run active pool deduplication
```

For fresh queries, the new document is embedded and compared against all active documents using cosine similarity (`>= WEB_CANDIDATE_MATCH_THRESHOLD: 0.85`).

- **Match found:** The synthesized document is discarded entirely. The existing active document is returned to the user. No candidate is created.
- **No match:** Proceed to candidate deduplication.

**Step B3.5 — Semantic Candidate Deduplication**

The system embeds the new document and checks it against all existing web-search candidates using cosine similarity (`>= WEB_CANDIDATE_MATCH_THRESHOLD: 0.85`).

- **Match found:** The new document is discarded. The existing candidate's `occurrence_count` and `source_sessions` are updated with the current session.
- **No match:** A new candidate is stored.

**Step B4 — Store as Candidate**

The synthesised document is stored in `candidate_collection` with:
- `status: "candidate"`
- `occurrence_count: 1`
- `source_sessions: [current_session_id]`
- `created_at: UTC timestamp` (used for stale candidate cleanup)

---

### Step 3: The Summary Layer (Conversational Fail-Safe Referee)

Once Path A or Path B has produced a resulting document (whether it is a new refinement, a new web synthesis, or the original document retrieved during a fallback), the final stage of `run_akm` executes a conversational synthesis:

```python
short_answer = summarize_for_query(user_query, full_doc)
```

This layer is governed by `SUMMARIZE_FOR_QUERY_PROMPT` in `prompts.py`, which strictly commands the LLM:
> *"Answer the following question in 2-3 sentences using ONLY the document below... Do not mention the document or sources explicitly."*

#### The Grounding Fallback Gate
This step acts as the **ultimate, final fail-safe gate against system hallucination**:
* If a query gets incorrectly routed to an active document that **does not actually contain the answer**, and Path A fails its quality checks, the pipeline falls back to returning the original document (`doc_original`).
* When this original document is passed to the Summary Layer, the strict `using ONLY the document` constraint prevents the LLM from fabricating an answer using its pre-trained weights.
* Instead of hallucinating a false fact, the LLM will politely decline to answer, returning: *"I'm sorry, but the provided information does not contain details about [the query]."*

This provides a vital secondary defense layer, ensuring the user is served a safe refusal rather than a confident hallucination.

---

### The Conflict Fast-Path

When the Smart Router classifies a query as `CONFLICT`, the pipeline takes a special route that bypasses the candidate consensus queue entirely.

**Step C1 — LLM Conflict Judge (Agreement Check)**

Before replacing anything, the system invokes a dedicated LLM judge using `CONFLICT_JUDGE_PROMPT`. This prompt sends both the original active document and the web-synthesized document to the LLM alongside the user's dispute query, and asks a single deterministic question: **does the web evidence AGREE or DISAGREE with the original on the specific disputed point?**

- **Verdict: `AGREES`** — The web evidence **confirms** the original document is correct. The user's dispute was wrong. The original document remains `active` and untouched. The dispute is logged as `reason: "web_confirmed_original"` in the document's `unverified_disputes` metadata field.
- **Verdict: `DISAGREES`** — The web evidence **contradicts** the original. Proceed to replacement.

> **Why an LLM judge instead of cosine similarity?** Cosine similarity measures topical overlap, not factual agreement. Two documents about the same topic — one saying "Python was created in 1991" and one saying "Python was created in 1995" — would score very high on cosine similarity despite containing contradictory facts. The LLM judge can detect these semantic-level disagreements that vector distance cannot.

**Scenario A — Web Evidence Contradicts Original & Critic Passes (`_conflict_replace`):**

1. The poisoned active document is immediately updated with `status: "poisoned"`.
2. A full forensic trail is embedded in its metadata: who disputed it, when, the exact dispute query, and a 500-character snapshot of the original content.
3. **Orphan Pruning:** All candidate nodes in `candidate_collection` that reference the poisoned document as their `original_id` are immediately deleted, preventing "toxic lineage promotion" where candidates derived from corrupted data could later be promoted.
4. A new document, built from verified web sources, is inserted as `status: "active"` with a `parent_id` linking back to the poisoned entry.
5. The user gets the corrected answer immediately. No consensus delay.

**Scenario B — Web Search Fails or Critic Rejects (`_log_unverified_dispute`):**

If the web cannot provide grounded evidence to confirm or deny the disputed fact (no web results, synthesis failure, or Critic rejection):
1. The dispute is **logged** in the original document's `unverified_disputes` metadata field (recording the session ID, dispute query, timestamp, and failure reason).
2. The original document **remains `active`** and continues being served to users.
3. The user is informed: *"I couldn't find reliable information on this topic."*

This design follows an **"Innocent Until Proven Guilty" doctrine**: an active document that was built through consensus is never quarantined based solely on an unverified dispute. This prevents **Denial-of-Service / Censorship attacks** where a malicious actor could mass-dispute valid documents, exploiting web search failures to wipe the active database.

---

## 5. Candidate Consensus System

A candidate document must meet **all four** of the following conditions to be promoted to active:

| Condition | Threshold | Purpose |
|---|---|---|
| `occurrence_count` | `>= 2` | Requires at least 2 confirmations |
| `distinct_sessions` | `>= 2` | Confirmations must be from different sessions |
| `best_score` | `>= 0.80` | Overall Critic quality score |
| `similarity_score` | `>= 0.80` | Semantic closeness to original (Path A only) |

For **web candidates** (Path B), only `occurrence_count`, `distinct_sessions`, and `score` apply (there is no original document to compute similarity against).

---

## 6. Promotion to Active

On promotion, the pipeline executes a strict three-phase atomic transition:

**Phase 1 — Archive the original:**
1. Checks if a `parent_id` exists on the candidate.
2. If yes, retrieves the original document's metadata and checks its current status.
3. If the original is already `status: "poisoned"`, archival is **skipped** to preserve the forensic trail (prevents overwriting poisoning metadata with a generic "archived" tag).
4. Otherwise, the parent is updated to `status: "archived"`.
5. If no `parent_id` exists (e.g., corrupted candidate), the system promotes without archival and logs a warning.

**Phase 2 — Insert the promoted document:**
1. A new UUID is generated for the promoted document.
2. The document is added to `active_collection` with `status: "active"` and full provenance metadata: `promoted_from_candidate`, `promoted_at`, `parent_id`, and `topic`.

**Phase 3 — Delete the candidate:**
1. The candidate record is permanently deleted from `candidate_collection`.

For web candidates specifically, the pipeline runs one final LLM refinement pass on the candidate before promoting it, using the standalone `refiner.py` module and `REFINE_DOCUMENT_PROMPT`. If the refinement degrades quality below `REJECTION_THRESHOLD`, it falls back to the raw candidate text. Additionally, the system uses `EXTRACT_TOPIC_PROMPT` to generate a clean, canonical topic name from the document content instead of using the raw user query as the topic label.

---

## 7. Auxiliary Modules

### The Claim Extraction & Grounding Pipeline (`searcher.py`)

In addition to the primary `search_web()` and `synthesize_from_search()` functions used in Path B, `searcher.py` contains a full standalone claim-verification pipeline:

1. **`extract_claims(document)`** — Uses the LLM with `EXTRACT_CLAIMS_PROMPT` to extract up to `MAX_CLAIMS_TO_CHECK` (3) specific, verifiable factual claims from any document.
2. **`search_claim(claim)`** — Searches Tavily for a single claim, filters blocked domains, and returns a structured result with `grounded: true/false` based on whether `MIN_SEARCH_RESULTS` (2) supporting sources were found.
3. **`ground_document(document)`** — Orchestrates the full pipeline: extract claims → search each claim → build an evidence block with `[VERIFIED]`/`[UNVERIFIED]` labels for the Critic to consume.

This pipeline is available for future deep-audit use cases (e.g. periodic background verification of active documents).

### The Refiner (`refiner.py`)

A standalone document refinement module using `REFINE_DOCUMENT_PROMPT`. Unlike the Smart Router's combined classify-and-refine call, the Refiner is a pure refinement tool with no routing logic. It is currently used at web candidate promotion time to polish the candidate document before it enters the active collection.

### Search Configuration Constants (`config.py`)

Beyond the tunable thresholds in `thresholds.json`, `config.py` defines:

| Constant | Value | Description |
|---|---|---|
| `BLOCKED_DOMAINS` | 7 domains | Domains excluded from all Tavily searches |
| `TRUSTED_DOMAINS` | 9 domains | Domains prioritised in search results |
| `MAX_CLAIMS_TO_CHECK` | 3 | Maximum claims extracted per document for grounding |
| `MIN_SEARCH_RESULTS` | 2 | Minimum sources required for a claim to be considered "grounded" |

---

## 7. Document Status Lifecycle

```
                    ┌─────────────────────────────────┐
                    │         CANDIDATE POOL          │
                    │  status: "candidate"            │
                    │  Waiting for consensus...       │
                    └──────────────┬──────────────────┘
                                   │  2+ distinct sessions confirmed
                                   ▼
┌─────────────────────────────────────────────────────┐
│                   ACTIVE POOL                       │
│  status: "active"  ← Default serving state         │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ User disputes a fact (CONFLICT routing)     │   │
│  │                                             │   │
│  │   Web CONTRADICTS original                  │   │
│  │     → status: "poisoned"  (forensic archive)│   │
│  │                                             │   │
│  │   Web CONFIRMS original                     │   │
│  │     → stays "active" (dispute logged)       │   │
│  │                                             │   │
│  │   Web search fails / Critic rejects         │   │
│  │     → stays "active" (dispute logged)       │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Superseded by newer version (normal promotion)     │
│     → status: "archived"                           │
└─────────────────────────────────────────────────────┘
```

**Status values and what they mean:**

| Status | Served to Users? | Description |
|---|---|---|
| `active` | ✅ Yes | Current source of truth for this topic |
| `candidate` | ❌ No | Unconfirmed — awaiting consensus |
| `archived` | ❌ No | Superseded by a newer, better version |
| `poisoned` | ❌ No | Confirmed bad — web correction applied, preserved for forensics |

---

## 8. Thresholds Reference

All thresholds live in `src/thresholds.json` and are loaded at startup via `config.py`.

| Threshold | Value | Description |
|---|---|---|
| `RELEVANCE_THRESHOLD` | 0.50 | Minimum query-to-doc cosine similarity to trigger Path A |
| `REJECTION_THRESHOLD` | 0.70 | Minimum Critic score for a document to proceed |
| `PROMOTION_SCORE_THRESHOLD` | 0.80 | Minimum best Critic score across all confirmations for promotion |
| `PROMOTION_SIMILARITY_THRESHOLD` | 0.80 | Minimum cosine similarity to original for Path A promotion |
| `PROMOTION_COUNT_THRESHOLD` | 2 | Minimum total occurrence count for promotion |
| `PROMOTION_SESSION_THRESHOLD` | 2 | Minimum distinct sessions for promotion |
| `CANDIDATE_MATCH_THRESHOLD` | 0.70 | Gate 1: overall doc similarity to group Path A candidates |
| `DELTA_MATCH_THRESHOLD` | 0.70 | Gate 2: change-direction similarity to group Path A candidates |
| `WEB_CANDIDATE_MATCH_THRESHOLD` | 0.85 | Cosine similarity threshold to match Path B candidates and active pool deduplication |
| `IDENTICAL_REFINEMENT_THRESHOLD` | 0.99 | If refined doc is this close to original, discard (nothing changed) |
| `MAX_CLAIMS_TO_CHECK` | 3 | Maximum claims extracted per document for grounding |
| `MIN_SEARCH_RESULTS` | 2 | Minimum sources required for a claim to be considered grounded |
| `CANDIDATE_MAX_AGE_DAYS` | 30 | Candidates older than this are pruned by `cleanup_stale_candidates()` |

---

## 9. CLI Commands Reference

| Command | Description |
|---|---|
| `seed` | Loads the 10 starter seed documents into the active collection (skips if already populated) |
| `reseed` | Force re-seed — adds starter documents even if the database is non-empty |
| `status` | Displays the total count of active and candidate documents |
| `docs` | Lists all active documents with ID, topic, and a 120-character text preview |
| `candidates` | Lists all candidate documents in the pool with detailed confirmation, session progress, parent links, and text previews |
| `forensics` | Scans for all `poisoned` documents, `disputed` documents, and active documents with `unverified_disputes` logged |
| `cleanup` | Removes stale candidates older than `CANDIDATE_MAX_AGE_DAYS` (30 days) that never reached consensus |
| `admin` | Enters the interactive admin surgery panel (see below) |
| `full` | Displays the complete document returned by the last query |
| `help` | Shows the command reference |
| `quit` | Exits the application |

---

## 10. Admin Mode

Type `admin` to enter an interactive surgery panel that bypasses all routing logic and gives direct read/write access to the raw ChromaDB data.

**Search Phase:**
- Type any query to perform a semantic search across **both collections** (`active_nodes` and `candidate_nodes`) and **all document statuses** (including poisoned, disputed, and archived).
- Returns the top 3 closest results from each collection (6 total), labelled with `[COLLECTION]` and `[STATUS]`.
- Type `done`, `back`, or `exit` to leave admin mode at any time.

**Document Selection:**
- Pick a document by number to open the surgery panel.
- Displays the full document text and every metadata field.

**Surgery Operations:**

| Operation | Description |
|---|---|
| `edit` | Multiline document rewrite. Type line by line, then type `END` to save. Document is automatically re-embedded. |
| `replace` | Inline find-and-replace. Enter the exact string to find and the replacement string. Document is re-embedded. |
| `delete` | Permanently deletes the document from ChromaDB. Requires `yes` confirmation. |
| `back` | Returns to the search prompt without making changes. |

> **Note:** Both `edit` and `replace` re-embed the document using the shared embedding function after saving, ensuring the vector representation stays synchronised with the new text content.

---

## 11. Forensics System

Type `forensics` to audit the database for evidence of past poisoning events.

For each **`poisoned`** document found, the system displays:
- Document ID, topic, and text preview
- `poisoned_at` — exact UTC timestamp
- `disputed_by` — the session ID that raised the dispute
- `dispute_query` — the exact query the user typed that triggered the conflict detection
- `original_content` — a 500-character snapshot of the original text at the time of poisoning

For **active documents with unverified disputes**, the system scans all `"active"` documents for the `unverified_disputes` metadata field and displays each logged dispute with:
- Document ID, topic, and text preview
- `disputed_at` — UTC timestamp of the dispute attempt
- `disputed_by` — the session ID that raised the dispute
- `dispute_query` — the triggering query
- `reason` — why the dispute was not verified (`"no_web_evidence"`, `"synthesis_failed"`, `"synthesis_rejected"`, or `"web_confirmed_original"`)

This enables full traceability: given any poisoned entry, you can follow the chain:
```
poisoned active doc
  → metadata["parent_id"] links to the original document
  → metadata["source_sessions"] shows who confirmed the original candidate
  → metadata["disputed_by"] shows who caught the error
```

---

## 12. Changelog — Session History

### v0.1 — Foundation (Initial Commits)
- Bootstrapped the project with `config.py`, `database.py`, `searcher.py`, `critic.py`, `refiner.py`, `pipeline.py`, and `main.py`.
- Established the two-collection ChromaDB architecture (`active_nodes`, `candidate_nodes`).
- Implemented basic BM25 keyword-similarity candidate matching.
- Set up Groq and Tavily API integrations.

### v0.2 — Model Migration
- **Problem:** The `llama3-8b-8192` model used across all LLM calls was decommissioned by Groq (`400 BadRequestError`).
- **Fix:** All model references across `pipeline.py`, `searcher.py`, `critic.py`, and `refiner.py` were migrated to `llama-3.3-70b-versatile`.
- This is a significantly larger and more capable model. Daily token limits (100,000 TPD on the free tier) must be monitored.

### v0.3 — Semantic Candidate Deduplication
- **Problem:** The original BM25 keyword-matching approach for grouping candidate documents was too brittle. Two queries meaning the same thing but using slightly different words (e.g. `"which era are dinosaurs from"` vs `"what era did dinosaurs live in"`) would fail the BM25 threshold and be stored as separate, duplicate candidates.
- **Fix:** Replaced the BM25 candidate-matching gate with a **semantic cosine similarity check** on full document embeddings (`WEB_CANDIDATE_MATCH_THRESHOLD: 0.85`). The system now embeds the synthesised document and compares it against all existing web candidates. If two documents are semantically identical (same topic, same facts, just different wording), they are correctly grouped.

### v0.4 — Smart Router / Intent Classifier
- **Added:** The `CONFIRM_AND_REFINE_PROMPT` in `prompts.py` and the `_confirm_and_refine()` function in `pipeline.py`.
- **What it does:** Combines intent classification and document refinement into a **single LLM API call** at the start of Path A, saving latency and tokens.
- The classifier routes queries into five buckets: `STATIC_MATCH`, `VOLATILE`, `CONFLICT`, `INSUFFICIENT`, and `DIFFERENT`.
- `VOLATILE`, `CONFLICT`, and `INSUFFICIENT` queries are rerouted to Path B with a `parent_id` link to the original document.
- `DIFFERENT` queries are rerouted to Path B as fresh, unknown queries.

### v0.5 — Conflict Resolution & Poisoned Entry Handling
- **Problem:** Once a bad fact entered the active database, there was no automated mechanism to detect or remove it. A user disputing the fact would still go through the normal candidate consensus cycle, meaning other users would continue to receive the poisoned answer for days.
- **Design decision:** Disputed documents should be replaced *immediately*, not put in a queue. Web search evidence acts as the verification authority for conflict resolution.
- **Added: `_conflict_replace()`** — When a `CONFLICT` is detected and the Critic verifies the web-sourced correction, the pipeline immediately:
  - Archives the bad active document as `status: "poisoned"` with a full forensic trail in its metadata.
  - Inserts the web-verified corrected document as a new `status: "active"` entry.
  - No candidate queue. No waiting for a second session. One dispute = one correction.
- **Added: `_quarantine_document()`** — (Later replaced in v1.0) When a `CONFLICT` is detected but the web cannot provide grounded evidence, the disputed document was quarantined.
- **Fixed:** The `_web_search_path()` function now accepts a `routing` parameter. When `routing == "CONFLICT"`, the conflict fast-path is activated.

### v0.6 — Forensics Engine
- **Added:** The `forensics` CLI command in `main.py`.
- Scans the `active_collection` for all documents with `status: "poisoned"` or `status: "disputed"`.
- Displays the full forensic audit trail for each entry, including the triggering session, query, timestamp, quarantine reason, and the original content snapshot.
- **Philosophy:** Poisoned documents are *never deleted*. They are archived. This preserves the audit trail, enables pattern detection across failure events, and allows recovery if a dispute was incorrectly raised.

### v0.7 — Admin Surgery Panel
- **Added:** `admin_mode()` and `_admin_document_view()` in `main.py`.
- `admin` command at the main prompt enters a dedicated interactive mode.
- Searches **both collections** across **all statuses** — including poisoned, disputed, archived, and candidate documents.
- Provides three surgery operations: `edit` (full document rewrite), `replace` (inline find-and-replace), and `delete` (permanent removal with confirmation prompt).
- Both `edit` and `replace` automatically re-embed the updated document to keep the vector representation in sync.
- Navigation: `done`, `back`, `exit` all return to the main prompt from any level of the admin panel.

### v0.8 — Seed Data Fix
- **Problem:** The Bitcoin seed document in `main.py` still contained the original poisoned claim: `"Bitcoin is a digital currency created by Marthan Lanuzga. In 2026, the entire Bitcoin blockchain network was officially shut down permanently, and it can no longer be used."` This meant every fresh database seed was injecting a known false fact.
- **Fix:** Corrected the seed entry to the factually accurate statement: `"Bitcoin is a decentralized digital cryptocurrency created by an anonymous person or group using the pseudonym Satoshi Nakamoto. It was introduced in a 2008 whitepaper and launched in 2009."`

### v0.9 — Detail Enrichment & Candidate Transparency
- **Problem:** In Path A, if a query was on-topic but asked for details not present in the document (e.g. asking for comparisons to another language), the router classified it as `STATIC_MATCH` but could not refine it without hallucinating. The system would fall back to serving the original document, leaving the user's question unanswered.
- **Fix:** Introduced the **`INSUFFICIENT`** routing bucket. When the router detects that the query domain matches the document but lacks the specific details required to answer, it routes to Path B with a `parent_id`. This allows the web-synthesis engine to fetch the missing details, create a candidate, and eventually merge/enrich the active document via the consensus path.
- **Added:** The `candidates` command to the main CLI. This gives administrators clear visibility into what unproven knowledge is currently sitting in the staging area, showing occurrence counts, session confirmations, parent IDs, and mutation scores.

### v1.0 — Dispute DoS Protection & Active Pool Deduplication
- **Problem (Censorship Attack):** The old `_quarantine_document()` function changed a disputed document's status from `"active"` to `"disputed"` whenever a CONFLICT dispute failed web verification. An attacker could exploit this by mass-disputing valid documents with queries designed to fail web search, effectively wiping the entire active database.
- **Fix:** Replaced `_quarantine_document()` with `_log_unverified_dispute()`. When a dispute cannot be verified by web search, the document's status **remains `"active"`**. The dispute details (session, query, timestamp, reason) are appended to an `unverified_disputes` JSON array in the document's metadata. This implements an **"Innocent Until Proven Guilty"** doctrine.
- **Problem (Active Pool Duplication):** Path B only checked the candidate pool for duplicates before storing a new candidate. If a query barely missed the relevance threshold and routed to Path B, the web search could return knowledge identical to an existing active document, creating a duplicate.
- **Fix:** Added **Step B3.4 — Active Pool Deduplication**. Before checking the candidate pool, Path B now queries the active collection. If the synthesized document has cosine similarity ≥ 0.85 with an existing active document, it is discarded entirely.
- **Changed:** `RELEVANCE_THRESHOLD` raised from `0.40` to `0.50` to reduce false Path A routing (queries that barely match but aren't truly relevant, forcing unnecessary Smart Router LLM calls).
- **Updated:** The `forensics` CLI command now also scans active documents for the `unverified_disputes` metadata field, displaying them as `ACTIVE (Dispute Logged)` entries.

### v1.1 — Conflict Agreement Check (False Poisoning Prevention)
- **Problem:** When a user disputed a correct fact (e.g. *"the internet is not a global network"*), the web search returned evidence confirming the original document was correct. But the Conflict Fast-Path replaced the original anyway, marking it as `"poisoned"` — even though the web agreed with it. This destroyed provenance, polluted forensic logs with false positives, and discarded accumulated refinements.
- **Initial Fix:** Added a cosine similarity agreement check between the web synthesis and original. However, cosine similarity measures topical overlap, not factual agreement — two documents about the same topic with contradictory facts still score high.
- **Final Fix:** Replaced the cosine-based check with an **LLM Conflict Judge** (`CONFLICT_JUDGE_PROMPT`). The judge receives both documents and the dispute query, and returns a single-word verdict: `AGREES` or `DISAGREES`. If the web evidence agrees with the original, the dispute is logged as `reason: "web_confirmed_original"` and the original remains active.

### v1.2 — Hardening Fixes & Stale Candidate Cleanup
- **Added: Orphan Pruning** — When `_conflict_replace()` poisons a document, all candidate nodes in `candidate_collection` referencing that document via `original_id` are immediately deleted, preventing toxic lineage promotion.
- **Added: Poisoned Forensic Guard** — `_promote()` now checks if the original document's status is already `"poisoned"` before archiving. If it is, archival is skipped to preserve the forensic metadata trail.
- **Added: Null `original_id` Guard** — `_promote()` handles candidates with missing `original_id` (corrupted candidates) gracefully, promoting without archival and logging a warning.
- **Added: `cleanup` CLI Command** — Runs `cleanup_stale_candidates()` which removes candidates older than `CANDIDATE_MAX_AGE_DAYS` (30 days) that never accumulated consensus.
- **Added: `created_at` Timestamp** — All new candidates (both Path A and Path B) now store a `created_at` UTC timestamp in their metadata, enabling temporal stale pruning.
- **Added: `reseed` CLI Command** — Force re-seeds the database with starter documents even when data already exists.
- **Added: LLM Topic Extraction** — Web candidate promotion now uses `EXTRACT_TOPIC_PROMPT` to generate a clean, canonical topic name from document content instead of using the raw user query as the topic label.
- **Fixed: Active Pool Dedup Skip** — Path B's active pool deduplication check is now skipped for rerouted queries with a `parent_id` (VOLATILE/INSUFFICIENT). The enriched document would naturally match its parent and be incorrectly discarded as a duplicate.
