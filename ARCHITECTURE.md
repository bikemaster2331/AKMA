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
│   ├── pipeline.py      # Core AKM routing logic, all paths, conflict & quarantine handlers
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
4. The score is compared against `RELEVANCE_THRESHOLD` (0.4):

```
Query similarity >= 0.4  →  PATH A (Refinement)
Query similarity <  0.4  →  PATH B (Web Search)
```

---

### Path A: Refinement (Known Topic)

Triggered when the query is deemed relevant to an existing active document.

**Step A1 — Smart Router (Intent Classifier)**

A single LLM call (`_confirm_and_refine`) performs two jobs simultaneously:
1. Classifies the user's intent into one of **four buckets**
2. If `STATIC_MATCH`, produces the refined document in the same response

| Classification | Meaning | Action |
|---|---|---|
| `STATIC_MATCH` | Same topic, stable facts | Continue refinement in Path A |
| `VOLATILE` | Same topic, time-sensitive facts (prices, current events) | Reroute to Path B with `parent_id` |
| `CONFLICT` | User is explicitly disputing a stored fact | Reroute to **Conflict Fast-Path** with `parent_id` |
| `DIFFERENT` | Vector DB false positive — actually a different topic | Reroute to Path B as a fresh query |

**Step A2 — Critic Scoring**

The Critic LLM reviews the refined document against the original. It checks:
- Were all original facts preserved?
- Are any facts softened, altered, or removed?
- Are additions coherent and logically consistent?

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

Tavily API is queried with the user's raw question. Returns up to 5 source URLs and their content. Blocked domains (Wikipedia, Reddit, Quora, Medium, social media) are filtered out.

**Step B2 — Synthesis**

The LLM reads the web sources and synthesises a clean, structured document answering the query. The synthesis is grounded strictly in the returned sources.

**Step B3 — Critic Scoring (Synthesis)**

The Critic scores the synthesised document against the raw web sources:
- Does the document cover key facts from the sources?
- Does the document introduce any fabricated claims not in the sources?
- Is the document internally consistent?

Score must meet `REJECTION_THRESHOLD` (0.70) to proceed.

**Step B3.5 — Semantic Candidate Deduplication**

Before storing a new candidate, the system embeds the new document and checks it against all existing web-search candidates using cosine similarity (`>= WEB_CANDIDATE_MATCH_THRESHOLD: 0.85`).

- **Match found:** The new document is discarded. The existing candidate's `occurrence_count` and `source_sessions` are updated with the current session.
- **No match:** A new candidate is stored.

This step replaces the old brittle BM25 keyword-matching approach, which failed when the same question was asked with slightly different wording.

**Step B4 — Store as Candidate**

The synthesised document is stored in `candidate_collection` with:
- `status: "candidate"`
- `occurrence_count: 1`
- `source_sessions: [current_session_id]`

---

### The Conflict Fast-Path

When the Smart Router classifies a query as `CONFLICT`, the pipeline takes a special route that bypasses the candidate consensus queue entirely.

**Scenario A — Web Evidence Found & Critic Passes (`_conflict_replace`):**

1. The poisoned active document is immediately updated with `status: "poisoned"`.
2. A full forensic trail is embedded in its metadata: who disputed it, when, the exact dispute query, and a 500-character snapshot of the original content.
3. A new document, built from verified web sources, is inserted as `status: "active"` with a `parent_id` linking back to the poisoned entry.
4. The user gets the corrected answer immediately. No consensus delay.

**Scenario B — Web Search Fails or Critic Rejects (`_quarantine_document`):**

If the web cannot provide grounded evidence to confirm or deny the disputed fact:
1. The disputed active document is updated with `status: "disputed"`.
2. Its `quarantine_reason` metadata records why (e.g. `"no_web_evidence"`, `"synthesis_rejected"`).
3. The document is immediately invisible to all future queries (the active collection filter only serves `status: "active"`).
4. The user is informed that no reliable information could be found.

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

When the final confirming session arrives:
1. The candidate is promoted and removed from `candidate_collection`.
2. The new active document is inserted with full provenance metadata.
3. If the candidate was created via a `CONFLICT` or `VOLATILE` reroute, the `parent_id` is used to archive the old active document.

---

## 6. Promotion to Active

On promotion, the pipeline:

1. Checks if a `parent_id` exists on the candidate (i.e., this knowledge is a correction/update of an older document).
2. If yes, the parent is updated to `status: "archived"`.
3. The new document is added to `active_collection` with `status: "active"` and provenance fields like `promoted_from_candidate`, `promoted_at`, and `parent_id`.
4. The candidate is deleted from `candidate_collection`.

For web candidates specifically, the pipeline also runs one final LLM refinement pass on the candidate before promoting it, to ensure the document is well-integrated. If the refinement degrades quality below `REJECTION_THRESHOLD`, it falls back to the raw candidate text.

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
│  │   Web confirms correction                   │   │
│  │     → status: "poisoned"  (forensic archive)│   │
│  │                                             │   │
│  │   Web cannot confirm correction             │   │
│  │     → status: "disputed"  (quarantine)      │   │
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
| `disputed` | ❌ No | Under dispute, web could not resolve — quarantined |
| `poisoned` | ❌ No | Confirmed bad — web correction applied, preserved for forensics |

---

## 8. Thresholds Reference

All thresholds live in `src/thresholds.json` and are loaded at startup via `config.py`.

| Threshold | Value | Description |
|---|---|---|
| `RELEVANCE_THRESHOLD` | 0.4 | Minimum query-to-doc cosine similarity to trigger Path A |
| `REJECTION_THRESHOLD` | 0.70 | Minimum Critic score for a document to proceed |
| `PROMOTION_SCORE_THRESHOLD` | 0.80 | Minimum best Critic score across all confirmations for promotion |
| `PROMOTION_SIMILARITY_THRESHOLD` | 0.80 | Minimum cosine similarity to original for Path A promotion |
| `PROMOTION_COUNT_THRESHOLD` | 2 | Minimum total occurrence count for promotion |
| `PROMOTION_SESSION_THRESHOLD` | 2 | Minimum distinct sessions for promotion |
| `CANDIDATE_MATCH_THRESHOLD` | 0.70 | Gate 1: overall doc similarity to group Path A candidates |
| `DELTA_MATCH_THRESHOLD` | 0.70 | Gate 2: change-direction similarity to group Path A candidates |
| `WEB_CANDIDATE_MATCH_THRESHOLD` | 0.85 | Cosine similarity threshold to match Path B candidates |
| `IDENTICAL_REFINEMENT_THRESHOLD` | 0.99 | If refined doc is this close to original, discard (nothing changed) |
| `BM25_QUERY_THRESHOLD` | 0.85 | Legacy BM25 query threshold (no longer used in routing) |

---

## 9. CLI Commands Reference

| Command | Description |
|---|---|
| `seed` | Loads the 10 starter seed documents into the active collection (skips if already populated) |
| `status` | Displays the total count of active and candidate documents |
| `docs` | Lists all active documents with ID, topic, and a 120-character text preview |
| `forensics` | Scans the active collection for all `poisoned` and `disputed` documents with full forensic metadata |
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

For each **`disputed`** document found, the system displays:
- Document ID, topic, and text preview
- `disputed_at` — exact UTC timestamp
- `disputed_by` — the session ID that raised the dispute
- `dispute_query` — the triggering query
- `quarantine_reason` — why the document could not be corrected (`"no_web_evidence"`, `"synthesis_failed"`, or `"synthesis_rejected"`)

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
- The classifier routes queries into four buckets: `STATIC_MATCH`, `VOLATILE`, `CONFLICT`, and `DIFFERENT`.
- `VOLATILE` and `CONFLICT` queries are rerouted to Path B with a `parent_id` link to the original document for forensic chaining.
- `DIFFERENT` queries are rerouted to Path B as fresh, unknown queries.

### v0.5 — Conflict Resolution & Poisoned Entry Handling
- **Problem:** Once a bad fact entered the active database, there was no automated mechanism to detect or remove it. A user disputing the fact would still go through the normal candidate consensus cycle, meaning other users would continue to receive the poisoned answer for days.
- **Design decision:** Disputed documents should be replaced *immediately*, not put in a queue. Web search evidence acts as the verification authority for conflict resolution.
- **Added: `_conflict_replace()`** — When a `CONFLICT` is detected and the Critic verifies the web-sourced correction, the pipeline immediately:
  - Archives the bad active document as `status: "poisoned"` with a full forensic trail in its metadata.
  - Inserts the web-verified corrected document as a new `status: "active"` entry.
  - No candidate queue. No waiting for a second session. One dispute = one correction.
- **Added: `_quarantine_document()`** — When a `CONFLICT` is detected but the web cannot provide grounded evidence (no results, synthesis failure, or Critic rejection), the disputed document is set to `status: "disputed"` and immediately stops being served.
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
