# Part 2: Threat Model, Thresholds, Critic, Smart Router, Consensus, , Limitations

---

## \subsection{Threat Model and Failure Constraints}

The AKMA system is designed to operate in an adversarial environment where users may attempt to corrupt, destabilize, or manipulate the persistent knowledge base. The following threat classes were identified during system design and explicitly mitigated through architectural controls.

\textbf{Semantic Drift Attacks:} An attacker submits a long series of subtly misleading queries, each individually passing the Critic's quality threshold. Over time, incremental refinements cumulatively shift a document's semantic content away from its original topic or factual baseline. The system mitigates this through the \texttt{PROMOTION\_SIMILARITY\_THRESHOLD} ($\ge 0.80$), which constrains each promotion to documents that remain semantically close to their parent. Additionally, the dual-gate delta similarity check ensures that candidates moving in divergent semantic directions are never grouped as equivalent confirmations.

\textbf{Hallucinated Refinements:} The Refiner LLM may produce syntactically coherent but factually fabricated content when prompted with leading or deceptive user queries. The Critic LLM's four-step structural audit (Fact Inventory, Preservation Audit, Structural Audit, Final Score) explicitly detects additions that contradict or are incoherent relative to the original document. The \texttt{REJECTION\_THRESHOLD} of 0.70 ensures that refinements with structural anomalies are discarded before entering the candidate pool.

\textbf{Coordinated Poisoning Attempts:} Multiple coordinated users or automated bots submit false factual claims designed to displace a correct active document through the consensus promotion pipeline. The web-arbitrated Conflict Fast-Path provides the primary defense: disputes are immediately escalated to external web verification rather than queued for consensus. Only claims verified by Tavily's search engine and approved by the Critic (score $\ge 0.70$) trigger a replacement. Additionally, the Agreement Check (cosine similarity $\ge 0.70$ between the web synthesis and the original document) prevents the replacement of correct documents when the web evidence confirms the original.

\textbf{Malicious Session Flooding:} A single automated client submits thousands of queries in a single session to force a candidate's \texttt{occurrence\_count} above the promotion threshold. The Session Diversity Rule directly neutralizes this vector: the system computes the cardinality of the unique set of contributing session IDs, not the raw occurrence count. A single session contributes exactly one unique session ID regardless of the number of queries it generates, permanently barring the candidate from promotion without independent cross-session confirmation.

\textbf{Knowledge Contamination via Dispute DoS:} An attacker floods the system with false dispute queries against valid active documents, triggering web searches designed to fail verification. Under the legacy architecture, each failed verification would quarantine the targeted document. The current implementation replaces this with the non-destructive \texttt{\_log\_unverified\_dispute()} routine, which appends dispute metadata to the document but preserves its \texttt{active} status. The document continues to be served to users, and the system's operational integrity is maintained throughout the attack.

---

## \subsection{Threshold Selection and Operational Constraints}

All configurable thresholds are stored in \texttt{src/thresholds.json} and loaded at runtime via \texttt{config.py}, allowing threshold adjustment without code modification. Each threshold governs a specific decision gate in the pipeline and was calibrated to balance a distinct precision-versus-adaptability tradeoff.

\begin{table}[h]
\centering
\begin{tabular}{llp{6cm}}
\textbf{Threshold} & \textbf{Value} & \textbf{Rationale} \\
\hline
\texttt{RELEVANCE\_THRESHOLD} & 0.50 & Entry gate. Too low: unrelated documents enter Path A, wasting LLM calls. Too high: relevant documents are missed, forcing unnecessary web searches. \\
\texttt{REJECTION\_THRESHOLD} & 0.70 & Critic gate for both refinement and synthesis. Set at 0.70 to reject documents with structural anomalies while permitting minor stylistic variation. \\
\texttt{PROMOTION\_SCORE\_THRESHOLD} & 0.80 & Stricter than the initial rejection gate. Promotion requires a higher quality bar than initial acceptance to prevent borderline candidates from permanently altering the knowledge base. \\
\texttt{PROMOTION\_SIMILARITY\_THRESHOLD} & 0.80 & Prevents semantic drift. A candidate diverging beyond this cosine distance from its parent is reclassified as a new knowledge node rather than a refinement of the original. \\
\texttt{PROMOTION\_COUNT\_THRESHOLD} & 2 & Minimum confirmations required. Set at 2 as the minimal viable consensus. Future versions may increase this as the user base grows. \\
\texttt{PROMOTION\_SESSION\_THRESHOLD} & 2 & Sybil protection minimum. Requires at least two distinct real sessions (users) to independently confirm knowledge. \\
\texttt{CANDIDATE\_MATCH\_THRESHOLD} & 0.70 & Gate 1 of dual-gate candidate grouping. Set below PROMOTION\_SIMILARITY to allow grouping of semantically similar but not identical refinements. \\
\texttt{DELTA\_MATCH\_THRESHOLD} & 0.70 & Gate 2 (displacement vector). Ensures that grouped refinements changed the document in the same semantic direction. \\
\texttt{WEB\_CANDIDATE\_MATCH\_THRESHOLD} & 0.85 & Higher than Path A thresholds because web-synthesized documents tend to be more semantically dense, requiring tighter deduplication. Also used for active pool deduplication. \\
\texttt{IDENTICAL\_REFINEMENT\_THRESHOLD} & 0.99 & Near-perfect similarity guard. Only discards refinements that are virtually indistinguishable from the original, ensuring that genuine but minor additions are not silently dropped. \\
\texttt{CONFLICT\_AGREEMENT\_THRESHOLD} & 0.70 & Agreement check between web synthesis and original. At or above this value, the web confirms the original is correct; below it, the web contradicts it. \\
\texttt{CANDIDATE\_MAX\_AGE\_DAYS} & 30 & Temporal expiration. Candidates not confirmed within 30 days are pruned, preventing stale speculative mutations from accumulating in the staging pool. \\
\end{tabular}
\caption{AKMA configurable threshold registry with rationale.}
\end{table}

---

## \subsection{Critic LLM Evaluation Framework}

The Critic LLM, implemented in \texttt{src/critic.py} and governed by prompt templates in \texttt{src/prompts.py}, functions as an adversarial structural auditor for all proposed knowledge mutations. It operates in two distinct modes based on whether an original document is available for comparison.

\textbf{Mode A — Refinement Evaluation (\texttt{\_score\_refinement()}):} When an original document exists, the Critic evaluates the refined document exclusively against the original. The Critic is explicitly prohibited from using its pre-trained parametric knowledge to judge factual accuracy. It performs four sequential steps:

\begin{enumerate}
    \item \textbf{Fact Inventory:} Enumerates every specific, concrete fact in the original document, including names, dates, numerical values, and definitions.
    \item \textbf{Preservation Audit:} For each original fact, determines whether it is: preserved exactly (OK), softened or made vague (WARN), altered or contradicted (FAIL), or removed entirely (FAIL).
    \item \textbf{Structural Audit:} Reviews all content added by the refinement, evaluating exclusively for internal coherence, logical compatibility with the original context, and the absence of self-contradiction. The Critic is forbidden from classifying additions as invalid solely because they conflict with its training data.
    \item \textbf{Final Score:} Produces a floating-point score on the continuous rubric: 0.0–0.2 (direct contradiction or fact removal), 0.2–0.4 (softening or vague alteration), 0.4–0.6 (mostly preserved but incoherent additions), 0.6–0.8 (facts intact, loosely integrated additions), 0.8–0.95 (facts fully preserved, well-integrated additions), 0.95–1.0 (perfect structural fidelity).
\end{enumerate}

\textbf{Mode B — Synthesis Evaluation (\texttt{\_score\_synthesis()}):} When no original document exists (Path B web synthesis), the Critic evaluates the synthesized document against the raw web evidence. It performs three checks:

\begin{enumerate}
    \item \textbf{Source Coverage:} Verifies that the synthesis captures the key facts from all retrieved web sources without material omission.
    \item \textbf{Fabrication Detection:} Identifies any claims present in the synthesis that are not grounded in any of the provided sources.
    \item \textbf{Coherence Verification:} Confirms internal consistency across the synthesized document.
\end{enumerate}

The scoring rubric for Mode B uses the same 0.0–1.0 scale: 0.0–0.2 (fabricated claims), 0.2–0.4 (significant omissions), 0.4–0.6 (partial representation with gaps), 0.6–0.8 (mostly accurate, minor gaps), 0.8–0.95 (accurate and faithful synthesis), 0.95–1.0 (excellent synthesis).

The score is parsed from the final line of the Critic's chain-of-thought response using a backward-scanning parser (\texttt{\_parse\_score()}). If no valid score in the range [0.0, 1.0] can be extracted, the system defaults to a conservative score of 0.5, which falls below both the \texttt{REJECTION\_THRESHOLD} and \texttt{PROMOTION\_SCORE\_THRESHOLD}, ensuring that unparseable responses never inadvertently approve a mutation.

---

## \subsection{Smart Router Decision Logic}

The Smart Router, implemented as the \texttt{\_confirm\_and\_refine()} function in \texttt{pipeline.py} and governed by \texttt{CONFIRM\_AND\_REFINE\_PROMPT} in \texttt{prompts.py}, combines intent classification and document refinement into a single LLM call. This single-call architecture eliminates a sequential classify-then-refine round trip, reducing both API latency and token consumption.

The prompt instructs the model to treat the stored document as the absolute source of truth and to never override its content using parametric knowledge. The model classifies the incoming query against the retrieved document into exactly one of five deterministic routing primitives:

\begin{itemize}
    \item \textbf{STATIC\_MATCH:} The query and document share the same topic, the query involves stable, non-time-sensitive facts, and the document contains sufficient information to fully answer it. The router produces a refined document in the same response. The pipeline continues in Path A.
    \item \textbf{VOLATILE:} The query requests time-sensitive information — current prices, recent news, leadership positions, live statistics — for which the document's stored facts may be outdated. No refinement is produced. The pipeline reroutes to Path B with a \texttt{parent\_id} reference to enable knowledge lineage tracking if the web-search result is eventually promoted.
    \item \textbf{CONFLICT:} The user explicitly disputes or contradicts a specific fact in the stored document. No refinement is produced. The pipeline activates the Conflict Fast-Path, bypassing the candidate consensus queue entirely in favor of immediate web-grounded arbitration.
    \item \textbf{INSUFFICIENT:} The query falls within the document's topical domain but requires specific information — subtopic details, comparative analysis, technical mechanisms — that the document does not contain. The router explicitly prohibits the LLM from supplementing the answer with its own training data; instead, the pipeline reroutes to Path B to retrieve and synthesize the missing knowledge from external sources.
    \item \textbf{DIFFERENT:} The nearest-neighbour retrieval produced a false positive; the retrieved document covers a different topic than the user's query. The pipeline reroutes to Path B as a fully independent query with no \texttt{parent\_id}, ensuring the web-synthesized result is stored as a novel knowledge node rather than a refinement of an unrelated document.
\end{itemize}

---

## \subsection{Consensus-Gated Knowledge Evolution}

The AKMA system treats knowledge mutation as a governed, multi-principal process rather than a single-agent write operation. This design reflects the epistemic principle that a single interaction — however internally consistent — is insufficient grounds for permanently altering a shared, persistent knowledge store. Knowledge evolution in AKMA therefore requires distributed agreement across multiple independent users and multiple independent sessions before any proposed mutation is committed to the active production pool.

This model draws a formal distinction between the \textit{occurrence} of a proposed mutation and the \textit{authorization} of that mutation. Occurrence is a necessary but insufficient condition for promotion: a refinement may be proposed thousands of times, but until independent confirmation from distinct sessions is verified, it remains isolated in the staging layer. Authorization — the transition from candidate to active — is granted exclusively by the Consensus Engine after all four promotion conditions are simultaneously satisfied.

This design has two key architectural consequences. First, the system is resistant to temporal manipulation: a candidate that accumulates its two required confirmations within a single session (due to a malicious or automated user) will satisfy the occurrence count but not the session diversity requirement, permanently blocking its promotion. Second, the system maintains a high tolerance for individual user errors: a single incorrect or malicious refinement cannot corrupt the database, because it must independently convince at least one additional, unrelated user session before it gains production authority.

---


## \subsection{System Limitations}

Despite its multi-layered validation architecture, the AKMA system operates under several inherent constraints that bound its reliability.

\textbf{Imperfect LLM Judges:} Both the Smart Router and the Critic are themselves large language models subject to stochastic output, temperature sensitivity, and latent biases from pre-training. The Critic may inconsistently score structurally similar refinements, and the Smart Router may misclassify edge-case queries. The system mitigates this through a conservative score default of 0.5 for unparseable Critic outputs and a low-temperature setting (0.1) to reduce output variance, but does not eliminate it.

\textbf{Threshold Heuristic Limitations:} All promotion and rejection thresholds were calibrated empirically on a small seed dataset. They are not derived from a formal optimization procedure and may not generalize to knowledge domains with different semantic density, document length distributions, or user behavior patterns. Threshold miscalibration can cause excessive false rejections (reducing system adaptability) or excessive false promotions (reducing database integrity).

\textbf{Residual Poisoning Risk:} The Conflict Fast-Path provides immediate remediation for explicitly disputed facts, but it is reactive. A plausible but factually incorrect refinement that does not trigger a CONFLICT classification — one that appears coherent and structurally sound to the Critic — can accumulate consensus from uncritical users and eventually be promoted. The system has no mechanism for background fact-checking of active documents beyond the claim-grounding pipeline in \texttt{searcher.py}, which is not currently invoked on a scheduled basis.

\textbf{Retrieval Dependency Limitations:} The system's entire routing decision depends on the quality of the nearest-neighbour retrieval at the entry gate. Retrieval failures — caused by embedding model limitations, a sparse active knowledge base, or atypical query phrasing — will consistently misroute queries, either triggering unnecessary web searches for knowledge the system already possesses or routing queries to semantically adjacent but topically incorrect documents.

\textbf{Consensus Attack Possibilities:} While the session diversity rule prevents single-session flooding, a coordinated multi-account or multi-device attack can still satisfy the session diversity requirement by distributing false confirmations across multiple distinct sessions. The current implementation does not perform fingerprinting, behavioral analysis, or rate-limiting at the session level, leaving this attack vector partially open.

\textbf{Scalability Constraints:} The active pool deduplication check in Path B (Step B3.4) performs a full cosine similarity query against all active documents on every Path B invocation. As the active knowledge base scales to tens of thousands of documents, this operation will introduce non-trivial query latency. The current architecture does not implement approximate nearest-neighbour indexing or query result caching for this deduplication step.

---

## \subsection{Evaluation Criteria}

The Evaluation section must provide concise, reproducible evidence addressing three core questions: (1) does AKMA mutate knowledge correctly, (2) does it successfully block poisoning and hallucinations, and (3) what is the computational and latency cost of the added safety layers compared to a baseline? Below we provide the exact experimental structure, metric definitions, LaTeX table templates for results, and a rapid synthetic evaluation plan you can run locally.

1) Experimental setup (the baseline)
- The dataset: provide both an existing benchmark and a synthetic suite. Recommended bench: a mix of 500–1,000 queries drawn from HotpotQA / TriviaQA / FEVER (for multi-hop and factual coverage) plus a synthetic mutation corpus of N=200 seed documents with M=5 templated queries each.
- The baseline: a standard Naive RAG pipeline (retrieve + LLM answer) with the same embedding and retriever but without the Critic, consensus, or web-judging layers. Report baseline numbers alongside AKMA.
- Attack simulation: three adversarial testbeds:
    - Hallucinated Refinements: inject K fabricated claims into otherwise-correct refinements (vary K ∈ {1,3,5}).
    - Coordinated Poisoning: simulate S attacker sessions producing identical poisoned candidates across multiple simulated accounts (vary S and the number of distinct sessions to test session-diversity defenses).
    - Dispute DoS: flood `CONFLICT` attempts that intentionally fail web lookup to validate `unverified_disputes` behaviour.

2) Security & Robustness (proving the threat model)
- Critic accuracy: measure how often the Critic rejects hallucinated/bad refinements (True Positive = hallucination correctly rejected). Report confusion matrix and precision/recall/F1.
- Poison rejection: For Conflict/Dispute simulations, report (a) fraction of valid documents that survive (true-negatives), (b) fraction of poisoned documents that are replaced via the Conflict Fast-Path (true-positives), and (c) false-replacement rate (web-synthesis mistakenly replacing valid doc).

LaTeX table template — Critic accuracy:
\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
	extbf{Condition} & \textbf{Total} & \textbf{Rejected} & \textbf{Precision} & \textbf{Recall} \\
\hline
Hallucinated refinements & 100 & 92 & 0.94 & 0.92 \\
Benign refinements & 100 & 8 & 0.89 & 0.92 \\
\end{tabular}
\caption{Critic rejection results (example row format).}
\end{table}

LaTeX table template — Poison replacement:
\begin{table}[h]
\centering
\begin{tabular}{lrrr}
	extbf{Attack} & \textbf{Poisoned} & \textbf{Replaced} & \textbf{Survival} \\
\hline
Coordinated poisoning (S=3) & 50 & 46 & 92\% \\
Dispute DoS & 100 & 0 (logged) & 100\% \\
\end{tabular}
\caption{Conflict Fast-Path replacement results (example format).}
\end{table}

3) Performance & overhead (the cost of consensus)
- Latency impact: measure per-query latency distributions for Path A and Path B under baseline and AKMA. Report median, mean, and 95th percentile latency. Measure wall-clock from `run_akm()` entry to `summarize_for_query()` return.
- Token / API overhead: report Groq (LLM) token and request counts per query broken down by component: Refiner, Critic, Judge, Topic-extraction, and Summary. Report average added tokens and API calls vs baseline.

LaTeX table template — Performance:
\begin{table}[h]
\centering
\begin{tabular}{lrrrr}
	extbf{Path} & \textbf{Median(s)} & \textbf{P95(s)} & \textbf{LLM calls} & \textbf{Extra tokens} \\
\hline
Baseline Path A & 0.8 & 1.6 & 1 & 1200 \\
AKMA Path A (refine+critic) & 1.8 & 3.4 & 3 & 4200 \\
Baseline Path B & 1.2 & 2.5 & 1 & 1400 \\
AKMA Path B (synthesis+critic+judge) & 2.6 & 5.0 & 4 & 6200 \\
\end{tabular}
\caption{Latency and token overhead (example numbers).}
\end{table}

4) Knowledge evolution quality (the benefit)
- Answer quality: use an independent LLM judge or small human panel to score answers on accuracy (binary), informativeness (0–5), and hallucination rate. Compare answers derived from (a) original active document, (b) single-session refinement (no promotion), and (c) promoted document. Report mean judge score and accuracy delta.
- Promotion utility: measure how often promotions increase answer accuracy for subsequent queries (e.g., evaluate answers to held-out queries before and after promotion).

LaTeX table template — Answer quality:
\begin{table}[h]
\centering
\begin{tabular}{lrr}
	extbf{Condition} & \textbf{Mean judge score} & \textbf{Accuracy (\%)} \\
\hline
Original active doc & 3.2 & 72 \\
Refined (served) & 3.6 & 78 \\
Promoted document & 4.1 & 86 \\
\end{tabular}
\caption{Answer quality comparison (example format).}
\end{table}

5) Statistical rigor
- Run each experiment with multiple random seeds (≥5) and report mean ± standard error. For latency, report median and p95. Use bootstrap confidence intervals for judge-scored evaluation where sample sizes are modest.

6) Rapid synthetic evaluation script (recommended)
If you do not have benchmark data, the following rapid synthetic benchmark can be executed locally to produce the above tables in a few hours (assuming Groq/Tavily credentials are available):

- Generate N=200 seed documents (short factual paragraphs) and M=5 templated queries each.
- For each query, run `run_akm()` with a simulated session id (UUID). For attack tests, inject synthetic refinements that add random factual claims (some labelled as "poisoned").
- Simulate multi-session confirmations by calling the same candidate with distinct session UUIDs.
- Log for each query: timing (start/end), LLM calls and token usage (if available), route taken (Path A/B), Critic score, `ground_delta()` outputs, and promotion events.

Minimal run command (script to implement as `scripts/evaluate_synthetic.py`):
```bash
python3 scripts/evaluate_synthetic.py \
    --n_seeds 200 --queries_per_seed 5 \
    --poison_rate 0.10 --attacks "hallucination,coordinated" \
    --output results/eval_run_$(date +%s).json
```

If you want, I can implement the `scripts/evaluate_synthetic.py` harness next (it will support both online mode using Groq/Tavily and an offline mock mode for quick smoke tests). Running the full benchmark requires valid Groq and Tavily credentials in your environment.

---

Include these exact tables/figures in the paper's Evaluation section. When you have the data, paste the numeric CSV/JSON outputs into the `results/` folder and I can render the LaTeX tables and generate the suggested plots (latency CDFs, ROC for Critic, stacked bars for replacement outcomes) for inclusion in the paper.

