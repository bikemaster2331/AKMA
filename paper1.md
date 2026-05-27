# Part 1: Atomic Promotion, Response Delivery, Algorithm Block

---

## \subsubsection{Conflict Fast-Path and Self-Healing}

When a user explicitly disputes a stored fact, AKMA activates a Conflict Fast-Path that bypasses the normal multi-session consensus queue and attempts immediate web arbitration. The objective is to neutralise provably poisoned entries quickly while preserving a complete forensic trail.

- \textbf{LLM Conflict Judge:} The system synthesises web evidence for the disputed query and invokes a deterministic LLM judge (\texttt{CONFLICT\_JUDGE\_PROMPT}) that compares the original active document and the web-synthesised document. The judge answers whether the web evidence \textbf{AGREES} or \textbf{DISAGREES} with the original on the specific disputed point.
- \textbf{Immediate Replacement (DISAGREES):} If the judge finds that the web evidence contradicts the active document, the pipeline executes an immediate replacement via \texttt{_conflict\_replace()}:
    - The original active document is updated to \texttt{status: "poisoned"} with a forensic payload (\texttt{poisoned\_at}, \texttt{disputed\_by}, \texttt{dispute\_query}, and a snapshot \texttt{original\_content}).
    - Any candidate nodes that reference the poisoned entry as their \texttt{original\_id} are pruned (orphan cleanup) to prevent toxic lineage promotion.
    - A corrected document, synthesised from web evidence and scored by the Critic, is inserted immediately as a new \texttt{status: "active"} entry with provenance metadata (\texttt{corrected\_at}, \texttt{corrected\_by}, \texttt{parent\_id}).

- \textbf{Confirmed Original (AGREES):} If the judge finds the web evidence supports the original, the original document remains \texttt{active} and the dispute is recorded in the document's \texttt{unverified\_disputes} metadata (reason: \texttt{"web\_confirmed\_original"}). The user still receives the web-synthesised explanation, but no archival or promotion occurs.

- \textbf{Unverified Disputes and DoS Protection:} If web search fails, synthesis fails, or the Critic rejects the correction, the dispute is logged (\texttt{_log\_unverified\_dispute()}) and the original remains \texttt{active}. This "innocent-until-proven-guilty" behaviour prevents an attacker from mass-disputing valid documents to induce censorship or denial-of-service on the active pool.

Together, these steps provide fast, auditable remediation: provably-bad entries are replaced immediately and preserved for forensic analysis, while unverifiable disputes are captured without disabling normal service.

---

## \subsubsection{Atomic Promotion and Knowledge Persistence}

When a candidate satisfies the consensus gates it enters a strictly atomic promotion sequence designed to preserve provenance and prevent partial state transitions. The system performs three atomic phases:

1. \textbf{Archive the original:} If the candidate references a parent (`parent_id`), the parent is retrieved and its metadata updated to \texttt{status: "archived"} (unless it is already \texttt{poisoned}, in which case archival is skipped to preserve forensic metadata). Archival preserves the full text and provenance but removes the document from normal serving.

2. \textbf{Insert the promoted document:} A new UUID is generated and the promoted document is added to the \texttt{active\_nodes} collection with \texttt{status: "active"} and rich provenance fields: \texttt{promoted\_from\_candidate}, \texttt{promoted\_at} (UTC), \texttt{parent\_id}, \texttt{topic}, and scoring metadata.

3. \textbf{Delete the candidate:} The promoted candidate is deleted from \texttt{candidate\_nodes}, removing it from the staging pool and preventing double-promotion.

This three-phase process enforces an \textbf{append-only mutation history}: every promotion produces a new, independently addressable node and the full evolution of any knowledge item can be reconstructed by following the \texttt{parent\_id} chain.

Final verification at promotion time: before committing the new active node, the system performs a fresh, targeted web verification of the candidate to ensure evidence freshness. If a parent exists the system extracts the delta (new claims only) and verifies those claims against the live web; if no parent exists it grounds the entire candidate document. Any unverified claim blocks promotion. This re-check protects against stale or incomplete synthesis even for candidates originally produced from Tavily.

Edge cases and safeguards:
- If the candidate contains no \texttt{original\_id} (corrupted or orphaned candidate), the system can promote without archival while logging a warning and reduced provenance.
- If the parent has \texttt{status: "poisoned"}, archival is skipped to preserve forensic traces.

---

## \subsubsection{Response Delivery and Knowledge State Resolution}

AKMA resolves each user query into one of four response states; which state is returned depends on routing, Critic scoring, immediate verification, and whether a promotion occurred:

\begin{enumerate}
        \item \textbf{Original Node Fallback:} When the Smart Router finds the active document relevant but the refinement fails (low Critic score), or an immediate delta check fails, the system returns the original active document. The Summary Layer then produces a grounded 2–3 sentence answer strictly from that document.

        \item \textbf{Validated Refinement Delivery:} If the Smart Router classifies the query as \texttt{STATIC\_MATCH}, the Critic approves the refinement, and the immediate delta verification (if any new claims were added) passes, the refined document is returned immediately to the user. The refinement is also staged as a candidate for later consensus — the user sees the up-to-date version while the system asynchronously processes promotion eligibility.

        \item \textbf{Promoted Mutation Delivery:} When a candidate completes the consensus gates and the final lazy verification at promotion time succeeds, the promoted document becomes the authoritative active node and is returned as the response.

        \item \textbf{Graceful Failure:} If no relevant active document exists and Path B web synthesis fails (no sources, synthesis rejected), the system returns a structured refusal indicating that reliable information could not be found. No speculative content is promoted.
\end{enumerate}

	extbf{Summary Layer and grounding fail-safe:} Every response is post-processed by the Summary Layer using \texttt{SUMMARIZE\_FOR\_QUERY\_PROMPT}, which constrains the LLM to answer using only the provided document. This final gate prevents user-facing hallucination: when evidence is absent or insufficient the model returns a refusal rather than inventing facts.

Operational note: the immediate delta verification executed during Path A is a fail-closed gate — if extraction/search/judgement fails or returns unverified claims, the refinement is rejected and the original document is returned. The lazy verification at promotion time is similarly conservative: any unverified claim blocks promotion. These layered checks balance responsiveness for single users with rigorous safety for the shared active store.

---

## \subsection{AKMA Operational Algorithm}

The following pseudocode formalizes the complete end-to-end operational flow of the AKMA pipeline. Each pseudocode function maps directly to an implemented function in \texttt{src/pipeline.py}.

\begin{algorithm}
\caption{AKMA Pipeline — End-to-End Knowledge Query and Mutation}
\begin{algorithmic}[1]

\Require user\_query $q$, session\_id $s$, active collection $\mathcal{A}$, candidate collection $\mathcal{C}$
\Ensure response $r$, updated $\mathcal{A}$, updated $\mathcal{C}$

\State $\vec{q} \leftarrow \texttt{embed}(q)$
\Comment{Encode query to 384-dim vector via DefaultEmbeddingFunction}

\State $(D, \vec{d}, sim) \leftarrow \texttt{retrieve}(\vec{q}, \mathcal{A})$
\Comment{Nearest-neighbour cosine search over active pool}

\If{$sim \ge \theta_{rel}$}
    \Comment{Path A: relevant document found ($\theta_{rel} = 0.50$)}
    \State $(routing, D') \leftarrow \texttt{classify\_and\_refine}(q, D)$
    \Comment{\texttt{\_confirm\_and\_refine()} — single LLM call}

    \If{$routing = \texttt{CONFLICT}$}
        \State \Return $\texttt{conflict\_fast\_path}(q, D, s)$
        \Comment{Bypass consensus; immediate web arbitration}
    \EndIf

    \If{$routing \ne \texttt{STATIC\_MATCH}$}
        \State \Return $\texttt{web\_search\_path}(q, routing, s)$
        \Comment{VOLATILE / INSUFFICIENT / DIFFERENT $\to$ Path B}
    \EndIf

    \State $score_{ref} \leftarrow \texttt{validate}(D, D')$
    \Comment{\texttt{score\_mutation()} — Critic Mode A: refinement}

    \If{$score_{ref} < \theta_{rej}$}
        \State \Return $\texttt{summarize}(q, D)$
        \Comment{Critic rejected — fall back to original}
    \EndIf

    \State $sim_{id} \leftarrow \texttt{cosine}(\vec{d}, \texttt{embed}(D'))$
    \If{$sim_{id} \ge \theta_{id}$}
        \State \Return $\texttt{summarize}(q, D)$
        \Comment{Identical refinement guard ($\theta_{id} = 0.99$)}
    \EndIf

    \State $result \leftarrow \texttt{stage\_candidate}(D', D, score_{ref}, s, \mathcal{C})$
    \Comment{Dual-gate matching; \texttt{\_increment\_and\_check()} or \texttt{\_insert\_candidate()}}

    \If{$result = \texttt{PROMOTED}$}
        \State $\texttt{promote}(D', D, \mathcal{A}, \mathcal{C})$
        \Comment{\texttt{\_promote()}: archive original, insert new active, delete candidate}
    \EndIf

\Else
    \Comment{Path B: no relevant document found}
    \State $result \leftarrow \texttt{web\_search\_path}(q, \texttt{NEW}, s)$
\EndIf

\State \Return $\texttt{summarize}(q, result)$
\Comment{\texttt{summarize\_for\_query()} — grounded 2-3 sentence answer}

\end{algorithmic}
\end{algorithm}

\textbf{Function-to-Implementation Mapping:}

\begin{tabular}{ll}
\textbf{Pseudocode Function} & \textbf{Implemented In} \\
\hline
\texttt{embed()} & \texttt{database.py} $\to$ \texttt{get\_embedding()} \\
\texttt{retrieve()} & \texttt{pipeline.py} $\to$ \texttt{run\_akm()} retrieval block \\
\texttt{classify\_and\_refine()} & \texttt{pipeline.py} $\to$ \texttt{\_confirm\_and\_refine()} \\
\texttt{validate()} & \texttt{critic.py} $\to$ \texttt{score\_mutation()} \\
\texttt{stage\_candidate()} & \texttt{pipeline.py} $\to$ \texttt{\_increment\_and\_check()} / \texttt{\_insert\_candidate()} \\
\texttt{promote()} & \texttt{pipeline.py} $\to$ \texttt{\_promote()} \\
\texttt{web\_search\_path()} & \texttt{pipeline.py} $\to$ \texttt{\_web\_search\_path()} \\
\texttt{conflict\_fast\_path()} & \texttt{pipeline.py} $\to$ \texttt{\_conflict\_replace()} / \texttt{\_log\_unverified\_dispute()} \\
\texttt{summarize()} & \texttt{searcher.py} $\to$ \texttt{summarize\_for\_query()} \\
\end{tabular}
