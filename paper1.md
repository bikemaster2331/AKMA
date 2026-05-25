# Part 1: Atomic Promotion, Response Delivery, Algorithm Block

---

## \subsubsection{Atomic Promotion and Knowledge Persistence}

When all consensus promotion requirements are satisfied, the system executes a structured, three-phase transition to transfer the validated knowledge from the staging area to the active production pool. These three operations are performed in strict sequential order, ensuring that the knowledge base never exists in a partially-updated or inconsistent state.

In the first phase, the system retrieves the metadata of the parent document from \texttt{active\_nodes} and updates its \texttt{status} field to \texttt{"archived"}. This preserves the document's full text and provenance within the active collection but renders it permanently invisible to all future user queries, which are filtered exclusively on \texttt{status = "active"}. 

Critically, the document is never physically deleted from the database; it is retained as an immutable historical record. This structural decision serves three vital functions:
\begin{enumerate}
    \item \textbf{Lineage Continuity:} If parent documents were physically purged, the \texttt{parent\_id} references of all subsequent mutations would become dangling pointers, breaking the historical chain of custody and making it impossible to map the epistemic trajectory of a topic back to its seed source.
    \item \textbf{Forensic Auditability:} Retaining historical states allows administrators to conduct forensic post-mortems of the database. It exposes slow-boil semantic drift attacks or coordinated poisoning campaigns that are only visible when analyzing document evolution over many iterations.
    \item \textbf{Instant Metadata Rollback:} Preserving the prior state allows for zero-loss recovery. If a flawed mutation or malicious candidate bypasses the consensus gate and gets promoted, administrators can instantly restore the system to its pre-compromised state by toggling the metadata \texttt{status} of the archived parent back to \texttt{"active"}, avoiding the need for expensive vector reconstructions or manual text restoration.
\end{enumerate}

In the second phase, the system generates a new globally unique identifier (UUID) and inserts the promoted candidate document as a fresh entry in \texttt{active\_nodes} with \texttt{status = "active"}. Along with the document text and its newly generated embedding vector, the system stores a rich provenance payload in the document's metadata, including \texttt{promoted\_from\_candidate} (linking to the deleted candidate record), \texttt{promoted\_at} (the UTC timestamp of the promotion event), \texttt{parent\_id} (linking to the archived predecessor), and the topic classification inherited from the parent document.

In the third and final phase, the candidate record is permanently deleted from \texttt{candidate\_nodes}, releasing its storage and removing it from the active staging pool.

This three-phase architecture enforces a strictly \textbf{append-only mutation history}. Since the original document is archived rather than overwritten, and the promoted document is inserted as a new record rather than replacing the original in-place, every state transition in the knowledge base produces a new, independently addressable document node. The full evolution of any knowledge node — from its initial seed entry, through every archived predecessor, to its current active state — can be reconstructed by traversing the \texttt{parent\_id} chain. This provides complete, tamper-evident lineage for administrative auditing, forensic analysis, and rollback procedures.

---

## \subsubsection{Response Delivery and Knowledge State Resolution}

The AKMA pipeline resolves each user query into exactly one of four possible response states, each reflecting the system's current knowledge confidence level for the queried topic. The governing function \texttt{run\_akm()} returns a document object that is subsequently passed to the Summary Layer for conversational formatting.

\begin{enumerate}
    \item \textbf{Original Node Fallback:} If the retrieved active document is relevant but the Smart Router determines the query adds no new information, or if the Critic rejects the produced refinement, the original active document is returned unchanged. The Summary Layer then generates a grounded, 2-to-3 sentence answer derived exclusively from the unmodified document. This state prioritises stability: the system provides a reliable, validated answer rather than exposing the user to a potentially degraded or hallucinated refinement.

    \item \textbf{Validated Refinement Delivery:} If the Smart Router classifies the query as \texttt{STATIC\_MATCH} and the Critic approves the refinement (score $\ge 0.70$), the refined document is returned. The candidate pipeline runs in parallel — the refinement is staged as a candidate without delaying the user's response. The user receives the most up-to-date, contextually enriched version of the knowledge immediately, while the system asynchronously processes the mutation's eligibility for permanent promotion.

    \item \textbf{Promoted Mutation Delivery:} If the incoming refinement or web synthesis triggers promotion (all consensus thresholds satisfied), the newly promoted document is the authoritative knowledge baseline for this response. The user's query directly precipitated the finalization of a multi-session consensus cycle, and the promoted document is returned and simultaneously committed to the active knowledge base.

    \item \textbf{Graceful Failure:} If no relevant document exists (Path B) and the web search returns no usable sources, or if the Critic rejects all synthesized documents, the pipeline returns a structured refusal. The user receives a clearly communicated message stating that the system could not locate reliable information on the requested topic. No LLM-generated speculation or parametric knowledge is inserted as a substitute.
\end{enumerate}

In all four states, the final output passes through the Summary Layer governed by \texttt{SUMMARIZE\_FOR\_QUERY\_PROMPT}, which strictly constrains the language model to answer exclusively from the provided document. This constraint functions as the system's final fail-safe against hallucination at the user-facing output boundary.

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
