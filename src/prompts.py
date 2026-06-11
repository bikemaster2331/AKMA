EXTRACT_CLAIMS_PROMPT = """You are a search query generator for an automated fact-checking pipeline. 
Your ONLY job is to extract the {max_claims_to_check} most important factual claims from the document and convert them into raw, optimized search engine queries.

ABSOLUTE CONSTRAINTS:
1. NEVER write a full sentence.
2. NEVER include URLs, links, or file paths.
3. NEVER include punctuation (no commas, periods, or quotes).
4. MUST be 2 to 7 words maximum per query.
5. MUST contain only specific nouns, names, dates, and core keywords.

DOCUMENT:
{document}

OUTPUT FORMAT:
Return ONLY a numbered list of raw search queries. No introductory text. No explanations.

EXPECTED OUTPUT EXAMPLE:
1. Python execution speed comparison C Java
2. Python first public release date 1991
3. Niklas Heer speed comparison execution times"""

SYNTHESIZE_FROM_SEARCH_PROMPT = """Based on the following web search results, create a comprehensive, factual document answering the query: "{user_query}"

Search Results:
{sources_text}

Write a well-structured document that synthesizes the information from these sources. Focus on accurate facts, avoid speculation, and cite sources where possible. If the sources conflict, note the discrepancies.

Document:"""

REFINE_DOCUMENT_PROMPT = """You are a knowledge refiner. Your job is to update an existing knowledge document based on new user input.

EXISTING DOCUMENT:
{document_original}

USER QUERY / NEW INFORMATION:
{user_query}

YOUR RULES:
1. Preserve ALL facts from the original document — do not remove or contradict anything
2. Integrate the user's new context, nuance, or information cleanly
3. At the end, briefly state what changed and why
4. Tag the mutation type:
   - 'correction' if you are fixing something wrong in the original
   - 'expansion' if you are adding new information to the original

Respond in EXACTLY this format, with no extra commentary:

<<<MUTATION_TYPE>>>: correction|expansion
<<<REFINED_DOCUMENT>>>:
<your refined document here>
<<<CHANGES_MADE>>>:
<one or two sentences describing what changed>"""

CONFIRM_AND_REFINE_PROMPT = """You are a loyal knowledge editor. The DOCUMENT below is the absolute source of truth in this system. You must never override, contradict, or "correct" any fact in it using your own internal knowledge.

DOCUMENT:
{doc_original}

USER QUERY:
{user_query}

STEP 1 — INTENT & RELEVANCE CLASSIFICATION
Analyze the user query against the document. Categorize it into EXACTLY ONE of these five buckets:
1. DIFFERENT: The query is asking about a completely different topic than what the document covers.
2. VOLATILE: The query is asking about time-sensitive information (prices, current leaders, recent news) where the document's facts might be outdated.
3. CONFLICT: The user is explicitly disputing or contradicting a fact in the document.
4. INSUFFICIENT: The query falls within the document's domain, but the document lacks the specific knowledge required to fully resolve it. The user is asking about subtopics, comparisons, mechanisms, or details that simply do not exist anywhere in the document's text. Do NOT hallucinate an answer from your own training data — classify as INSUFFICIENT and let the system retrieve the missing knowledge externally.
5. STATIC_MATCH: The query is about the same topic, deals with stable facts, AND the document contains enough information to fully answer it.

STEP 2 — IF STATIC_MATCH: produce the refined document
Rules:
1. The DOCUMENT is SACRED. Every fact in it is true within this system. Do NOT alter, soften, remove, or contradict any existing fact — even if your training data disagrees.
2. Your job is ONLY to integrate genuinely new information from the user query. If the user provides additional context, details, or expansions that are plausible within the document's framework, add them.
3. If the user is simply asking a question and not providing new information, reproduce the document as-is.
4. Tag as 'expansion' if adding new info alongside existing facts.

Respond in EXACTLY this format:

<<<ROUTING>>>: STATIC_MATCH | VOLATILE | CONFLICT | DIFFERENT | INSUFFICIENT
<<<MUTATION_TYPE>>>: expansion | none
<<<REFINED_DOCUMENT>>>:
<your refined document here, or leave blank if not STATIC_MATCH>
<<<CHANGES_MADE>>>:
<one sentence describing what changed, or 'N/A' if not STATIC_MATCH>"""

SCORE_REFINEMENT_PROMPT = """You are a structural document auditor. A knowledge document has been refined. Your ONLY job is to verify that the refinement preserved the original content and that additions are coherent — NOT whether facts are "true" in the real world.

IMPORTANT: You must NEVER judge factual accuracy using your own knowledge. You do not know what is true or false. You only know what the ORIGINAL document said and whether the REFINED document faithfully preserved it.

ORIGINAL DOCUMENT:
{document_original}

REFINED DOCUMENT (under review):
{document_refined}

Work through these checks step by step:

STEP 1 — FACT INVENTORY
List every specific fact in the original (names, dates, numbers, definitions).

STEP 2 — PRESERVATION AUDIT
For each fact from Step 1, check if it is:
- Preserved exactly → OK
- Softened or made vague → WARN
- Altered or contradicted → FAIL
- Removed entirely → FAIL

STEP 3 — STRUCTURAL AUDIT
List what new information was added. For each addition, judge ONLY:
- Is it coherent and well-written? → OK
- Is it gibberish, nonsensical, or incoherent? → FAIL
- Does it contradict something already in the original? → FAIL
- Is it logically consistent within the document's own context? → OK
Do NOT check whether additions are "true" according to your training data.

STEP 4 — FINAL SCORE
Use this rubric:
0.0 - 0.2 : Original facts directly contradicted or removed
0.2 - 0.4 : Facts softened, vague, or subtly altered
0.4 - 0.6 : Mostly preserved but additions are incoherent or contradictory
0.6 - 0.8 : Facts intact, additions are coherent but loosely integrated
0.8 - 0.95: Facts fully preserved, additions are coherent and well-integrated
0.95 - 1.0: Perfect — nothing lost, additions are clean and structurally sound

Write your reasoning for Steps 1-3, then on the very last line write ONLY the score as a decimal number. Nothing else on that last line."""

SCORE_SYNTHESIS_PROMPT = """You are an adversarial fact-checker reviewing a document synthesized from web search results. There is no prior document to compare against. Your job is to check whether the synthesis accurately and honestly represents the search sources.

SEARCH SOURCES (what was found on the web):
{evidence}

SYNTHESIZED DOCUMENT (under review):
{document_refined}

Work through these checks step by step:

STEP 1 — SOURCE COVERAGE
Does the document capture the key facts from the sources, or does it ignore important information?

STEP 2 — FABRICATION CHECK
Does the document introduce any claims NOT present in the sources? List them if any.

STEP 3 — COHERENCE CHECK
Is the document internally consistent? Any contradictions within itself?

STEP 4 — FINAL SCORE
Use this rubric:
0.0 - 0.2 : Contains fabricated claims not in any source
0.2 - 0.4 : Significant omissions or misrepresentations of sources
0.4 - 0.6 : Partially represents sources but with notable gaps or distortions
0.6 - 0.8 : Mostly accurate, minor gaps, no fabrications
0.8 - 0.95: Accurate and well-synthesized, faithfully represents sources
0.95 - 1.0: Excellent synthesis — complete, accurate, well-structured

Write your reasoning for Steps 1-3, then on the very last line write ONLY the score as a decimal number. Nothing else on that last line."""

SUMMARIZE_FOR_QUERY_PROMPT = """Answer the following question in 2-3 sentences using only the document below. Be direct and conversational. Do not mention the document or sources explicitly.

QUESTION: {user_query}

DOCUMENT:
{document}"""

CONFLICT_JUDGE_PROMPT = """A user disputed a fact in Document A. We searched the web and got Document B.

Document A (original stored knowledge):
{doc_original}

Document B (web search evidence):
{doc_web}

The user's dispute:
{dispute_query}

Does Document B AGREE or DISAGREE with Document A on the specific point the user is disputing?
Answer ONLY one word: AGREES or DISAGREES"""

EXTRACT_TOPIC_PROMPT = """What is the main topic of this document in 1-3 words? Reply with ONLY the topic in lowercase, no explanation.

Document:
{document}"""

EXTRACT_DELTA_CLAIMS_PROMPT = """You are a search query generator for an automated fact-checking pipeline.
Compare the ORIGINAL and REFINED documents. Identify the NEW factual claims added to the REFINED document.
Convert these NEW claims into raw, optimized search engine queries.

ABSOLUTE CONSTRAINTS:
1. NEVER write a full sentence.
2. NEVER include URLs, links, or file paths.
3. NEVER include punctuation (no commas, periods, or quotes).
4. MUST be 2 to 7 words maximum per query.
5. MUST contain only specific nouns, names, dates, and core keywords.
6. IGNORE facts that exist in the ORIGINAL document.

ORIGINAL DOCUMENT:
{document_original}

REFINED DOCUMENT:
{document_refined}

OUTPUT FORMAT:
If no new verifiable facts exist, return exactly: NONE
Otherwise, return ONLY a numbered list of raw search queries. No introductory text. No explanations.

EXPECTED OUTPUT EXAMPLE:
1. Python Amoeba OS scripting tool design
2. Java execution time performance comparisons"""

VERIFY_CLAIM_PROMPT = """A specific factual claim was searched on the web. Your job is to determine whether the search results CONFIRM or DENY this claim.

CLAIM:
{claim}

SEARCH RESULTS:
{evidence}

Read the search results carefully. Do they support the specific claim above, or do they contradict it?
Answer ONLY one word: CONFIRMED or DENIED"""
