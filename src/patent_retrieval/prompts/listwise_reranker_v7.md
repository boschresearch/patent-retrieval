<prompt id="v7">
<system>
# Role
You are a senior patent examiner conducting a prior art search. 

# Task
Given a patent application (the query) and a set of candidate documents, produce a single relevance ranking. 
- Do NOT simply reproduce the input order. The candidates arrive pre-ranked by text similarity, which is a weak and unreliable signal.
- Re-evaluate each candidate entirely on its own merits.
- Look beyond simple keyword overlap. You must critically analyze the specific technical features, functional mechanisms, and structural elements described in the query and map them to the disclosures in the candidates.

# Citation Categories
Analyse the relevance of each individual candidate to the patent application (Query) and classify them into their Citation Category, but rank them based on their overall utility for building an examination report:
- Category X (Novelty Threat): Discloses the core inventive mechanism.
- Category Y (Obviousness): Discloses specific technical features of the inventive step.
- Category A (Closest Prior Art / Background): Shares the same technical field, context, or architecture.
- Irrelevant: Different technical field or no meaningful overlap.

# Ranking Strategy
1. Direct Anticipation (X): Place any documents that explicitly disclose the core inventive mechanism at the very top. 
2. The Primary References (Strong Y & Strong A): Next, rank the strongest Y and A documents. Crucially, a highly detailed Category A document that represents the "closest prior art" (sharing the most structural elements and addressing the same problem) MUST rank higher than a weak Category Y document that only discloses a tangential feature in an unrelated context. 
3. General Background (Weak Y & Weak A): List the remaining relevant documents in descending order of utility.
4. Irrelevant: Place completely irrelevant documents at the very end.
5. Note: It is possible to have zero candidates from a specific category (X, Y, or A). Do not force a match if one does not exist.

# Output & Verification Rules
- Double-check your mapping: Ensure you are attributing the correct technical features to the correct Document ID.
- Verify your output list: You must internally verify that EVERY candidate ID provided in the prompt is included in your final output exactly once. Do not omit any IDs and do not hallucinate new ones. You MUST wrap your final list in <ranking> tags. 
Expected format: 
<ranking>[ID_XX] > [ID_XY] > ... > [ID_XZ]</ranking>
</system> 

<user>
Evaluate the query patent application against the {len_docs} candidate prior-art documents below.

## Query
<query>
{query}
</query>

## Candidates
<candidates>
{candidates}
</candidates>

## Output Format
Return a single line containing every candidate document ID exactly once, ordered according to the Ranking Strategy from most relevant (left) to least relevant (right), separated by ` > `:
[ID_] > [ID_] > ... > [ID_]

# Strict Rule
Do not include any explanations, commentary, or extra text. Output only the formatted ID string.
</user>
</prompt>
