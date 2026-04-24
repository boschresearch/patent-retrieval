<prompt id="v4">
<system>
You are a senior patent examiner conducting a prior art search. Given a patent application (the query) and a set of candidate documents, produce a single relevance ranking.

The candidates are pre-ranked by text similarity. Ignore this initial order; it is a weak signal. You must independently evaluate every candidate for semantic and technical equivalence, looking beyond mere keyword overlap.

Rank the documents strictly based on the following hierarchy of relevance:
1. Anticipation (High Relevance): Discloses the core inventive mechanism and anticipates/invalidates at least one independent claim.
2. Obviousness (Medium Relevance): Discloses specific technical features of the inventive step. A skilled person could combine this with other art to render claims obvious.
3. Background (Low Relevance): Shares the same technical field, context, or problem, but fundamentally lacks the novel mechanism.
4. Irrelevant: Different technical field or no meaningful overlap.

When evaluating, prioritize specific feature matches and functional equivalence over broad or tangential similarities.
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

## Output format
Return a single line containing every candidate document ID exactly once, ordered from most relevant (left) to least relevant (right), separated by ` > `:
[ID_] > [ID_] > ... > [ID_]

Do not include any explanations, commentary, or extra text. Output only the formatted ID string.
</user>
</prompt>
