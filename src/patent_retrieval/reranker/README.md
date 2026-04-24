# Reranker

This directory contains reranker implementations for re-scoring retrieved patent candidates. All rerankers implement the `BaseReranker` interface and return `(List[Tuple[candidate_id, score]], error_flag)` sorted by descending relevance.

## Interface

All rerankers share the same input/output contract:

- **Input:** query string + dictionary of `{doc_id: doc_text}` + optional `top_n`
- **Output:** list of `(candidate_id, score)` tuples (higher = more relevant) + boolean error flag

## Implementations

### 1. Pointwise Reranker
**File:** `pointwise_reranker.py`

Pointwise cross-encoder reranker that scores each document independently against the query. Uses the Cohere Rerank API (also compatible with local vLLM servers).

| Parameter | Default | Description |
|---|---|---|
| `base_url` | `None` | API endpoint (e.g. C `http://localhost:80000/` for a local vLLM server) |
| `api_key` | `"EMPTY"` | API key |
| `model_name` | `"rerank-english-v3.0"` | Model identifier |

C
### 2. LLM-Based Listwise Reranker
**File:** `listwise_reranker.py`

Prompts an LLM to rank a list of documents by relevance. Supports OpenAI-compatible and Azure endpoints.

| Parameter | Default | Description |
|---|---|---|
| `base_url` | — | OpenAI-compatible API endpoint |
| `model_name` | — | Model identifier |
| `backend` | `"openai"` | `"openai"` or `"azure"` |
| `api_key` | `"EMPTY"` | API key |
| `thinking` | `True` | Enable LLM thinking/reasoning mode |
| `mode` | `"simple"` | Reranking strategy: `simple`, `sliding_window`, or `tournament` |
| `n` | `20` | Window/group size for sliding_window and tournament |
| `passes` | `1` | Number of iterative passes (sliding_window) |
| `remap_ids` | `True` | Replace patent IDs with random hex aliases to prevent positional bias |
| `prompt_id` | `"v1"` | Prompt template identifier (loaded from `prompts/listwise_reranker_{id}.md`) |

**Reranking modes:**

- **`simple`** — Single LLM call with all documents. Falls back to tournament if input exceeds context length.
- **`tournament`** — Documents are shuffled into groups of size `n`; top half advances each round until a final ranking is produced.
- **`sliding_window`** (experimental) — Ascending-sorted sliding windows of size `n` with 50% overlap, re-ranked from worst to best. Supports multiple passes.
- **`cluster_tournament`** (experimental) — Pre-groups documents by cluster, reranks each independently, merges via round-robin.

### 3. HuggingFace CrossEncoder Reranker
**File:** `hf_reranker.py`

Pointwise reranker using a HuggingFace `CrossEncoder` model. Scores each `(query, document)` pair independently via `model.predict()`.

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"cross-encoder/ms-marco-MiniLM-L6-v2"` | CrossEncoder model ID |
| `device` | `"cpu"` | Compute device (auto-switches to CUDA if available) |

## Running the Reranking Pipeline

The main entry point is `02_reranker_async.py` in the parent directory. 

All parameters below are set in the `cfg` block inside the script.

### Reranker selection

| Option | Default | Description |
|---|---|---|
| `type` | `"listwise"` | Reranker type: `"pointwise"` or `"listwise"` |
| `backend` | `"openai"` | Pointwise: `"cohere"` or `"huggingface"`. Listwise: `"openai"` or `"azure"` |
| `model_name` | — | Model identifier (e.g. `"Qwen/Qwen3.5-397B-A17B-FP8"` for listwise, `"Qwen/Qwen3-Reranker-4B"` for pointwise) |

### Listwise-specific

| Option | Default | Description |
|---|---|---|
| `mode` | `"tournament"` | Reranking strategy: `"simple"`, `"sliding_window"`, or `"tournament"` |
| `thinking` | `True` | Enable LLM thinking/reasoning mode |
| `window_size` | `50` | Window/group size (`n`) for sliding_window and tournament |
| `passes` | `1` | Number of iterative passes (sliding_window / multi-pass RRF) |
| `remap_ids` | `True` | Replace patent IDs with random hex aliases to prevent positional bias |
| `prompt_id` | `"v7"` | Prompt template version (loaded from `prompts/listwise_reranker_{id}.md`) |
| `use_cluster_tournament` | `False` | Enable experimental cluster tournament mode |
| `clusters_path` | `None` | Path to Leiden cluster JSON (required if `use_cluster_tournament=True`) |

### Data & topics

| Option | Default | Description |
|---|---|---|
| `candidates_path` | — | Path to first-stage retrieval results CSV |
| `retrieval_model` | — | Name of the retrieval model (used in run directory naming) |
| `db_path` | — | Path to the patent SQLite database |
| `test_topics_path` | — | Path to the CLEF-IP test topics file |
| `topk` | `100` | Number of candidates to rerank per topic |
| `q` | `500` | Number of topics to evaluate |
| `topics` | `None` | Explicit list of topic IDs (overrides `q`) |

### Query construction

| Option | Default | Description |
|---|---|---|
| `query_columns` | `["abstract","claims"]` | Patent fields used to build the query text |
| `doc_columns` | `["title","abstract","claims"]` | Patent fields used to build candidate document text |
| `claims` | `None` | Claims variant (e.g. prefix for modified claims columns) |
| `independent_claims` | `True` | Use only independent claims |

### Context augmentation

| Option | Default | Description |
|---|---|---|
| `relevance` | `False` | Append pre-computed relevance assessments to candidate text |
| `relevance_analysis_path` | — | Path to prefilter relevance JSON |
| `summary` | `False` | Append pre-computed candidate summaries to candidate text |
| `candidates_summary_path` | — | Path to candidate summary JSON |

### Multi-pass & fusion

| Option | Default | Description |
|---|---|---|
| `passes` | `1` | Number of reranking passes (>1 enables RRF fusion) |
| `rrf_k` | `60` | RRF fusion constant |
| `seed_file_path` | `"src/patent_retrieval/seeds.txt"` | File with per-pass shuffle seeds |

### Evaluation & logging

| Option | Default | Description |
|---|---|---|
| `wandb` | `True` | Enable Weights & Biases logging |
| `tags` | `["rerank","listwise","async"]` | W&B run tags |
| `run_name` | — | Name prefix for the output directory and W&B run |
| `bootstrapp` | `True` | Compute bootstrap confidence intervals |
| `bootstrap_confidence` | `0.95` | Bootstrap confidence level |
| `bootstrap_seed` | `42` | Bootstrap random seed |
| `semaphore` | `100` | Max concurrent async reranking tasks |

### Output

Results are saved to `reranking/runs/<run_name>/` including:
- `results.csv` — final reranked results
- `metrics.json` — evaluation metrics
- `per_topic_metrics.csv` — per-topic breakdown
- `bootstrap_metrics.json` — bootstrap confidence intervals
- `config.yaml` — full configuration snapshot