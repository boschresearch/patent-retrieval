# Graph-Based Post-Retrieval Workflow

## Overview

This pipeline builds similarity graphs over retrieved patent candidates and applies Leiden community detection to identify cohesive clusters. These clusters serve as an unsupervised post-retrieval filter — selecting the most cohesive clusters removes noisy candidates and improves precision without using any ground-truth labels.

---

## Pipeline Stages

### Stage 1: Load FAISS Index & Retrieval Candidates

- Load the pre-built **FAISS vector store** containing patent embeddings.
- Read `results.csv` from a prior retrieval run — each row is `(topic, doc_number, score)`.
- Take the **top-K candidates** per topic (configurable via `cfg.k`, e.g., 100–500).

### Stage 2: Compute Pairwise Similarity Matrices

For each topic's candidate set:

1. **Extract embeddings** from FAISS by reconstructing vectors for each candidate doc.
2. *(Optional)* **PCA dimensionality reduction** — reduce embedding dimensions while retaining `pca_target_cumsum` (e.g., 95%) of variance. Re-normalize after reduction.
3. **Cosine similarity** — compute the full N×N similarity matrix via dot product (embeddings are L2-normalized).

**Output:** `similarity_matrices.npz` — one matrix per topic, `topic_doc_ids.json` — ordered doc IDs per topic.

### Stage 3: Build Similarity Graph

Convert the similarity matrix into a weighted undirected graph using **two edge rules** (union):

| Rule | Description | Parameter |
|------|-------------|-----------|
| **Threshold** | Connect pair `(i,j)` if `sim[i,j] >= threshold` | `leiden_similarity_threshold` |
| **kNN** | Connect each node to its k most similar neighbors | `leiden_knn_k` |

The union ensures no node is completely isolated even when the threshold is strict. Edge weights are the cosine similarity values.

### Stage 4: Leiden Community Detection

Run the **Leiden algorithm** (`leidenalg.RBConfigurationVertexPartition`) on the graph:

- **Resolution parameter** controls granularity: lower → fewer, larger clusters; higher → more, smaller clusters.
- **n_iterations = -1** means iterate until convergence.
- Uses edge weights so stronger connections have more influence on cluster assignments.

**Output per topic:**
- `membership`: cluster ID for each document
- `quality`: modularity score
- `n_clusters`, `n_edges`: graph statistics

### Stage 5: Compute Cluster Features

For each cluster:

1. **Medoid** — the document with the highest average similarity to all others in its cluster (the cluster "center").
2. **Similarity-to-medoid** — each doc's cosine similarity to its cluster's medoid (used as a within-cluster ranking signal).

**Output:** `leiden_clusters.csv`, `leiden_cluster_medoids.csv`, `leiden_cluster_stats.csv`, `leiden_clusters.json`.

### Stage 6: Cluster-Based Selection & Evaluation

1. **Merge** retrieval scores with cluster assignments.
2. **Score each cluster** with an unsupervised priority:

   ```
   priority = 0.30 × mean_retrieval_score + 0.70 × graph_cohesion
   ```

   where `graph_cohesion = actual_intra_edges / possible_intra_edges` (fraction of within-cluster pairs above the similarity threshold).

3. **Select top 2–3 clusters** per topic (enough to cover ≥100 candidates).
4. **Rank documents** within selected clusters by retrieval score.
5. **Evaluate** the filtered ranking against qrels (precision, recall, nDCG at various cutoffs).

**Output:** `cluster_balanced_top{k}.csv`, `metrics_top{k}.json`.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 300 | Top-K candidates per topic to build the graph over |
| `q` | 500 | Number of topics to process |
| `leiden_similarity_threshold` | 0.7 | Min cosine similarity to create a threshold edge |
| `leiden_knn_k` | 10 | Number of nearest neighbors per node |
| `leiden_resolution` | 0.8 | Leiden resolution (lower = fewer clusters) |
| `leiden_n_iterations` | -1 | Iterations (-1 = until convergence) |
| `leiden_seed` | 678 | Random seed for reproducibility |
| `pca_enable` | False | Whether to apply PCA before similarity computation |
| `pca_target_cumsum` | 0.95 | Cumulative variance target for PCA |
| `topic_workers` | 1 | Parallel threads for topic processing |

### Tuning for ≤5 clusters per topic

```
leiden_similarity_threshold = 0.8   # sparse, high-confidence edges only
leiden_knn_k = 5                    # minimal neighbor connectivity
leiden_resolution = 0.4             # coarse partitioning (most impactful knob)
```

---

## Output Artifacts

| File | Contents |
|------|----------|
| `similarity_matrices.npz` | Per-topic N×N cosine similarity matrices |
| `topic_doc_ids.json` | Ordered doc ID lists per topic |
| `pairwise_similarities.parquet` | All pairwise similarities (upper triangle) |
| `leiden_clusters.csv` | Per-doc cluster assignment, medoid flag, sim-to-medoid |
| `leiden_cluster_medoids.csv` | Per-cluster medoid doc, size, mean sim-to-medoid |
| `leiden_cluster_stats.csv` | Per-topic: n_docs, n_edges, n_clusters, quality |
| `leiden_clusters.json` | `{topic: {cluster_id: [doc_ids]}}` |
| `cluster_balanced_top{k}.csv` | Final ranked docs from selected clusters |
| `metrics_top{k}.json` | IR evaluation metrics |
| `config.yaml` | Run configuration snapshot |

---

## Key Finding

Relevant patents naturally form **dense subgraphs** in the embedding similarity space. Across 455 evaluated topics:

- **78.9%** have relevant docs concentrated in ≤2 clusters (purity ≥ 0.6)
- Median top-cluster share = **1.0**
- Graph cohesion (subgraph density) is a stronger relevance signal than individual retrieval scores

This means selecting the most internally connected clusters is an effective unsupervised strategy for filtering retrieval noise in patent search.
