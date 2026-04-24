# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import rich.syntax

import hydralette as hl
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from pyrootutils import setup_root

from patent_retrieval import utils as utils, encoder as encoder

load_dotenv()

root = setup_root(__file__)
logger = utils.get_logger(__name__)


def _read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read an optional CSV artifact; return None if missing or empty."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Optional artifact is empty: {path}")
        return None
    return df

def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.query_columns.copy()

    asymmetric = set(cfg.query_columns) != set(cfg.doc_columns)

    if "claims" in search_columns and cfg.claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"


    out_dir: Path = root / "graphs"/ "runs" / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.language else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.k}"))
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.language else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.k}")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def get_index_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.doc_columns.copy()

    if "claims" in search_columns and cfg.index_claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.index_claims}claims"""

    index_dir: Path = Path("/home")/"alm3rng"/"scratch" /  "clef_ip_2011" / (f"{cfg.run_name}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")) 
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)

    return index_dir
#    candidates_path = "/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_500topics_top1000/results.csv",
#    candidates_path = "/home/alm3rng/patent-retrieval/embeddings/runs/qwen3-emb-4b_db-v4_all-topics_abstract-claims_aysm_top1000_2026-02-13-12:50:34/results.csv",

cfg = hl.Config(
    type="dense",
    tokenizer= "/home/alm3rng/patent-retrieval/finetuning/runs/optuna/trial_11/checkpoint-95",#"/home/alm3rng/patent-retrieval/finetuning/runs/qwen3_lora_12epochs/v1"
    run_name="patQwen3-emb-4b-v2_db-v4", #"bm25" # qwen3_emb_8b
    suffix="_500-topics",
    embedding_model="patQwen3-emb-4b-v2",#"qwen-finetuned",#"Qwen/Qwen3-Embedding-4B" # mpi-inno-comp/paecter
    candidates_path = "/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_500topics_top1000/results.csv",    
    base_url="http://localhost:59749/v1",
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    claims=None,
    index_claims=None,
    independent_claims=False,
    index_dir= hl.Field(reference=get_index_dir, type=Path),
    doc_columns=hl.Field(
        default=["title","abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    query_columns=hl.Field(
        default=["abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    wandb=False,
    tags=[],
    store_type="faiss",
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    k=300,
    q=500,
    topics=None,
    leiden_enable=True,
    leiden_similarity_threshold=0.7,
    leiden_knn_k=10,
    leiden_resolution=0.8,
    leiden_n_iterations=-1,
    leiden_seed=678,
    pca_enable=False,
    pca_target_cumsum=0.95,
    pca_random_state=42,
    topic_workers=1,

   # notes="top200 fusion 150 from original + 50 from english query",
)


def load_vector_store(cfg):
    """Load FAISS vector store and return it, or None if not found."""
    patent_encoder = encoder.get_encoder(
        model_name=cfg.embedding_model, tokenizer=cfg.tokenizer,
        index_dir=cfg.index_dir, base_url=cfg.base_url
    )
    index_suffix = os.path.join(str(cfg.index_dir), "index.faiss")
    if not os.path.exists(index_suffix):
        logger.error(f"No FAISS index found at {cfg.index_dir}")
        return None
    patent_encoder.load_index(path=str(cfg.index_dir), store_type=cfg.store_type)
    logger.info(f"Loaded FAISS index with {patent_encoder.vector_store.index.ntotal} vectors")
    return patent_encoder.vector_store


def build_id_mapping(vs) -> dict:
    """Build patent_number -> faiss_int_idx mapping from vector store."""
    id_to_idx = {docstore_id: faiss_idx for faiss_idx, docstore_id in vs.index_to_docstore_id.items()}
    logger.info(f"Built mapping for {len(id_to_idx)} documents")
    return id_to_idx


def compute_topic_similarity(
    vs,
    id_to_idx: dict,
    doc_ids: List[str],
    topic: str,
    pca_enable: bool = False,
    pca_target_cumsum: float = 0.95,
    pca_random_state: int = 42,
):
    """Compute pairwise cosine similarity for a single topic's candidates.
    Returns (sim_matrix, valid_ids, csv_rows) or None if no valid docs."""
    valid_ids = [d for d in doc_ids if d in id_to_idx]
    if len(valid_ids) < len(doc_ids):
        logger.warning(f"Topic {topic}: {len(doc_ids) - len(valid_ids)} docs not found in index")
    if not valid_ids:
        logger.warning(f"Topic {topic}: no docs found, skipping")
        return None

    faiss_indices = [id_to_idx[d] for d in valid_ids]
    embeddings = np.stack([vs.index.reconstruct(i) for i in faiss_indices])

    sim_embeddings = embeddings
    if pca_enable:
        n_samples, n_features = embeddings.shape
        max_components = min(n_samples, n_features)

        if max_components > 1:
            target_cumsum = float(np.clip(pca_target_cumsum, 0.0, 1.0))
            pca = PCA(
                n_components=max_components,
                svd_solver="full",
                random_state=pca_random_state,
            )
            reduced = pca.fit_transform(embeddings)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            selected_components = int(np.searchsorted(cumsum, target_cumsum, side="left") + 1)
            selected_components = min(selected_components, max_components)
            sim_embeddings = reduced[:, :selected_components]

            # Re-normalize so cosine similarity can still be computed as dot product.
            norms = np.linalg.norm(sim_embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            sim_embeddings = sim_embeddings / norms

            logger.info(
                f"Topic {topic}: PCA reduced {n_features} -> {selected_components} "
                f"(cumsum={cumsum[selected_components - 1]:.4f}, target={target_cumsum:.2f})"
            )
        else:
            logger.warning(
                f"Topic {topic}: skipping PCA because max_components={max_components}"
            )

    # Cosine similarity = dot product (vectors are L2-normalized)
    sim_matrix = sim_embeddings @ sim_embeddings.T

    # Build CSV rows (upper triangle, no self-similarity)
    rows = []
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            rows.append({
                "topic": topic,
                "doc_i": valid_ids[i],
                "doc_j": valid_ids[j],
                "cosine_sim": float(sim_matrix[i, j])
            })

    logger.info(f"Topic {topic}: {len(valid_ids)} docs, sim_matrix {sim_matrix.shape}")
    return sim_matrix, valid_ids, rows


def _build_similarity_edges(
    sim_matrix: np.ndarray,
    threshold: float,
    knn_k: Optional[int],
) -> Dict[Tuple[int, int], float]:
    """Build undirected weighted edges using threshold OR kNN union rule."""
    n_nodes = sim_matrix.shape[0]
    edge_weights: Dict[Tuple[int, int], float] = {}

    # Threshold edges from upper triangle.
    triu_i, triu_j = np.triu_indices(n_nodes, k=1)
    triu_vals = sim_matrix[triu_i, triu_j]
    keep = triu_vals >= threshold
    for i, j, w in zip(triu_i[keep], triu_j[keep], triu_vals[keep]):
        edge_weights[(int(i), int(j))] = float(w)

    # Add kNN edges (union with threshold edges).
    if knn_k is not None and knn_k > 0 and n_nodes > 1:
        k = min(knn_k, n_nodes - 1)
        for i in range(n_nodes):
            row = sim_matrix[i]
            # Get k+1 to safely drop self if included.
            candidates = np.argpartition(row, -(k + 1))[-(k + 1):]
            for j in candidates:
                if i == j:
                    continue
                u, v = (int(i), int(j)) if i < j else (int(j), int(i))
                w = float(sim_matrix[u, v])
                prev = edge_weights.get((u, v))
                if prev is None or w > prev:
                    edge_weights[(u, v)] = w

    return edge_weights


def run_leiden_clustering(
    sim_matrix: np.ndarray,
    doc_ids: List[str],
    threshold: float,
    knn_k: Optional[int],
    resolution: float,
    n_iterations: int,
    seed: int,
):
    """Run Leiden clustering for one topic and return assignments and stats."""
    n_nodes = len(doc_ids)
    if n_nodes == 0:
        return [], {"n_docs": 0, "n_edges": 0, "n_clusters": 0, "quality": 0.0}
    if n_nodes == 1:
        return [0], {"n_docs": 1, "n_edges": 0, "n_clusters": 1, "quality": 0.0}

    try:
        ig = importlib.import_module("igraph")
        leidenalg = importlib.import_module("leidenalg")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Leiden clustering requires 'igraph' and 'leidenalg'. "
            "Install them first, then rerun."
        ) from exc

    edge_weights = _build_similarity_edges(sim_matrix=sim_matrix, threshold=threshold, knn_k=knn_k)
    edges = list(edge_weights.keys())
    weights = list(edge_weights.values())

    if not edges:
        membership = list(range(n_nodes))
        return membership, {
            "n_docs": n_nodes,
            "n_edges": 0,
            "n_clusters": n_nodes,
            "quality": 0.0,
        }

    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    weights = np.maximum(0, weights)
    graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )

    membership = [int(x) for x in partition.membership]
    n_clusters = len(set(membership))
    stats = {
        "n_docs": n_nodes,
        "n_edges": len(edges),
        "n_clusters": n_clusters,
        "quality": float(partition.quality()),
    }
    return membership, stats


def compute_cluster_medoid_features(
    sim_matrix: np.ndarray,
    doc_ids: List[str],
    membership: List[int],
):
    """Compute per-doc similarity-to-medoid and per-cluster medoid summaries."""
    by_cluster: Dict[int, List[int]] = {}
    for idx, cluster_id in enumerate(membership):
        by_cluster.setdefault(int(cluster_id), []).append(idx)

    per_doc_rows = []
    medoid_rows = []

    for cluster_id, indices in by_cluster.items():
        sub = sim_matrix[np.ix_(indices, indices)]
        # Medoid = doc with highest average similarity to all docs in its cluster.
        local_medoid_pos = int(np.argmax(sub.mean(axis=1)))
        medoid_idx = indices[local_medoid_pos]
        medoid_doc_id = doc_ids[medoid_idx]

        sim_to_medoid = sim_matrix[indices, medoid_idx]
        medoid_rows.append(
            {
                "cluster_id": cluster_id,
                "medoid_doc_id": medoid_doc_id,
                "cluster_size": len(indices),
                "mean_sim_to_medoid": float(np.mean(sim_to_medoid)),
            }
        )

        for doc_idx, sim_val in zip(indices, sim_to_medoid):
            per_doc_rows.append(
                {
                    "doc_id": doc_ids[doc_idx],
                    "cluster_id": cluster_id,
                    "is_medoid": bool(doc_idx == medoid_idx),
                    "sim_to_medoid": float(sim_val),
                }
            )

    return per_doc_rows, medoid_rows


def _process_topic(
    topic: str,
    doc_ids: List[str],
    vs,
    id_to_idx: dict,
    pca_enable: bool,
    pca_target_cumsum: float,
    pca_random_state: int,
    leiden_enable: bool,
    leiden_similarity_threshold: float,
    leiden_knn_k: Optional[int],
    leiden_resolution: float,
    leiden_n_iterations: int,
    leiden_seed: int,
):
    result = compute_topic_similarity(
        vs,
        id_to_idx,
        doc_ids,
        topic,
        pca_enable=pca_enable,
        pca_target_cumsum=pca_target_cumsum,
        pca_random_state=pca_random_state,
    )
    if result is None:
        return None

    sim_matrix, valid_ids, rows = result
    payload = {
        "topic": topic,
        "sim_matrix": sim_matrix,
        "valid_ids": valid_ids,
        "rows": rows,
        "cluster_rows": [],
        "cluster_medoid_rows": [],
        "cluster_stats": None,
        "topic_cluster_docs": None,
    }

    if not leiden_enable:
        return payload

    try:
        membership, stats = run_leiden_clustering(
            sim_matrix=sim_matrix,
            doc_ids=valid_ids,
            threshold=leiden_similarity_threshold,
            knn_k=leiden_knn_k,
            resolution=leiden_resolution,
            n_iterations=leiden_n_iterations,
            seed=leiden_seed,
        )
        per_doc_rows, medoid_rows = compute_cluster_medoid_features(
            sim_matrix=sim_matrix,
            doc_ids=valid_ids,
            membership=membership,
        )

        topic_cluster_docs = {}
        for doc_id, cluster_id in zip(valid_ids, membership):
            cluster_key = f"cluster_{int(cluster_id)}"
            topic_cluster_docs.setdefault(cluster_key, []).append(str(doc_id))

        payload["cluster_rows"] = [{"topic": topic, **row} for row in per_doc_rows]
        payload["cluster_medoid_rows"] = [{"topic": topic, **row} for row in medoid_rows]
        payload["cluster_stats"] = {
            "topic": topic,
            "n_docs": stats["n_docs"],
            "n_edges": stats["n_edges"],
            "n_clusters": stats["n_clusters"],
            "quality": stats["quality"],
            "threshold": leiden_similarity_threshold,
            "knn_k": leiden_knn_k,
            "resolution": leiden_resolution,
        }
        payload["topic_cluster_docs"] = topic_cluster_docs
        logger.info(
            f"Topic {topic}: Leiden produced {stats['n_clusters']} clusters with {stats['n_edges']} edges"
        )
    except Exception as exc:
        logger.exception(f"Topic {topic}: Leiden clustering failed: {exc}")

    return payload


def compute_cos(cfg, top_k: int = 100):
    """Extract embeddings from FAISS index and compute pairwise cosine similarity per topic."""
    vs = load_vector_store(cfg)
    if vs is None:
        return

    id_to_idx = build_id_mapping(vs)

    # Load candidates and take top_k per topic
    df = pd.read_csv(cfg.candidates_path)
    topics = df.groupby("topic").head(top_k).groupby("topic")["number"].apply(list).to_dict()

    if cfg.topics:
        if isinstance(cfg.topics, int):
            topics = dict(list(topics.items())[: cfg.topics])
        elif isinstance(cfg.topics, list):
            topics = {topic: topics[topic] for topic in cfg.topics if topic in topics}

    logger.info(f"Processing {len(topics)} topics, top {top_k} candidates each")

    sim_matrices = {}
    doc_ids_map = {}
    all_rows = []
    cluster_rows = []
    cluster_medoid_rows = []
    cluster_stats_rows = []
    cluster_docs_by_topic = {}

    topic_items = list(topics.items())
    topic_workers = max(1, int(cfg.topic_workers))

    def collect_topic_payload(payload):
        if payload is None:
            return

        topic = payload["topic"]
        sim_matrices[topic] = payload["sim_matrix"]
        doc_ids_map[topic] = payload["valid_ids"]
        all_rows.extend(payload["rows"])

        if payload["cluster_rows"]:
            cluster_rows.extend(payload["cluster_rows"])
        if payload["cluster_medoid_rows"]:
            cluster_medoid_rows.extend(payload["cluster_medoid_rows"])
        if payload["cluster_stats"] is not None:
            cluster_stats_rows.append(payload["cluster_stats"])
        if payload["topic_cluster_docs"] is not None:
            cluster_docs_by_topic[str(topic)] = payload["topic_cluster_docs"]

    if topic_workers > 1 and len(topic_items) > 1:
        logger.info(f"Parallel topic processing enabled with {topic_workers} workers")
        with ThreadPoolExecutor(max_workers=topic_workers) as executor:
            futures = [
                executor.submit(
                    _process_topic,
                    topic,
                    doc_ids,
                    vs,
                    id_to_idx,
                    cfg.pca_enable,
                    cfg.pca_target_cumsum,
                    cfg.pca_random_state,
                    cfg.leiden_enable,
                    cfg.leiden_similarity_threshold,
                    cfg.leiden_knn_k,
                    cfg.leiden_resolution,
                    cfg.leiden_n_iterations,
                    cfg.leiden_seed,
                )
                for topic, doc_ids in topic_items
            ]

            for future in as_completed(futures):
                collect_topic_payload(future.result())
    else:
        for topic, doc_ids in topic_items:
            payload = _process_topic(
                topic,
                doc_ids,
                vs,
                id_to_idx,
                cfg.pca_enable,
                cfg.pca_target_cumsum,
                cfg.pca_random_state,
                cfg.leiden_enable,
                cfg.leiden_similarity_threshold,
                cfg.leiden_knn_k,
                cfg.leiden_resolution,
                cfg.leiden_n_iterations,
                cfg.leiden_seed,
            )
            collect_topic_payload(payload)

    # Save outputs
    out_dir = cfg.output_dir
    logger.info(f"Saving results to {out_dir}")

    np.savez(out_dir / "similarity_matrices.npz", **{str(t): m for t, m in sim_matrices.items()})

    with open(out_dir / "topic_doc_ids.json", "w") as f:
        json.dump(doc_ids_map, f, indent=2)

    pd.DataFrame(all_rows).to_parquet(out_dir / "pairwise_similarities.parquet", index=False)

    if cfg.leiden_enable:
        logger.info(f"Saving Leiden clustering results for {len(cluster_stats_rows)} topics")
        cluster_columns = ["topic", "doc_id", "cluster_id", "is_medoid", "sim_to_medoid"]
        medoid_columns = ["topic", "cluster_id", "medoid_doc_id", "cluster_size", "mean_sim_to_medoid"]
        stats_columns = ["topic", "n_docs", "n_edges", "n_clusters", "quality", "threshold", "knn_k", "resolution"]

        pd.DataFrame(cluster_rows, columns=cluster_columns).to_csv(out_dir / "leiden_clusters.csv", index=False)

        pd.DataFrame(cluster_medoid_rows, columns=medoid_columns).to_csv(out_dir / "leiden_cluster_medoids.csv", index=False)

        pd.DataFrame(cluster_stats_rows, columns=stats_columns).to_csv(out_dir / "leiden_cluster_stats.csv", index=False)

        with open(out_dir / "leiden_clusters.json", "w") as f:
            json.dump(cluster_docs_by_topic, f, indent=2)


    logger.info(f"Done. Saved {len(sim_matrices)} topic matrices to {out_dir}")

def load_graph_artifacts(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    matrices_path = run_dir / "similarity_matrices.npz"
    ids_path = run_dir / "topic_doc_ids.json"
    clusters_path = run_dir / "leiden_clusters.csv"
    medoids_path = run_dir / "leiden_cluster_medoids.csv"
    stats_path = run_dir / "leiden_cluster_stats.csv"
    clusters_json_path = run_dir / "leiden_clusters.json"

    #if not matrices_path.exists() or not ids_path.exists():
     #   raise FileNotFoundError("Missing required files: similarity_matrices.npz and topic_doc_ids.json")

    matrices = np.load(matrices_path)
    with open(ids_path, "r") as f:
        topic_doc_ids = json.load(f)

    clusters_df = _read_optional_csv(clusters_path)
    medoids_df = _read_optional_csv(medoids_path)
    stats_df = _read_optional_csv(stats_path)
    if clusters_json_path.exists():
        with open(clusters_json_path, "r") as f:
            clusters_json = json.load(f)
    else:
        clusters_json = None

    return {
        "run_dir": run_dir,
        "matrices": matrices,
        "topic_doc_ids": topic_doc_ids,
        "clusters_df": clusters_df,
        "medoids_df": medoids_df,
        "stats_df": stats_df,
        "clusters_json": clusters_json,
    }

def select_cluster_balanced_topk(
    topic_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    topic_doc_ids: List[str],
    top_k: int = 100,
) -> pd.DataFrame:
    """Select top-k with per-cluster quota, then retrieval-score backfill."""
    if topic_df.empty:
        return pd.DataFrame(columns=["number", "score", "cluster_id", "is_medoid", "rank"])

    required_cols = {"number", "retrieval_score", "cluster_id", "is_medoid"}
    missing = required_cols - set(topic_df.columns)
    if missing:
        raise ValueError(f"topic_df is missing required columns: {sorted(missing)}")

    # Keep one row per document, prioritizing the highest retrieval score.
    candidates_df = (
        topic_df.sort_values("retrieval_score", ascending=False)
        .drop_duplicates(subset=["number"], keep="first")
        .copy()
    )
    if candidates_df.empty:
        return pd.DataFrame(columns=["number", "score", "cluster_id", "is_medoid", "rank"])

    idx_by_doc = {str(doc_id): i for i, doc_id in enumerate(topic_doc_ids)}
    candidates_df["number_str"] = candidates_df["number"].astype(str)
    candidates_df = candidates_df[candidates_df["number_str"].isin(idx_by_doc)].copy()
    if candidates_df.empty:
        return pd.DataFrame(columns=["number", "score", "cluster_id", "is_medoid", "rank"])

    candidates_df["matrix_idx"] = candidates_df["number_str"].map(idx_by_doc)
    candidates_df = candidates_df.sort_values("retrieval_score", ascending=False).reset_index(drop=True)

    n_target = min(top_k, len(candidates_df))

    # Build per-cluster candidate lists sorted by retrieval score.
    cluster_groups = {
        int(cluster): grp.sort_values("retrieval_score", ascending=False).reset_index(drop=True)
        for cluster, grp in candidates_df.groupby("cluster_id", sort=False)
    }
    n_clusters = len(cluster_groups)
    if n_clusters == 0:
        return pd.DataFrame(columns=["number", "score", "cluster_id", "is_medoid", "rank"])

    # Divide target budget by number of clusters.
    per_cluster_target = max(1, n_target // n_clusters)

    # Stable order: strongest retrieval seed first.
    cluster_order = sorted(
        cluster_groups.keys(),
        key=lambda c: float(cluster_groups[c].iloc[0]["retrieval_score"]),
        reverse=True,
    )

    # Start with n=1 top-retrieval seed per cluster.
    selected_numbers = []
    seen = set()
    seed_by_cluster = {}
    active_clusters = cluster_order[: min(n_target, n_clusters)]
    for cluster in active_clusters:
        grp = cluster_groups[cluster]
        seed_idx = 0
        while seed_idx < len(grp) and grp.iloc[seed_idx]["number"] in seen:
            seed_idx += 1
        if seed_idx >= len(grp):
            continue

        seed_row = grp.iloc[seed_idx]
        seed_number = seed_row["number"]
        selected_numbers.append(seed_number)
        seen.add(seed_number)
        seed_by_cluster[cluster] = seed_row

    taken_per_cluster = {cluster: 1 for cluster in seed_by_cluster}

    topic_label = topic_df["topic"].iloc[0] if "topic" in topic_df.columns else "<unknown>"
    logger.info(
        f"Selecting top {n_target} candidates for topic {topic_label} "
        f"across {n_clusters} clusters with per-cluster target {per_cluster_target}"
    )

    # For each seed, precompute same-cluster neighbors sorted by similarity.
    neighbor_lists = {}
    for cluster, seed_row in seed_by_cluster.items():
        seed_number = seed_row["number"]
        seed_idx = int(seed_row["matrix_idx"])
        grp = cluster_groups[cluster]
        neighbors = []
        for _, cand_row in grp.iterrows():
            cand_number = cand_row["number"]
            if cand_number == seed_number:
                continue
            cand_idx = int(cand_row["matrix_idx"])
            neighbors.append(
                (
                    cand_number,
                    float(sim_matrix[seed_idx, cand_idx]),
                    float(cand_row["retrieval_score"]),
                )
            )
        neighbors.sort(key=lambda x: (x[1], x[2]), reverse=True)
        neighbor_lists[cluster] = neighbors

    ptr = {cluster: 0 for cluster in neighbor_lists}

    # Expand around seeds by first skipping ineligible clusters, then round-robin picking.
    rr_start = 0
    while len(selected_numbers) < n_target:
        eligible_clusters = []
        for cluster in active_clusters:
            if cluster not in seed_by_cluster:
                continue
            if taken_per_cluster.get(cluster, 0) >= per_cluster_target:
                continue

            neighbors = neighbor_lists.get(cluster, [])
            i = ptr[cluster]
            while i < len(neighbors) and neighbors[i][0] in seen:
                i += 1
            ptr[cluster] = i
            if i >= len(neighbors):
                continue

            eligible_clusters.append(cluster)

        if not eligible_clusters:
            break

        start = rr_start % len(eligible_clusters)
        rr_order = eligible_clusters[start:] + eligible_clusters[:start]
        rr_start += 1

        for cluster in rr_order:
            neighbors = neighbor_lists[cluster]
            i = ptr[cluster]
            if i >= len(neighbors):
                continue

            chosen_number = neighbors[i][0]
            selected_numbers.append(chosen_number)
            seen.add(chosen_number)
            ptr[cluster] += 1
            taken_per_cluster[cluster] = taken_per_cluster.get(cluster, 0) + 1

            if len(selected_numbers) >= n_target:
                break

    # Fallback fill by retrieval score if neighbor expansion exhausted.
    if len(selected_numbers) < n_target:
        for number in candidates_df["number"].tolist():
            if number in seen:
                continue
            selected_numbers.append(number)
            seen.add(number)
            if len(selected_numbers) >= n_target:
                break

    selected_df = pd.DataFrame({"number": selected_numbers})
    out = selected_df.merge(
        candidates_df[["number", "cluster_id", "is_medoid"]],
        on="number",
        how="left",
    )
    out["rank"] = range(1, len(out) + 1)
    # Enforce ranking order during evaluation by using inverse rank as score.
    out["score"] = (len(out) - out["rank"] + 1).astype(float)

    return out[["number", "score", "cluster_id", "is_medoid", "rank"]]

def main(cfg: hl.Config):
    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    compute_cos(cfg, top_k=cfg.k)
    logger.info("Finished computing similarities and clusters. Now loading artifacts for selection and evaluation.")
    artifacts = load_graph_artifacts(cfg.output_dir)
    if artifacts.get("clusters_df") is None:
        raise ValueError("leiden_clusters.csv not found in RUN_DIR. Run clustering first.")

    retrieval_df = pd.read_csv(cfg.candidates_path)
    retrieval_df = retrieval_df[["topic", "number", "score"]].copy().rename(columns={"score": "retrieval_score"})
    retrieval_df["retrieval_score"] = retrieval_df["retrieval_score"].astype(float)

    cluster_df = artifacts["clusters_df"][["topic", "doc_id", "cluster_id", "is_medoid", "sim_to_medoid"]].rename(columns={"doc_id": "number"})
    cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)
    cluster_df["is_medoid"] = cluster_df["is_medoid"].astype(bool)
    cluster_df["sim_to_medoid"] = cluster_df["sim_to_medoid"].astype(float)

    scored_clustered = retrieval_df.merge(cluster_df, on=["topic", "number"], how="inner")

    # Build topic ranking by retrieval-score seeds + nearest-neighbor expansion.
    selected_topics = []
    for topic, grp in scored_clustered.groupby("topic", sort=False):
        topic_key = str(topic)
        if topic_key not in artifacts["topic_doc_ids"] or topic_key not in artifacts["matrices"]:
            logger.warning(f"Topic {topic}: missing similarity artifacts, skipping")
            continue

        topic_ranked = select_cluster_balanced_topk(
            grp,
            sim_matrix=artifacts["matrices"][topic_key],
            topic_doc_ids=artifacts["topic_doc_ids"][topic_key],
            top_k=cfg.k,
        )
        if topic_ranked.empty:
            continue
        topic_ranked.insert(0, "topic", topic)
        selected_topics.append(topic_ranked)

    if not selected_topics:
        logger.warning("No topics produced cluster-balanced rankings. Nothing to save/evaluate.")
        return

    cluster_top100_df = pd.concat(selected_topics, ignore_index=True)
    cluster_top100_df = cluster_top100_df.merge(
        retrieval_df[["topic", "number", "retrieval_score"]],
        on=["topic", "number"],
        how="left",
    )

    # Evaluate using existing utility.
    # calculate_metrics expects either path or DataFrame with 3 columns -> [q_id, doc_id, score].
    eval_df = cluster_top100_df[["topic", "number", "score"]].copy()

    metrics = utils.calculate_metrics(results=eval_df, topk=cfg.k, test_topics_path=cfg.test_topics_path)

    metrics_json = json.dumps(metrics, indent=4)
    logger.info(metrics_json)
    #cfg.output_dir.joinpath(f"metrics_top{cfg.k}.json").write_text(metrics_json)

    # Optional: save ranking and metrics for reproducibility.
    out_rank_path = artifacts["run_dir"] / f"cluster_balanced_top{cfg.k}.csv"
    out_metrics_path = artifacts["run_dir"] / f"metrics_top{cfg.k}.json"
    cluster_top100_df.to_csv(out_rank_path, index=False)
    out_metrics_path.write_text(json.dumps(metrics, indent=4))

    logger.info(f"\nSaved ranking to: {out_rank_path}")
    logger.info(f"Saved metrics to: {out_metrics_path}")


if __name__ == "__main__":
    cfg.apply()
    cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())
    #candidates_path = "/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_30topics_top1000_2026-02-25-11:16:02/results.csv"
    main(cfg)
