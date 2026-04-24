# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import hydralette as hl
import numpy as np
import pandas as pd
import rich.syntax
import wandb
from dotenv import load_dotenv
from pyrootutils import setup_root
from tqdm import tqdm
from patent_retrieval import dataset as dataset, encoder as encoder, utils as utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()

root = setup_root(__file__)
os.environ["WANDB_DIR"] = str(root)
logger = utils.get_logger(__name__)


def get_output_dir(cfg: hl.Config) -> Path:
    out_dir = (
        root
        / "post_retrieval"
        / "runs"
        / f"{cfg.run_name}{cfg.suffix}_post-retrieval_seed{cfg.seed_n}_top{cfg.k}"
    )
    if out_dir.exists():
        out_dir = out_dir.parent / (
            f"{cfg.run_name}{cfg.suffix}_post-retrieval_seed{cfg.seed_n}_top{cfg.k}"
            f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_index_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.doc_columns.copy()
    if "claims" in search_columns and cfg.index_claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.index_claims}claims"

    index_dir: Path = Path("/home") / "alm3rng" / "scratch" / "clef_ip_2011" / (
        f"{cfg.run_name}_{'-'.join(search_columns)}"
        + (f"_{cfg.language}" if cfg.unilingual_index else "")
    )
    return index_dir

#    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_all-topics_abstract-claims_aysm_top1000/results.csv",
BASE_URL = "http://localhost:59749/v1"
cfg = hl.Config(
    type="dense",
    backend="openai",
    store_type="faiss",
    tokenizer="/home/alm3rng/patent-retrieval/finetuning/runs/optuna/trial_11/checkpoint-95",
    run_name="patQwen3-emb-4b-v2_db-v4",
    suffix="_all-topics",
    embedding_model="patQwen3-emb-4b-v2",#"patQwen3-emb-4b-v2",
    base_url=BASE_URL,
    q=500,
    candidates_path = "/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_all-topics_abstract-claims_aysm_top1000/results.csv",
    output_filename="results_centroid.csv",
    dropped_filename="dropped_candidates.csv",
    metrics_filename="metrics_centroid.json",
    seed_n=3,
    alpha=1,
    beta=1,
    k=1000,
    claims=None,
    independent_claims=False,
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    index_claims=None,
    query_columns=hl.Field(default=["abstract", "claims"],
                           convert=lambda x: x.split(",")),
    doc_columns=hl.Field(default=["title", "abstract", "claims"], 
                         convert=lambda x: x.split(",")),
    mode="recompute_sim",#"full_index",#"recompute_sim"
    unilingual_index=False,
    language="",
    wandb=True,
    tags=[],
    index_dir=hl.Field(reference=get_index_dir, type=Path),
    output_dir=hl.Field(reference=get_output_dir, type=Path),
)


def load_vector_store(cfg: hl.Config):
    patent_encoder = encoder.get_encoder(
        type=cfg.type,
        backend=cfg.backend,
        store_type=cfg.store_type,
        model_name=cfg.embedding_model,
        tokenizer=cfg.tokenizer,
        index_dir=cfg.index_dir,
        base_url=cfg.base_url,
    )
    index_suffix = os.path.join(str(cfg.index_dir), "index.faiss")
    if not os.path.exists(index_suffix):
        raise FileNotFoundError(f"No FAISS index found at {cfg.index_dir}")
    patent_encoder.load_index(path=str(cfg.index_dir), store_type=cfg.store_type)
    logger.info(f"Loaded FAISS index with {patent_encoder.vector_store.index.ntotal} vectors")
    return patent_encoder, patent_encoder.vector_store


def get_topic_query_vector(
    topic_id: str,
    patent_encoder,
    cfg: hl.Config,
    cache: Dict[str, Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    if topic_id in cache:
        return cache[topic_id]

    topic_file = next(
        cfg.test_topics_path.parent.glob(f"PAC_topics/files/{topic_id}*.xml"),
        None,
    )
    if topic_file is None:
        logger.warning(f"Topic {topic_id}: query xml not found")
        cache[topic_id] = None
        return None

    try:
        topic_patent = dataset.parse_patent([topic_file])[0]
        query_text = dataset.extract_query_text(
            topic_patent,
            cfg.query_columns,
            cfg.claims,
            independent_only=cfg.independent_claims,
        )
        query_vec = np.array(patent_encoder.encode_query(query_text), dtype=np.float32)
        query_vec = l2_normalize(query_vec)
    except Exception as exc:
        logger.warning(f"Topic {topic_id}: processing failed ({exc})")
        cache[topic_id] = None
        return None

    if query_vec is None or not query_text:
        logger.warning(f"Topic {topic_id}: invalid query result")
        cache[topic_id] = None
        return None

    cache[topic_id] = query_vec
    return query_vec


def build_weighted_centroid(
    query_vec: Optional[np.ndarray],
    seed_matrix: np.ndarray,
    alpha: float,
    beta: float,
) -> Optional[np.ndarray]:
    """
    Build a weighted centroid of the seed document vectors and the query vector (if available).
    The seed_matrix is weighted by beta, and the query_vec is weighted by alpha. The resulting centroid is L2-normalized.
    If the resulting centroid has zero norm, returns None.
    This allows flexible interpolation between the original query vector and the seed document vectors for reranking.
     For example:
      - alpha=1, beta=0: use only the query vector (no seed influence)
      - alpha=0, beta=1: use only the seed document centroid (no query influence)
      - alpha=1, beta=1: equal weighting of query and seed centroid
    
    """
    pooled_matrix = beta * seed_matrix

    if query_vec is not None:
        pooled_matrix = np.vstack([pooled_matrix, alpha * query_vec])

    return l2_normalize(pooled_matrix.mean(axis=0))


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Candidates file is empty: {path}")
    return normalize_candidate_schema(df)


def normalize_candidate_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "q_id" in df.columns:
        rename_map["q_id"] = "topic"
    if "doc_id" in df.columns:
        rename_map["doc_id"] = "number"
    df = df.rename(columns=rename_map)

    required = {"topic", "number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidates CSV missing required columns: {sorted(missing)}")

    if "score" not in df.columns:
        # Preserve input order as fallback score when the source has no score column.
        df = df.copy()
        df["score"] = -np.arange(len(df), dtype=np.float32)

    out = df[["topic", "number", "score"]].copy()
    out["topic"] = out["topic"].astype(str)
    out["number"] = out["number"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])
    return out


def build_id_mapping(vector_store) -> Dict[str, int]:
    """
    Build a mapping from document ID (as string) to FAISS index (as int).
    """
    id_to_idx = {
        docstore_id: faiss_idx
        for faiss_idx, docstore_id in vector_store.index_to_docstore_id.items()
    }
    logger.info(f"Built id mapping for {len(id_to_idx)} indexed documents")
    return id_to_idx


def l2_normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return vec / norm


def search_full_index_with_centroid(
    vector_store,
    topic_id: str,
    centroid: np.ndarray,
    topk: int,
) -> List[Dict[str, object]]:
    """
    Perform a FAISS search over the full index using the centroid as the query vector.
     Returns a list of dicts with keys: topic, number, score.
    
    """
    query_vec = centroid.astype(np.float32).reshape(1, -1)
    scores, faiss_indices = vector_store.index.search(query_vec, topk)

    rows: List[Dict[str, object]] = []
    for faiss_idx, score in zip(faiss_indices[0], scores[0]):
        if int(faiss_idx) < 0:
            continue
        doc_id = vector_store.index_to_docstore_id.get(int(faiss_idx))
        if doc_id is None:
            continue
        rows.append({"topic": topic_id, "number": doc_id, "score": float(score)})

    return rows


def get_document_vector(
    vector_store,
    id_to_idx: Dict[str, int],
    doc_id: str,
    cache: Dict[str, Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    if doc_id in cache:
        return cache[doc_id]

    faiss_idx = id_to_idx.get(doc_id)
    if faiss_idx is None:
        cache[doc_id] = None
        return None

    vec = np.array(vector_store.index.reconstruct(faiss_idx), dtype=np.float32)
    vec = l2_normalize(vec)
    cache[doc_id] = vec
    return vec


def rerank_topic(
    topic_df: pd.DataFrame,
    patent_encoder,
    cfg: hl.Config,
    vector_store,
    id_to_idx: Dict[str, int],
    doc_cache: Dict[str, Optional[np.ndarray]],
    query_cache: Dict[str, Optional[np.ndarray]],
    seed_n: int,
    alpha: float,
    beta: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    topic_id = str(topic_df.iloc[0]["topic"])

    # Keep a single row per candidate id (highest original score) and keep order by original score.
    topic_df = (
        topic_df.groupby("number", as_index=False)["score"]
        .max()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    docs: List[str] = topic_df["number"].tolist()

    vectors: Dict[str, np.ndarray] = {}
    dropped_rows: List[Dict[str, object]] = []
    for doc_id in docs:
        vec = get_document_vector(vector_store, id_to_idx, doc_id, doc_cache)
        if vec is None:
            dropped_rows.append({"topic": topic_id, "number": doc_id, "reason": "missing_in_index"})
            continue
        vectors[doc_id] = vec

    if not vectors:
        return [], dropped_rows

    available_docs = [doc_id for doc_id in docs if doc_id in vectors]
    seed_ids = available_docs[:seed_n]
    if not seed_ids:
        return [], dropped_rows

    seed_matrix = np.stack([vectors[doc_id] for doc_id in seed_ids])
    query_vec = get_topic_query_vector(
        topic_id=topic_id,
        patent_encoder=patent_encoder,
        cfg=cfg,
        cache=query_cache,
    )

    centroid = build_weighted_centroid(
        query_vec=query_vec,
        seed_matrix=seed_matrix,
        alpha=alpha,
        beta=beta,
    )

    if centroid is None:
        return [], dropped_rows

    if cfg.mode == "full_index":
        return (
            search_full_index_with_centroid(
                vector_store=vector_store,
                topic_id=topic_id,
                centroid=centroid,
                topk=cfg.k,
            ),
            dropped_rows,
        )

    rescored_rows = []
    for doc_id in available_docs:
        cosine_sim = float(vectors[doc_id] @ centroid)
        rescored_rows.append({"topic": topic_id, "number": doc_id, "score": cosine_sim})

    rescored_rows.sort(key=lambda row: row["score"], reverse=True)
    return rescored_rows, dropped_rows


def rerank_candidates(
    candidates_df: pd.DataFrame,
    patent_encoder,
    cfg: hl.Config,
    vector_store,
    id_to_idx: Dict[str, int],
    seed_n: int,
    alpha: float,
    beta: float,
):
    results: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    doc_cache: Dict[str, Optional[np.ndarray]] = {}
    query_cache: Dict[str, Optional[np.ndarray]] = {}

    total_topics = candidates_df["topic"].nunique()
    pbar = utils.RichTableProgress(total=total_topics, print_every=25)
    for topic, group in tqdm(candidates_df.groupby("topic", sort=False), total=total_topics):
        topic_group = group[["topic", "number", "score"]].copy()
        topic_rows, dropped_rows = rerank_topic(
            topic_group,
            patent_encoder=patent_encoder,
            cfg=cfg,
            vector_store=vector_store,
            id_to_idx=id_to_idx,
            doc_cache=doc_cache,
            query_cache=query_cache,
            seed_n=seed_n,
            alpha=alpha,
            beta=beta,
        )
        if not topic_rows:
            logger.warning(f"Topic {topic}: skipped (no valid vectors for reranking)")
        results.extend(topic_rows)
        dropped.extend(dropped_rows)
        pbar.update()

    return pd.DataFrame.from_records(results), pd.DataFrame.from_records(dropped)


def main(cfg: hl.Config, wandb_run=None):
    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())

    candidates_df = load_candidates(Path(cfg.candidates_path))
    if cfg.q:
        try:
            topics = utils.read_topics(cfg.q)
            candidates_df = candidates_df[candidates_df["topic"].isin(topics)]
            logger.info(f"Using  topics from {cfg.q}topics.txt")
        except Exception as e:
            logger.error(f"Failed to read topics from {cfg.q}topics.txt: {e}")
            topics = candidates_df["topic"].unique().tolist()
            candidates_df = candidates_df[candidates_df["topic"].isin(topics)]
            logger.info(f"Falling back to all topics from candidates: {len(topics)} topics")
    else:
        logger.info(f"Using all topics from candidates: {len(topics)} topics")

    logger.info(f"Loaded {len(candidates_df)} candidate rows from {cfg.candidates_path}")

    if cfg.alpha < 0 or cfg.beta < 0 or (cfg.alpha == 0 and cfg.beta == 0):
        raise ValueError("alpha and beta must be non-negative, and not both zero")

    allowed_modes = {"recompute_sim", "full_index"}
    # 
    if cfg.mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}, got: {cfg.mode}")

    patent_encoder, vector_store = load_vector_store(cfg)
    id_to_idx = build_id_mapping(vector_store)

    reranked_df, dropped_df = rerank_candidates(
        candidates_df=candidates_df,
        patent_encoder=patent_encoder,
        cfg=cfg,
        vector_store=vector_store,
        id_to_idx=id_to_idx,
        seed_n=cfg.seed_n,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )

    if reranked_df.empty:
        raise RuntimeError("No reranked results generated. Check candidates/index compatibility.")

    output_csv = cfg.output_dir / cfg.output_filename
    reranked_df.to_csv(output_csv, index=False)
    logger.info(f"Saved reranked results: {output_csv}")

    metrics = utils.calculate_metrics(
        results=output_csv,
        test_topics_path=cfg.test_topics_path,
        topk=cfg.k,
    )
    metrics_path = cfg.output_dir / cfg.metrics_filename
    metrics_path.write_text(json.dumps(metrics, indent=4))
    logger.info(f"Saved reranked metrics: {metrics_path}")
    logger.info(f"Reranking complete. Metrics:\n{json.dumps(metrics, indent=4)}")

    if cfg.wandb and wandb_run is not None:
        wandb.log(metrics)

    if not dropped_df.empty:
        dropped_csv = cfg.output_dir / cfg.dropped_filename
        dropped_df.to_csv(dropped_csv, index=False)
        logger.info(f"Saved dropped-candidate report: {dropped_csv}")

    if cfg.wandb and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    cfg.apply()
    run = None

    if cfg.wandb:
        settings = wandb.Settings(
            save_code=False,
            x_disable_meta=True,
            x_disable_stats=True,
        )
        run = wandb.init(
            project="post_retriever",
            name=cfg.run_name,
            config=cfg.to_dict(),
            group="post_retrieval",
            config_exclude_keys=[
                "test_topics_path",
                "wandb",
                "tags",
                "base_url",
            ],
            settings=settings,
            tags=cfg.tags,
        )

    main(cfg, wandb_run=run)


