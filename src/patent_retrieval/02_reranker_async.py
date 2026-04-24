# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
import asyncio
from pyrootutils import setup_root
import re
from datetime import datetime
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import hydralette as hl
import pandas as pd
import rich.syntax
import sqlalchemy as sqla
import sqlmodel as sqlm
from pyrootutils import setup_root
from langchain_core.documents import Document
import traceback
import wandb

import json
import numpy as np
from itertools import islice
from sklearn.metrics import precision_score, recall_score, f1_score
from asyncer import asyncify
from dotenv import load_dotenv

from tqdm.asyncio import tqdm

from patent_retrieval import utils as utils, dataset as dataset, reranker as reranker, agents as agents
import torch
load_dotenv()
torch.cuda.empty_cache()

root = setup_root(__file__)
logger = utils.get_logger(__name__)

import random
from collections import defaultdict
#os.environ["WANDB_DIR"] = str(root)

def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.query_columns.copy()

    asymmetric = set(cfg.query_columns) != set(cfg.doc_columns)

    if "claims" in search_columns and cfg.claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims" if cfg.claims is not None else f"claims"


    out_dir: Path = root / "reranking"/ "runs" / (f"{cfg.run_name}_"+(f"thinking" if cfg.thinking else "non-thinking")+f"_ret-{cfg.retrieval_model}_{'-'.join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}"))
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}_"+(f"thinking" if cfg.thinking else "non-thinking")+f"_ret-{cfg.retrieval_model}_{'-'.join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



#BASE_URL=os.getenv("AZURE_OPENAI_ENDPOINT") 


BASE_URL="http://rng-dl01-w26a01:55047/v1" #openai
#BASE_URL="http://localhost:8000" #cohere 

API_KEY="EMPTY"
cfg = hl.Config(
    type="listwise",
    backend="openai",
    mode="tournament", # "tournament" or "simple"
    
    #method="",
    run_name="qwen3.5-397b_db-v4",#"qwen3_rerank_4b_v4_100topics_rewrite",#"qwen3_30b_instruct_3topics_top100",#"bm25" # qwen3_emb_4b
    retrieval_model="patQwen3-emb-4b",
    model_name="Qwen/Qwen3.5-397B-A17B-FP8",# "Qwen/Qwen3-30B-A3B-Instruct-2507",#"Qwen/Qwen3-Reranker-4B" # mpi-inno-comp/paecter
    #model_name="Qwen/Qwen3.5-397B-A17B-FP8",
    thinking=True,
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_500topics_top1000/results.csv",
    claims=None,
    independent_claims=True,
    target_claims=None,
    relevance=False,
   # relevance_analysis_path="/home/alm3rng/patent-retrieval/prefilter/runs/qwen3.5-397b_db-v4_patQwen3-based_claims_judgev1_aysm_500topics_top200/results.json",
    summary=False,
    candidates_summary_path="/home/alm3rng/patent-retrieval/summary/runs/qwen3.5-397b_db-v4_patQwen3-based_title-abstract-claims-description_judgev2_500topics_top100/results.json",
    clusters_path=None,
    use_cluster_tournament=False,
    #index_dir= hl.Field(reference=get_index_dir, type=Path),
    doc_columns=hl.Field(
        default=["title","abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    query_columns=hl.Field(
        default=["abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    wandb=True,
    tags=["rerank","listwise","async"],
    remap_ids=True,
    prompt_id="v7_tournament",
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    topk=100,
    q=500,
    window_size=20,
    semaphore=100,
    passes=1,    
    topics=None,
    bootstrapp=True,
    bootstrap_confidence=0.95,
    bootstrap_seed=42,
    rrf_k=60,
    seed_file_path="src/patent_retrieval/utils/seeds.txt",
)




def _normalize_patent_id(value: object) -> str:
    patent_id = "" if value is None else str(value).strip()
    return patent_id


def _base_patent_id(value: object) -> str:
    patent_id = _normalize_patent_id(value)
    return re.sub(r"-[A-Z]\d+$", "", patent_id)


def _add_relevance_assessment(topic_id, candidate_docs, relevance_analysis_path=None):
    logger.info(f"Extending candidate contexts for topic {topic_id}")
    if relevance_analysis_path is None:
        logger.warning("No relevance analysis path provided; skipping relevance augmentation.")
        return candidate_docs

    with open(relevance_analysis_path, "r", encoding="utf-8") as f:
        analysis_json = json.load(f)

    topic_key = _normalize_patent_id(topic_id)
    topic_analysis = analysis_json.get(topic_key)
    if topic_analysis is None:
        topic_analysis = analysis_json.get(_base_patent_id(topic_key), [])

    if isinstance(topic_analysis, dict):
        topic_entries = [topic_analysis]
    elif isinstance(topic_analysis, list):
        topic_entries = [entry for entry in topic_analysis if isinstance(entry, dict)]
    else:
        topic_entries = []

    # Allow matching by full ID and by base ID without kind code suffix.
    candidate_key_map: Dict[str, str] = {}
    for raw_id in candidate_docs.keys():
        full_id = _normalize_patent_id(raw_id)
        base_id = _base_patent_id(raw_id)
        candidate_key_map.setdefault(full_id, raw_id)
        candidate_key_map.setdefault(base_id, raw_id)

    matched_candidates = 0
    appended_candidates = 0

    for candidate_entry in topic_entries:
        for raw_candidate_id, candidate_data in candidate_entry.items():
            if not isinstance(candidate_data, dict):
                continue

            candidate_id = _normalize_patent_id(raw_candidate_id)
            target_key = candidate_key_map.get(candidate_id) or candidate_key_map.get(
                _base_patent_id(candidate_id)
            )
            if target_key is None:
                continue

            matched_candidates += 1
            relevance_lines = ["\n\n<Relevance Assessment>"]

            for key, value in candidate_data.items():
                if "query" in str(key).lower():
                    continue

                if isinstance(value, list):
                    value_str = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = "" if value is None else str(value)

                if value_str == "":
                    continue

                label = str(key).replace("_", " ").strip().capitalize()
                relevance_lines.append(f"\n {label}: {value_str}")

            if len(relevance_lines) > 1:
                candidate_docs[target_key] += "".join(relevance_lines)
                appended_candidates += 1

    return candidate_docs

def _add_candidate_summary(topic_id, candidate_docs, summary_path=None):
    logger.info(f"Adding candidate summaries for topic {topic_id}")
    if summary_path is None:
        logger.warning("No summary path provided; skipping summary augmentation.")
        return candidate_docs

    summary_file = Path(summary_path)
    if not summary_file.exists():
        logger.warning(f"Summary file not found at {summary_file}; skipping summary augmentation.")
        return candidate_docs

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_json = json.load(f)

    if not isinstance(summary_json, dict):
        logger.warning("Summary payload is not a dictionary; skipping summary augmentation.")
        return candidate_docs

    summary_key_map: Dict[str, str] = {}
    for raw_id in summary_json.keys():
        full_id = _normalize_patent_id(raw_id)
        base_id = _base_patent_id(raw_id)
        summary_key_map.setdefault(full_id, raw_id)
        summary_key_map.setdefault(base_id, raw_id)

    matched_candidates = 0
    appended_candidates = 0

    for candidate_id in candidate_docs.keys():
        candidate_key = _normalize_patent_id(candidate_id)
        summary_lookup_key = summary_key_map.get(candidate_key) or summary_key_map.get(
            _base_patent_id(candidate_key)
        )
        if summary_lookup_key is None:
            continue

        summary = summary_json.get(summary_lookup_key)
        if not summary:
            continue

        matched_candidates += 1
        summary_lines = ["\n\n<Summary>"]

        if isinstance(summary, dict):
            for key, value in summary.items():
                if "query" in str(key).lower():
                    continue

                if isinstance(value, list):
                    value_str = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value_str = json.dumps(value, ensure_ascii=False)
                else:
                    value_str = "" if value is None else str(value)

                if value_str == "":
                    continue

                label = str(key).replace("_", " ").strip().capitalize()
                summary_lines.append(f"\n {label}: {value_str}")
        else:
            value_str = str(summary).strip()
            if value_str:
                summary_lines.append(f"\n {value_str}")

        if len(summary_lines) > 1:
            candidate_docs[candidate_id] += "".join(summary_lines)
            appended_candidates += 1

    logger.info(
        "Topic %s summary augmentation: matched=%s appended=%s total_candidates=%s",
        topic_id,
        matched_candidates,
        appended_candidates,
        len(candidate_docs),
    )

    return candidate_docs


def _load_clusters_map(clusters_path: str) -> Dict[str, Dict[str, List[str]]]:
    path = Path(clusters_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        logger.warning(f"Cluster file not found: {path}. Falling back to standard reranking.")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load cluster file {path}: {exc}. Falling back to standard reranking.")
        return {}

    if not isinstance(data, dict):
        logger.warning(f"Invalid cluster file format at {path}. Falling back to standard reranking.")
        return {}

    return data


def _build_topic_cluster_groups(
    topic_id: str,
    candidate_ids: List[str],
    clusters_map: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    topic_clusters = clusters_map.get(str(topic_id), {})
    cluster_groups: Dict[str, List[str]] = {}

    candidate_set = set(candidate_ids)
    assigned = set()
    duplicate_count = 0

    if isinstance(topic_clusters, dict):
        for cluster_name, cluster_items in topic_clusters.items():
            if not isinstance(cluster_items, list):
                continue

            filtered_cluster = []
            for candidate_id in cluster_items:
                cid = str(candidate_id)
                if cid not in candidate_set:
                    continue
                if cid in assigned:
                    duplicate_count += 1
                    continue
                filtered_cluster.append(cid)
                assigned.add(cid)

            if filtered_cluster:
                cluster_groups[str(cluster_name)] = filtered_cluster

    # Keep every candidate eligible by assigning missing docs to singleton clusters.
    for cid in candidate_ids:
        if cid not in assigned:
            cluster_groups[f"singleton_{cid}"] = [cid]

    if duplicate_count:
        logger.warning(
            f"Topic {topic_id}: {duplicate_count} duplicate candidate IDs found across clusters; keeping first assignment."
        )

    return cluster_groups


def _load_pass_seeds(seed_file_path: str, passes: int) -> List[int]:
    if passes <= 1:
        return []

    path = Path(seed_file_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(
            f"Seed file not found at {path}. Multi-pass mode requires a valid seed file."
        )

    seeds: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            value = raw_line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                seeds.append(int(value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid seed value '{value}' in {path}. Expected one integer per line."
                ) from exc

    if not seeds:
        raise ValueError(f"Seed file {path} does not contain any usable seeds.")

    return seeds


def _topic_shuffle_seed(base_seed: int, topic_id: str) -> int:
    digest = hashlib.md5(str(topic_id).encode("utf-8")).hexdigest()
    topic_offset = int(digest[:8], 16)
    return int(base_seed) + topic_offset




def _fuse_rrf_rankings(
    topic_pass_rankings: Dict[str, List[List[str]]],
    rrf_k: int,
) -> pd.DataFrame:
    fused_rows: List[Dict[str, object]] = []
    for topic_num, pass_rankings in topic_pass_rankings.items():
        score_map: Dict[str, float] = defaultdict(float)
        for ranking in pass_rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                score_map[doc_id] += 1.0 / (rrf_k + rank)

        sorted_docs = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        fused_rows.extend(
            {"topic": topic_num, "number": doc_id, "score": score}
            for doc_id, score in sorted_docs
        )

    return pd.DataFrame.from_records(fused_rows)



async def main(cfg: hl.Config,retrieve=True,evaluate=True) -> None:

    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    logger.info(f"Output directory: {cfg.output_dir}")
    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
    
    topics = utils.load_topics(cfg.test_topics_path)
    topics = [t for t in topics if t in cfg.topics] if cfg.topics else topics

    logger.info(f"Loaded {len(topics)} topics")
    candidates_dict = utils.load_retreived_docs(cfg.candidates_path,k=cfg.topk)
    eval_topics = list(candidates_dict.keys()) if cfg.topics is None else cfg.topics

    if cfg.q and len(eval_topics) > cfg.q and cfg.topics is None:
        try:
            eval_topics = utils.read_topics(cfg.q)
            logger.info(f"Using  topics from {cfg.q}topics.txt")
        except Exception as e:
            logger.error(f"Failed to read topics from {cfg.q}topics.txt: {e}")
            eval_topics = eval_topics[:cfg.q]

        cfg.topics = eval_topics
        #cfg.topics = list(eval_topics) if cfg.topics is None else cfg.topics
    logger.info(f"Evaluating on {len(eval_topics)} topics")
    cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())
    pbar = utils.RichTableProgress(total=cfg.passes*len(eval_topics), print_every=5)

    use_cluster_tournament = bool(getattr(cfg, "use_cluster_tournament", False))
    if use_cluster_tournament and cfg.type.lower() != "listwise":
        logger.warning("Cluster tournament is currently supported only for listwise reranker. Falling back to standard reranking.")
        use_cluster_tournament = False

    clusters_map = {}
    if use_cluster_tournament:
        clusters_map = _load_clusters_map(cfg.clusters_path)
        logger.info(f"Cluster tournament enabled for {len(clusters_map)} topics from {cfg.clusters_path}")

    patent_reranker=reranker.get_reranker(
        type=cfg.type,
        backend=cfg.backend,
        model_name=cfg.model_name,
        mode=cfg.mode,
        thinking=cfg.thinking,
        api_key=API_KEY,
        base_url=BASE_URL, 
        n=cfg.window_size,
        passes=cfg.passes,
        remap_ids=cfg.remap_ids,
        prompt_id=cfg.prompt_id,
    )

    topic_inputs = []
    for i, topic_num in enumerate(eval_topics):
        topic_file = next(
            cfg.test_topics_path.parent.glob(f"PAC_topics/files/{topic_num}*.xml"),
            None,
        )
        if topic_file is None:
            logger.warning(
                "Skipping topic %s because no topic XML was found under PAC_topics/files.",
                topic_num,
            )
            continue

        topic_patent = dataset.parse_patent([topic_file])[0]
        with sqlm.Session(engine) as session:
            candidates_ids = candidates_dict[topic_num]
           #logger.info(f"Preparing topic {topic_num} with {len(candidates_ids)} candidates")
            candidate_patents = list(
                session.exec(
                    sqlm.select(dataset.Patent)
                    .where(dataset.Patent.number.in_(candidates_ids))
                )
            )

            if not candidate_patents:
                logger.warning("Skipping topic %s because no candidate patents were found in DB.", topic_num)
                continue

            # Reorder to match candidates_ids order
            candidate_rank = {candidate_id: idx for idx, candidate_id in enumerate(candidates_ids)}
            fallback_rank = len(candidate_rank)
            candidate_patents = sorted(
                candidate_patents,
                key=lambda p: candidate_rank.get(p.number, fallback_rank),
            )

            query = dataset.extract_query_text(
                topic_patent,
                cfg.query_columns,
                cfg.claims,
                independent_only=cfg.independent_claims,
            )
            candidate_docs = {
                patent.number: dataset.extract_query_text(
                    patent,
                    cfg.query_columns,
                    cfg.claims,
                    independent_only=cfg.independent_claims,
                )
                for patent in candidate_patents
            }

            relevance = cfg.relevance and cfg.relevance_analysis_path is not None
            if relevance:
                candidate_docs = _add_relevance_assessment(
                    topic_id=topic_num,
                    candidate_docs=candidate_docs,
                    relevance_analysis_path=cfg.relevance_analysis_path
                )
            elif cfg.summary and cfg.candidates_summary_path is not None:
                candidate_docs = _add_candidate_summary(
                    topic_id=topic_num,
                    candidate_docs=candidate_docs,
                    summary_path=cfg.candidates_summary_path,
                )
            else:
                pass

            cluster_groups = None
            if use_cluster_tournament:
                cluster_groups = _build_topic_cluster_groups(
                    topic_id=topic_num,
                    candidate_ids=list(candidate_docs.keys()),
                    clusters_map=clusters_map,
                )

        topic_inputs.append((topic_num, query, candidate_docs, cluster_groups))

    logger.info(f"All {len(topic_inputs)} topics prepared. Starting async reranking ({cfg.passes} pass(es))...")

    # Build immutable base topic data for multi-pass reranking.
    topic_docs = {
        topic_num: {
            "query": query,
            "docs": candidate_docs,
            "clusters": cluster_groups,
            "original_order": list(candidate_docs.keys()),
        }
        for topic_num, query, candidate_docs, cluster_groups in topic_inputs
    }

    pass_seeds = _load_pass_seeds(cfg.seed_file_path, cfg.passes)
    if cfg.passes > 1:
        logger.info(
            "Loaded %s seeds from %s. Pass 1 keeps initial order; later passes use seeded shuffles.",
            len(pass_seeds),
            cfg.seed_file_path,
        )

    semaphore = asyncio.Semaphore( max(20, cfg.semaphore) )
    pass_df = None
    pass_metrics = None
    pass_metrics_json = None
    num_failed_topics = []
    failure_rates = []
    topic_pass_rankings: Dict[str, List[List[str]]] = defaultdict(list)
    for pass_num in range(1, cfg.passes + 1):
        logger.info(f"=== Pass {pass_num}/{cfg.passes} ===")

        pass_seed = None
        if cfg.passes > 1 and pass_num > 1:
            pass_seed = pass_seeds[(pass_num - 2) % len(pass_seeds)]
            logger.info("[Pass %s] Shuffle seed: %s", pass_num, pass_seed)


        async def _rerank_topic(topic_id, data, _pass=pass_num):
            async with semaphore:
                if _pass == 1:
                    rerank_docs = data["docs"]
                else:
                    shuffled_ids = list(data["original_order"])
                    rng = random.Random(_topic_shuffle_seed(int(pass_seed or 0), str(topic_id)))
                    rng.shuffle(shuffled_ids)
                    rerank_docs = {
                        doc_id: data["docs"][doc_id]
                        for doc_id in shuffled_ids
                        if doc_id in data["docs"]
                    }

                logger.info(f"[Pass {_pass}] Reranking topic {topic_id}...")
                try:
                    rerank_results, failed = await asyncify(patent_reranker.rerank)(
                        query=data["query"],
                        docs=rerank_docs,
                        cluster_groups=data["clusters"] if cfg.use_cluster_tournament else None,
                        cluster_tournament=cfg.use_cluster_tournament,
                    )

                except Exception as e:
                    logger.warning(f"[Pass {_pass}] Reranking failed for topic {topic_id}: {e}")
                    logger.debug(traceback.format_exc())
                    rerank_results, failed = [], True


                logger.info(f"[Pass {_pass}] Topic {topic_id} done.")
                return topic_id, rerank_results, failed

        tasks = [_rerank_topic(tn, topic_docs[tn]) for tn in topic_docs]
        all_results = await tqdm.gather(*tasks, desc=f"Pass: {pass_num}/{cfg.passes}")

        # Collect pass results; next pass ordering is independently built from original order and seed.
        pass_results = []
        pass_failed_topics = []
        for topic_num, rerank_results, failed in all_results:
            pass_results.extend(
                {"topic": topic_num, "number": mid, "score": score}
                for mid, score in rerank_results
            )
            topic_pass_rankings[topic_num].append([mid for mid, _ in rerank_results])
            if failed:
                pass_failed_topics.append(topic_num)
            pbar.update()

        # Save intermediate results
        
        pass_df = pd.DataFrame.from_records(pass_results)
        pass_num_failed_topics = len(pass_failed_topics)
        pass_failure_rate = (pass_num_failed_topics / len(topic_docs) * 100) if topic_docs else 0.0

        if pass_failed_topics:
            logger.warning(f"Pass {pass_num} had failures in {pass_num_failed_topics} topics: {pass_failed_topics}")
            logger.warning(f"Pass {pass_num} failure rate: {pass_failure_rate:.2f}%")

        num_failed_topics.append(pass_num_failed_topics)
        failure_rates.append(pass_failure_rate)
            
        if cfg.passes > 1:
            pass_file = cfg.output_dir / f"results_pass_{pass_num}.csv"
            pass_df.to_csv(pass_file, index=False)
            logger.info(f"Pass {pass_num} results saved to {pass_file}")

            if evaluate:  # Only evaluate after the first pass to save time, can be changed as needed
                pass_metrics = utils.calculate_metrics(
                    results=pass_df,
                    test_topics_path=cfg.test_topics_path,
                    topk=cfg.topk,
                )
                #pass_metrics["pass"] = pass_num
                pass_metrics["num_failed_topics"] = pass_num_failed_topics
                pass_metrics["failure_rate"] = pass_failure_rate

                pass_metrics_json = json.dumps(pass_metrics, indent=4)
                print(f"Pass {pass_num} metrics:\n{pass_metrics_json}")
                cfg.output_dir.joinpath(f"metrics_pass_{pass_num}.json").write_text(pass_metrics_json)

    if cfg.passes > 1:
        results = _fuse_rrf_rankings(topic_pass_rankings, int(getattr(cfg, "rrf_k", 60)))
        logger.info(
            f"Fused rankings across {cfg.passes} pass(es) with RRF."
        )
    else:
        results = pass_df
        
        

    if evaluate:
        pass_metrics = utils.calculate_metrics(
            results=results,
            test_topics_path=cfg.test_topics_path,
            topk=cfg.topk,
        )
        pass_metrics["num_failed_topics"] = sum(num_failed_topics)/len(num_failed_topics) if num_failed_topics else 0
        pass_metrics["failure_rate"] = sum(failure_rates)/len(failure_rates) if failure_rates else 0.0
        pass_metrics_json = json.dumps(pass_metrics, indent=4)

    results_file = cfg.output_dir / "results.csv"
    results.to_csv(results_file, index=False)
    logger.info(f"Final results saved to {results_file}")

    if evaluate:
        if pass_metrics_json is not None:
            cfg.output_dir.joinpath("metrics.json").write_text(pass_metrics_json)
            print(pass_metrics_json)

        bootstrap_payload = None
        bootstrap_file = None
        if cfg.bootstrapp:
            if results.empty:
                logger.warning("Bootstrap enabled but final results are empty; skipping bootstrap computation.")
            else:
                per_topic_metrics_df = utils.calculate_per_topic_metrics(
                    results=results,
                    topk=cfg.topk,
                    test_topics_path=cfg.test_topics_path,
                )
                per_topic_metrics_df.to_csv(cfg.output_dir / "per_topic_metrics.csv", index=False)
                bootstrap_payload = utils.bootstrap_recall_ndcg(
                    per_topic_metrics_df=per_topic_metrics_df,
                    confidence_level=float(getattr(cfg, "bootstrap_confidence", 0.95)),
                    seed=int(getattr(cfg, "bootstrap_seed", 42)),
                    n_bootstrap=max(1000, int(per_topic_metrics_df["q_id"].nunique())) if not per_topic_metrics_df.empty else 0,
                )
                bootstrap_file = cfg.output_dir / "bootstrap_metrics.json"
                bootstrap_file.write_text(json.dumps(bootstrap_payload, indent=4))
                logger.info(f"Bootstrap metrics saved to {bootstrap_file}")
                logger.info(f"Bootstrap results:\n{json.dumps(bootstrap_payload, indent=4)}")

        if cfg.wandb:

            if pass_metrics["failure_rate"] < 95:
                wandb.log(pass_metrics)
                if bootstrap_payload is not None:
                    wandb.log({"bootstrap": bootstrap_payload})
                    if bootstrap_file is not None and bootstrap_file.exists():
                        bootstrap_artifact = wandb.Artifact(
                            name=f"bootstrap-metrics-{get_wandb_name(str(cfg.output_dir))}",
                            type="evaluation",
                        )
                        bootstrap_artifact.add_file(str(bootstrap_file))
                        wandb.log_artifact(bootstrap_artifact)
            else:
                logger.warning(f"Overall failure rate {pass_metrics['failure_rate']:.2f}%; skipping wandb logging.")
                wandb.log(
                    {
                        "num_failed_topics": pass_metrics["num_failed_topics"],
                        "failure_rate": pass_metrics["failure_rate"]
                     
                    }
                    )
            if run is not None:
                run.finish()
        logger.info(f"Final results saved to {results_file}")


def get_wandb_name(s):
        return re.sub(r'_\d{4}-\d{2}-\d{2}(-\d{2}:\d{2}:\d{2})?$', '', os.path.basename(s))

if __name__ == "__main__":
    cfg.apply()
    #print(cfg.to_yaml())
    if cfg.wandb:
        settings = wandb.Settings(
                save_code=False,      
                x_disable_meta=True,
                x_disable_stats=True,  
                
            )
        
        
        
        
        run = wandb.init(project="patent-reranking-v2",
                    name=get_wandb_name(cfg.output_dir), 
                    config=cfg.to_dict(),
                    #group=cfg.model_name,
                    config_exclude_keys=["db_path","test_topics_path","tags","unilingual_index","wandb","seed_file_path",],
                    settings=settings,
                    tags=cfg.tags
                    )

    asyncio.run(main(cfg))