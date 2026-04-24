# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import asyncio
from collections import defaultdict
import os
from pyrootutils import setup_root
import re
from datetime import datetime
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List, Optional
import hydralette as hl
import pandas as pd
import rich.syntax
import sqlalchemy as sqla
import sqlmodel as sqlm
from pyrootutils import setup_root
import traceback
import wandb
import json
import numpy as np
from itertools import islice
from sklearn.metrics import precision_score, recall_score, f1_score
from dotenv import load_dotenv
from patent_retrieval import utils as utils, dataset as dataset, reranker as reranker, agents as agents
import torch
from multiprocessing import Pool
from functools import partial
load_dotenv()
torch.cuda.empty_cache()


root = setup_root(__file__)
logger = utils.get_logger(__name__)


os.environ["WANDB_DIR"] = str(root)


def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.query_columns.copy()

    asymmetric = set(cfg.query_columns) != set(cfg.doc_columns)

    if "claims" in search_columns and cfg.claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"


    out_dir: Path = root / "prefilter"/ "runs" / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_judge{cfg.judge_version}" if cfg.judge_version else "")+(f"_{cfg.language}" if cfg.language else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}"))
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_judge{cfg.judge_version}" if cfg.judge_version else "")+(f"_{cfg.language}" if cfg.language else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


#BASE_URL= #BASE_URL=os.getenv("AZURE_OPENAI_ENDPOINT")


BASE_URL="http://rng-dl01-w26a02:42799/v1"
API_KEY="EMPTY"
#candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_30topics_top1000_2026-02-25-11:16:02/results.csv"
#candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_9topics_top1000/results.csv",


cfg = hl.Config(
    type="prefilter",
    #method="",
    run_name="qwen3.5-397b_db-v4",#"qwen3_rerank_4b_v4_100topics_rewrite",#"qwen3_30b_instruct_3topics_top100",#"bm25" # qwen3_emb_4b
    backend="openai",#"openai"
    suffix="_patQwen3-based",
    model_name="Qwen/Qwen3.5-397B-A17B-FP8",# "Qwen/Qwen3-30B-A3B-Instruct-2507",#"Qwen/Qwen3-Next-80B-A3B-Thinking" # mpi-inno-comp/paecter
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    #candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_30topics_top1000_2026-02-25-11:16:02/results.csv",
    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_500topics_top1000/results.csv",
    claims=None,
    independent_claims=True,
    target_claims=None,
    desc_max_tokens=500,
    #index_dir= hl.Field(reference=get_index_dir, type=Path),
    doc_columns=hl.Field(
        default=["abstract","claims","description"],
        convert=lambda x: x.split(","),
    ),
    query_columns=hl.Field(
        default=["claims" ],
        convert=lambda x: x.split(","),
    ),
    wandb=True,
    judge_version="v5",
    tags=["prefilter","prompt_v1"],
    #store_type="",
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    topk=100,
    q=500,
    semaphore=50,

    topics=None
)


def process_candidate_worker(candidate, *, query: str, doc_columns: List[str], claims: int, db_path: str, judge_model: str, topic_language: Optional[str] = None):
    """Worker function to be pickled and run in child processes.

    Opens its own DB session, loads the candidate by ID, constructs the candidate text,
    runs the judge, and returns a mapping {candidate_number: judgment}.
    """
    try:

        candidate_text = dataset.extract_query_text(candidate, doc_columns, claims, independent_only=cfg.independent_claims)

        # Create a local judge instance per worker to avoid sharing non-picklable state
        local_judge = agents.PatentJudgeAgent(
            model=cfg.model_name,
            base_url=BASE_URL,
            api_key=API_KEY
        )
        
        judgment = local_judge.judge(query, candidate_text)

        return {candidate.number: judgment}
    except Exception as e:
        traceback.print_exc()
        return {candidate.number: {"error": str(e)}}


def extract_single_score(judgment: dict) -> float:
    """Extract exactly one numeric score-like key from a judgment dict."""
    if not isinstance(judgment, dict):
        raise ValueError(f"Expected dict judgment, got {type(judgment).__name__}")

    score_keys = [key for key in judgment.keys() if "score" in str(key).lower()]
    if len(score_keys) != 1:
        raise ValueError(f"Expected exactly one score key, found {score_keys}")

    score_key = score_keys[0]
    try:
        return float(judgment[score_key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Score value for key '{score_key}' is not numeric: {judgment[score_key]}") from e


async def main(cfg: hl.Config,retrieve=True,evaluate=True) -> None:

    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    
    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
    #topics = utils.load_topics(cfg.test_topics_path)
    
   # topics = [t for t in topics if t in cfg.topics] if cfg.topics else topics
    
    cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())
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
    #eval_topics = [t for t in eval_topics if t in cfg.topics] if cfg.topics else eval_topics
    results_json = cfg.output_dir / "results.json"
    results = defaultdict(list)
    persisted_results = {}
    """
    if results_json.exists():
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                loaded_results = json.load(f)
            if isinstance(loaded_results, dict):
                persisted_results = loaded_results
                logger.info(f"Resuming from existing results file: {results_json}")
            else:
                logger.warning(
                    f"Expected dict in {results_json}, got {type(loaded_results).__name__}. Starting with empty results."
                )
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {results_json}: {e}. Cleaning and starting with empty results.")
            with open(results_json, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    existing_topic_keys = {str(topic_id) for topic_id in persisted_results.keys()}
    original_topic_count = len(eval_topics)
    eval_topics = [topic for topic in eval_topics if str(topic) not in existing_topic_keys]
    skipped_topics = original_topic_count - len(eval_topics)

    logger.info(
        f"Evaluating {len(eval_topics)} pending topics (skipped {skipped_topics} already present in {results_json.name})"
    )
    
    

    
    for topic_id, topic_results in persisted_results.items():
        if isinstance(topic_results, list):
            results[topic_id] = topic_results

    # Ensure output exists even if no topics are pending.
    if not results_json.exists():
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(persisted_results, f, ensure_ascii=False, indent=2)
    """
    pbar = utils.RichTableProgress(total=len(eval_topics), print_every=5)
    judge = agents.PatentJudgeAgent(
                    model=cfg.model_name,
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    backend=cfg.backend
                )
    
    for i,topic_num in enumerate(eval_topics):
        topic_file = next(
                cfg.test_topics_path.parent.glob(f"PAC_topics/files/{topic_num}*.xml")
            )
        topic_patent = dataset.parse_patent([topic_file])[0]
 
        with sqlm.Session(engine) as session:
                
            candidate_patents =  list(
                    session.exec(
                        sqlm.select(dataset.Patent)
                        .where(dataset.Patent.number.in_(candidates_dict[topic_num]))
                        #.offset(3)
                    )
                )
            
            candidate_texts = list(map(lambda candidate: dataset.extract_query_text(candidate, cfg.doc_columns, cfg.target_claims,
                                    independent_only=cfg.independent_claims, desc_max_tokens=cfg.desc_max_tokens), candidate_patents))
            
            query = dataset.extract_query_text(
                    topic_patent,
                    cfg.query_columns,
                    cfg.claims,
                    independent_only=cfg.independent_claims,
                )
        
        
        semaphore = asyncio.Semaphore(cfg.semaphore)
        try:
            judge_func = getattr(judge, f"judge_{cfg.judge_version}")
        except AttributeError:
            logger.error(f"Judge version {cfg.judge_version} not found. Falling back to judge_v1.")
            judge_func = judge.judge_v1
        
        async def sem_judge_candidate(candidate_text):
            async with semaphore:
                try:
                    #judgment = 
                    return await judge_func(query, candidate_text)
                except Exception as e:
                    return traceback.print_exc()
        
        #tasks = [sem_judge_candidate(candidate_text if len(candidate_text.split()) < 100000 else ' '.join(candidate_text.split()[:100000])) for candidate_text in candidate_texts]
        tasks = [sem_judge_candidate(candidate_text) for candidate_text in candidate_texts]
        judgments = await asyncio.gather(*tasks)
        
        for candidate, judgment in zip(candidate_patents, judgments):
            results[topic_num].append({candidate.number: judgment})

        # Persist this topic immediately to avoid losing progress on long runs.
        persisted_results[str(topic_num)] = results[topic_num]
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(persisted_results, f, ensure_ascii=False, indent=2)

        #results[topic_num].extend(judgments)
        
        pbar.update()

    pbar.update()
    

    cleaned_prefilter_results = {}
    for topic, candidates in results.items(): 
        cleaned_candidates = []
        for item in candidates: 
            cleaned_item = {}
            for number, judgment in item.items():
                if isinstance(judgment, dict):
                    cleaned_item[number] = judgment
            if cleaned_item:  # Only add if there are valid entries
                cleaned_candidates.append(cleaned_item)
        if cleaned_candidates:  # Only add topic if it has valid candidates
            cleaned_prefilter_results[topic] = cleaned_candidates
    records = []
    dropped_candidates = 0
    for topic, candidates in cleaned_prefilter_results.items():
        for item in candidates:
            for number, judgment in item.items():
                try:
                    records.append(
                        {
                            "topic": topic,
                            "number": number,
                            "score": extract_single_score(judgment),
                        }
                    )
                except ValueError as e:
                    dropped_candidates += 1
                    logger.warning(f"Skipping candidate {number} for topic {topic}: {e}")

    if dropped_candidates:
        logger.warning(f"Dropped {dropped_candidates} candidates due to invalid score payloads.")

    results_df = pd.DataFrame.from_records(records, columns=["topic", "number", "score"])
    if results_df.empty:
        logger.warning("No scored records were produced. Writing empty results and skipping metrics.")

    results_csv = cfg.output_dir / "results.csv"

    results_df.to_csv(results_csv, index=False)
    logger.info(f"Results saved to {results_csv}")
    

    if evaluate and not results_df.empty:
        metrics  = utils.calculate_metrics(
                results=results_csv,
                test_topics_path=cfg.test_topics_path,
                topk=cfg.topk,
            )
        
        metrics_json = json.dumps(metrics, indent=4)
        print(metrics_json)
        cfg.output_dir.joinpath("metrics.json").write_text(metrics_json)

        if cfg.wandb:
            wandb.log(metrics)
            if run is not None:
                run.finish()
        logger.info(f"Results saved to {results_csv}")
    elif evaluate:
        logger.warning("Metrics were skipped because results.csv is empty.")



def get_wandb_name(s):
        return re.sub(r'_\d{4}-\d{2}-\d{2}(-\d{2}:\d{2}:\d{2})?$', '', os.path.basename(s))

if __name__ == "__main__":
    cfg.apply()
    #print(cfg.to_yaml())
    if cfg.wandb:
        settings = wandb.Settings(
                save_code=False,      
                x_disable_meta=True,
                x_disable_stats=True  
            )

        
        
        run = wandb.init(project="relevance_analysis-prefilter",
                    name=get_wandb_name(cfg.output_dir), 
                    config=cfg.to_dict(),
                    #group=cfg.model_name,
                    config_exclude_keys=["db_path","test_topics_path","tags","unilingual_index","wandb"],
                    settings=settings,
                    tags=cfg.tags
                    )
    
    asyncio.run(main(cfg))

