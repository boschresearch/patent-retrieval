# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
import asyncio
from pyrootutils import setup_root
import re
from datetime import datetime
import os
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from typing import List, Optional
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
import gc
from sklearn.metrics import precision_score, recall_score, f1_score
from asyncer import asyncify
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
#import patent_retrieval
from patent_retrieval import utils as utils, dataset as dataset, reranker as reranker, agents as agents
import torch
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
        search_columns[idx] = f"{cfg.claims}claims" if cfg.claims is not None else f"claims"


    out_dir: Path = root / "reranking"/ "runs" / (f"{cfg.run_name}_{'-'.join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}"))
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}_{'-'.join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_index_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.doc_columns.copy()

    #if "claims" in search_columns:
     #   idx = search_columns.index("claims")
     #   search_columns[idx] = f"{cfg.claims}claims" if cfg.claims is not None else f"claims"

    index_dir: Path = Path("/home")/"alm3rng"/"scratch" /  "clef_ip_2011" / (f"{cfg.run_name}_{'-'.join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")) 
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)

    return index_dir
BASE_URL="http://rng-dl01-w26n09:53072/v1"
API_KEY="EMPTY"
#    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_9topics_top1000/results.csv",
#    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_all-topics_abstract-claims_aysm_top1000/results.csv",
candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_30topics_top1000_2026-02-25-11:16:02/results.csv"
cfg = hl.Config(
    type="listwise",
    backend="openai",
    mode="sliding_window", # "batch" or "sliding_window"
    #method="",
    run_name="qwen3.5-397b_8b_v4",#"qwen3_rerank_4b_v4_100topics_rewrite",#"qwen3_30b_instruct_3topics_top100",#"bm25" # qwen3_emb_4b
    model_name="Qwen/Qwen3.5-397B-A17B-FP8",# "Qwen/Qwen3-30B-A3B-Instruct-2507",#"Qwen/Qwen3-Reranker-4B" # mpi-inno-comp/paecter
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4.db",
    candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_abstract-claims_aysm_30topics_top1000_2026-02-25-11:16:02/results.csv",
    claims=None,
    independent_claims=True,
    target_claims=None,
    #index_dir= hl.Field(reference=get_index_dir, type=Path),
    doc_columns=hl.Field(
        default=["abstract","claims","description"],
        convert=lambda x: x.split(","),
    ),
    query_columns=hl.Field(
        default=["abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    wandb=True,
    tags=["rerank","listwise","async"],
    #store_type="",
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    topk=200,
    q=30,
    topics=None
)




def evaluate_docs(path:str ):
    utils.evaluate_docs(
        path=path,
        test_topics_path=cfg.test_topics_path,
        topk=cfg.topk,
        output_dir=cfg.output_dir,
        enable_wandb=cfg.wandb,
        run=globals().get("run"),
    )
    return

async def main(cfg: hl.Config,retrieve=True,evaluate=True) -> None:

    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    
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
    pbar = utils.RichTableProgress(total=len(eval_topics), print_every=5)
    results = []

    patent_reranker=reranker.get_reranker(
        type=cfg.type,
        backend=cfg.backend,
        model_name=cfg.model_name,
        mode=cfg.mode,
        api_key=API_KEY,
        base_url=BASE_URL, 
        n=10,
    )

    topic_inputs = []  
    for i, topic_num in enumerate(eval_topics):
        topic_file = next(
            cfg.test_topics_path.parent.glob(f"PAC_topics/files/{topic_num}*.xml")
        )
        topic_patent = dataset.parse_patent([topic_file])[0]
        with sqlm.Session(engine) as session:
            candidates_ids = candidates_dict[topic_num]
            logger.info(f"Preparing topic {topic_num} with {len(candidates_ids)} candidates")
            candidate_patents = list(
                session.exec(
                    sqlm.select(dataset.Patent)
                    .where(dataset.Patent.number.in_(candidates_ids))
                )
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
        topic_inputs.append((topic_num, query, candidate_docs))

    logger.info(f"All {len(topic_inputs)} topics prepared. Starting async reranking...")


    semaphore = asyncio.Semaphore(20)  

    async def _process_topic(topic_num, query, candidate_docs):
        async with semaphore:
            logger.info(f"Reranking topic {topic_num}...")
            rerank_results = await asyncify(patent_reranker.rerank)(query=query, docs=candidate_docs)
            logger.info(f"Topic {topic_num} done.")
            return [
                {"topic": topic_num, "number": match_num, "score": score}
                for match_num, score in rerank_results
            ]

    all_tasks = [
        _process_topic(topic_num, query, candidate_docs)
        for topic_num, query, candidate_docs in topic_inputs
    ]
    topic_results = await asyncio.gather(*all_tasks)

    results = []
    for topic_result in topic_results:
        results.extend(topic_result)


    results = pd.DataFrame.from_records(results)
    results_file = cfg.output_dir / "results.csv"
    results.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")

    if evaluate:
        metrics = utils.calculate_metrics(
            results=results_file,
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
        logger.info(f"Results saved to {results_file}")


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
        
        
        
        
        run = wandb.init(project="patent-reranking-v2",
                    name=get_wandb_name(cfg.output_dir), 
                    config=cfg.to_dict(),
                    #group=cfg.model_name,
                    config_exclude_keys=["db_path","test_topics_path","tags","unilingual_index","wandb"],
                    settings=settings,
                    tags=cfg.tags
                    )

    asyncio.run(main(cfg))