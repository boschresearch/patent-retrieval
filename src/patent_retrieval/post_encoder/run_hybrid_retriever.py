# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import rich.syntax

import hydralette as hl
import pandas as pd
import wandb
from pyrootutils import setup_root

from patent_retrieval import dataset, utils
from patent_retrieval.post_encoder.hybrid_retriever import IndexSpec, build_hybrid_retriever

root = setup_root(__file__)
os.environ["WANDB_DIR"] = str(root)
logger = utils.get_logger(__name__)


def get_run_dir(cfg: hl.Config) -> Path:
    out_dir = root / "embeddings" / "runs" / f"{cfg.run_name}_{cfg.fusion_method}_{cfg.q if cfg.q is not None else 'all'}-topics_top{cfg.k}"
    if out_dir.exists():
        out_dir = out_dir.parent / f"{out_dir.name}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_run_name(cfg: hl.Config) -> Path:
    
    return f"hybrid_{cfg.index1_name.split('_')[0]}_{cfg.index2_name.split('_')[0]}_{cfg.fusion_method}_{cfg.q if cfg.q is not None else 'all'}-topics_top{cfg.k}"

def get_index_dir(cfg: hl.Config,index_name) -> Path:
    search_columns = cfg.doc_columns.copy()

    if "claims" in search_columns and cfg.index_claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.index_claims}claims"""

    index_dir: Path = Path("/home")/"alm3rng"/"scratch" /  "clef_ip_2011" / (f"{index_name}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")) 
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)

    return index_dir
BASE_URL1 = "http://localhost:59749/v1"
BASE_URL2 = "http://localhost:59689/v1"
cfg = hl.Config(
    
    index1_type="dense",
    index1_name="patQwen3-emb-4b-v2_db-v4",
    index2_type="dense",
    index2_name="llama-nemo-8b_db-v4",
    run_name=hl.Field(reference=get_run_name),
    fusion_method="min_max", # rrf or min_max
    rrf_k=0,
    min_max_weights=[0.7,0.3],
    k=1000,
    q=500,
    topics=None,
    claims=None,
    independent_claims=False,
    index_claims=None, 
    doc_columns=hl.Field(
        default=["title","abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    query_columns=hl.Field(
        default=["abstract","claims"],
        convert=lambda x: x.split(","),
    ),
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs.txt"
    ),
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    unilingual_index=False,
    language="",
    wandb=True,
    tags=[],
   
    index1_path=hl.Field(reference=lambda cfg: get_index_dir(cfg, cfg.index1_name)),
    index1_backend="openai",
    index1_model_name="patQwen3-emb-4b-v2",
    index1_tokenizer="/home/alm3rng/patent-retrieval/finetuning/runs/optuna/trial_11/checkpoint-95",
    index1_base_url=BASE_URL1,
    
    
    index2_path=hl.Field(reference=lambda cfg: get_index_dir(cfg, cfg.index2_name)),
    index2_backend="openai",
    index2_model_name="nvidia/llama-embed-nemotron-8b",
    index2_tokenizer="nvidia/llama-embed-nemotron-8b",
    index2_base_url=BASE_URL2,

)


def _build_index_specs(config: hl.Config) -> List[IndexSpec]:
    return [
        IndexSpec(
            type=str(config.index1_type).lower(),
            path=Path(config.index1_path),
            backend=str(config.index1_backend).lower(),
            model_name=config.index1_model_name,
            tokenizer=config.index1_tokenizer,
            base_url=config.index1_base_url,
        ),
        IndexSpec(
            type=str(config.index2_type).lower(),
            path=Path(config.index2_path),
            backend=str(config.index2_backend).lower(),
            model_name=config.index2_model_name,
            tokenizer=config.index2_tokenizer,
            base_url=config.index2_base_url,
        ),
    ]



def run_hybrid_experiment(config: hl.Config, wandb_run=None) -> None:
    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))

    specs = _build_index_specs(config)
    logger.info("Building hybrid retriever with the following index specs:")
    for spec in specs:
        logger.info(f"Index type={spec.type}, backend={spec.backend}, model={spec.model_name}, path={spec.path}")
    retriever = build_hybrid_retriever(
        index_specs=specs,
        fusion_method=config.fusion_method,
        weights=config.min_max_weights,
        rrf_k=config.rrf_k,
    )
    logger.info(f"Hybrid retriever built successfully.{retriever}")

    topics = utils.load_topics(config.test_topics_path)
    if config.topics:
        allowed = set(config.topics)
        topics = [topic for topic in topics if topic in allowed]
    elif config.q:
        try:
            topics = utils.read_topics(config.q)
            logger.info(f"Using  topics from {config.q}topics.txt")
        except Exception as e:
            logger.error(f"Failed to read topics from {config.q}topics.txt: {e}")
            topics = topics[:config.q]
    config.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())

    logger.info("Loaded %d topics for hybrid retrieval.", len(topics))
    pbar = utils.RichTableProgress(total=len(topics), print_every=100)

    rows = []
    for topic_num in topics:    
        topic_file = next(
            config.test_topics_path.parent.glob(f"PAC_topics/files/{topic_num}*.xml"),
            None,
        )
        if topic_file is None:
            logger.warning("Topic file not found for topic=%s", topic_num)
            pbar.update()
            continue
        topic_patent = dataset.parse_patent([topic_file])[0]
        query = dataset.extract_query_text(
            patent=topic_patent,
            search_columns=config.query_columns,
            kclaims=config.claims,
            independent_only=config.independent_claims,
        )
        search_results = retriever.search(query=query, k=config.k)

        rows.extend(
            {"topic": topic_num, "number": doc_id, "score": score}
            for doc_id, score in search_results
        )
        
        pbar.update()

    results_df = pd.DataFrame.from_records(rows)
    results_file = config.output_dir / "results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info("Results saved to %s", results_file)

    metrics = utils.calculate_metrics(
        results=results_file,
        topk=config.k,
        test_topics_path=config.test_topics_path,
    )
    metrics_json = json.dumps(metrics, indent=4)
    print(metrics_json)
    config.output_dir.joinpath("metrics.json").write_text(metrics_json)

    if config.wandb and wandb_run is not None:
        wandb.log(metrics)
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
            project="patent-retrieval-hybrid",
            name=cfg.run_name,
            config=cfg.to_dict(),
            group="hybrid",
            config_exclude_keys=["test_topics_path", "wandb", "tags"],
            settings=settings,
            tags=cfg.tags,
        )

    run_hybrid_experiment(cfg, wandb_run=run)
