# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import hydralette as hl
import pandas as pd
import rich
import rich.syntax
import sqlmodel as sqlm
import wandb
from dotenv import load_dotenv
from pyrootutils import setup_root

from patent_retrieval import agents as agents
from patent_retrieval import dataset as dataset
from patent_retrieval import utils as utils

load_dotenv()

root = setup_root(__file__)
logger = utils.get_logger(__name__)
os.environ["WANDB_DIR"] = str(root)

BASE_URL = "http://rng-dl01-w26a02:42799/v1"
API_KEY = "EMPTY"

def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.doc_columns.copy()



    if "claims" in search_columns and cfg.claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"


    out_dir: Path = root / "summary"/ "runs" / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_judge{cfg.version}" if cfg.version else "")+(f"_{cfg.language}" if cfg.language else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}"))
    
    #if out_dir.exists():
    #    out_dir = (
     #       out_dir.parent
     #       / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_judge{cfg.version}" if cfg.version else "")+(f"_{cfg.language}" if cfg.language else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.topk}")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
     #   )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

cfg = hl.Config(
    type="summarizer",
    run_name="qwen3.5-397b_db-v4",
    backend="openai",
    suffix="_patQwen3-based",
    model_name="Qwen/Qwen3.5-397B-A17B-FP8",
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
    desc_max_tokens=1000,
    doc_columns=hl.Field(
        default=["title","abstract", "claims", "description"],
        convert=lambda x: x.split(","),
    ),
    wandb=False,
    version="v2",
    tags=[],
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    topk=100,
    q=500,
    semaphore=200,
    topics=None,
)




def load_unique_candidate_numbers(candidates_path: Path, topk: int | None) -> list[str]:
    candidates_df = pd.read_csv(candidates_path, )
    if "number" not in candidates_df.columns:
        raise ValueError(f"Expected 'number' column in candidates CSV: {candidates_path}")

    numbers = candidates_df["number"].astype(str).dropna().drop_duplicates().tolist()
    if topk and topk > 0:
        return numbers[:topk]
    return numbers


def select_summarizer_func(summarizer: object, version: str):
    summarize_func_name = f"summarize_{version}"
    try:
        summarize_func = getattr(summarizer, summarize_func_name)
    except AttributeError:
        logger.error(
            f"Summarizer version {version} not found. Falling back to summarize_v1."
        )
        summarize_func = summarizer.summarize_v1

    return summarize_func


async def main(cfg: hl.Config) -> None:
    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))

    cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())
    summaries_json = cfg.output_dir / "results.json"

    summaries: dict[str, object] = {}
    if summaries_json.exists():
        try:
            existing_results = json.loads(summaries_json.read_text(encoding="utf-8"))
            if isinstance(existing_results, dict):
                summaries = {str(k): v for k, v in existing_results.items()}
                logger.info(
                    f"Found existing results with {len(summaries)} IDs. They will be skipped."
                )
            else:
                logger.warning(
                    "Existing results.json is not a dict. Ignoring it and starting fresh."
                )
        except Exception as e:
            logger.warning(f"Failed to load existing results.json: {e}")

    candidates_dict = utils.load_retreived_docs(cfg.candidates_path,k=cfg.topk)
    eval_topics = list(candidates_dict.keys()) if cfg.topics is None else cfg.topics

    if cfg.q and len(eval_topics) > cfg.q and cfg.topics is None:
        try:
            eval_topics = utils.read_topics(cfg.q)
            logger.info(f"Using  topics from {cfg.q}topics.txt")
        except Exception as e:
            eval_topics = eval_topics[:cfg.q]
            logger.error(f"Failed to read topics from {cfg.q}topics.txt: {e}")
    
        candidates_dict = {
            topic: candidates_dict[topic]
            for topic in eval_topics
            if topic in candidates_dict
        }

    candidate_numbers = list(
        dict.fromkeys(
            patent_num
            for topic_candidates in candidates_dict.values()
            for patent_num in topic_candidates
        )
    )

    pending_candidate_numbers = [
        candidate_num
        for candidate_num in candidate_numbers
        if candidate_num not in summaries
    ]
    logger.info(f"Loaded candidates for {candidate_numbers}")
    logger.info(
        f"Loaded {len(candidate_numbers)} unique candidates, {len(pending_candidate_numbers)} pending summarization"
    )

    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")
    summarizer = agents.PatentSummarizerAgent(
        model=cfg.model_name,
        base_url=BASE_URL,
        api_key=API_KEY,
        backend=cfg.backend,
    )
    
    summarize_func = select_summarizer_func(
        summarizer, cfg.version
    )

    preload_pbar = utils.RichTableProgress(total=len(pending_candidate_numbers), print_every=500)
    candidate_patents = []
    with sqlm.Session(engine) as session:
        try:
            
            BATCH_SIZE = 1000
            for i in range(0, len(pending_candidate_numbers), BATCH_SIZE):
                end_idx = min(i + BATCH_SIZE, len(pending_candidate_numbers))
                batch_numbers = pending_candidate_numbers[i:end_idx]
                candidate_batch=  list(
                    session.exec(
                        sqlm.select(dataset.Patent)
                        .where(dataset.Patent.number.in_(batch_numbers))
                        #.offset(3)
                    )
                )
                candidate_patents.extend(candidate_batch)
                preload_pbar.update()
        except Exception as e:
            logger.error(f"Failed to load patent texts for candidates from DB: {e}")

    candidates = {
        str(candidate.number): dataset.extract_query_text(
            candidate,
            cfg.doc_columns,
            cfg.target_claims,
            independent_only=cfg.independent_claims,
            desc_max_tokens=cfg.desc_max_tokens,
        )
        for candidate in candidate_patents
    }
    logger.info(
        f"Preloaded {len(candidates)} candidates. Starting async summarization with {cfg.semaphore} workers using {summarize_func.__name__}."
    )
    

    semaphore = asyncio.Semaphore(cfg.semaphore)
    summarize_pbar = utils.RichTableProgress(
        total=len(candidates), print_every=500
    )
    persist_lock = asyncio.Lock()

    async def persist_results() -> None:
        async with persist_lock:
            with open(summaries_json, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)

    async def sem_summarize(candidate_num: str, candidate_text: str) -> None:
        async with semaphore:
            try:
                summaries[candidate_num] = await summarize_func(candidate_text)
                await persist_results()

            except Exception as e:
                logger.error(f"Failed to summarize {candidate_num}")
                summaries[candidate_num] = None
            finally:
                summarize_pbar.update()
            
    tasks = [
        sem_summarize(candidate_num, candidate_text)
        for candidate_num, candidate_text in candidates.items()
    ]

    await asyncio.gather(*tasks)

    with open(summaries_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved candidate summaries to {summaries_json}")
    
def get_wandb_name(path: str) -> str:
    return re.sub(r"_\d{4}-\d{2}-\d{2}(-\d{2}:\d{2}:\d{2})?$", "", os.path.basename(path))


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
            project="relevance_analysis-prefilter",
            name=get_wandb_name(str(cfg.output_dir)),
            config=cfg.to_dict(),
            config_exclude_keys=["db_path", "test_topics_path", "tags", "unilingual_index", "wandb"],
            settings=settings,
            tags=cfg.tags,
        )

    asyncio.run(main(cfg))

    if run is not None:
        run.finish()
