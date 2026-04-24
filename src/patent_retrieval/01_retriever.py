# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from pyrootutils import setup_root
import re
from datetime import datetime
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import List

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
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
#import patent_retrieval
from patent_retrieval import utils as utils, dataset as dataset, encoder as encoder
import torch
load_dotenv()
torch.cuda.empty_cache()

root = setup_root(__file__)
logger = utils.get_logger(__name__)


def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.query_columns.copy()

    asymmetric = set(cfg.query_columns) != set(cfg.doc_columns)

    if "claims" in search_columns and cfg.claims:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"


    out_dir: Path = root / "embeddings"/ "runs" / (f"{cfg.run_name}{cfg.suffix}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.language else "")+(f"_aysm" if asymmetric else "")+(f"_{cfg.q}topics" if cfg.q else "")+(f"_top{cfg.k}"))
    
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



cfg = hl.Config(
    type="dense",
    tokenizer= "/home/alm3rng/patent-retrieval/finetuning/runs/optuna/trial_11/checkpoint-95",#"/home/alm3rng/patent-retrieval/finetuning/runs/qwen3_lora_12epochs/v1"
    run_name="patQwen3-emb-4b-v2_db-v4", #"bm25" # qwen3_emb_8b
    suffix="_rewrite",
    backend="openai", # "instructor" # "sentence-transformers" # "huggingface"
    embedding_model="patQwen3-emb-4b-v2",#"qwen-finetuned",#"Qwen/Qwen3-Embedding-4B" # mpi-inno-comp/paecter
    base_url="http://localhost:59749/v1",
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_abs_en.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v4_en.db",
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
    wandb=True,
    tags=["instruct"],
    store_type="faiss",
    unilingual_index=True,
    language="EN",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    k=1000,
    q=30,
    topics=None,
    rewrite_query=False

   # notes="top200 fusion 150 from original + 50 from english query",
)


def setup_index(
    index_path: Path,
    search_columns: list[str],
    engine: sqla.Engine,
    language: str ='',
    batch_size: int = 10000,
    rewrite_index: bool = False
) :

    try:

        patent_encoder = encoder.get_encoder(type=cfg.type,backend=cfg.backend,store_type=cfg.store_type,model_name=cfg.embedding_model,
                                       tokenizer=cfg.tokenizer,index_dir=cfg.index_dir,base_url=cfg.base_url)
       # print(encoder.)
        conditions = []
        index_suffix = os.path.join(index_path, "index.faiss") if cfg.type =="dense" else os.path.join(index_path, "bm25.index")
        if os.path.exists(index_suffix) and not rewrite_index:
            logger.info(" index found. Loading existing index...")
            patent_encoder.load_index(path=index_path,store_type=cfg.store_type)
            if cfg.type=="sparse":
                return patent_encoder
            index_ids = patent_encoder.get_indices()
            logger.info(f"FIASS {len(index_ids)} embeddings loaded...")
         #   conditions.append(dataset.Patent.number.not_in(index_ids))
            
        else:
            index_ids = []
            logger.info("No index loaded. Proceeding to create a new one...")

        with sqlm.Session(engine) as session:

            
            if cfg.unilingual_index:
                conditions.append(dataset.Patent.language == language.upper())

            num_rows = session.exec(
                sqlm.select(sqlm.func.count()).select_from(dataset.Patent)
           #     .where(*conditions) 
            ).one()
            
            if num_rows==len(index_ids):
                logger.info(f"Patents exist already in FAISS index. Return encoder..")
                return patent_encoder

            
            pbar = utils.RichTableProgress(
                total=num_rows-len(index_ids), print_every=10000
            )
            offset = len(index_ids)
            batch_size=10000
            while True:
                batch = list(
                    session.exec(
                        sqlm.select(dataset.Patent)
                        .where(*conditions)
                        .offset(offset)
                        .limit(batch_size)
                    )
                )
              
                if not batch:
                    break
                # Extract text and metadata, create Langchain Document objects
                docs = list(map(lambda x: Document(
                    page_content=dataset.extract_query_text(patent=x,search_columns=search_columns, kclaims=cfg.index_claims, independent_only=cfg.independent_claims), 
                    metadata={"number":x.number,"language":x.language,"jurisdiction": x.jurisdiction,"publication_date":x.publication_date}
                ),batch))

                patent_encoder.encode_docs(docs=docs)
                pbar.update(len(batch))
                offset += batch_size     
                if offset % 50000 == 0:
                    logger.info(f"Saving index at {offset} docs...")
                    patent_encoder.save_index(path=index_path)
                    
                gc.collect()
                torch.cuda.empty_cache()    
                

            logger.info(f"Patents Encoding is done")
            patent_encoder.save_index(path=index_path)
            logger.info(f"FAISS Index saved to {index_path}")

            return patent_encoder

    except Exception as e:
        print("Error: ",e)
        traceback.print_exc()
        

def prepend_instruct(task_description: str="", query: str="") -> str:
    if not task_description:
            task_description = ""
    #return f'Instruct: Given a patent, retrieve prior art documents that describe the same technical invention regardless of the language. \nQuery:{query}'
    return f'<Instruct>: Given a patent, perform a prior art search and identify relevant existing patents regardless of the language. \n<Query>:{query}'


def main(cfg: hl.Config,retrieve=True,evaluate=True) -> None:

    rich.print(rich.syntax.Syntax(cfg.to_yaml(), "yaml"))
    print(cfg.index_dir)
    engine = sqlm.create_engine(f"sqlite:///{cfg.db_path}")

    encoder = setup_index(index_path=cfg.index_dir,search_columns=cfg.doc_columns,engine=engine,language=cfg.language)
    
    

    if retrieve:
        topics = utils.load_topics(cfg.test_topics_path)
        logger.info(f"Loaded {len(topics)} topics")
        topics = [t for t in topics if t in cfg.topics] if cfg.topics else topics
        if cfg.q and cfg.topics is None:
            try:
                topics = utils.read_topics(cfg.q)
                logger.info(f"Using  topics from {cfg.q}topics.txt")
            except Exception as e:
                logger.error(f"Failed to read topics from {cfg.q}topics.txt: {e}")
                topics = topics[:cfg.q]
            cfg.topics = topics
        cfg.output_dir.joinpath("config.yaml").write_text(cfg.to_yaml())
        pbar = utils.RichTableProgress(total=len(topics), print_every=100)
        results = []

        for i,topic_num in enumerate(topics):
            

            topic_file = next(
                cfg.test_topics_path.parent.glob(f"PAC_topics/files/{topic_num}*.xml")
            )
            topic_patent = dataset.parse_patent([topic_file])[0]
            if cfg.language and topic_patent.language.lower() != cfg.language.lower():
                logger.warning(f"Skipping topic {topic_num} with language {topic_patent.language} as it does not match the specified language {cfg.language}")
                continue
            else:

                query = dataset.extract_query_text(
                    topic_patent,
                    cfg.query_columns,
                    cfg.claims,
                    independent_only=cfg.independent_claims
                )

                                
            metadata_filter = None  
            if cfg.language:
                metadata_filter={"language":cfg.language}

            search_results = encoder.search(query, k=cfg.k,fetch_k=cfg.k,filter_dict=metadata_filter,rewrite=cfg.rewrite_query)

            
            results.extend(
                {"topic": topic_num, "number": match_num, "score": score}
                for match_num, score in search_results
            )

            if not search_results:
                logger.warning(f"No search results found for topic {topic_num}")

            pbar.update()
        pbar.update()
        
        results = pd.DataFrame.from_records(results)
        #candidates_path="/home/alm3rng/patent-retrieval/embeddings/runs/patQwen3-emb-4b-v2_db-v4_all-topics_abstract-claims_aysm_top1000/results.csv"

        #results = pd.read_csv(candidates_path)
        results_file = cfg.output_dir / "results.csv"
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
        
        if evaluate:
            metrics  = utils.calculate_metrics(
                    results=results_file,
                    test_topics_path=cfg.test_topics_path,
                    topk=cfg.k,
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
    if cfg.wandb:
        settings = wandb.Settings(
                save_code=False,      
                x_disable_meta=True,
                x_disable_stats=True  
            )

        
        run = wandb.init(project="patent-retrieval-v2",
                    name=get_wandb_name(cfg.output_dir), 
                    config=cfg.to_dict(),
                    group=cfg.embedding_model,
                    config_exclude_keys=["db_path","test_topics_path","tags","unilingual_index","wandb","suffix","base_url"],
                    settings=settings,
                    tags=cfg.tags
                    )
        
    main(cfg)
