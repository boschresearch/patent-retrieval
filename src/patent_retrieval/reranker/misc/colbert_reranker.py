# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import os
from pyrootutils import setup_root
import re
from datetime import datetime
import os
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from typing import List, Tuple
import hydralette as hl
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from langchain_core.documents import Document

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from patent_retrieval import utils as utils, dataset as dataset, encoder as _encoder

root = setup_root(__file__)
logger = utils.get_logger(__name__)

os.environ["WANDB_DIR"] = str(root)

def get_run_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.query_columns.copy()

    asymmetric = set(cfg.query_columns) != set(cfg.doc_columns)

    if "claims" in search_columns:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"


    out_dir: Path = root / "reranking"/ "runs" / (f"{cfg.run_name}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else ""))
    
    if out_dir.exists():
        out_dir = (
            out_dir.parent
            / (f"{cfg.run_name}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")+(f"_aysm" if asymmetric else "")+f"_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_index_dir(cfg: hl.Config) -> Path:
    search_columns = cfg.doc_columns.copy()

    if "claims" in search_columns:
        idx = search_columns.index("claims")
        search_columns[idx] = f"{cfg.claims}claims"

    index_dir: Path = Path("/home")/"alm3rng"/"scratch" /  "clef_ip_2011" / (f"{cfg.run_name}_{"-".join(search_columns)}"+ (f"_{cfg.language}" if cfg.unilingual_index else "")) 
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)

    return index_dir
"""
cfg = hl.Config(
    type="dense",
    method="",
    run_name="colbert_rerank_db100k",#"bm25" # qwen3_emb_4b
    backend="openai",#"openai"
    embedding_model="Qwen/Qwen3-Embedding-4B",#"Qwen/Qwen3-Embedding-4B" # mpi-inno-comp/paecter
    test_topics_path=(
        Path(os.environ["CLEF_IP_LOCATION"])
        / "02_topics"
        / "test-pac"
        / "relass_clef-ip-2011-PAC_50topics.txt"
    ),
    db_path=Path(os.environ["CLEF_IP_LOCATION"]) / "patents_v2_50topics_100k.db",
    claims="",
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
    tags=["rerank"],
    store_type="faiss",
    unilingual_index=False,
    language="",
    output_dir=hl.Field(reference=get_run_dir, type=Path),
    k=1000,
)
"""
class ColBERTReranker:
    
    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        index_name: str = "colbert_index",
        index_root: str = "./colbert_indexes",
        experiment: str = "default"
    ):
        self.checkpoint = checkpoint
        self.index_name = index_name
        self.index_root = index_root
        self.experiment = experiment
        self.searcher = None
        self.doc_map = {}
    
    def build_index(self, documents: List[Document], nbits: int = 2):
        collection = [doc.page_content for doc in documents]
        self.doc_map = {i: doc for i, doc in enumerate(documents)}
        
        with Run().context(RunConfig(nranks=1, experiment=self.experiment, root=self.index_root)):
            config = ColBERTConfig(
                nbits=1,
                checkpoint=self.checkpoint,
                root=self.index_root,
                reranker=True,
                kmeans_niters = 1,
                bsize=32
            )
            
            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(name=self.index_name, collection=collection, overwrite=True)
    
    def load_index(self):
        with Run().context(RunConfig(experiment=self.experiment, root=self.index_root)):
            self.searcher = Searcher(index=self.index_name, checkpoint=self.checkpoint)
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        if self.searcher is None:
            self.load_index()
        
        collection = [doc.page_content for doc in documents]
        temp_doc_map = {i: doc for i, doc in enumerate(documents)}
        
        results = self.searcher.search(query, k=min(top_k, len(documents)))
        
        ranked_docs = []
        for passage_id, passage_rank, passage_score in zip(*results):
            if passage_id in temp_doc_map:
                ranked_docs.append(temp_doc_map[passage_id])
        
        return ranked_docs[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if self.searcher is None:
            self.load_index()
        
        results = self.searcher.search(query, k=top_k)
        
        ranked_results = []
        for passage_id, passage_rank, passage_score in zip(*results):
            if passage_id in self.doc_map:
                ranked_results.append((self.doc_map[passage_id], passage_score))
        
        return ranked_results


if __name__=='__main__':
        # Build index
    reranker = ColBERTReranker(
        checkpoint="colbert-ir/colbertv2.0",
        index_name="my_docs",
        index_root="./indexes"
    )

    docs = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning with PyTorch"),
        Document(page_content="Science Mr White"),
        Document(page_content="Power is power"),
        Document(page_content="Dogs are the best"),
        Document(page_content="I like pizza"),
        Document(page_content="Sun is gone"),
        Document(page_content="Ronaldo is the goat"),
    ]

    reranker.build_index(docs)

    # Rerank retrieved documents
    query = "python programming"
    #retrieved_docs = [...]  # from your retriever
    reranked = reranker.rerank(query, docs, top_k=5)

    # Or search directly
    results = reranker.search(query, top_k=5)
