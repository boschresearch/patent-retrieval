# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from sentence_transformers import CrossEncoder
from typing import Dict, List, Tuple, Optional
import torch
from .reranker import BaseReranker

class HuggingfaceReranker(BaseReranker):

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", device: str = "cpu",model_kwargs=None,tokenizer_kwargs=None,**kwargs):

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device,model_kwargs=model_kwargs,tokenizer_kwargs=tokenizer_kwargs,**kwargs)
        
        tok = self.model.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
       # print(tok.pad_token)
       # print(tok.pad_token_id)
        # Some ST models use model-specific tokenizer
        if hasattr(self.model.model, "tokenizer"):
            hf_tok = self.model.model.tokenizer
            if hf_tok.pad_token is None:
                hf_tok.pad_token = tok.pad_token
                hf_tok.pad_token_id = tok.pad_token_id

        # Transformers may use config.pad_token_id during batching
        cfg = self.model.model.config
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tok.pad_token_id

        

    def preprocess(self, query: str, docs: Dict[str, str]) -> List[Tuple[str, str]]:

        return [(query, doc_text) for _, doc_text in docs.items()]

    def score(self, query: str, docs: Dict[str, str], batch_size: int = 32) -> List[float]:

        pairs = self.preprocess(query, docs)
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return scores

    def rerank(self, query: str, docs: Dict[str, str], top_n: Optional[int] = None, **kwargs) -> tuple[List[Tuple[str,float]], bool]:

        scores = self.score(query, docs)
        doc_scores = [(doc_id, float(score)) for doc_id, score in zip(docs.keys(), scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            doc_scores = doc_scores[:top_n]
        return doc_scores, False


if __name__ == "__main__":
    reranker = HuggingfaceReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")

    query = "What causes gravity?"
    candidate_docs = {
        "doc-1": "Gravity is a force that attracts two bodies toward each other.",
        "doc-2": "The Earth rotates around the Sun.",
        "doc-3": "Gravity affects how objects fall to the ground."
    }

    ranked_docs = reranker.rerank(query, candidate_docs, top_n=2)
    for doc_id, score in ranked_docs:
        print(f"Score: {score:.4f} | ID: {doc_id}")
