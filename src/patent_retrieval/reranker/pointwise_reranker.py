# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from typing import List, Dict, Any, Optional, Tuple
import cohere
from transformers import AutoTokenizer
from patent_retrieval import utils
from .reranker import BaseReranker

logger = utils.get_logger(__name__)

class PointwiseReranker(BaseReranker):

    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: str="EMPTY", 
        model_name: str = "rerank-english-v3.0"
    ):
        
        self.client = cohere.ClientV2(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_tokens = 40950
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_length=self.max_tokens)
    
    def _truncate(self,query:str, text: str) -> str:
        """
        Internal helper: Tokenize -> Truncate -> Decode
        """
        # Encode with the ORIGINAL model's tokenizer
        tokens = self.tokenizer.encode(query+text)
       # logger.debug(f"Document token length: {len(tokens)}")
        if len(tokens) > self.max_tokens-10:
            query_tokens = self.tokenizer.encode(query)
            text_tokens = self.tokenizer.encode(text)
            # Truncate to safe limit
        #    logger.warning(f"Truncating document from {len(tokens)} to {self.max_tokens} tokens.")
            tokens = text_tokens[:self.max_tokens-len(query_tokens)-10]
            # Decode back to string for the API
            text = self.tokenizer.decode(tokens)
            print(f"Truncated text to {len(tokens)} tokens.")
        return text

    def rerank(
        self, 
        query: str, 
        docs: dict,
        top_n: Optional[int] = None,
        **kwargs,
    ) -> tuple[List[Tuple[str, float]], bool]:

        
        instruction = "Judge whether a candidate document is relevant to the Query patent application. Answer 'yes' if it meets at least one condition: (1) it discloses, explicitly or inherently, every element of at least one independent claim of the query application (anticipation); (2) it discloses a substantial portion of the claimed elements and, combined with common general knowledge, would make the invention obvious to a person skilled in the art; (3) it establishes the technological background or common general knowledge against which the claims are assessed (background/A-class), provided it is directly relevant to the technical field or problem addressed by the query. Answer 'no' if the document shares surface-level terminology with the query but addresses a distinct technical problem or achieves a different technical effect."

        # Combine them using the Qwen3 format
        query = f"<Instruct>: {instruction}\n<Query>: {query}"
        if not docs:
            return [], True
        id_mapping = {idx: doc_id for idx, (doc_id, doc_text) in enumerate(docs.items())}
        text_docs=[f"<Document>: {self._truncate(query,doc_text)}" for doc_id, doc_text in docs.items()]
        top_n = len(docs) if top_n is None else top_n
        try:
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=text_docs,
                top_n=top_n
                # Removed max_tokens_per_doc as it often breaks local vLLM V2 endpoints
            )

            # Safety check: if response or response.results is None
            if not response or not getattr(response, 'results', None):
                logger.warning(f"Warning: API returned no results. Response: {response}")
                return [],True

            results = []
            
            for result in response.results:
                # result.index is the integer index in the text_docs list
                doc_id = id_mapping[result.index]
                results.append((doc_id, result.relevance_score))
            
            return results,False

        except Exception as e:
            print(f"Rerank failed: {e}")
            return [],True


# Example usage
if __name__ == "__main__":
    reranker = PointwiseReranker(
        #api_key="your-api-key",
        base_url="http://localhost:59749/",  # Optional
        model_name="Qwen/Qwen3-Reranker-4B"
    )
    
    docs = {
        "doc-0": "Python is a high-level programming language.",
        "doc-1": "Machine learning is a subset of artificial intelligence.",
        "doc-2": "Cohere provides natural language processing APIs.",
        "doc-3": "Deep learning uses neural networks with multiple layers.",
    }
    
    query = "What is machine learning?"
    
    reranked_results = reranker.rerank(query, docs, top_n=3)
    print("\nReranked Results:")
    for doc_id, score in reranked_results:
        print(f"  {doc_id}: {score:.4f}")