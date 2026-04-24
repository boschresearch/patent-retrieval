# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from typing import List, Optional, Dict
from pathlib import Path
import logging
from txtai.scoring import ScoringFactory
from langchain_core.documents import Document
from patent_retrieval import utils as utils, encoder as encoder

logger = logging.getLogger(__name__)

class SparseEncoder(encoder.Encoder):
    
    def __init__(self, model_name: str = "bm25", index_dir: str = "index"):

        logger.info(f"Sparse encoder method: {model_name}")
        
        # Initialize scoring method
        self.method = model_name.lower()    
        self.scoring = ScoringFactory.create({
            "method": self.method,
            "terms": True,
            "content": True
        })
        
        self.documents: List[Dict] = []
        self.doc_ids: List[str] = []
        self.indexed: bool = False
        
        self.index_dir = Path(index_dir)
        
    def encode_docs(self, docs: List[Document]) -> None:

        docs = [(doc.metadata['number'], doc.page_content,None) for doc in docs]
        self.scoring.index(docs)

        self.documents.extend(docs)
        logger.info(f"Indexed {len(docs)} new documents. Total: {len(self.documents)}")
    

    def encode_query(self, text: str) -> Dict:

        if not self.indexed:
            raise ValueError("No documents indexed yet.")
        return self.scoring.weights(text)
    
    def search(self, query: str, k: int = 10,fetch_k:int=100, filter_dict: Dict = None) -> List[tuple]:

        matching_docs = self.scoring.search(query, limit=k)
        return [(doc['id'],doc['score']) for doc in matching_docs]
        
    
    def save_index(self, path: str = "sparse_idx"):
        index_path =Path(path)/ f"{self.method}.index"

        # Convert to string here ↓↓↓
        self.scoring.save(str(index_path))
        logger.info(f"Saved sparse index to: {path}")
    
    def load_index(self, path: str = "sparse_idx.index", store_type: str = "txtai") -> None:
        index_path = Path(path)/ f"{self.method}.index"
        try:
            self.scoring.load(str(index_path))
            logger.info(f"Loaded sparse index from: {path} with {len(self.documents)} documents")

        except:    
            logger.error(f"Index not found at: {path}")
        
        
        
 
    def get_indices(self) -> List[str]:
        return self.doc_ids
    
def main():
    """Test the SparseEncoder class with sample documents."""
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={"number": "doc1", "category": "programming"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"number": "doc2", "category": "AI"}
        ),
    ]
    
    print("Initializing SparseEncoder...")
    encoder = SparseEncoder(model_name="bm25")
    
    # Index documents
    print("\nIndexing documents...")
    encoder.encode_docs(sample_docs)
    
    # Test search
    print("\n" + "="*50)
    query = "artificial intelligence and machine learning"
    print(f"Query: '{query}'")
    print("="*50)
    
    results = encoder.search(query, k=3)
    print(f"\nTop {len(results)} results:")
    for i, (doc, score) in enumerate(results):
        print(f"\n{i}. Score: {score:.4f}")

    
    # Test with filter

    
    # Test save and load
    print("\n" + "="*50)
    print("Testing save/load functionality...")
    print("="*50)
    
    print("Testing with filter (category='AI')...")
    print("="*50)
    
    filtered_results = encoder.search(query, k=3, filter_dict={"category": "AI"})
    print(f"\nFiltered results: {len(filtered_results)}")
    for i, (doc, score) in enumerate(filtered_results, 1):
        print(f"\n{i}. Score: {score:.4f}")

    save_path = "test_sparse_index"
    encoder.save_index(save_path)
    
    # Create new encoder and load
    new_encoder = SparseEncoder(model_name="bm25")
    new_encoder.load_index(save_path)
    print("\n" + "="*50)

    print("Testing with filter (category='AI')...")
    print("="*50)
    
    filtered_results = encoder.search(query, k=3, filter_dict={"category": "AI"})
    print(f"\nFiltered results: {len(filtered_results)}")
    for i, (doc, score) in enumerate(filtered_results, 1):
        print(f"\n{i}. Score: {score:.4f}")




if __name__ == "__main__":
    main()