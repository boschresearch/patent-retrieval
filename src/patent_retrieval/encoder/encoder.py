# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document


class Encoder(ABC):
    """
    Abstract base class for encoders. Defines the interface for encoding documents and queries, as well as searching and managing indices.
    """
    def __init__(self, index_dir: str = "index"):
        self.index_dir = index_dir
        self.indexed = False
    
    @abstractmethod
    def encode_docs(self, docs: List[Document], duplicates: bool = False) -> None:
        
        pass
    
    @abstractmethod
    def encode_query(self, text: str) -> Any:

        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs) -> List[Tuple[str, float]]:

        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:

        pass
    
    @abstractmethod
    def load_index(self, path: str,store_type:str) -> None:

        pass
    
    @abstractmethod
    def get_indices(self) -> List[str]:

        pass