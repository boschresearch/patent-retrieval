# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.


from sentence_transformers import SentenceTransformer
import numpy as np

from pyrootutils import setup_root
root = setup_root(__file__)

from patent_retrieval.encoder import BaseEncoder
from patent_retrieval.utils import get_logger
logger = get_logger(__name__)

class HFEncoder(BaseEncoder):
    """Encoder using Hugging Face SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
     #   self.normalize = normalize
        self._dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        embeddings = self.model.encode(texts,convert_to_numpy=True)
        return embeddings

    @property
    def dimension(self):
        return self._dimension