# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from openai import OpenAI
import numpy as np

from pyrootutils import setup_root
root = setup_root(__file__)

from patent_retrieval.encoder import BaseEncoder
from patent_retrieval.utils import get_logger
logger = get_logger(__name__)


class OpenAIEncoder(BaseEncoder):
    """Encoder using OpenAI embedding models."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Thinking-2507",
                 api_key :str = "EMPTY",
                 base_url: str ="http://localhost:59549/v1",
                 normalize: bool = True
        ):

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.normalize = normalize
        self._dimension = None

    def encode(self, texts):
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = np.array([d.embedding for d in response.data])

        return embeddings

    @property
    def dimension(self):
        return self._dimension or 0
