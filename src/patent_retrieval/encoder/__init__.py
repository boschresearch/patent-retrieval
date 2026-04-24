# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from typing import Union
from .encoder import Encoder
from .dense_encoder import DenseEncoder
from .sparse_encoder import SparseEncoder


def get_encoder(
    type: str = "dense",
    backend: str = "openai", 
    store_type: str = "faiss",
    model_name: str = None, 
    index_dir: str = "index",
    tokenizer: str = "Qwen/Qwen3-Embedding-4B",
    base_url: str = None,
    api_key: str = "EMPTY",

) -> Union[DenseEncoder, SparseEncoder]:
    """
    Factory function to get an encoder instance based on the specified type.

    Returns:
        An instance of DenseEncoder or SparseEncoder based on the specified type.
    """
    if type == "dense":
        return DenseEncoder(backend=backend, model_name=model_name, index_dir=index_dir,
                            store_type=store_type,tokenizer=tokenizer, base_url=base_url, api_key=api_key)
    elif type == "sparse":
        return SparseEncoder(model_name=model_name, index_dir=index_dir)
    else:
        raise ValueError(f"Unsupported encoder type: {type}. Use 'dense' or 'sparse'.")


__all__ = ["BaseEncoder", "DenseEncoder", "SparseEncoder", "Encoder"]