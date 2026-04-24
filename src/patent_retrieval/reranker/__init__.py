# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from .hf_reranker import HuggingfaceReranker
from .pointwise_reranker import PointwiseReranker
from .listwise_reranker import ListwiseReranker
from .reranker import BaseReranker

def get_reranker(
    type: str = "listwise",
    base_url: str = "", 
    model_name: str = "", 
    backend: str = "openai",
    api_key: str = "EMPTY",
    mode: str = "simple",
    n: int = 20,
    thinking: bool = True,
    passes: int = 1,
    remap_ids: bool = False,
    prompt_id: str = "v1",
) -> BaseReranker:

    if type.lower() == "pointwise":
        if backend.lower() == "cohere":
            return PointwiseReranker(base_url=base_url, model_name=model_name, api_key=api_key)
        if backend.lower() == "huggingface":
            return HuggingfaceReranker(model_name=model_name, device="cpu")
        else:
            raise ValueError(f"Unsupported backend for pointwise reranker: {backend}. Use 'cohere' or 'huggingface'.")
    elif type.lower() == "listwise":
        return ListwiseReranker(base_url=base_url, model_name=model_name,backend=backend, api_key=api_key, mode=mode, n=n,thinking=thinking, passes=passes,remap_ids=remap_ids, prompt_id=prompt_id)
    else:
        raise ValueError(f"Unsupported reranker type: {type}. Use 'pointwise' or 'listwise'.")


__all__ = ["PointwiseReranker", "ListwiseReranker", "HuggingfaceReranker", "BaseReranker"]