# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from patent_retrieval import encoder as encoder
from patent_retrieval import utils as utils

logger = utils.get_logger(__name__)

VALID_FUSION_METHODS = ("rrf", "min_max")
VALID_ENCODER_PAIRS = (("dense", "dense"), ("dense", "sparse"))


@dataclass(frozen=True)
class IndexSpec:
    """Configuration for loading one index-backed encoder."""

    type: str
    path: str | Path
    backend: str = "openai"
    store_type: str = "faiss"
    model_name: Optional[str] = None
    tokenizer: str = "Qwen/Qwen3-Embedding-4B"
    base_url: Optional[str] = None
    api_key: str = "EMPTY"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexSpec":
        return cls(
            type=str(data["type"]).lower(),
            path=data["path"],
            backend=str(data.get("backend", "openai")).lower(),
            store_type=str(data.get("store_type", "faiss")).lower(),
            model_name=data.get("model_name"),
            tokenizer=str(data.get("tokenizer", "Qwen/Qwen3-Embedding-4B")),
            base_url=data.get("base_url"),
            api_key=str(data.get("api_key", "EMPTY")),
        )


def _expected_artifact(spec: IndexSpec) -> Path:
    index_path = Path(spec.path)

    if spec.type == "dense":
        if spec.store_type != "faiss":
            raise ValueError(
                f"Unsupported dense store_type '{spec.store_type}'. "
                "HybridRetriever currently expects FAISS-backed dense indices."
            )
        return index_path / "index.faiss"

    if spec.type == "sparse":
        method = (spec.model_name or "bm25").lower()
        return index_path / f"{method}.index"

    raise ValueError(
        f"Unsupported encoder type '{spec.type}'. Choose 'dense' or 'sparse'."
    )


def _load_encoder(spec: IndexSpec) -> encoder.Encoder:
    index_path = Path(spec.path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index path not right: '{index_path}' does not exist.")

   # artifact = _expected_artifact(spec)
    #if not artifact.exists():
        

    if spec.type == "dense":
        try: 
            enc = encoder.get_encoder(
                type="dense",
                backend=spec.backend,
                store_type=spec.store_type,
                model_name=spec.model_name,
                index_dir=str(index_path),
                tokenizer=spec.tokenizer,
                base_url=spec.base_url,
                api_key=spec.api_key,
            )
            enc.load_index(path=str(index_path), store_type=spec.store_type)
            logger.info("Loaded dense index from %s", index_path)
            return enc
        except Exception as e: 
            logger.error(f"Failed to load dense index from {index_path}: {e}")
            raise RuntimeError(f"Failed to load dense index from {index_path}") from e
    elif spec.type == "sparse":
        try:
            sparse_method = spec.model_name or "bm25"
            enc = encoder.get_encoder(
                type="sparse",
                model_name=sparse_method,
                index_dir=str(index_path),
            )
            enc.load_index(path=str(index_path))
            logger.info("Loaded sparse index from %s with method=%s", index_path, sparse_method)
            return enc
        except Exception as e:
            logger.error(f"Failed to load sparse index from {index_path}: {e}")
            raise RuntimeError(f"Failed to load sparse index from {index_path}") from e

    else:
        raise ValueError(
            f"Unsupported encoder type '{spec.type}'. Choose 'dense' or 'sparse'."
        )

class HybridRetriever:
    """Fuse results from two loaded retrievers using min-max or RRF."""

    def __init__(
        self,
        encoders: Sequence[encoder.Encoder],
        weights: Optional[Sequence[float]] = None,
        fusion_method: str = "min_max",
        rrf_k: int | None = 60,
    ):
        if len(encoders) != 2:
            raise ValueError("HybridRetriever expects exactly two encoders.")

        if fusion_method not in VALID_FUSION_METHODS:
            raise ValueError(
                f"Unsupported fusion_method '{fusion_method}'. "
                f"Choose one of: {', '.join(VALID_FUSION_METHODS)}"
            )

        if weights is None:
            weights = [1.0, 1.0]
        if len(weights) != len(encoders):
            raise ValueError(
                f"weights length ({len(weights)}) must match encoders length ({len(encoders)})."
            )

        self.encoders = list(encoders)
        self.weights = [float(w) for w in weights]
        self.fusion_method = fusion_method
        self.rrf_k = max(1, rrf_k)

    @classmethod
    def from_index_specs(
        cls,
        index_specs: Sequence[IndexSpec | Dict[str, Any]],
        weights: Optional[Sequence[float]] = None,
        fusion_method: str = "min_max",
        rrf_k: int | None = 60,
    ) -> "HybridRetriever":
        specs: List[IndexSpec] = [
            spec if isinstance(spec, IndexSpec) else IndexSpec.from_dict(spec)
            for spec in index_specs
        ]

        if len(specs) != 2:
            raise ValueError("Exactly two index specs are required.")

        pair = tuple(sorted(spec.type for spec in specs))
        if pair not in VALID_ENCODER_PAIRS:
            raise ValueError(
                "Unsupported encoder pair. Supported pairs are: "
                "dense+sparse or dense+dense."
            )

        encoders = [_load_encoder(spec) for spec in specs]
        return cls(
            encoders=encoders,
            weights=weights,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
        )

    def search(
        self,
        query: str,
        k: int = 100,
        fetch_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if k <= 0:
            return []

        per_encoder_k = max(k, fetch_k or k)
        metadata_filter = filter_dict or {}
        results_list: List[List[Tuple[str, float]]] = []

        for retriever in self.encoders:

            results = retriever.search(
                query=query,
                k=per_encoder_k,
                fetch_k=per_encoder_k,
                filter_dict=metadata_filter,
            )
            results_list.append(results)

        if self.fusion_method == "rrf":
            return self._fuse_rrf(results_list, top_k=k)
        return self._fuse_min_max(results_list, top_k=k)

    def _fuse_rrf(
        self,
        results_list: Sequence[Sequence[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        doc_scores: Dict[str, float] = {}

        for results, weight in zip(results_list, self.weights):
            for rank, (doc_id, _score) in enumerate(results, start=1):
                fused_score = float(weight) / float(self.rrf_k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + fused_score

        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _fuse_min_max(
        self,
        results_list: Sequence[Sequence[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        doc_scores: Dict[str, float] = {}

        for results, weight in zip(results_list, self.weights):
            if not results:
                continue

            values = [score for _doc_id, score in results]
            min_score = min(values)
            max_score = max(values)
            score_range = max_score - min_score if max_score > min_score else 1.0

            for doc_id, score in results:
                normalized = (score - min_score) / score_range
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + float(weight) * normalized

        return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def build_hybrid_retriever(
    index_specs: Sequence[IndexSpec | Dict[str, Any]],
    fusion_method: str = "min_max",
    weights: Optional[Sequence[float]] = None,
    rrf_k: int | None = 60,
) -> HybridRetriever:
    return HybridRetriever.from_index_specs(
        index_specs=index_specs,
        weights=weights,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
    )
