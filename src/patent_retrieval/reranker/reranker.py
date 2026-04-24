# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BaseReranker(ABC):
    """Abstract base class for all reranker implementations.

    Every reranker must accept a query and a ``{candidate_id: text}`` dict
    and return ``[(candidate_id, score)]`` sorted by descending relevance.
    Higher scores indicate greater relevance.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: Dict[str, str],
        top_n: Optional[int] = None,
    ) -> tuple[List[Tuple[str, float]], bool]:
        """Return (candidate_id, score) sorted by descending relevance."""
        ...
