# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import math
import random
import os
import re
from typing import Dict, List, Optional, Tuple
import secrets
from openai import OpenAI

from patent_retrieval import utils as utils
from .reranker import BaseReranker

logger = utils.get_logger(__name__)

VALID_MODES = ("simple", "sliding_window", "tournament")

PROMPT_MD_PATH =  os.getenv("PROMPT_MD_PATH", "/home/alm3rng/patent-retrieval/src/patent_retrieval/prompts")+"/listwise_reranker_{id}.md"

class ListwiseReranker(BaseReranker):
    """LLM-based listwise reranker with three operating modes.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible API endpoint.
    model_name : str
        Model identifier served behind *base_url*.
    api_key : str
        API key (default ``"EMPTY"`` for local servers).
    mode : str
        Reranking strategy – one of:

        * ``"simple"``         – single LLM call with all documents.
        * ``"sliding_window"`` – ascending-sorted sliding window with overlap.
        * ``"tournament"``     – tournament-style elimination rounds.
    n : int
        Window / group size used by *sliding_window* and *tournament* modes
        (ignored for *simple*).  Defaults to ``20``.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        backend: str = "openai",
        api_key: str = "EMPTY",
        thinking: bool = True,
        mode: str = "simple",
        n: int = 20,
        passes: int = 1,
        remap_ids: bool = True,
        prompt_id: str = "v1",

        **kwargs,
        
    ):
        if backend.lower() == "openai":
            self.api_key = api_key or "EMPTY"
            self.client = OpenAI(base_url=base_url, api_key=api_key,timeout=1800)
        elif backend.lower() == "azure":
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AzureOpenAI

            token_provider = get_bearer_token_provider(
                    AzureCliCredential(),
                    "https://cognitiveservices.azure.com/.default"
                )
   
            self.client = AzureOpenAI(
                    azure_endpoint=base_url,
                    azure_ad_token_provider=token_provider,
                    api_version="2025-01-01-preview",
            )

        else:
            raise ValueError(f"Unsupported backend '{backend}'. Choose 'openai' or 'azure'.")


        if mode.lower() not in VALID_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose one of: {', '.join(VALID_MODES)}")

        self.model_name = model_name
        self.thinking = thinking
        self.backend = backend.lower()
        self.mode = mode.lower()
        self.n = n
        self.passes = max(1, passes)
        self.remap_ids = remap_ids
        self.prompt_id = prompt_id



    def rerank(
        self,
        query: str,
        docs: Dict[str, str],
        top_n: Optional[int] = None,
        cluster_groups: Optional[Dict[str, List[str]]] = None,
        cluster_tournament: bool = False,
        cluster_top_c: int = 5,
    ) -> tuple[List[Tuple[str, float]], bool]:

        if not docs:
            return [], True
        try:
            if cluster_tournament:
                results = self._rerank_cluster_tournament(
                    query=query,
                    docs=docs,
                    cluster_groups=cluster_groups,
                    top_n=top_n,
                    cluster_top_c=cluster_top_c,
                )
            elif self.mode == "simple":
                results = self._rerank_simple(query, docs, top_n)
            elif self.mode == "sliding_window":
                results = self._rerank_sliding_window(query, docs, top_n)
            elif self.mode == "tournament":
                results = self._rerank_tournament(query, docs, top_n)
            else:
                # Should never happen thanks to __init__ validation.
                raise ValueError(f"Unknown mode '{self.mode}'.")

            return results, False
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return [(doc_id, 1.0 - rank / len(docs)) for rank, doc_id in enumerate(docs.keys())],True

    # ------------------------------------------------------------------
    # Mode 1 – Simple (single-shot)
    # ------------------------------------------------------------------

    def _rerank_simple(
        self,
        query: str,
        docs: Dict[str, str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Pass **all** documents to the LLM in a single call."""
        doc_ids = list(docs.keys())
        #logger.info(f"initial document order :\n{[(i+1, did) for i, did in enumerate(doc_ids)]}")

        ranked_ids = self._call_llm(query, docs)
        #logger.info(f"LLM ranked IDs: {ranked_ids}")
        #logger.info(f"LLM ranked IDs: {ranked_ids}")
        return self._ordered_ids_to_scores(ranked_ids, top_n=top_n)

    # ------------------------------------------------------------------
    # Mode 2 – Sliding window
    # ------------------------------------------------------------------

    def _rerank_sliding_window(
        self,
        query: str,
        docs: Dict[str, str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:

        window_size = self.n
        step = max(1, window_size // 2)  # 50 % overlap
        doc_ids = list(docs.keys())
        #logger.info(f"initial document order :\n{[(i+1, did) for i, did in enumerate(doc_ids)]}")
        total = len(doc_ids)
        # Reverse: incoming order is desc → make it ascending (worst first).
        doc_ids = list(reversed(doc_ids))
        #logger.info(f"initial document order (worst → best):\n{[(i+1, did) for i, did in enumerate(doc_ids)]}")
        total = len(doc_ids)
        effective_window = min(window_size, total)

        for pass_idx in range(self.passes):
            if pass_idx > 0:
                # Previous pass left doc_ids in best-first order; reverse back
                # to worst-first so the sliding window can promote upward again.
                doc_ids = list(reversed(doc_ids))
                logger.info(f"Sliding-window pass {pass_idx + 1}/{self.passes}")

            # Build start indices from the back towards the front.
            start_indices = list(range(total - effective_window, -1, -step))
            if not start_indices:
                start_indices = [0]
            # Guarantee we always include a window starting at 0.
            if start_indices[-1] != 0:
                start_indices.append(0)

            for start in start_indices:
                end = min(start + effective_window, total)
                window_ids = doc_ids[start:end]
                logger.info(f"Reranking window (pass {pass_idx + 1}):\n{[(i+1, did) for i, did in enumerate(window_ids)]}")
                window_docs = {did: docs[did] for did in window_ids}

                ranked_window = self._call_llm(query, window_docs)

                # Replace the window slice with the LLM's ordering.
                doc_ids[start:end] = ranked_window
                logger.info(f"Updated document order (pass {pass_idx + 1}):\n{[(i+1, did) for i, did in enumerate(doc_ids)]}")

        # Deduplicate (keeps first occurrence → best position).
        doc_ids = list(dict.fromkeys(doc_ids))
        return self._ordered_ids_to_scores(doc_ids, top_n=top_n)

    # ------------------------------------------------------------------
    # Mode 3 – Tournament
    # ------------------------------------------------------------------

    def _rerank_tournament(
        self,
        query: str,
        docs: Dict[str, str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        
        group_size = self.n

        all_ids = list(docs.keys())
        random.shuffle(all_ids)
        #logger.info(f"Starting tournament with {len(all_ids)} documents,\n {[(i+1, candidate_id) for i, candidate_id in enumerate(all_ids)]}...")
        # Collect eliminated IDs across rounds (earliest eliminated = least
        # relevant, so they are appended at the very end in reverse order).
        eliminated: List[str] = []

        remaining = list(all_ids)

        while len(remaining) > group_size:
            # --- one round ---
            groups = [
                remaining[i : i + group_size]
                for i in range(0, len(remaining), group_size)
            ]

            next_round: List[str] = []
            for group in groups:
                group_docs = {did: docs[did] for did in group}
                ranked_group = self._call_llm(query, group_docs)

                # If the group is smaller than expected (last/remainder group)
                # keep ceil(len(group)/2) survivors.
                k = math.ceil(len(ranked_group) / 2)
                next_round.extend(ranked_group[:k])
                eliminated.extend(ranked_group[k:])

            remaining = next_round

        # Final round – rank the last batch.
        if remaining:
            final_docs = {did: docs[did] for did in remaining}
            remaining = self._call_llm(query, final_docs)

        # Build full ranking: survivors (best) + eliminated (worst-last).
        full_ranking = remaining + list(reversed(eliminated))

        return self._ordered_ids_to_scores(full_ranking, top_n=top_n)

    def _rerank_cluster_tournament(
        self,
        query: str,
        docs: Dict[str, str],
        cluster_groups: Optional[Dict[str, List[str]]] = None,
        top_n: Optional[int] = None,
        cluster_top_c: int = 5,
    ) -> List[Tuple[str, float]]:
        if not cluster_groups:
            logger.warning("Cluster tournament requested without cluster groups. Falling back to tournament mode.")
            return self._rerank_tournament(query, docs, top_n=top_n)

        ordered_doc_ids = list(docs.keys())
        valid_doc_ids = set(ordered_doc_ids)
        cluster_to_docs: Dict[str, List[str]] = {}
        assigned = set()
        duplicate_count = 0

        for cluster_name, cluster_doc_ids in cluster_groups.items():
            logger.info(f"Processing cluster '{cluster_name}' with {len(cluster_doc_ids)} documents.")
            if not cluster_doc_ids:
                continue

            filtered_ids: List[str] = []
            for doc_id in cluster_doc_ids:
                cid = str(doc_id)
                if cid not in valid_doc_ids:
                    continue
                if cid in assigned:
                    duplicate_count += 1
                    continue
                filtered_ids.append(cid)
                assigned.add(cid)

            if filtered_ids:
                cluster_to_docs[str(cluster_name)] = filtered_ids

        for doc_id in ordered_doc_ids:
            logger.info(f"Checking document ID '{doc_id}' for cluster assignment...")
            if doc_id not in assigned:
                cluster_to_docs[f"singleton_{doc_id}"] = [doc_id]

        if duplicate_count:
            logger.warning(
                f"Cluster tournament found {duplicate_count} duplicate IDs across clusters. Keeping first assignment."
            )

        if not cluster_to_docs:
            return self._rerank_tournament(query, docs, top_n=top_n)

        cluster_rankings: Dict[str, List[str]] = {}

        for cluster_name, cluster_ids in cluster_to_docs.items():
            logger.info(f"Reranking cluster '{cluster_name}' with {len(cluster_ids)} documents...")
            cluster_docs = {doc_id: docs[doc_id] for doc_id in cluster_ids}
            cluster_results = self._rerank_simple(query, cluster_docs)
            ranked_cluster_ids = [doc_id for doc_id, _ in cluster_results]
            if not ranked_cluster_ids:
                continue

            cluster_rankings[cluster_name] = ranked_cluster_ids

        if not cluster_rankings:
            return self._rerank_simple(query, docs, top_n=top_n)

        final_order: List[str] = []
        seen = set()
        cluster_names = list(cluster_rankings.keys())

        # Final merge: pure round-robin over full per-cluster rankings.
        max_depth = max(len(ids) for ids in cluster_rankings.values())
        for depth in range(max_depth):
            for cluster_name in cluster_names:
                ranked_ids = cluster_rankings.get(cluster_name, [])
                if depth >= len(ranked_ids):
                    continue
                doc_id = ranked_ids[depth]
                if doc_id not in seen:
                    final_order.append(doc_id)
                    seen.add(doc_id)

        for doc_id in ordered_doc_ids:
            if doc_id not in seen:
                final_order.append(doc_id)
                seen.add(doc_id)

        return self._ordered_ids_to_scores(final_order, top_n=top_n)

    # ------------------------------------------------------------------
    # LLM interaction helpers
    # ------------------------------------------------------------------

    def _call_llm(self, query: str, docs: Dict[str, str]) -> List[str]:
        """Send a single ranking request to the LLM.

        Returns an ordered list of **original** document IDs (best first).
        Internally, IDs are replaced with shuffled neutral labels (D01, D02, …)
        to prevent the model from developing sequential bias based on the
        real patent numbers.  On failure the original insertion order is
        returned unchanged.
        """
        original_docs = dict(docs)
        original_doc_ids = list(original_docs.keys())
        alias_to_real: Dict[str, str] = {}

        if self.remap_ids:
            docs, alias_to_real = self._encode_ids(docs)
       # logger.info(f"Initial order of documents sent to LLM:\n{[(i+1, did) for i, did in enumerate(docs.keys())]}")
        prompt = utils.read_md_prompt(self.prompt_id, PROMPT_MD_PATH.format(id=self.prompt_id))
        if not prompt or not prompt.get("system") or not prompt.get("user"):
            logger.error(
                "Prompt '%s' is missing or invalid at path '%s'. Falling back to original order.",
                self.prompt_id,
                PROMPT_MD_PATH.format(id=self.prompt_id),
            )
            raise
            #return original_doc_ids
        user_prompt = self._construct_prompt2(query, docs,user_prompt=prompt["user"])
        try:

            call_args = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": user_prompt},
                ]
                }

            # 2. Add 'extra_body' only if a condition is met
            # For example, only if thinking mode is explicitly enabled
            if self.backend == "openai":
                call_args["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": self.thinking},
                }
                call_args["max_tokens"] = 50000
                call_args["temperature"] = 0.1

            # 3. Call the method using the ** operator to unpack the dictionary
            response = self.client.chat.completions.create(**call_args)
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.thinking},
                },
                max_tokens=50000,
            )
            """
           # print(f"LLM response: {response.choices[0]}\n-------------------")
            if self.thinking:
                raw = response.choices[0].message.content
            else:
                raw = response.choices[0].message.reasoning
            #raw = response.choices[0].message.content
        except Exception as e:
            error_message = str(e).lower()
            is_input_too_long = any(
                marker in error_message
                for marker in (
                    "a maximum input length",
                    "reduce the length",
                    "context length is only",
                    "too many tokens",
                    
                )
            )

            if self.mode == "simple" and is_input_too_long:
                fallback_top_n = max(1, len(original_doc_ids) // 2)
                logger.warning(
                    "LLM input too long in simple mode; falling back to _rerank_tournament with top_n=%d.",
                    fallback_top_n,
                )
                previous_n = self.n
                try:
                    fallback_results = self._rerank_tournament(
                        query=query,
                        docs=original_docs,
                        top_n=fallback_top_n,
                    )
                    fallback_doc_ids = [doc_id for doc_id, _ in fallback_results]
                    if fallback_doc_ids:
                        return fallback_doc_ids
                except Exception as fallback_error:
                    logger.error("Fallback tournament rerank failed: %s", fallback_error)
                finally:
                    self.n = previous_n

            logger.error("LLM call failed: %s", e)
            raise
            #return original_doc_ids
        #logger.info(f"LLM raw output: {raw}")
        ranked_ids = self._parse_ids(raw, docs)

        if self.remap_ids:
            return self._decode_ids(ranked_ids, alias_to_real)
        
        return ranked_ids

    # ------------------------------------------------------------------
    # ID aliasing helpers
    # ------------------------------------------------------------------
    

    

    @staticmethod
    def _encode_ids(docs: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Replace real patent IDs with non-sequential, random hex aliases.
        Preserves the input order of the dictionary for the LLM prompt.
        """
        # 1. Prepare to store mappings
        alias_to_real = {}
        aliased_docs = {}
        
        # 2. Iterate through docs in their current order
        for real_id, text in docs.items():
            # Generate a short, random hex ID (e.g., 'X8R2') 
            # This implies no order, unlike 'D01'
            alias = secrets.token_hex(2).upper()
            
            # Ensure uniqueness (very low collision chance, but good practice)
            while alias in alias_to_real:
                alias = secrets.token_hex(2).upper()
            
            # 3. Map them
            # Format as [ID_X8R2] so the LLM treats it as a tag
            tag = f"ID_{alias}"
            alias_to_real[tag] = real_id
            aliased_docs[tag] = text
            
        return aliased_docs, alias_to_real

    @staticmethod
    def _decode_ids(ranked_aliases: List[str], alias_to_real: Dict[str, str]) -> List[str]:
        """Map a ranked list of aliases back to original patent IDs."""
        decoded: List[str] = []
        unknown_aliases: List[str] = []

        for alias in ranked_aliases:
            real_id = alias_to_real.get(alias)
            if real_id is None:
                unknown_aliases.append(alias)
                continue
            decoded.append(real_id)

        if unknown_aliases:
            logger.warning(
                "Found %d unknown aliases in LLM output during decode. They will be ignored: %s",
                len(unknown_aliases),
                unknown_aliases,
            )

        # Ensure every candidate is still present exactly once.
        seen = set(decoded)
        for real_id in alias_to_real.values():
            if real_id not in seen:
                decoded.append(real_id)
                seen.add(real_id)

        return decoded

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _construct_prompt2(query: str, docs: Dict[str, str],user_prompt) -> str:
        formatted = "\n".join(
            f'<patent id="{pid}">\n{text}\n</patent>'
            for pid, text in docs.items()
        )
        return user_prompt.format(
            len_docs=len(docs),
            query=query,
            candidates=formatted
        )

    @staticmethod
    def _construct_prompt(query: str, docs: Dict[str, str]) -> str:
        formatted = "\n".join(
            f'<patent id="{pid}">\n{text}\n</patent>'
            for pid, text in docs.items()
        )
        return f"""\
Below is a patent application (the **query**) and {len(docs)} candidate prior-art documents. The candidates arrive in an approximate retrieval order — this order is unreliable.

Re-rank ALL {len(docs)} candidates from most to least relevant to the query, applying the criteria from your instructions.

## Query (patent application)
<query>
{query}
</query>

## Candidates (initial retrieval order — treat as approximate)
<candidates>
{formatted}
</candidates>

## Output format
Return a single line with every document ID exactly once, most relevant first, separated by ` > `:
[ID_] > [ID_] > ... > [ID_]

## Rules
- Include every document ID exactly once.
- Most relevant on the left, least relevant on the right.
- Do NOT simply reproduce the input order — you must re-evaluate each candidate independently.
- No explanations, commentary, or extra text.
"""

    # ------------------------------------------------------------------
    # Parsing / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ids(llm_output: str, docs: Dict[str, str]) -> List[str]:
        """Extract document IDs from LLM output.

        Handles both neutral aliases (``[ID_01] > [ID_02]``) and real patent IDs
        (``[EP-123] > [WO-456]``).  Deduplicates while preserving order and
        appends any IDs present in *docs* but missing from the output.

        If a ``<final_ranking>...</final_ranking>`` block is present, it is
        treated as the authoritative source to avoid picking IDs mentioned in
        explanatory analysis text.
        """
        #logger.info(f"LLM output: {llm_output}")

        # Try neutral alias pattern first, then fall back to real patent IDs.
        # Prefer parsing from <final_ranking> when available.
        try:
            ranking_source = llm_output
            ranking_match = re.search(
                r"<\s*final_ranking\s*>(.*?)<\s*/\s*final_ranking\s*>",
                llm_output,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if ranking_match:
                ranking_source = ranking_match.group(1)

            # First parse explicit ranking separators in <final_ranking> content.
            matches: List[str] = []
            if ranking_match:
                for item in ranking_source.split(">"):
                    token = item.strip().strip("[]")
                    token = token.strip(" \t\r\n.,;:(){}")
                    if token:
                        matches.append(token)

            # Fallback regex extraction for plain text responses.
            if not matches:
                alias_matches = re.findall(
                    r"\[?(ID_[A-Z0-9]+)\]?",
                    ranking_source,
                    flags=re.IGNORECASE,
                )
                if alias_matches:
                    matches = [m.upper() for m in alias_matches]
                else:
                    # Handles IDs like EP-1243277-A1, WO-2020-123456, US-12345.
                    matches = re.findall(
                        r"\[?([A-Za-z]{2,}-\d+(?:-[A-Za-z0-9]+)*)\]?",
                        ranking_source,
                    )

            if not matches:
                logger.warning("No valid IDs found in LLM output. Returning original order.")
                #return list(docs.keys())
                raise ValueError("No valid IDs found in LLM output.")
            
            #logger.info(f"Extracted IDs from LLM output: {matches}")
            valid = [pid for pid in matches if pid in docs]
            if len(valid) != len(matches):
                logger.warning(f"Invalid IDs in LLM output: {set(matches) - set(valid)}")
            #else:
                #logger.info("All IDs in LLM output are valid.")
            
            # Deduplicate, keeping first (= best) occurrence.
            ranked = list(dict.fromkeys(valid))
            #logger.info(f"Validated ranked IDs: {ranked}")
            # Append any candidates the LLM forgot.
            missing = [pid for pid in docs if pid not in set(ranked)]
            if len(missing) > 0:
                logger.warning(f"LLM output is missing {len(missing)} candidates. Appending them at the end: {missing}")
                ranked.extend(missing)
            #logger.info(f"Final ranked IDs after appending missing candidates: {ranked}")
            return ranked
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}. Returning original order.")
            logger.info(f"LLM output that caused parsing error: {llm_output}")
            #return list(docs.keys())
            raise ValueError(f"Error parsing LLM output: {e}")
    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _ordered_ids_to_scores(ordered_ids: List[str],top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Convert a ranked list of IDs to ``(id, score)`` tuples.

        Score is linearly normalised: best gets ``1.0``, worst gets ``1/n``.
        """

        if not ordered_ids:
            return []

        n = len(ordered_ids)
        results = [
            (doc_id, float((n - rank) / n))
            for rank, doc_id in enumerate(ordered_ids)
        ]
        if top_n:
            return results[:top_n]
        return results

