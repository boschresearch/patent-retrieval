# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

"""
Patent Summarizer Agent - Summarizes patent documents using LLM
"""
import json
import os
from typing import Dict, Optional

from json_repair import repair_json
from openai import AsyncOpenAI

from patent_retrieval import utils as utils

logger = utils.get_logger(__name__)

PROMPT_MD_PATH = os.getenv(
    "PROMPT_MD_PATH",
    "/home/alm3rng/patent-retrieval/src/patent_retrieval/prompts",
) + "/patent_summarizer_{id}.md"


class PatentSummarizerAgent:
    """Agent for summarizing patent documents using OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url: str = "http://localhost:59349/v1",
        backend: str = "openai",
    ):
        """
        Initialize the Patent Summarizer Agent.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for summarization
            base_url: OpenAI-compatible API base URL
            backend: Backend name (kept for interface compatibility)
        """
        self.api_key = api_key or "EMPTY"
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set or passed as api_key parameter")

        _ = backend
        self.chat_client = AsyncOpenAI(api_key=self.api_key, base_url=base_url,timeout=1800)
        self.model = model

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """
        Extract the first balanced JSON object from free-form text.
        Handles braces inside quoted strings and escaped quotes.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    @staticmethod
    def _failed_summary_v2_payload() -> Dict[str, object]:
        return {
            "technical_problem_solved": "Failed to parse JSON output even after repair.",
            "key_structural_features": [
                "Failed to parse JSON output even after repair.",
                "Failed to parse JSON output even after repair.",
                "Failed to parse JSON output even after repair.",
            ],
            "novelty_indicators": ["Failed to parse JSON output even after repair."],
            "technical_depth_score": 0,
            "core_contribution_summary": "Failed to parse JSON output even after repair.",
        }

    def _parse_summary_json(self, result: str) -> Dict[str, object]:
        """
        Parse a JSON object from noisy LLM output.
        Parse order: direct -> extracted object -> repaired extracted -> repaired full text.
        """
        cleaned = result.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        for candidate in (cleaned, self._extract_first_json_object(cleaned)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(repair_json(candidate))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

        parsed = json.loads(repair_json(cleaned))
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("summary output is not an object", result, 0)
        return parsed



    async def summarize_v1(self, query: str, max_tokens: int = 40000) -> str:
        """
        Summarize a patent document.
        
        Args:
            query: The full text of the patent document
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Rewritten query text for retrieval
        """
        prompt = utils.read_md_prompt("v1", PROMPT_MD_PATH.format(id="v1"))
        if not prompt or not prompt.get("system") or not prompt.get("user"):
            raise RuntimeError("Prompt 'v1' is missing or invalid at path: " + PROMPT_MD_PATH.format(id="v1"))

        user_content = prompt["user"].format(patent_text=query)

        try:
            response = await self.chat_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.5,
                max_tokens=max_tokens,
            )

            full_output = response.choices[0].message.content or ""

            # Simple parser to separate the thought process from the search query
            if "[QUERY]" in full_output:
                logger.info("Rewriting query for dense retrieval...")
                parts = full_output.split("[QUERY]")
                final_query = parts[1].strip()

                return final_query
            else:
                # Fallback if format is missed
                return query
        except Exception as e:
            raise RuntimeError(f"Failed to summarize patent: {str(e)}")

    async def summarize_v2(self, patent_text: str, max_tokens: int = 40000) -> Dict[str, object]:
        """
        Denoise a patent document into a compact technical JSON summary.
        
        Args:
            patent_text: The full text of the patent document
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Dictionary containing the denoised technical summary
        """
        prompt = utils.read_md_prompt("v2", PROMPT_MD_PATH.format(id="v2"))
        if not prompt or not prompt.get("system") or not prompt.get("user"):
            raise RuntimeError("Prompt 'v2' is missing or invalid at path: " + PROMPT_MD_PATH.format(id="v2"))

        user_content = prompt["user"].format(patent_text=patent_text)

        try:
            response = await self.chat_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": user_content}
                ],
               # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                temperature=0.1,
                max_tokens=max_tokens,
            )

            full_output = response.choices[0].message.content or ""
              # Log the beginning of the output for debugging
            if "[FINAL OUTPUT]" in full_output:

                parts = full_output.split("[FINAL OUTPUT]")
                result = parts[-1].strip()
                #logger.info(f"output from summarizer: {result}...")
                try:
                    return self._parse_summary_json(result)

                except json.JSONDecodeError as e2:
                    logger.info(f"Summarizer output: {result}")
                    logger.error(f"Failed to parse JSON output even after repair: {str(e2)}")
                    logger.error(f"Raw output: {result}")
                    return self._failed_summary_v2_payload()
            else:
                # Fallback if format is missed
                return self._failed_summary_v2_payload()

        except Exception as e:
            logger.error(f"Traceback for summarization error: ", exc_info=True)
            raise RuntimeError(f"Failed to summarize patent: {str(e)}")
        