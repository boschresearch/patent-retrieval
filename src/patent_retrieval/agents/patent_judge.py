# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

"""
Patent Judge Agent - Judges patent relevance using LLM
"""
import math
import os
import re
import json
from typing import Dict, List, Optional

from openai import OpenAI, AsyncOpenAI
from json_repair import repair_json

from patent_retrieval import utils as utils

logger = utils.get_logger(__name__)

PROMPT_MD_PATH = os.getenv(
    "PROMPT_MD_PATH",
    "/home/alm3rng/patent-retrieval/src/patent_retrieval/prompts",
) + "/patent_judge_{id}.md"
class PatentJudgeAgent:
    
    def __init__(self,  model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
                 base_url:str="http://localhost:59839/v1",api_key: str = "EMPTY",backend: str = "openai"):
        
        """
        Initialize the Patent Judge Agent.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for summarization
        """
        self.backend = backend.lower()
        if backend.lower() == "openai":
            self.api_key = api_key or "EMPTY"
            
            
            self.chat_client = AsyncOpenAI(api_key=self.api_key,base_url=base_url)

        elif backend.lower() == "azure":
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AsyncAzureOpenAI
            
            token_provider = get_bearer_token_provider(
                AzureCliCredential(),
                os.getenv("AZURE_SCOPE")
            )
     
            self.chat_client = AsyncAzureOpenAI(
                azure_endpoint=base_url,
                azure_ad_token_provider=token_provider,
                api_version="2025-01-01-preview",
            )
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
    def _failed_judgment_v1_payload() -> Dict[str, object]:
        return {
  "query_core_mechanism": "Failed to parse JSON output even after repair.",
  "candidate_mechanism": "Failed to parse JSON output even after repair.",
  "matched_features": ["Failed to parse JSON output even after repair.", "Failed to parse JSON output even after repair."],
  "missing_features": ["Failed to parse JSON output even after repair."],
  "category": "0",
  "score": 0,
  "utility_summary": "Failed to parse JSON output even after repair."
}


    @staticmethod
    def _failed_judgment_v2_payload() -> Dict[str, str]:
        return {
            "reasoning": "Failed to parse judge_v2 output.",
            "verdict": "DROP",
        }
    @staticmethod
    def _failed_judgment_v3_payload() -> Dict[str, str]:
        return {
            "primary_invention_mechanism": "Failed to parse JSON output even after repair.",
            "technical_problem_solved": "Failed to parse JSON output even after repair.",
            "key_structural_features": ["Failed to parse JSON output even after repair.", "Failed to parse JSON output even after repair.", "Failed to parse JSON output even after repair."],
            "novelty_indicators": ["Failed to parse JSON output even after repair."],
            "technical_domain": "Failed to parse JSON output even after repair.",
            "technical_depth_score": 0,
            "core_contribution_summary": "Failed to parse JSON output even after repair."
        }

    def _parse_judge_json(self, result: str) -> Dict[str, object]:
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
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(repair_json(candidate))
                except json.JSONDecodeError:
                    continue

        return json.loads(repair_json(cleaned))

    def _parse_judge_v2_json(self, result: str) -> Dict[str, str]:
        parsed = self._parse_judge_json(result)

        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("judge_v2 output is not an object", result, 0)

        reasoning = str(parsed.get("reasoning", "")).strip()
        verdict_raw = str(parsed.get("verdict", "")).strip().upper()

        if verdict_raw not in {"KEEP", "DROP"}:
            raise json.JSONDecodeError("judge_v2 verdict must be KEEP or DROP", result, 0)

        return {
            "reasoning": reasoning,
            "verdict": verdict_raw,
        }
    
    async def judge_v1(self, target: str, candidate: str, max_tokens: int = 40000) -> Dict[str, str]:
        """
        Judge the relevance of a candidate document to a given patent.
        
        Args:
            target: The full text of the target patent claim
            candidate: The full text of the candidate document
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Dictionary containing the summary and key information
        """

        prompt = utils.read_md_prompt("v1", PROMPT_MD_PATH.format(id="v1"))
        system_prompt = prompt["system"]
        query = prompt["user"].format(target=target, candidate=candidate)
        try:
            response = await self.chat_client.chat.completions.create(
            model=self.model, 
            messages=[
                    {"role": "system", "content": system_prompt},
                    
                    {"role": "user", "content": query}
                ],
                temperature=0.1, 
                max_tokens=40000,

            )
            
            full_output = response.choices[0].message.content
            logger.info(f"Judge output: {full_output}")
            # Simple parser to separate the thought process from the search query
            if "[FINAL OUTPUT]" in full_output:
                parts = full_output.split("[FINAL OUTPUT]")
               # reasoning = parts[0].replace("[THOUGHT PROCESS]", "").strip()
                result = parts[-1].strip()
                
                # Optional: Print reasoning to console for debugging/education
                # print(f"DEBUG REASONING:\n{reasoning}\n")
                #logger.info(f"Rewritten Query: {final_query}")
                # Parse JSON from the result
                try:
                    return self._parse_judge_json(result)

                except json.JSONDecodeError as e2:
                    logger.info(f"Judgment output: {result}")
                    logger.error(f"Failed to parse JSON output even after repair: {str(e2)}")
                    logger.error(f"Raw output: {result}")
                    return self._failed_judgment_payload()
            else:
                #logger.info("Could not find [QUERY] section in LLM output.")
                # Fallback if format is missed
                return self._failed_judgment_v1_payload()

        except Exception as e:
            raise RuntimeError(f"Failed to summarize patent: {str(e)}")
        
    async def judge_v5(self, target: str, candidate: str, max_tokens: int = 40000) -> Dict[str, str]:
        """
        Judge the relevance of a candidate document to a given patent.
        
        Args:
            target: The full text of the target patent claim
            candidate: The full text of the candidate document
            max_tokens: Maximum tokens for the summary
            
        Returns:
            Dictionary containing the summary and key information
        """

        prompt = utils.read_md_prompt("v5", PROMPT_MD_PATH.format(id="v5"))
        system_prompt = prompt["system"]
        query = prompt["user"].format(target=target)
        try:
            response = await self.chat_client.chat.completions.create(
            model=self.model, 
            messages=[
                    {"role": "system", "content": system_prompt},
                    
                    {"role": "user", "content": query}
                ],
                temperature=0.1, 
                max_tokens=40000,

            )
            
            full_output = response.choices[0].message.content

            if "[FINAL OUTPUT]" in full_output:
                parts = full_output.split("[FINAL OUTPUT]")
                result = parts[-1].strip()
                

                try:
                    return self._parse_judge_json(result)

                except json.JSONDecodeError as e2:
                    logger.info(f"Judgment output: {result}")
                    logger.error(f"Failed to parse JSON output even after repair: {str(e2)}")
                    logger.error(f"Raw output: {result}")
                    return self._failed_judgment_payload()
            else:
                #logger.info("Could not find [QUERY] section in LLM output.")
                # Fallback if format is missed
                return self._failed_judgment_v3_payload()

        except Exception as e:
            raise RuntimeError(f"Failed to summarize patent: {str(e)}")
        

    async def judge_v2(self, target: str, candidate: str, max_tokens: int = 20000) -> Dict[str, str]:

            
        prompt = utils.read_md_prompt("v2", PROMPT_MD_PATH.format(id="v2"))
        system_prompt = prompt["system"]
        query = prompt["user"].format(target=target, candidate=candidate)
        try:
            response = await self.chat_client.chat.completions.create(
            model=self.model, 
            messages=[
                    {"role": "system", "content": system_prompt},
                  #  {"role": "assistant", "content": """Understood. I will respond with a strict lowercase 'yes' or 'no'."""},
                    
                    {"role": "user", "content": query}
                ],
                temperature=0.1, 
                max_tokens=max_tokens,
            )

            full_output = response.choices[0].message.content or ""
            if "[FINAL OUTPUT]" in full_output:
                result = full_output.split("[FINAL OUTPUT]")[-1].strip()
            else:
                result = full_output.strip()

            try:
                return self._parse_judge_v2_json(result)
            except json.JSONDecodeError as e2:
                logger.info(f"judge_v2 output: {result}")
                logger.error(f"Failed to parse judge_v2 output: {str(e2)}")
                return self._failed_judgment_v2_payload()
            
        except Exception as e:
            raise RuntimeError(f"Failed to summarize patent: {str(e)}")

    async def judge_v4(self, target: str, candidate: str, max_tokens: int = 40000 ) -> dict:
        
        def _parse_verdict( output: str) -> str:
            """
            Parse the verdict from the response output.
            Extracts the content between <verdict> tags.
            
            Args:
                output: The full response text from the LLM
                
            Returns:
                The verdict as a string: 'yes' or 'no'
            """
            match = re.search(r'<verdict>\s*(yes|no)\s*</verdict>', output, re.IGNORECASE)
            if match:
                if "yes" in match.group(1).strip().lower():
                    return 1
                else:
                    return 0
            
            # Fallback: check for the word at the end
            last_line = output.strip().split('\n')[-1].lower()
            if 'yes' in last_line:
                return 1
            elif 'no' in last_line:
                return 0
            
            logger.warning(f"Could not parse verdict from output: {output}")
            return -1
        

            
        prompt = utils.read_md_prompt("v4", PROMPT_MD_PATH.format(id="v4"))
        system_prompt = prompt["system"]
        query = prompt["user"].format(target=target, candidate=candidate)
        
        try:
            response = await self.chat_client.chat.completions.create(
                model=self.model, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3 if "openai" in self.backend else 1.0, 
            )
            full_output = response.choices[0].message.content
            relevance_assessment_match = re.search(r"(?i)Relevance Assessment[:\*]*\s*(.*?)(?=\s*(?:\*\*)*Final output|$)", full_output, re.DOTALL)
            summary_match = re.search(r'<summary>(.*?)</summary>', full_output, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else "No summary provided"
            verdict = _parse_verdict(full_output)
            relevance_assessment = relevance_assessment_match.group(1).strip() if relevance_assessment_match else "No relevance assessment provided"
            return {"verdict": verdict, "summary": summary, "relevance_assessment": relevance_assessment}
            
        except Exception as e:
            raise RuntimeError(f"Failed to judge patent: {str(e)}")

        


    async def judge_v3(self, target: str, candidate: str, max_tokens: int = 150) -> float:
        """
        Optimized for Qwen3 30b-instruct using Chain-of-Thought validation.
        Returns the probability of the 'yes' token after a reasoning step.
        """
        def get_yes_prop(top_logprobs):
            """
            Helper to extract the linear probability of 'yes' (or 'Yes', 'YES') tokens.
            """
            yes_prob = 0.0
            no_prob = 0.0
            
            for token_data in top_logprobs:
                token_str = token_data.token.strip().lower()
                if token_str == 'yes':
                    yes_prob += math.exp(token_data.logprob)
                elif token_str == 'no':
                    no_prob += math.exp(token_data.logprob)
                    
            # Normalize if both are present, or just return raw yes_prob
            if yes_prob + no_prob > 0:
                return yes_prob / (yes_prob + no_prob)
            return yes_prob
        # 1. System Prompt: Role + Specific Thinking Steps
        # Qwen follows detailed steps well. We ask it to check specific alignment points.
        prompt = utils.read_md_prompt("v3", PROMPT_MD_PATH.format(id="v3"))
        system_prompt = prompt["system"]
        query = prompt["user"].format(target=target, candidate=candidate)

        try:
            response = await self.chat_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,  # Strict logic
                max_tokens=1,     # Single token output
                logprobs=True,
                top_logprobs=5
            )

            # We return the probability of the "yes" token.
            # Since the prompt biases heavily toward "yes", 
            # a low score here means the model is VERY confident it's junk.
            return get_yes_prop(response.choices[0].logprobs.content[0].top_logprobs)

        except Exception as e:
            raise RuntimeError(f"Failed to judge patent: {str(e)}")

    