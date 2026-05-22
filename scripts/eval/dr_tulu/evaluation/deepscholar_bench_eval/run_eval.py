#!/usr/bin/env python3
"""
DeepScholar-Bench (63-item Related Works generation) evaluation script.

Evaluates model-generated "Related Works" sections for academic papers against
ground truth data.  Six metrics are supported:

  - **reference_coverage**: fraction of important citations that appear among
    the model's retrieved references.
  - **cite_p**: citation precision – whether each cited reference actually
    supports the sentence it annotates (LLM-graded).
  - **nugget_coverage**: fraction of ground-truth nuggets covered by the
    generated text (LLM-graded).
  - **organization**: pairwise comparison of organization/coherence between
    generated and ground-truth related works sections (LLM-graded).
  - **claim_coverage**: citation precision by claim support – whether cited
    references support the claims in the generated text (LLM-graded).
  - **coverage_relevance_rate**: reference relevance scoring – how relevant
    each retrieved reference is to the paper (LLM-graded).

Usage:
    python scripts/eval/dr_tulu/evaluation/deepscholar_bench_eval/run_eval.py \\
        --input_file path/to/results.jsonl \\
        --task_name my_model \\
        --dataset_dir deepscholar/dataset
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``open_instruct`` is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[5])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import csv
import json
import os
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

from open_instruct import logger_utils
from open_instruct.scholar_evaluator import (
    clean_rich_media_reference,
    custom_sent_tokenize,
    extract_document_infos_from_text,
    remove_citations,
)

logger = logger_utils.setup_logger(__name__)

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DEFAULT_GEMINI_RACE_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_FACT_MODEL = "gemini-2.5-flash"
DEFAULT_INTERNAL_JUDGE_URL = "https://ivavmlgq.fn.bytedance.net"
DEFAULT_INTERNAL_PROXY = "http://sys-proxy-rd-relay.byted.org:8118"


# ============================================================================
# LLM Judge Client
# ============================================================================
class JudgeLLMClient:
    """Judge client for DeepScholar-Bench evaluation.

    Supports:
    - ``openai_compatible`` chat completions API (default)
    - ``internal_http`` endpoint
    - ``gemini`` for Gemini models
    """

    def __init__(
        self,
        *,
        backend: str = "openai_compatible",
        api_key: str | None = None,
        api_base: str | None = None,
        race_model: str | None = None,
        timeout_s: int = 600,
    ):
        self.backend = backend
        self.timeout_s = timeout_s
        self.race_model = race_model or os.environ.get("DRB_RACE_MODEL")

        if self.backend == "gemini":
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise ImportError(
                    "google-genai package required for Gemini backend. "
                    "Install with: pip install google-genai"
                )

            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
                )

            self.client = genai.Client(
                api_key=self.api_key, http_options={"timeout": timeout_s * 1000}
            )
            self._types = types
            self.race_model = self.race_model or DEFAULT_GEMINI_RACE_MODEL
            return

        if self.backend == "internal_http":
            self.internal_judge_url = (
                os.environ.get("DRB_INTERNAL_JUDGE_URL")
                or os.environ.get("SCHOLAR_JUDGE_URL")
                or DEFAULT_INTERNAL_JUDGE_URL
            )
            self.internal_proxy = (
                os.environ.get("DRB_INTERNAL_PROXY")
                or os.environ.get("SCHOLAR_JUDGE_PROXY")
                or DEFAULT_INTERNAL_PROXY
            )
            self.race_model = self.race_model or "internal_http"
            return

        if self.backend != "openai_compatible":
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                "Use 'openai_compatible', 'internal_http', or 'gemini'."
            )

        self.api_base = (
            api_base
            or os.environ.get("DRB_JUDGE_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).rstrip("/")
        if not self.api_base:
            raise ValueError(
                "OpenAI-compatible judge backend requires DRB_JUDGE_API_BASE "
                "(or OPENAI_BASE_URL / OPENAI_API_BASE)."
            )
        if not self.api_base.endswith("/v1"):
            self.api_base = f"{self.api_base}/v1"

        self.api_key = (
            api_key
            or os.environ.get("DRB_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.race_model = self.race_model or os.environ.get("DRB_JUDGE_MODEL")
        if not self.race_model:
            raise ValueError(
                "OpenAI-compatible judge backend requires DRB_RACE_MODEL or DRB_JUDGE_MODEL."
            )

    @property
    def grader_model(self) -> str:
        """Backward-compatible alias for the primary judge model name."""
        return self.race_model

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
    ) -> str:
        """Generate text response from the configured judge backend."""
        model_to_use = model or self.race_model
        if not model_to_use and self.backend != "internal_http":
            raise ValueError("Judge model is not configured.")

        if self.backend == "gemini":
            contents = []
            if system_prompt:
                contents.append({"role": "system", "parts": [{"text": system_prompt}]})
            contents.append({"role": "user", "parts": [{"text": user_prompt}]})

            response = self.client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    thinking_config=self._types.ThinkingConfig(thinking_budget=16000)
                ),
            )
            return response.text

        if self.backend == "internal_http":
            content = f"{system_prompt}{user_prompt}" if system_prompt else user_prompt
            payload = {"messages": [{"role": "user", "content": content}]}
            proxies = None
            if self.internal_proxy:
                proxies = {
                    "http": self.internal_proxy,
                    "https": self.internal_proxy,
                }
            response = requests.post(
                self.internal_judge_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                proxies=proxies,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("response")
            if not content:
                raise ValueError(f"Missing 'response' in internal judge response: {data}")
            return content

        # openai_compatible
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": model_to_use,
            "messages": messages,
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not content:
            raise ValueError(f"Missing message content in judge response: {data}")
        return content


def _judge_generate_with_retry(
    judge: JudgeLLMClient,
    user_prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    max_retries: int = 5,
) -> str:
    """Call ``judge.generate()`` with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return judge.generate(user_prompt, system_prompt=system_prompt, model=model)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "Judge call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("Judge call failed after %d attempts: %s", max_retries, e)
                raise
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(input_file: str) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """Load ``papers_with_related_works.csv``."""
    csv_path = os.path.join(dataset_dir, "papers_with_related_works.csv")
    return pd.read_csv(csv_path)


def load_important_citations(dataset_dir: str) -> dict[str, list[dict]]:
    """Load ``important_citations.csv`` grouped by parent_paper_arxiv_id."""
    csv_path = os.path.join(dataset_dir, "important_citations.csv")
    df = pd.read_csv(csv_path)
    grouped: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        parent_id = str(row["parent_paper_arxiv_id"])
        grouped.setdefault(parent_id, []).append(row.to_dict())
    return grouped


def load_nuggets(dataset_dir: str, file_id: str) -> dict | None:
    """Load ground truth nuggets for a paper."""
    nuggets_path = os.path.join(dataset_dir, "gt_nuggets_outputs", str(file_id), "res.json")
    if not os.path.exists(nuggets_path):
        return None
    with open(nuggets_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------


def parse_model_output(result: dict) -> dict:
    """Parse a single model output into the evaluation format.

    Returns a dict with keys:
      - clean_text: answer text with ``[N]`` style citations.
      - docs: mapping  *id → {"title": str, "sent": str}*.
      - citation_ids: list of integer citation IDs extracted from cites.
      - citations_for_cite_quality: list of ``(title, abstract)`` tuples for
        each cited document.
    """
    final_response = result.get("final_response", "")

    # Clean the response and collect integer citation IDs.
    clean_text, citation_ids = clean_rich_media_reference(final_response, return_citation_ids=True)

    # Build docs mapping from full_traces.tool_calls.
    docs: dict[int, dict[str, str]] = {}

    tool_calls = result.get("full_traces", {}).get("tool_calls", [])

    # Strategy 1: parse original XML from raw_text (``<document reference_id=…>``).
    all_raw_text = ""
    for tc in tool_calls:
        raw_text = tc.get("raw_output", {}).get("raw_text", "")
        if raw_text:
            all_raw_text += raw_text + "\n"

    if all_raw_text:
        doc_infos = extract_document_infos_from_text(all_raw_text)
        for di in doc_infos:
            if di["id"] is not None:
                docs[di["id"]] = {
                    "title": di.get("title", ""),
                    "sent": di.get("abstract", ""),
                }

    # Strategy 2: parse structured tool_outputs (handles rewritten IDs).
    for tc in tool_calls:
        for to_item in tc.get("raw_output", {}).get("tool_outputs", []):
            ref_id = to_item.get("reference_id", "")
            title = to_item.get("title", "")
            snippet = to_item.get("snippet", "")
            # Try to extract trailing numeric index from e.g. "call_xxx-5".
            m = re.search(r"(\d+)$", ref_id)
            if m:
                num_id = int(m.group(1))
                if num_id not in docs:
                    docs[num_id] = {"title": title, "sent": snippet}
            # Also try the first integer (for simple numeric IDs).
            m2 = re.search(r"(\d+)", ref_id)
            if m2:
                num_id2 = int(m2.group(1))
                if num_id2 not in docs:
                    docs[num_id2] = {"title": title, "sent": snippet}

    # Build citations_for_cite_quality list.
    citations_for_cite_quality: list[tuple[str, str]] = []
    for cid in citation_ids:
        if cid in docs:
            d = docs[cid]
            citations_for_cite_quality.append((d["title"], d["sent"]))

    return {
        "clean_text": clean_text,
        "docs": docs,
        "citation_ids": citation_ids,
        "citations_for_cite_quality": citations_for_cite_quality,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

_COMMON_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by",
}


def _normalize_title(title: str) -> str:
    """Normalize title for comparison (matches deepscholar logic)."""
    if not title:
        return ""
    normalized = re.sub(r"[^\w\s]", "", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    words = [w for w in normalized.split() if w not in _COMMON_WORDS and len(w) > 2]
    return " ".join(words)


def evaluate_reference_coverage(
    parsed_outputs: list[dict],
    paper_rows: list,
    important_citations: dict[str, list[dict]],
) -> list[float]:
    """Compute reference coverage for each paper.

    For each paper's important citations, check if the model's docs contain a
    matching reference (by arxiv_id substring or title similarity > 0.8).
    """
    scores: list[float] = []
    for parsed, paper_row in zip(parsed_outputs, paper_rows):
        arxiv_id = str(paper_row.get("arxiv_id", "")) if hasattr(paper_row, "get") else ""
        paper_cites = important_citations.get(arxiv_id, [])

        if not paper_cites:
            scores.append(0.0)
            continue

        docs = parsed["docs"]
        doc_titles = [d.get("title", "").lower() for d in docs.values()]
        doc_contents = [d.get("sent", "").lower() for d in docs.values()]

        covered = 0
        for cite in paper_cites:
            cite_arxiv = str(cite.get("cited_paper_arxiv_link", "") or cite.get("arxiv_link", ""))
            cite_title = str(cite.get("cited_paper_title", "") or cite.get("title", ""))
            found = False

            # Check arxiv_id substring match (extract DDDD.DDDDD pattern).
            if cite_arxiv:
                m = re.search(r"(\d+\.\d+)(?:v\d+)?", cite_arxiv)
                arxiv_id_num = m.group(1) if m else ""
                if arxiv_id_num:
                    for content in doc_contents + doc_titles:
                        if arxiv_id_num in content:
                            found = True
                            break

            # Check title similarity (with normalization).
            if not found and cite_title:
                norm_cite = _normalize_title(cite_title)
                for doc_title in doc_titles:
                    norm_doc = _normalize_title(doc_title)
                    if norm_cite and norm_doc:
                        sim = SequenceMatcher(None, norm_cite, norm_doc).ratio()
                    else:
                        sim = SequenceMatcher(None, cite_title.lower(), doc_title).ratio()
                    if sim > 0.8:
                        found = True
                        break

            if found:
                covered += 1

        scores.append(covered / len(paper_cites))

    return scores


def evaluate_cite_p(
    parsed_outputs: list[dict],
    judge: JudgeLLMClient,
) -> list[float]:
    """Compute citation precision for each paper.

    For each sentence with ``[N]`` citations (length >= 50 chars), ask the LLM
    whether each cited reference supports the claim.  Per-sentence precision is
    ``correct / total``; the paper score is the mean over all cited sentences.
    """
    scores: list[float] = []
    for parsed in parsed_outputs:
        clean_text = parsed["clean_text"]
        docs = parsed["docs"]

        if not clean_text:
            scores.append(0.0)
            continue

        sentences = custom_sent_tokenize(clean_text)

        sentence_precisions: list[float] = []
        for sentence in sentences:
            if len(sentence) < 50:
                continue

            cite_matches = re.findall(r"\[(\d+)\]", sentence)
            if not cite_matches:
                continue

            claim = remove_citations(sentence)
            correct = 0
            total = 0

            for cite_id_str in cite_matches:
                cite_id = int(cite_id_str)
                total += 1

                if cite_id not in docs:
                    continue

                doc = docs[cite_id]
                title = doc.get("title", "")
                abstract = doc.get("sent", "")

                prompt = (
                    "You are an Attribution Validator. Your task is to verify "
                    "whether a given reference can support the given claim.\n\n"
                    f"Claim: {claim}\n"
                    f"Reference: Title: {title}\n{abstract}\n\n"
                    "Does the reference support the claim? Answer '1' if it "
                    "supports the claim, or '0' if it does not.\n"
                    "Do not explain your answer, just return '1' or '0'.\n"
                    "Answer:"
                )

                try:
                    response = _judge_generate_with_retry(judge, prompt)
                    m = re.search(r"[01]", response)
                    if m and m.group(0) == "1":
                        correct += 1
                except Exception as e:
                    logger.warning("cite_p LLM call failed: %s", e)

            if total > 0:
                sentence_precisions.append(correct / total)

        if sentence_precisions:
            scores.append(float(np.mean(sentence_precisions)))
        else:
            scores.append(0.0)

    return scores


def evaluate_nugget_coverage(
    parsed_outputs: list[dict],
    orig_ids: list[str],
    dataset_dir: str,
    judge: JudgeLLMClient,
) -> list[float]:
    """Compute nugget coverage for each paper.

    Loads ground-truth nuggets from ``gt_nuggets_outputs/{id}/res.json`` and
    uses the LLM to judge which nuggets are covered by the generated text.
    Returns ``strict_all_score = count(support) / count(all_nuggets)`` per paper.
    """
    scores: list[float] = []
    for parsed, orig_id in zip(parsed_outputs, orig_ids):
        clean_text = parsed["clean_text"]

        nuggets_data = load_nuggets(dataset_dir, orig_id)
        if nuggets_data is None:
            logger.warning("No nuggets found for id=%s", orig_id)
            scores.append(0.0)
            continue

        query = nuggets_data.get("query", "")
        # Prefer "supported_nuggets" (ground-truth validated), fall back to "nuggets"
        nuggets = nuggets_data.get("supported_nuggets", []) or nuggets_data.get("nuggets", [])

        if not nuggets:
            scores.append(0.0)
            continue

        nugget_texts = [n.get("text", "") for n in nuggets]

        # Use the official Nuggetizer assigner prompt format (SUPPORT_GRADE_3)
        instruction = (
            f"Based on the query and passage, label each of the {len(nuggets)} nuggets "
            f"either as support, partial_support, or not_support using the following criteria. "
            f"A nugget that is fully captured in the passage should be labeled as support. "
            f"A nugget that is partially captured in the passage should be labeled as partial_support. "
            f"If the nugget is not captured at all, label it as not_support. "
            f"Return the list of labels in a Pythonic list format (type: List[str]). "
            f"The list should be in the same order as the input nuggets. "
            f"Make sure to provide a label for each nugget."
        )

        prompt = (
            f"{instruction}\n\n"
            f"Search Query: {query}\n"
            f"Passage: {clean_text}\n"
            f"Nugget List: {nugget_texts}\n"
            f"Only return the list of labels (List[str]). Do not explain.\n"
            f"Labels:"
        )

        try:
            response = _judge_generate_with_retry(judge, prompt)
            labels = _parse_label_list(response, len(nuggets))
            support_count = sum(1 for label in labels if label == "support")
            score = support_count / len(nuggets)
        except Exception as e:
            logger.warning("nugget_coverage LLM call failed for id=%s: %s", orig_id, e)
            score = 0.0

        scores.append(score)

    return scores


def _parse_label_list(response: str, expected_len: int) -> list[str]:
    """Parse a list of labels from LLM response."""
    import ast as _ast

    # Try to extract a Python list literal.
    m = re.search(r"\[.*?\]", response, re.DOTALL)
    if m:
        try:
            labels = _ast.literal_eval(m.group(0))
            if isinstance(labels, list) and len(labels) == expected_len:
                return [str(l).strip().lower().replace(" ", "_") for l in labels]
        except Exception:
            pass

    # Fallback: scan each line for label keywords.
    labels: list[str] = []
    for line in response.split("\n"):
        line_lower = line.strip().lower()
        if "not_support" in line_lower or "not support" in line_lower:
            labels.append("not_support")
        elif "partial_support" in line_lower or "partial support" in line_lower:
            labels.append("partial_support")
        elif "support" in line_lower:
            labels.append("support")
        if len(labels) == expected_len:
            return labels

    # Pad with not_support if we could not parse enough labels.
    while len(labels) < expected_len:
        labels.append("not_support")
    return labels[:expected_len]


# ---------------------------------------------------------------------------
# New metrics: organization, claim_coverage, coverage_relevance_rate
# ---------------------------------------------------------------------------

_ORGANIZATION_PROMPT_TEMPLATE = """\
You will receive the **title and abstract** of a research paper, together with \
two candidate **related-work sections** (A and B) written for that paper.
Do not consider the formatting of the text e.g., latex, markdown, etc. Only consider the content.

Task: Decide which section—A or B—exhibits better organization and coherence.
Return only one letter: A or B.

How to judge (organization only)
Ignore breadth of coverage, citation accuracy, and analytic depth. Assess:
Logical structure – Clear introduction, grouping of related themes, and smooth progression of ideas.
Paragraph cohesion – Each paragraph develops a single topic and flows naturally to the next.
Clarity & readability – Minimal redundancy or contradictions; transitions guide the reader.
Signposting – Helpful headings, topic sentences, or discourse markers (if provided).

Pick the section that is easier to follow and better structured—no ties.

Output format: return a single token: A or B
A – Section A is better organized.
B – Section B is better organized.

### Paper under assessment
{paper_title}
{paper_abstract}

### Candidate related-work section A
{related_work_a}

### Candidate related-work section B
{related_work_b}

Output your answer as a JSON dictionary:
"decision": "A" or "B"
"explanation": "One sentence clearly explaining the key differences."
Only output the dictionary."""


def _parse_organization_decision(response: str) -> str | None:
    """Extract the ``decision`` field ('A' or 'B') from the LLM JSON response."""
    # Try JSON parse first.
    try:
        # The LLM sometimes wraps in ```json ... ```
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"```\s*$", "", cleaned)
        data = json.loads(cleaned)
        decision = str(data.get("decision", "")).strip().upper()
        if decision in ("A", "B"):
            return decision
    except Exception:
        pass
    # Fallback: look for "decision" key pattern.
    m = re.search(r'"decision"\s*:\s*"([ABab])"', response)
    if m:
        return m.group(1).upper()
    # Last resort: first standalone A or B.
    m = re.search(r"\b([AB])\b", response)
    if m:
        return m.group(1).upper()
    return None


def evaluate_organization(
    parsed_outputs: list[dict],
    paper_rows: list,
    judge: JudgeLLMClient,
) -> list[float]:
    """Pairwise comparison of organization between generated and GT related works.

    Runs 2 trials with permuted order to reduce position bias.
    Score = avg(trial1, trial2) where each trial is 1 if generated wins, else 0.
    """
    scores: list[float] = []
    for parsed, paper_row in zip(parsed_outputs, paper_rows):
        generated_text = parsed["clean_text"]
        gt_text = str(paper_row.get("related_works_section", "")) if hasattr(paper_row, "get") else ""
        title = str(paper_row.get("title", "")) if hasattr(paper_row, "get") else ""
        abstract = str(paper_row.get("abstract", "")) if hasattr(paper_row, "get") else ""

        if not generated_text or not gt_text:
            scores.append(0.0)
            continue

        trial_scores: list[float] = []

        # Trial 1: A=generated, B=gt  → generated wins if decision is A
        prompt_1 = _ORGANIZATION_PROMPT_TEMPLATE.format(
            paper_title=title,
            paper_abstract=abstract,
            related_work_a=generated_text,
            related_work_b=gt_text,
        )
        try:
            resp_1 = _judge_generate_with_retry(judge, prompt_1)
            dec_1 = _parse_organization_decision(resp_1)
            trial_scores.append(1.0 if dec_1 == "A" else 0.0)
        except Exception as e:
            logger.warning("organization trial 1 failed: %s", e)
            trial_scores.append(0.0)

        # Trial 2: A=gt, B=generated → generated wins if decision is B
        prompt_2 = _ORGANIZATION_PROMPT_TEMPLATE.format(
            paper_title=title,
            paper_abstract=abstract,
            related_work_a=gt_text,
            related_work_b=generated_text,
        )
        try:
            resp_2 = _judge_generate_with_retry(judge, prompt_2)
            dec_2 = _parse_organization_decision(resp_2)
            trial_scores.append(1.0 if dec_2 == "B" else 0.0)
        except Exception as e:
            logger.warning("organization trial 2 failed: %s", e)
            trial_scores.append(0.0)

        scores.append(float(np.mean(trial_scores)))

    return scores


_SUPPORT_PROMPT_TEMPLATE = """\
You are an Attribution Validator. Your task is to verify whether a given reference can support the given claim.

Claim: {claim}
Reference: {reference}

Does the reference support the claim? Answer '1' if it supports the claim, or '0' if it does not.
Do not explain your answer, just return '1' or '0'.
Answer:"""


def evaluate_claim_coverage(
    parsed_outputs: list[dict],
    judge: JudgeLLMClient,
) -> list[float]:
    """Citation precision by claim support with sliding window.

    For each sentence with citations (length >= 50 chars), check if cited
    references in a sliding window (current + 1 neighbor) support the claim.
    If any reference supports, the sentence is marked as supported.
    Final score = mean(supported / total) per paper.
    """
    scores: list[float] = []
    for parsed in parsed_outputs:
        clean_text = parsed["clean_text"]
        docs = parsed["docs"]

        if not clean_text:
            scores.append(0.0)
            continue

        sentences = custom_sent_tokenize(clean_text)
        sentence_results: list[float] = []

        for idx, sentence in enumerate(sentences):
            if len(sentence) < 50:
                continue

            cite_matches = re.findall(r"\[(\d+)\]", sentence)
            if not cite_matches:
                continue

            claim = remove_citations(sentence)

            # Collect citation IDs from sliding window (current + 1 neighbor).
            window_cite_ids: set[int] = set()
            for cite_id_str in cite_matches:
                window_cite_ids.add(int(cite_id_str))

            # Check neighbor sentence for additional citation context.
            if idx + 1 < len(sentences):
                neighbor_cites = re.findall(r"\[(\d+)\]", sentences[idx + 1])
                for cid_str in neighbor_cites:
                    window_cite_ids.add(int(cid_str))

            supported = False
            for cid in window_cite_ids:
                if cid not in docs:
                    continue
                doc = docs[cid]
                title = doc.get("title", "")
                abstract = doc.get("sent", "")
                reference_text = f"Title: {title}\n{abstract}"

                prompt = _SUPPORT_PROMPT_TEMPLATE.format(
                    claim=claim,
                    reference=reference_text,
                )
                try:
                    response = _judge_generate_with_retry(judge, prompt)
                    m = re.search(r"[01]", response)
                    if m and m.group(0) == "1":
                        supported = True
                        break
                except Exception as e:
                    logger.warning("claim_coverage LLM call failed: %s", e)

            sentence_results.append(1.0 if supported else 0.0)

        if sentence_results:
            scores.append(float(np.mean(sentence_results)))
        else:
            scores.append(0.0)

    return scores


_RELEVANCE_PROMPT_TEMPLATE = """\
Given the title and abstract of a paper under assessment, the paper's ground-truth related-work section, and the title and abstract of a candidate reference paper, determine the relevance of the candidate reference to the related-work section.

Return a graded relevance score:
'2' – Highly relevant
'1' – Somewhat relevant
'0' – Irrelevant

Paper under assessment:
{paper_title}
{paper_abstract}

Ground-truth related-work section:
{paper_related_work}

Candidate reference paper:
{ref_title}
{ref_abstract}

Return only the score: ##final score: <0, 1, or 2>"""


def evaluate_coverage_relevance_rate(
    parsed_outputs: list[dict],
    paper_rows: list,
    judge: JudgeLLMClient,
) -> list[float]:
    """Reference relevance scoring.

    For each unique retrieved reference, check relevance to the paper.
    Compute mean relevance across all unique docs, normalized to [0, 1] by / 2.
    """
    scores: list[float] = []
    for parsed, paper_row in zip(parsed_outputs, paper_rows):
        docs = parsed["docs"]
        title = str(paper_row.get("title", "")) if hasattr(paper_row, "get") else ""
        abstract = str(paper_row.get("abstract", "")) if hasattr(paper_row, "get") else ""
        gt_related_work = (
            str(paper_row.get("related_works_section", "")) if hasattr(paper_row, "get") else ""
        )

        if not docs:
            scores.append(0.0)
            continue

        relevance_scores: list[float] = []
        seen_titles: set[str] = set()

        for doc in docs.values():
            ref_title = doc.get("title", "")
            ref_abstract = doc.get("sent", "")
            # Deduplicate by normalized title.
            norm_key = _normalize_title(ref_title)
            if norm_key in seen_titles:
                continue
            seen_titles.add(norm_key)

            prompt = _RELEVANCE_PROMPT_TEMPLATE.format(
                paper_title=title,
                paper_abstract=abstract,
                paper_related_work=gt_related_work,
                ref_title=ref_title,
                ref_abstract=ref_abstract,
            )
            try:
                response = _judge_generate_with_retry(judge, prompt)
                # Parse "##final score: N"
                m = re.search(r"##final\s+score:\s*([012])", response)
                if m:
                    relevance_scores.append(int(m.group(1)))
                else:
                    # Fallback: look for standalone 0, 1, or 2
                    m2 = re.search(r"\b([012])\b", response)
                    if m2:
                        relevance_scores.append(int(m2.group(1)))
                    else:
                        relevance_scores.append(0.0)
            except Exception as e:
                logger.warning("coverage_relevance_rate LLM call failed: %s", e)
                relevance_scores.append(0.0)

        if relevance_scores:
            # Normalize to [0, 1] by dividing by max score of 2.
            scores.append(float(np.mean(relevance_scores)) / 2.0)
        else:
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(
    output_dir: str,
    task_name: str,
    per_paper_results: list[dict],
    summary: dict[str, float],
) -> None:
    """Save per-paper CSV and summary text."""
    results_dir = os.path.join(output_dir, "deepscholar_bench", task_name)
    os.makedirs(results_dir, exist_ok=True)

    # Per-paper CSV.
    csv_path = os.path.join(results_dir, "results.csv")
    if per_paper_results:
        keys = list(per_paper_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_paper_results)

    # Summary text.
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for metric, value in summary.items():
            f.write(f"{metric}: {value:.4f}\n")

    logger.info("Results saved to %s", results_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepScholar-Bench evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSONL results file")
    parser.add_argument("--task_name", type=str, required=True, help="Task/model name for output")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="Output directory")
    parser.add_argument("--dataset_dir", type=str, default="deepscholar/dataset", help="Dataset directory")
    parser.add_argument(
        "--llm_backend",
        type=str,
        choices=["openai_compatible", "internal_http", "gemini"],
        default=os.environ.get("DRB_JUDGE_BACKEND", "openai_compatible"),
        help="Judge LLM backend",
    )
    parser.add_argument(
        "--llm_api_base",
        type=str,
        default=os.environ.get("DRB_JUDGE_API_BASE"),
        help="API base URL for OpenAI-compatible backend",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.environ.get("DRB_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        help="API key for judge backend",
    )
    parser.add_argument(
        "--grader_model",
        type=str,
        default=os.environ.get("DRB_JUDGE_MODEL") or os.environ.get("DRB_RACE_MODEL"),
        help="Judge model name",
    )
    parser.add_argument(
        "--evals",
        nargs="+",
        default=[
            "reference_coverage",
            "cite_p",
            "nugget_coverage",
            "organization",
            "claim_coverage",
            "coverage_relevance_rate",
        ],
        help="Metrics to compute",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build judge client
    # ------------------------------------------------------------------
    judge = JudgeLLMClient(
        backend=args.llm_backend,
        api_key=args.llm_api_key,
        api_base=args.llm_api_base,
        race_model=args.grader_model,
    )
    logger.info("Judge client: backend=%s, model=%s", judge.backend, judge.grader_model)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading results from %s", args.input_file)
    results = load_results(args.input_file)
    logger.info("Loaded %d results", len(results))

    logger.info("Loading dataset from %s", args.dataset_dir)
    dataset = load_dataset(args.dataset_dir)
    logger.info("Dataset has %d papers", len(dataset))

    # ------------------------------------------------------------------
    # Map results to papers & parse model outputs
    # ------------------------------------------------------------------
    parsed_outputs: list[dict] = []
    paper_rows: list = []
    orig_ids: list[str] = []

    for result in results:
        orig_id = str(result.get("original_data", {}).get("orig_id", ""))
        orig_ids.append(orig_id)

        try:
            row_idx = int(orig_id)
            paper_row = dataset.iloc[row_idx]
        except (ValueError, IndexError):
            paper_row = pd.Series()
        paper_rows.append(paper_row)

        parsed_outputs.append(parse_model_output(result))

    logger.info("Parsed %d model outputs", len(parsed_outputs))

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    summary: dict[str, float] = {}
    per_paper_results: list[dict] = [{"orig_id": oid} for oid in orig_ids]

    if "reference_coverage" in args.evals:
        logger.info("Evaluating reference_coverage...")
        important_citations = load_important_citations(args.dataset_dir)
        ref_cov_scores = evaluate_reference_coverage(parsed_outputs, paper_rows, important_citations)
        for i, score in enumerate(ref_cov_scores):
            per_paper_results[i]["reference_coverage"] = score
        summary["reference_coverage"] = float(np.mean(ref_cov_scores)) if ref_cov_scores else 0.0
        logger.info("reference_coverage: %.4f", summary["reference_coverage"])

    if "cite_p" in args.evals:
        logger.info("Evaluating cite_p...")
        cite_p_scores = evaluate_cite_p(parsed_outputs, judge)
        for i, score in enumerate(cite_p_scores):
            per_paper_results[i]["cite_p"] = score
        summary["cite_p"] = float(np.mean(cite_p_scores)) if cite_p_scores else 0.0
        logger.info("cite_p: %.4f", summary["cite_p"])

    if "nugget_coverage" in args.evals:
        logger.info("Evaluating nugget_coverage...")
        nugget_scores = evaluate_nugget_coverage(parsed_outputs, orig_ids, args.dataset_dir, judge)
        for i, score in enumerate(nugget_scores):
            per_paper_results[i]["nugget_coverage"] = score
        summary["nugget_coverage"] = float(np.mean(nugget_scores)) if nugget_scores else 0.0
        logger.info("nugget_coverage: %.4f", summary["nugget_coverage"])

    if "organization" in args.evals:
        logger.info("Evaluating organization...")
        org_scores = evaluate_organization(parsed_outputs, paper_rows, judge)
        for i, score in enumerate(org_scores):
            per_paper_results[i]["organization"] = score
        summary["organization"] = float(np.mean(org_scores)) if org_scores else 0.0
        logger.info("organization: %.4f", summary["organization"])

    if "claim_coverage" in args.evals:
        logger.info("Evaluating claim_coverage...")
        claim_scores = evaluate_claim_coverage(parsed_outputs, judge)
        for i, score in enumerate(claim_scores):
            per_paper_results[i]["claim_coverage"] = score
        summary["claim_coverage"] = float(np.mean(claim_scores)) if claim_scores else 0.0
        logger.info("claim_coverage: %.4f", summary["claim_coverage"])

    if "coverage_relevance_rate" in args.evals:
        logger.info("Evaluating coverage_relevance_rate...")
        rel_scores = evaluate_coverage_relevance_rate(parsed_outputs, paper_rows, judge)
        for i, score in enumerate(rel_scores):
            per_paper_results[i]["coverage_relevance_rate"] = score
        summary["coverage_relevance_rate"] = float(np.mean(rel_scores)) if rel_scores else 0.0
        logger.info("coverage_relevance_rate: %.4f", summary["coverage_relevance_rate"])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_results(args.output_dir, args.task_name, per_paper_results, summary)

    logger.info("=== DeepScholar-Bench Evaluation Summary ===")
    for metric, value in summary.items():
        logger.info("%s: %.4f", metric, value)


if __name__ == "__main__":
    main()
