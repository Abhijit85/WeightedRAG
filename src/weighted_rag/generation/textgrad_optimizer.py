"""TextGrad-inspired prompt optimization utilities."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from ..types import Query

if TYPE_CHECKING:  # pragma: no cover
    from ..pipeline import WeightedRAGPipeline


@dataclass
class TextGradExample:
    """Single supervision example for prompt optimization."""

    question: str
    reference_answer: str
    query_id: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TextGradSettings:
    """Controls the optimization procedure."""

    max_steps: int = 3
    min_gain: float = 0.01
    sample_size: Optional[int] = None
    mutations_per_step: int = 4
    candidate_instructions: Sequence[str] = field(
        default_factory=lambda: [
            "Respond with concise bullet points and cite chunk ids inline.",
            "Lead with a single-sentence executive summary before details.",
            "Highlight any location-specific policies explicitly.",
            "Include a short 'Recommended Actions' section at the end.",
            "Surface confidentiality levels and departments alongside each fact.",
            "Prefer quoting the original text verbatim when referencing procedures.",
        ]
    )


def load_textgrad_examples(path: Path) -> List[TextGradExample]:
    """Loads examples from a JSON or JSONL file."""

    if path.suffix.lower() == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        payloads = [json.loads(line) for line in lines if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            payloads = data.get("examples", [])
        elif isinstance(data, list):
            payloads = data
        else:
            raise ValueError("Unsupported JSON structure for TextGrad examples.")

    examples: List[TextGradExample] = []
    for idx, payload in enumerate(payloads):
        question = payload.get("question")
        reference = payload.get("reference_answer") or payload.get("ideal_answer") or payload.get("answer")
        if not question or not reference:
            continue
        query_id = payload.get("query_id") or f"textgrad-{idx:04d}"
        metadata = payload.get("metadata") or {}
        examples.append(TextGradExample(question=question, reference_answer=reference, query_id=query_id, metadata=metadata))
    return examples


class TextGradPromptOptimizer:
    """Lightweight textual gradient search for prompt improvements."""

    def __init__(self, pipeline: "WeightedRAGPipeline", settings: TextGradSettings):
        self.pipeline = pipeline
        self.settings = settings

    def optimize(self, examples: Sequence[TextGradExample], base_prompt: str) -> Tuple[str, List[Dict[str, object]]]:
        if not examples:
            raise ValueError("No TextGrad examples provided.")

        prompt = base_prompt.strip()
        history: List[Dict[str, object]] = []
        current_score = self._score_prompt(prompt, examples)
        history.append({"step": 0, "score": current_score, "prompt": prompt})

        for step in range(1, self.settings.max_steps + 1):
            best_score = current_score
            best_prompt = prompt

            for candidate in self._generate_candidates(prompt):
                score = self._score_prompt(candidate, examples)
                if score > best_score + self.settings.min_gain:
                    best_score = score
                    best_prompt = candidate

            if best_prompt == prompt:
                break

            prompt = best_prompt
            current_score = best_score
            history.append({"step": step, "score": current_score, "prompt": prompt})

        return prompt, history

    def _generate_candidates(self, prompt: str) -> Iterable[str]:
        unseen = [instr for instr in self.settings.candidate_instructions if instr.lower() not in prompt.lower()]
        random.shuffle(unseen)
        count = min(len(unseen), self.settings.mutations_per_step)
        for idx in range(count):
            addition = unseen[idx].strip()
            if not addition:
                continue
            yield f"{prompt.strip()}\n\n{addition}"

    def _score_prompt(self, prompt: str, examples: Sequence[TextGradExample]) -> float:
        subset = list(examples)
        if self.settings.sample_size and self.settings.sample_size < len(subset):
            subset = random.sample(subset, self.settings.sample_size)

        total = 0.0
        for example in subset:
            query = Query(query_id=example.query_id, text=example.question, metadata=example.metadata)
            retrieval = self.pipeline.retrieve(query)
            result = self.pipeline.generator.generate(retrieval, system_prompt=prompt)
            total += self._f1_score(example.reference_answer, result.answer)
        return total / max(len(subset), 1)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _f1_score(self, reference: str, candidate: str) -> float:
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        if not ref_tokens or not cand_tokens:
            return 0.0
        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)
        overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
