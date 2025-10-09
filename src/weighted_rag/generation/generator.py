"""LLM generation component with context-grounding prompt."""

from __future__ import annotations

from typing import List

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

from ..config import GenerationConfig
from ..types import GenerationResult, RetrievalResult
from .prompt import PromptBuilder


class LLMGenerator:
    """Wraps a causal language model for answer generation."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.prompt_builder = PromptBuilder(system_prompt=config.system_prompt)
        self._pipeline = None
        if pipeline is not None and AutoTokenizer is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(config.model_name)
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                )
            except Exception:
                self._pipeline = None

    def generate(self, retrieval: RetrievalResult) -> GenerationResult:
        prompt = self.prompt_builder.build(retrieval.query.text, retrieval.chunks)
        if self._pipeline is None:
            answer = self._fallback_answer(retrieval)
        else:
            output = self._pipeline(
                prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
            )
            answer = output[0]["generated_text"][len(prompt) :].strip()
        references = [item.chunk.chunk_id for item in retrieval.chunks]
        return GenerationResult(
            query=retrieval.query,
            answer=answer,
            references=references,
            metadata={"retrieved_chunks": float(len(retrieval.chunks))},
        )

    def _fallback_answer(self, retrieval: RetrievalResult) -> str:
        lines = [
            "No generation model available. Here is the retrieved context summary:",
        ]
        for item in retrieval.chunks:
            lines.append(f"- [{item.chunk.chunk_id}] {item.chunk.text[:150]}...")
        return "\n".join(lines)
