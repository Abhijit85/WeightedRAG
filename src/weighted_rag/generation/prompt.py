"""Prompt construction utilities for the generation stage."""

from __future__ import annotations

from typing import Iterable, List

from ..types import RetrievedChunk


class PromptBuilder:
    """Formats context and query into an instruction-following prompt."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(self, question: str, chunks: Iterable[RetrievedChunk]) -> str:
        context_lines: List[str] = []
        for item in chunks:
            context_lines.append(f"[{item.chunk.chunk_id}] {item.chunk.text.strip()}")
        context_str = "\n\n".join(context_lines)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context_str or 'No relevant context retrieved.'}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return prompt
