"""LLM generation component with context-grounding prompt."""

from __future__ import annotations

import os
from typing import List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover
    requests = None

from ..config import GenerationConfig
from ..types import GenerationResult, RetrievalResult
from .prompt import PromptBuilder


class LLMGenerator:
    """Wraps a causal language model for answer generation."""

    def __init__(self, config: GenerationConfig):
        env_model = os.getenv("model_name")
        if env_model:
            config.model_name = env_model
        self.config = config
        self.prompt_builder = PromptBuilder(system_prompt=config.system_prompt)

        self._pipeline = None
        self._backend = "hf"
        self._openrouter_session: Optional["requests.Session"] = None
        self._openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
        self._openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "https://openrouter.ai/api/v1")
        self._openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME", "WeightedRAG")

        if self._openrouter_api_key and requests is not None:
            self._backend = "openrouter"
            self._openrouter_session = requests.Session()
        elif self._openrouter_api_key and requests is None:
            print("⚠️ OpenRouter API key provided, but the 'requests' library is missing. Falling back to transformers.")

        if self._backend == "hf" and pipeline is not None and AutoTokenizer is not None:
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

    def update_system_prompt(self, prompt: str) -> None:
        self.prompt_builder.set_system_prompt(prompt)

    def generate(self, retrieval: RetrievalResult, system_prompt: Optional[str] = None) -> GenerationResult:
        effective_system_prompt = system_prompt or self.prompt_builder.system_prompt
        prompt = self.prompt_builder.build(retrieval.query.text, retrieval.chunks, system_prompt=system_prompt)

        if self._backend == "openrouter":
            answer = self._generate_openrouter(prompt, effective_system_prompt)
            if not answer:
                answer = self._fallback_answer(retrieval)
        elif self._pipeline is None:
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

    def _generate_openrouter(self, prompt: str, system_prompt: str) -> str:
        if not self._openrouter_session:
            return ""
        url = f"{self._openrouter_site_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "HTTP-Referer": self._openrouter_site_url,
            "X-Title": self._openrouter_site_name,
        }
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        try:
            response = self._openrouter_session.post(url, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message", {}).get("content", "")
            return (message or "").strip()
        except Exception as exc:
            print(f"⚠️ OpenRouter request failed: {exc}")
            return ""

    def _fallback_answer(self, retrieval: RetrievalResult) -> str:
        lines = [
            "No generation model available. Here is the retrieved context summary:",
        ]
        for item in retrieval.chunks:
            lines.append(f"- [{item.chunk.chunk_id}] {item.chunk.text[:150]}...")
        return "\n".join(lines)
