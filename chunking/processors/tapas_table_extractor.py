"""TAPAS-backed table extractor for structured representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from transformers import TapasTokenizer  # type: ignore
except Exception:  # pragma: no cover
    TapasTokenizer = None


@dataclass
class TAPASTable:
    headers: List[str]
    rows: List[List[str]]
    linearized: str
    model_name: str


class TAPASTableExtractor:
    """Normalizes HTML tables using the TAPAS tokenizer for consistent cell text."""

    def __init__(self, model_name: str = "google/tapas-large-finetuned-wtq"):
        if TapasTokenizer is None:
            raise ImportError("transformers.TapasTokenizer is required for TAPAS extraction")
        self.model_name = model_name
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)

    def extract(self, html_table: str) -> Optional[TAPASTable]:
        try:
            dataframes = pd.read_html(html_table)
        except ValueError:
            return None
        except Exception:
            return None

        if not dataframes:
            return None

        dataframe = dataframes[0].fillna("")
        headers = [self._normalize_cell(col) for col in dataframe.columns]
        rows = [
            [self._normalize_cell(cell) for cell in row]
            for row in dataframe.itertuples(index=False, name=None)
        ]

        linearized = self._linearize(headers, rows)
        return TAPASTable(headers=headers, rows=rows, linearized=linearized, model_name=self.model_name)

    def _normalize_cell(self, value: Any) -> str:
        tokens = self.tokenizer.tokenize(str(value))
        if not tokens:
            return ""
        return self.tokenizer.convert_tokens_to_string(tokens).strip()

    def _linearize(self, headers: List[str], rows: List[List[str]]) -> str:
        parts: List[str] = []
        header_line = " | ".join(header for header in headers if header)
        if header_line:
            parts.append(header_line)
        for row in rows:
            row_line = " | ".join(cell for cell in row if cell)
            if row_line:
                parts.append(row_line)
        return " || ".join(parts)
