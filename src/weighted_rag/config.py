"""Configuration dataclasses for the WeightedRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, TypeVar, get_args, get_origin


@dataclass  
class ChunkingConfig:

    max_tokens: int = 200
    overlap_tokens: int = 20
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    enabled_chunk_types: List[str] = field(default_factory=lambda: [
        "full_table", "table_row"
    ])
    chunk_type_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexStageConfig:
    name: str
    dimension: int
    top_k: int
    weight: float
    normalize: bool = True
    index_factory: str = "HNSW32"
    ef_search: int = 64
    ef_construction: int = 200
    quantize: Optional[str] = None
    model_name: Optional[str] = None  # Allow different models per stage


@dataclass
class RetrievalConfig:
    stages: List[IndexStageConfig] = field(
        default_factory=lambda: [
            IndexStageConfig(
                name="coarse", 
                dimension=384, 
                top_k=200, 
                weight=1.0, 
                index_factory="HNSW32",
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        ]
    )
    lambda_similarity: float = 0.70  # Increased primary weight
    alpha_reliability: float = 0.10  # Increased metadata weight
    beta_temporal: float = 0.00
    gamma_domain: float = 0.00
    delta_structure: float = 0.05
    epsilon_bm25: float = 0.15  # Reduced BM25 weight to prevent dominance


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/embedding-matryoshka"
    device: str = "cuda"
    batch_size: int = 64
    normalize: bool = True
    truncate_dims: Optional[List[int]] = None
    use_fp16: bool = True


@dataclass
class GenerationConfig:
    model_name: str = "meta-llama/Llama-2-13b-chat-hf"
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 512
    system_prompt: str = (
        "You are a grounded assistant. Use only the provided context.\n"
        "Cite evidence by chunk id when possible."
    )


@dataclass
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    storage_dir: str = "storage"
    run_name: str = "default"
    use_graph_rerank: bool = False
    cross_encoder: Optional["CrossEncoderConfig"] = None


@dataclass
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    device: str = "cpu"
    batch_size: int = 16
    top_n: int = 5


T = TypeVar("T")


def _build_dataclass(cls: Type[T], payload: Dict[str, Any]) -> T:
    params: Dict[str, Any] = {}
    for entry in fields(cls):
        value = payload.get(entry.name)
        if value is None:
            continue
        field_type = entry.type
        origin = get_origin(field_type)
        if origin in (list, List) and isinstance(value, list):
            args = get_args(field_type)
            if args and hasattr(args[0], "__dataclass_fields__"):
                params[entry.name] = [_build_dataclass(args[0], item) for item in value]
            else:
                params[entry.name] = value
        else:
            target_cls = None
            if hasattr(field_type, "__dataclass_fields__"):
                target_cls = field_type
            else:
                for arg in get_args(field_type):
                    if hasattr(arg, "__dataclass_fields__"):
                        target_cls = arg
                        break
            if isinstance(value, dict) and target_cls is not None:
                params[entry.name] = _build_dataclass(target_cls, value)
            else:
                params[entry.name] = value
    return cls(**params)  # type: ignore[arg-type]


def pipeline_config_from_dict(payload: Dict[str, Any]) -> PipelineConfig:
    return _build_dataclass(PipelineConfig, payload)
