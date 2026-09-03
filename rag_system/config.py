import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    api_key_deepseek: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    api_key_qwen: str = field(
        default_factory=lambda: os.getenv("QWEN_API_KEY", "")
    )
    api_base_deepseek: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    )
    api_base_qwen: str = field(
        default_factory=lambda: os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )

    pdf_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    chunks_dir: Path = Path("data/chunks")
    index_dir: Path = Path("data/indexes")
    questions_dir: Path = Path("data/questions")
    output_dir: Path = Path("output")

    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    deepseek_model: str = "deepseek-chat"
    qwen_model: str = "qwen-plus"

    generator_temperature: float = 0.1
    generator_max_tokens: int = 2048
    request_timeout: int = 120

    chunk_size: int = 512
    chunk_overlap: int = 128

    rrf_k: int = 60
    top_n_bm25: int = 20
    top_n_embedding: int = 20
    top_n_rerank: int = 5
    context_expand_neighbors: int = 1

    min_chunk_length: int = 50
    min_chunks_per_doc: int = 3
    max_chunks_per_doc: int = 8
    max_chunks_easy: int = 5
    max_chunks_medium: int = 10
    max_chunks_hard: int = 15
    top_expand: int = 2
    context_expand_enabled: bool = True
    bm25_score_threshold: float = 0.1
    embedding_score_threshold: float = 0.3

    max_retries: int = 3
    retry_base_delay: float = 1.0

    parser_backend: str = "pymupdf"
    skip_embedding: bool = False
    cloud_embedding: bool = True
    cloud_embedding_model: str = "text-embedding-v3"
    cloud_embedding_dim: int = 1024
    cloud_embedding_batch_size: int = 10
    parser_workers: int = 2
    retrieval_workers: int = 4
    api_workers: int = 2

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        if self.max_retries <= 0:
            raise ValueError(
                f"max_retries ({self.max_retries}) must be > 0"
            )
        if self.top_n_rerank <= 0:
            raise ValueError(
                f"top_n_rerank ({self.top_n_rerank}) must be > 0"
            )
        if self.request_timeout <= 0:
            raise ValueError(
                f"request_timeout ({self.request_timeout}) must be > 0"
            )

        if not self.api_key_deepseek and not self.api_key_qwen:
            warnings.warn(
                "No API keys configured. Set DEEPSEEK_API_KEY or "
                "QWEN_API_KEY in environment variables."
            )
