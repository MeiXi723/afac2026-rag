import os
import warnings
from pathlib import Path

import pytest

from rag_system.config import RAGConfig


class TestRAGConfig:
    def test_default_values(self):
        cfg = RAGConfig()
        assert cfg.embedding_model == "BAAI/bge-large-zh-v1.5"
        assert cfg.reranker_model == "BAAI/bge-reranker-large"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 128
        assert cfg.rrf_k == 60
        assert cfg.top_n_bm25 == 20
        assert cfg.top_n_rerank == 5
        assert cfg.generator_temperature == 0.1
        assert cfg.generator_max_tokens == 2048
        assert cfg.request_timeout == 120
        assert cfg.max_retries == 3
        assert cfg.context_expand_neighbors == 1
        assert cfg.parser_workers == 2
        assert cfg.retrieval_workers == 4
        assert cfg.api_workers == 2

    def test_api_keys_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-ds")
        monkeypatch.setenv("QWEN_API_KEY", "sk-test-qw")
        cfg = RAGConfig()
        assert cfg.api_key_deepseek == "sk-test-ds"
        assert cfg.api_key_qwen == "sk-test-qw"

    def test_base_urls(self):
        cfg = RAGConfig()
        assert "deepseek.com" in cfg.api_base_deepseek
        assert "aliyuncs.com" in cfg.api_base_qwen

    def test_paths_are_absolute(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "test")
        cfg = RAGConfig()
        assert isinstance(cfg.pdf_dir, Path)
        assert isinstance(cfg.output_dir, Path)
        assert cfg.chunks_dir.name == "chunks"
        assert cfg.index_dir.name == "indexes"


class TestRAGConfigValidation:
    def test_valid_config_passes_post_init(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        cfg = RAGConfig()
        assert cfg.chunk_size == 512

    def test_chunk_overlap_not_less_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RAGConfig(chunk_size=128, chunk_overlap=128)

    def test_chunk_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RAGConfig(chunk_size=128, chunk_overlap=200)

    def test_max_retries_zero_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            RAGConfig(max_retries=0)

    def test_max_retries_negative_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            RAGConfig(max_retries=-1)

    def test_top_n_rerank_zero_raises(self):
        with pytest.raises(ValueError, match="top_n_rerank"):
            RAGConfig(top_n_rerank=0)

    def test_top_n_rerank_negative_raises(self):
        with pytest.raises(ValueError, match="top_n_rerank"):
            RAGConfig(top_n_rerank=-1)

    def test_request_timeout_zero_raises(self):
        with pytest.raises(ValueError, match="request_timeout"):
            RAGConfig(request_timeout=0)

    def test_request_timeout_negative_raises(self):
        with pytest.raises(ValueError, match="request_timeout"):
            RAGConfig(request_timeout=-1)

    def test_no_api_keys_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RAGConfig(api_key_deepseek="", api_key_qwen="")
            assert len(w) == 1
            assert "No API keys configured" in str(w[0].message)

    def test_deepseek_key_only_no_warning(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RAGConfig()
            api_key_warnings = [
                x for x in w
                if "No API keys configured" in str(x.message)
            ]
            assert len(api_key_warnings) == 0

    def test_qwen_key_only_no_warning(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "sk-test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RAGConfig()
            api_key_warnings = [
                x for x in w
                if "No API keys configured" in str(x.message)
            ]
            assert len(api_key_warnings) == 0
