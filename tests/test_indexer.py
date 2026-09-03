import json
import tempfile
from pathlib import Path

from rag_system.indexer import Indexer
from rag_system.chunker import Chunk


class TestIndexer:
    @staticmethod
    def _make_chunks(n: int = 5) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"insurance_test_{i}",
                doc_id="test_doc",
                domain="insurance",
                text=f"第{i}条 保险公司应按照规定{i}缴纳保证金。",
                section_path=f"第一章 > 第{i}条",
                prev_chunk_id=f"insurance_test_{i - 1}" if i > 0 else None,
                next_chunk_id=f"insurance_test_{i + 1}" if i < n - 1 else None,
            )
            for i in range(n)
        ]

    def test_build_bm25_index(self):
        cfg = type("cfg", (), {"index_dir": Path(tempfile.mkdtemp())})()
        indexer = Indexer(cfg)
        chunks = self._make_chunks()
        indexer.build_bm25("insurance", chunks)
        bm25_path = cfg.index_dir / "insurance" / "bm25.pkl"
        assert bm25_path.exists()

    def test_build_embedding_index(self):
        cfg = type("cfg", (), {"index_dir": Path(tempfile.mkdtemp()),
                               "embedding_model": "BAAI/bge-large-zh-v1.5"})()
        indexer = Indexer(cfg)
        chunks = self._make_chunks()
        indexer.build_embedding("insurance", chunks)
        emb_dir = cfg.index_dir / "insurance" / "embedding"
        assert emb_dir.exists()
        assert (emb_dir / "done.semaphore").exists()

    def test_build_doc_profiles(self):
        cfg = type("cfg", (), {"index_dir": Path(tempfile.mkdtemp())})()
        indexer = Indexer(cfg)
        chunks = self._make_chunks()
        indexer.build_doc_profiles({"insurance": chunks})
        profiles_path = cfg.index_dir / "doc_profiles.json"
        assert profiles_path.exists()
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        assert "insurance" in profiles
        assert "test_doc" in profiles["insurance"]
        profile = profiles["insurance"]["test_doc"]
        assert "title" in profile
        assert "total_chunks" in profile
        assert profile["total_chunks"] == len(chunks)
        assert "preview" in profile
        assert "keywords" in profile
        assert len(profile["keywords"]) <= 5

    def test_index_skips_if_done(self):
        cfg = type("cfg", (), {"index_dir": Path(tempfile.mkdtemp()),
                               "embedding_model": "BAAI/bge-large-zh-v1.5"})()
        indexer = Indexer(cfg)
        chunks = self._make_chunks()
        indexer.build_embedding("insurance", chunks)
        modified_time = (cfg.index_dir / "insurance" / "embedding" / "done.semaphore").stat().st_mtime
        indexer.build_embedding("insurance", chunks)
        new_time = (cfg.index_dir / "insurance" / "embedding" / "done.semaphore").stat().st_mtime
        assert modified_time == new_time
