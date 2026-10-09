import json
import tempfile
from pathlib import Path

from rag_system.retriever import Retriever
from rag_system.chunker import Chunk


class TestRetriever:
    @staticmethod
    def _build_test_index(tmpdir: str) -> tuple[Retriever, list[Chunk]]:
        index_dir = Path(tmpdir) / "indexes"
        chunks_dir = Path(tmpdir) / "chunks"
        index_dir.mkdir(parents=True)
        chunks_dir.mkdir(parents=True)
        (index_dir / "insurance").mkdir(parents=True)

        chunks = [
            Chunk(chunk_id=f"insurance_test_{i}", doc_id="test_doc", domain="insurance",
                  text=text, section_path=f"第一章 > 第{i}条",
                  prev_chunk_id=f"insurance_test_{i-1}" if i > 0 else None,
                  next_chunk_id=f"insurance_test_{i+1}" if i < 4 else None)
            for i, text in enumerate([
                "保险公司应当缴纳保证金。保证金的数额为注册资本总额的百分之二十。",
                "保险合同的变更应当由投保人和保险人协商一致。变更后的合同效力自变更之日起计算。",
                "保险标的转让的，保险标的的受让人承继被保险人的权利和义务。",
                "被保险人死亡后，保险金作为被保险人的遗产，由保险人依照继承法的规定履行给付保险金的义务。",
                "以死亡为给付保险金条件的合同，未经被保险人书面同意并认可保险金额的，合同无效。",
            ])
        ]

        chunks_path = chunks_dir / "insurance_chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "domain": c.domain,
                    "text": c.text, "section_path": c.section_path,
                    "prev_chunk_id": c.prev_chunk_id, "next_chunk_id": c.next_chunk_id,
                }, ensure_ascii=False) + "\n")

        from rag_system.indexer import Indexer
        fake_cfg = type("cfg", (), {
            "index_dir": index_dir,
            "chunks_dir": chunks_dir,
            "embedding_model": "BAAI/bge-large-zh-v1.5",
        })()
        indexer = Indexer(fake_cfg)
        indexer.build_bm25("insurance", chunks)

        doc_profiles = {
            "insurance": {
                "test_doc": {
                    "title": "保险法",
                    "total_chunks": 5,
                    "total_chars": 200,
                    "preview": "保险公司应当缴纳保证金。",
                    "keywords": ["保险", "保证金", "合同", "被保险人", "保险金"],
                }
            }
        }
        profiles_path = index_dir / "doc_profiles.json"
        profiles_path.write_text(json.dumps(doc_profiles, ensure_ascii=False, indent=2), encoding="utf-8")

        retriever_cfg = type("cfg", (), {
            "index_dir": index_dir,
            "chunks_dir": chunks_dir,
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "reranker_model": "BAAI/bge-reranker-large",
            "top_n_bm25": 20,
            "top_n_embedding": 20,
            "top_n_rerank": 5,
            "rrf_k": 60,
            "retrieval_workers": 2,
            # 与 RAGConfig 保持同步的检索参数（夹具曾缺以下字段导致 12 例失败）
            "bm25_score_threshold": 0.0,
            "embedding_score_threshold": 0.0,
            "min_chunk_length": 10,
            "min_chunks_per_doc": 0,
            "max_chunks_per_doc": 8,
            "max_chunks_easy": 5,
            "max_chunks_medium": 10,
            "max_chunks_hard": 15,
            "context_expand_neighbors": 1,
            "skip_embedding": True,
            "cloud_embedding": False,
        })()
        retriever = Retriever(retriever_cfg)
        return retriever, chunks

    def test_domain_routing(self):
        retriever, _ = self._build_test_index(tempfile.mkdtemp())
        domain = retriever._route_domain("insurance", "保险公司保证金比例是多少")
        assert domain == "insurance"

    def test_bm25_search(self):
        retriever, _ = self._build_test_index(tempfile.mkdtemp())
        results = retriever._bm25_search("insurance", "保险保证金", 20)
        assert len(results) > 0
        assert any("保证金" in r["text"] for r in results)

    def test_hybrid_search_returns_results(self):
        retriever, _ = self._build_test_index(tempfile.mkdtemp())
        question = {
            "qid": "test_001", "domain": "insurance",
            "question": "保险保证金的数额比例是多少",
            "options": {"A": "10%", "B": "20%", "C": "30%", "D": "50%"},
            "doc_ids": ["test_doc"],
        }
        results = retriever.search(question)
        assert results is not None
        assert 1 <= len(results) <= 5

    def test_fallback_when_few_results(self):
        retriever, chunks = self._build_test_index(tempfile.mkdtemp())
        question = {
            "qid": "test_002", "domain": "insurance",
            "question": "非常罕见的问题关键词abcdefg",
            "options": {"A": "x", "B": "y"},
            "doc_ids": ["test_doc"],
        }
        results = retriever.search(question)
        assert results is not None
        assert len(results) <= 5

    def test_entity_extraction(self):
        retriever, _ = self._build_test_index(tempfile.mkdtemp())
        entities = retriever._extract_entities("某公司2019年营收8亿元，资产负债率75.5%，根据第四十七条规定")
        assert len(entities) > 0
        assert "8亿元" in entities or "8亿" in "".join(entities)
