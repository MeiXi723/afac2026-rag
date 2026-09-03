from rag_system.chunker import Chunker, Chunk


class TestChunker:
    @staticmethod
    def _make_records() -> list[dict]:
        return [
            {"doc_id": "test_doc", "text": "# 第一章 总则\n第一条 本法适用于", "block_type": "title"},
            {"doc_id": "test_doc", "text": "金融合同纠纷的处理。\n当事人应当遵守法律法规。", "block_type": "paragraph"},
            {"doc_id": "test_doc", "text": "| 项目 | 金额 |\n|------|------|\n| 收入 | 100 |", "block_type": "table"},
            {"doc_id": "test_doc", "text": "A token " * 200, "block_type": "paragraph"},
        ]

    def test_chunk_count(self):
        cfg = type("cfg", (), {"chunk_size": 512, "chunk_overlap": 128})()
        chunker = Chunker(cfg)
        chunks = chunker.chunk_document("insurance", self._make_records())
        assert len(chunks) >= 3

    def test_chunk_structure(self):
        cfg = type("cfg", (), {"chunk_size": 512, "chunk_overlap": 128})()
        chunker = Chunker(cfg)
        chunks = chunker.chunk_document("insurance", self._make_records())
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id
            assert chunk.doc_id == "test_doc"
            assert chunk.text != ""
            assert chunk.domain == "insurance"

    def test_table_preserved(self):
        cfg = type("cfg", (), {"chunk_size": 512, "chunk_overlap": 128})()
        chunker = Chunker(cfg)
        chunks = chunker.chunk_document("insurance", self._make_records())
        table_chunks = [c for c in chunks if "| 项目 |" in c.text]
        assert len(table_chunks) >= 1

    def test_prev_next_linking(self):
        cfg = type("cfg", (), {"chunk_size": 512, "chunk_overlap": 128})()
        chunker = Chunker(cfg)
        chunks = chunker.chunk_document("insurance", self._make_records())
        for i, chunk in enumerate(chunks):
            if i == 0:
                assert chunk.prev_chunk_id is None
            else:
                assert chunk.prev_chunk_id == chunks[i - 1].chunk_id
            if i == len(chunks) - 1:
                assert chunk.next_chunk_id is None
            else:
                assert chunk.next_chunk_id == chunks[i + 1].chunk_id

    def test_section_path_included(self):
        cfg = type("cfg", (), {"chunk_size": 512, "chunk_overlap": 128})()
        chunker = Chunker(cfg)
        chunks = chunker.chunk_document("insurance", self._make_records())
        path_chunks = [c for c in chunks if "第一章" in c.section_path or "总则" in c.section_path]
        assert len(path_chunks) >= 1
