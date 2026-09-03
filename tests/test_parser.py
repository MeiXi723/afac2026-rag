import json
import os
import tempfile
from pathlib import Path

import pytest

from rag_system.parser import PDFParser
from rag_system.config import RAGConfig


class TestPDFParser:
    @staticmethod
    def _get_test_pdf() -> Path:
        pdf = Path("data/raw/insurance/7.pdf")
        return pdf

    def test_parse_single_pdf_creates_jsonl(self):
        cfg = RAGConfig()
        cfg.parsed_dir = Path(tempfile.mkdtemp())
        parser = PDFParser(cfg)
        pdf_path = self._get_test_pdf()
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")
        results = list(parser.parse_all([pdf_path]))
        output_path = cfg.parsed_dir / "insurance" / "7.jsonl"
        if not output_path.exists():
            pytest.skip("MinerU may not be installed")
        assert len(results) > 0
        assert output_path.exists()
        done_path = cfg.parsed_dir / "insurance" / "7.done"
        assert done_path.exists()
        with open(output_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) > 0
        first = json.loads(lines[0])
        assert "text" in first
        assert "block_type" in first
        assert "page_num" in first

    def test_output_structure(self):
        cfg = RAGConfig()
        cfg.parsed_dir = Path(tempfile.mkdtemp())
        parser = PDFParser(cfg)
        pdf_path = self._get_test_pdf()
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")
        results = list(parser.parse_all([pdf_path]))
        if not results:
            pytest.skip("MinerU may not be installed")
        record = results[0]
        assert isinstance(record, dict)
        for key in ["text", "block_type", "page_num", "doc_id"]:
            assert key in record, f"Missing key: {key}"
        assert record["block_type"] in ("paragraph", "table", "title")

    def test_skip_already_parsed(self):
        cfg = RAGConfig()
        cfg.parsed_dir = Path(tempfile.mkdtemp())
        parser = PDFParser(cfg)
        pdf_path = self._get_test_pdf()
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")
        results1 = list(parser.parse_all([pdf_path]))
        if not results1:
            pytest.skip("MinerU may not be installed")
        results2 = list(parser.parse_all([pdf_path]))
        assert len(results2) == len(results1)

    def test_handle_parse_failure(self):
        cfg = RAGConfig()
        cfg.parsed_dir = Path(tempfile.mkdtemp())
        parser = PDFParser(cfg)
        bad_pdf = cfg.parsed_dir / "insurance" / "not_a_real.pdf"
        bad_pdf.parent.mkdir(parents=True, exist_ok=True)
        bad_pdf.write_text("not a pdf")
        results = list(parser.parse_all([bad_pdf]))
        assert len(results) == 0
        error_log = cfg.parsed_dir / "error_log.jsonl"
        assert error_log.exists()


class TestSplitBlocks:
    def test_split_paragraphs(self):
        parser = PDFParser(RAGConfig())
        content = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        blocks = parser._split_blocks(content)
        assert len(blocks) == 3
        assert all(b["block_type"] == "paragraph" for b in blocks)

    def test_split_headings(self):
        parser = PDFParser(RAGConfig())
        content = "# 第一章\n\n这是正文内容。\n\n## 第一节\n\n更多正文。"
        blocks = parser._split_blocks(content)
        assert blocks[0]["block_type"] == "title"
        assert blocks[0]["text"] == "第一章"
        assert blocks[1]["block_type"] == "paragraph"
        assert blocks[2]["block_type"] == "title"
        assert blocks[2]["text"] == "第一节"

    def test_split_table(self):
        parser = PDFParser(RAGConfig())
        content = "前面一段文字。\n\n| 项目 | 金额 |\n|------|------|\n| 收入 | 100 |\n\n后面一段文字。"
        blocks = parser._split_blocks(content)
        assert blocks[0]["block_type"] == "paragraph"
        assert blocks[1]["block_type"] == "table"
        assert "| 项目 | 金额 |" in blocks[1]["text"]
        assert blocks[2]["block_type"] == "paragraph"

    def test_filter_table_separator(self):
        parser = PDFParser(RAGConfig())
        content = "| 项目 | 金额 |\n|------|------|\n| 收入 | 100 |"
        blocks = parser._split_blocks(content)
        table_text = blocks[0]["text"]
        assert "|---" not in table_text

    def test_mixed_content(self):
        parser = PDFParser(RAGConfig())
        content = "# 标题\n\n正文段落。\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n结尾段落。"
        blocks = parser._split_blocks(content)
        types = [b["block_type"] for b in blocks]
        assert types == ["title", "paragraph", "table", "paragraph"]
