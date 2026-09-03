import json
import tempfile
from pathlib import Path

from rag_system.config import RAGConfig
from rag_system.pipeline import Pipeline


class TestPipelineSmoke:
    def test_pipeline_instantiation(self):
        cfg = RAGConfig()
        cfg.pdf_dir = Path(tempfile.mkdtemp())
        cfg.parsed_dir = Path(tempfile.mkdtemp())
        cfg.output_dir = Path(tempfile.mkdtemp())
        pipe = Pipeline(cfg)
        assert pipe.parser is not None
        assert pipe.chunker is not None
        assert pipe.indexer is not None
        assert pipe.generator is not None
        assert pipe.validator is not None

    def test_collect_pdfs_empty_dir(self):
        cfg = RAGConfig()
        cfg.pdf_dir = Path(tempfile.mkdtemp())
        cfg.output_dir = Path(tempfile.mkdtemp())
        pipe = Pipeline(cfg)
        pdfs = pipe._collect_pdfs()
        assert pdfs == []

    def test_load_questions_empty_dir(self):
        cfg = RAGConfig()
        cfg.questions_dir = Path(tempfile.mkdtemp())
        cfg.output_dir = Path(tempfile.mkdtemp())
        pipe = Pipeline(cfg)
        questions = pipe._load_questions()
        assert questions == []

    def test_write_answer_csv(self):
        cfg = RAGConfig()
        cfg.output_dir = Path(tempfile.mkdtemp())
        pipe = Pipeline(cfg)
        pipe._write_answer_csv({
            "q001": {"answer": "B", "confidence": "high"},
            "q002": {"answer": "AC", "confidence": "low"},
        })
        csv_path = cfg.output_dir / "answer.csv"
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "q001,B" in content
        assert "q002,AC" in content
