from rag_system.generator import (
    Generator,
    _extract_json_answer,
    _neutralize_tf_options,
    _assemble_multi_answer,
    _expand_context,
)


class TestJSONExtraction:
    def test_direct_json(self):
        raw = '{"answer": "B", "reasoning": "test", "evidence_ids": ["c1"]}'
        result = _extract_json_answer(raw)
        assert result["answer"] == "B"
        assert result["reasoning"] == "test"

    def test_code_block_json(self):
        raw = 'Some text\n```json\n{"answer": "AC", "reasoning": "r", "evidence_ids": []}\n```\nMore text'
        result = _extract_json_answer(raw)
        assert result["answer"] == "AC"

    def test_brace_extraction(self):
        raw = 'The answer is {"answer": "B", "reasoning": "test reasoning", "evidence_ids": ["a", "b"]} end'
        result = _extract_json_answer(raw)
        assert result["answer"] == "B"

    def test_fallback_uppercase(self):
        raw = "I believe the correct answer is B based on the evidence."
        result = _extract_json_answer(raw)
        assert result["answer"] == "B"
        assert result.get("parse_fallback") is True

    def test_all_fail(self):
        raw = "no letters or json here"
        result = _extract_json_answer(raw)
        assert result["answer"] == ""
        assert result.get("parse_error") is True


class TestTFNeutralization:
    def test_neutralize_options(self):
        options = {"A": "正确", "B": "错误"}
        new_opts, mapping = _neutralize_tf_options(options)
        assert "选项1" in new_opts
        assert "选项2" in new_opts
        assert mapping["选项1"] == "A"
        assert mapping["选项2"] == "B"


class TestMultiAnswerAssembly:
    @staticmethod
    def _make_judgments(judgments: list[dict]) -> list[dict]:
        return [
            {
                "letter": j["letter"],
                "judgment": j["judgment"],
                "reasoning": j.get("reasoning", ""),
                "evidence_ids": j.get("evidence_ids", []),
            }
            for j in judgments
        ]

    def test_assemble(self):
        judgments = self._make_judgments([
            {"letter": "A", "judgment": "YES", "evidence_ids": ["c1"]},
            {"letter": "B", "judgment": "NO", "evidence_ids": ["c2"]},
            {"letter": "C", "judgment": "YES", "evidence_ids": ["c3"]},
            {"letter": "D", "judgment": "NO", "evidence_ids": ["c4"]},
        ])
        result = _assemble_multi_answer(judgments, {})
        assert result["answer"] == "AC"
        assert len(result["evidence_ids"]) == 2

    def test_all_no_fallback(self):
        judgments = self._make_judgments([
            {"letter": "A", "judgment": "NO"},
            {"letter": "B", "judgment": "NO"},
            {"letter": "C", "judgment": "NO"},
            {"letter": "D", "judgment": "NO"},
        ])
        result = _assemble_multi_answer(judgments, {})
        assert result["needs_retry"] is True


class TestContextExpansion:
    def test_expand_context(self):
        chunks = [
            {"chunk_id": "c0", "text": "text0", "doc_id": "d1",
             "prev_chunk_id": None, "next_chunk_id": "c1", "section_path": "p0"},
            {"chunk_id": "c1", "text": "text1", "doc_id": "d1",
             "prev_chunk_id": "c0", "next_chunk_id": "c2", "section_path": "p1"},
            {"chunk_id": "c2", "text": "text2", "doc_id": "d1",
             "prev_chunk_id": "c1", "next_chunk_id": None, "section_path": "p2"},
        ]
        evidence = [{"chunk_id": "c1", "text": "text1"}]
        expanded = _expand_context(evidence, chunks)
        assert len(expanded) == 3
        ids = {e["chunk_id"] for e in expanded}
        assert ids == {"c0", "c1", "c2"}
