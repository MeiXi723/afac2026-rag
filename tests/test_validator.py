from rag_system.validator import Validator, _normalize_answer


class TestNormalizeAnswer:
    def test_uppercase(self):
        assert _normalize_answer("a") == "A"

    def test_sort_multi(self):
        assert _normalize_answer("DBA") == "ABD"

    def test_dedup(self):
        assert _normalize_answer("AAB") == "AB"

    def test_empty(self):
        assert _normalize_answer("") is None

    def test_tf_true_false_mapping(self):
        options = {"A": "正确", "B": "错误"}
        assert _normalize_answer("True", options) == "A"
        assert _normalize_answer("正确", options) == "A"
        assert _normalize_answer("False", options) == "B"
        assert _normalize_answer("对", options) == "A"
        assert _normalize_answer("是", options) == "A"
        assert _normalize_answer("错", options) == "B"
        assert _normalize_answer("否", options) == "B"

    def test_tf_lowercase_option_values(self):
        options = {"A": "true", "B": "false"}
        assert _normalize_answer("True", options) == "A"
        assert _normalize_answer("False", options) == "B"

    def test_no_options_no_change(self):
        assert _normalize_answer("B") == "B"


class TestValidator:
    def test_high_confidence_match(self):
        v = Validator()
        ds_result = {"answer": "B", "reasoning": "r1", "evidence_ids": ["c1"]}
        qw_result = {"answer": "B", "reasoning": "r2", "evidence_ids": ["c2"]}
        result = v.validate(ds_result, qw_result, {})
        assert result["answer"] == "B"
        assert result["confidence"] == "high"

    def test_medium_confidence_one_model(self):
        v = Validator()
        ds_result = {"answer": "A", "reasoning": "r1", "evidence_ids": []}
        qw_result = {"answer": "", "reasoning": "", "evidence_ids": [], "parse_error": True}
        result = v.validate(ds_result, qw_result, {})
        assert result["answer"] == "A"
        assert result["confidence"] == "medium"

    def test_low_confidence_mismatch(self):
        v = Validator()
        ds_result = {"answer": "A", "reasoning": "r1", "evidence_ids": []}
        qw_result = {"answer": "B", "reasoning": "r2", "evidence_ids": []}
        result = v.validate(ds_result, qw_result, {})
        assert result["confidence"] == "low"
        assert "deepseek_answer" in result

    def test_failed_both_models(self):
        v = Validator()
        ds_result = {"answer": "", "reasoning": "", "evidence_ids": [], "parse_error": True}
        qw_result = {"answer": "", "reasoning": "", "evidence_ids": [], "parse_error": True}
        result = v.validate(ds_result, qw_result, {})
        assert result["confidence"] == "failed"
        assert result["answer"] == ""

    def test_multi_answer_normalized_match(self):
        v = Validator()
        ds_result = {"answer": "DBA", "reasoning": "r1", "evidence_ids": []}
        qw_result = {"answer": "ABD", "reasoning": "r2", "evidence_ids": []}
        result = v.validate(ds_result, qw_result, {}, answer_format="multi")
        assert result["confidence"] == "high"
        assert result["answer"] == "ABD"
