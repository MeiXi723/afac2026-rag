from typing import Optional


class Validator:
    def validate(
        self, ds_result: dict, qw_result: dict, question: dict,
        answer_format: str = ""
    ) -> dict:
        if not answer_format:
            answer_format = question.get("answer_format", "mcq")

        options = question.get("options", {})
        ds_answer = _normalize_answer(ds_result.get("answer", ""), options)
        qw_answer = _normalize_answer(qw_result.get("answer", ""), options)

        ds_valid = ds_answer is not None
        qw_valid = qw_answer is not None

        base = {
            "deepseek_answer": ds_result.get("answer", ""),
            "qwen_answer": qw_result.get("answer", ""),
        }

        if not ds_valid and not qw_valid:
            return {
                **base,
                "answer": "", "confidence": "failed",
                "evidence_ids": list(dict.fromkeys(
                    ds_result.get("evidence_ids", []) + qw_result.get("evidence_ids", [])
                )),
            }

        if not ds_valid:
            return {
                **base,
                "answer": qw_answer, "confidence": "medium",
                "evidence_ids": qw_result.get("evidence_ids", []),
            }

        if not qw_valid:
            return {
                **base,
                "answer": ds_answer, "confidence": "medium",
                "evidence_ids": ds_result.get("evidence_ids", []),
            }

        match = _answers_match(ds_answer, qw_answer, answer_format)
        evidence_ids = list(dict.fromkeys(
            ds_result.get("evidence_ids", []) + qw_result.get("evidence_ids", [])
        ))

        confidence = "high" if match else "low"
        if match and answer_format == "multi":
            if ds_result.get("_numerical_warning") or qw_result.get("_numerical_warning"):
                confidence = "medium"

        if match:
            return {
                **base,
                "answer": ds_answer, "confidence": confidence,
                "evidence_ids": evidence_ids,
            }

        resolved = self._resolve_disagreement(
            ds_answer, qw_answer, ds_result, qw_result, answer_format
        )
        return {
            **base,
            "answer": resolved, "confidence": "low",
            "evidence_ids": evidence_ids,
        }

    def _resolve_disagreement(
        self, ds_answer: str, qw_answer: str,
        ds_result: dict, qw_result: dict, answer_format: str,
    ) -> str:
        if answer_format == "multi":
            # 并集会系统性膨胀多选答案；改保守交集，完全无交集时才保留候选并集
            common = set(ds_answer) & set(qw_answer)
            if common:
                return "".join(sorted(common))
            return "".join(sorted(set(ds_answer) | set(qw_answer)))
        if answer_format == "tf":
            return ds_answer
        return ds_answer


def _normalize_answer(answer: str, options: Optional[dict] = None) -> Optional[str]:
    if not answer or not answer.strip():
        return None
    a = answer.strip().upper()
    if a in ("无", "NONE", "没有", "/", "-", "无正确答案"):
        return None
    if options:
        a = _map_tf_answer(a, options)
    if all(c in "ABCD" for c in a):
        return "".join(sorted(set(a)))
    return a


def _map_tf_answer(answer: str, options: dict) -> str:
    if answer in ("TRUE", "正确", "对", "是"):
        for letter, text in options.items():
            if text in ("正确", "对", "是", "TRUE", "True", "true"):
                return letter
    if answer in ("FALSE", "错误", "错", "否"):
        for letter, text in options.items():
            if text in ("错误", "错", "否", "FALSE", "False", "false"):
                return letter
    return answer


def _answers_match(a: str, b: str, answer_format: str) -> bool:
    if answer_format == "multi":
        return set(a) == set(b)
    return a == b
