import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system.config import RAGConfig
from rag_system.generator import _expand_context
from rag_system.retriever import Retriever


def _load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").strip().split("\n")[1:]:
            if "," in line:
                qid, rest = line.split(",", 1)
                answers[qid] = rest.strip()
    return answers


def _load_cache(cache_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                cache[data["qid"]] = data
            except Exception:
                pass
    return cache


def _build_model_reasoning(cached: dict | None) -> str:
    if not cached:
        return ""
    reasoning = cached.get("reasoning", "")
    if reasoning:
        return reasoning
    parts = []
    for model in ("deepseek_answer", "qwen_answer"):
        val = cached.get(model, "")
        if isinstance(val, dict):
            r = val.get("reasoning", "")
            if r:
                parts.append(f"[{model.split('_')[0]}] {r}")
        elif isinstance(val, str) and val:
            parts.append(f"[{model.split('_')[0]}] {val}")
    return "; ".join(parts)


def main() -> None:
    config = RAGConfig()
    retriever = Retriever(config)

    questions: list[dict] = []
    q_dir = config.questions_dir / "group_a"
    for qf in sorted(q_dir.glob("*_questions.json")):
        data = json.loads(qf.read_text(encoding="utf-8"))
        if isinstance(data, list):
            questions.extend(data)

    answers = _load_answers(config.output_dir / "answer.csv")
    cache = _load_cache(config.output_dir / "cache")

    out_path = Path("docs/questions_review.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            qid = q["qid"]
            cached = cache.get(qid, {})
            confidence = cached.get("confidence", "unknown")

            try:
                evidence = retriever.search(q)
                chunks: list[dict] = []
                if evidence:
                    domain_name = q.get("domain", "insurance")
                    evidence = _expand_context(
                        evidence,
                        retriever._load_domain_chunks(domain_name),
                        config,
                    )
                    for ev in evidence:
                        chunks.append({
                            "chunk_id": ev.get("chunk_id", ""),
                            "doc_id": ev.get("doc_id", ""),
                            "text": ev.get("text", "")[:500],
                            "section_path": ev.get("section_path", ""),
                            "score": round(ev.get("rerank_score", ev.get("score", 0)), 4),
                        })
            except Exception:
                chunks = []

            record = {
                "qid": qid,
                "domain": q.get("domain", ""),
                "question_type": q.get("answer_format", ""),
                "doc_ids": q.get("doc_ids", []),
                "question_text": q.get("question", ""),
                "options": q.get("options", {}),
                "final_answer": answers.get(qid, ""),
                "confidence": confidence,
                "model_reasoning": _build_model_reasoning(cached),
                "retrieved_chunks": chunks,
                "human_label": None,
                "error_category": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    size_kb = out_path.stat().st_size / 1024
    print(f"Written: {out_path} ({size_kb:.0f} KB, {count} questions)")


if __name__ == "__main__":
    main()
