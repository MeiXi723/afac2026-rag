import json
import os
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import QwenAgent


def run_eval(eval_path: str | None = None) -> None:
    if eval_path is None:
        eval_path = str(PROJECT_ROOT / "data" / "eval_questions.json")
    if not os.path.exists(eval_path):
        print(f"[ERROR] {eval_path} not found. Run: python scripts/eval.py --sample")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    agent = QwenAgent()
    correct = 0
    total = 0
    total_token = 0
    results: list[dict] = []

    for q in questions:
        token_before = agent.token_counter.snapshot()
        t0 = time.time()
        try:
            result = agent.answer_single(
                qid=q["qid"], domain=q["domain"], question=q["question"],
                options=q["options"], answer_format=q["answer_format"],
                doc_ids=q.get("doc_ids"),
            )
            elapsed = round(time.time() - t0, 1)
            answer = result["answer"]
            expected = q.get("answer", "")
            is_correct = answer == expected if expected else None

            if is_correct:
                correct += 1
            total += 1

            token_after = agent.token_counter.snapshot()
            q_tokens = (token_after[0] - token_before[0]) + (token_after[1] - token_before[1])
            total_token += q_tokens

            status = "✓" if is_correct else ("✗" if is_correct is False else "?")
            print(f"  [{status}] {q['qid']}  expected={expected or '?'}  got={answer}  "
                  f"{result['confidence']}  {q_tokens:,} tokens  {elapsed}s")

            results.append({
                "qid": q["qid"], "domain": q["domain"], "type": q["answer_format"],
                "expected": expected, "got": answer, "correct": is_correct,
                "confidence": result["confidence"], "tokens": q_tokens,
            })
        except Exception as e:
            print(f"  [FAIL] {q['qid']}: {e}")
            total += 1
            results.append({"qid": q["qid"], "error": str(e)})

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Total tokens: {total_token:,}")
    print(f"Results -> eval_results.json")
    with open(str(PROJECT_ROOT / "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total,
                    "total_tokens": total_token, "results": results},
                   f, ensure_ascii=False, indent=2)


def sample_questions(output_path: str | None = None) -> None:
    """Sample 30 questions (6 per domain, mix of types) for local eval."""
    import random

    if output_path is None:
        output_path = str(PROJECT_ROOT / "data" / "eval_questions.json")

    QUESTIONS_DIR = str(PROJECT_ROOT / "public_dataset_upload" / "questions" / "group_a")
    DOMAINS = ["insurance", "regulatory", "financial_contracts", "financial_reports", "research"]
    sampled: list[dict] = []

    for domain in DOMAINS:
        fpath = os.path.join(QUESTIONS_DIR, f"{domain}_questions.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            qs = json.load(f)

        mcq = [q for q in qs if q["answer_format"] == "mcq"]
        multi = [q for q in qs if q["answer_format"] == "multi"]
        tf = [q for q in qs if q["answer_format"] == "tf"]

        picks: list[dict] = []
        picks.extend(random.sample(mcq, min(2, len(mcq))))
        picks.extend(random.sample(multi, min(3, len(multi))))
        picks.extend(random.sample(tf, min(1, len(tf))))
        random.shuffle(picks)
        sampled.extend(picks)

    random.shuffle(sampled)
    sampled = sampled[:30]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Sampled {len(sampled)} questions -> {output_path}")
    print("Fill in the 'answer' field for each question with the correct answer.")
    print("Types summary:")
    type_counts = {}
    for q in sampled:
        type_counts[q["answer_format"]] = type_counts.get(q["answer_format"], 0) + 1
    for t, c in type_counts.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        sample_questions()
    else:
        run_eval()
