import json
import logging
import time
from pathlib import Path

from rag_system.chunker import Chunker
from rag_system.config import RAGConfig
from rag_system.generator import Generator, _extract_json_answer
from rag_system.indexer import Indexer
from rag_system.parser import PDFParser
from rag_system.validator import Validator

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.parser = PDFParser(config)
        self.chunker = Chunker(config)
        self.indexer = Indexer(config)
        self.generator = Generator(config)
        self.validator = Validator()

    def run(self) -> dict:
        self._ensure_dirs()
        results: dict[str, int] = {"total": 0, "answered": 0, "high_confidence": 0, "failed": 0}

        docs = self._collect_documents()
        logger.info("Step 1/5: Parsing %d documents", len(docs))
        all_records: dict[str, list] = {}
        for file_path, domain in docs:
            records = list(self.parser.parse_all([file_path]))
            all_records.setdefault(domain, []).extend(records)

        logger.info("Step 2/5: Chunking")
        all_chunks: dict[str, list] = {}
        for domain, records in all_records.items():
            doc_groups: dict[str, list] = {}
            for r in records:
                doc_groups.setdefault(r["doc_id"], []).append(r)
            chunks = []
            for doc_id, doc_records in doc_groups.items():
                chunks.extend(self.chunker.chunk_document(domain, doc_records))
            all_chunks[domain] = chunks
            self._write_chunks(domain, chunks)

        logger.info("Step 3/5: Building indexes")
        for domain, chunks in all_chunks.items():
            self.indexer.build_bm25(domain, chunks)
        for domain, chunks in all_chunks.items():
            try:
                self.indexer.build_embedding(domain, chunks)
            except Exception:
                logger.exception(
                    "Embedding build failed for domain=%s, continuing with BM25-only", domain
                )
            try:
                self.indexer.build_field_index(domain, chunks)
            except Exception:
                logger.exception(
                    "Field index build failed for domain=%s, continuing", domain
                )
        self.indexer.build_doc_profiles(all_chunks)

        logger.info("Step 4/5: Answering questions")
        questions = self._load_questions()
        results["total"] = len(questions)

        cache_dir = self.config.output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        answers: dict[str, dict] = {}
        evidence: dict[str, list] = {}
        run_log_entries: list[dict] = []

        for q in questions:
            cache_path = cache_dir / f"{q['qid']}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                answers[q["qid"]] = cached
                if cached.get("confidence") == "high":
                    results["high_confidence"] += 1
                if cached.get("answer"):
                    results["answered"] += 1
                continue

            t0 = time.time()
            gen_result = self.generator.answer_question(q)

            if "deepseek" in gen_result:
                ds_raw = gen_result["deepseek"]
                qw_raw = gen_result["qwen"]
                ds_parsed = _extract_json_answer(ds_raw) if isinstance(ds_raw, str) else ds_raw
                qw_parsed = _extract_json_answer(qw_raw) if isinstance(qw_raw, str) else qw_raw
                validated = self.validator.validate(ds_parsed, qw_parsed, q)
            else:
                validated = dict(gen_result)
                if "confidence" not in validated:
                    validated["confidence"] = "medium" if validated.get("answer") else "failed"

            validated["qid"] = q["qid"]
            validated["domain"] = q.get("domain", "")
            validated["elapsed"] = time.time() - t0

            cache_path.write_text(json.dumps(validated, ensure_ascii=False), encoding="utf-8")
            answers[q["qid"]] = validated
            evidence[q["qid"]] = validated.get("evidence_ids", [])

            if validated.get("confidence") == "high":
                results["high_confidence"] += 1
            if validated.get("answer"):
                results["answered"] += 1
            if validated.get("confidence") == "failed":
                results["failed"] += 1

            run_log_entries.append({
                "qid": q["qid"], "domain": q.get("domain", ""),
                "answer": validated["answer"], "confidence": validated["confidence"],
                "elapsed": validated["elapsed"],
            })

        logger.info("Step 5/5: Writing output")
        self._write_answer_csv(answers)
        self._write_evidence(evidence)
        self._write_run_log(run_log_entries)

        return results

    def _collect_documents(self) -> list[tuple[Path, str]]:
        docs: list[tuple[Path, str]] = []
        for ext in ("*.pdf", "*.PDF", "*.html", "*.htm", "*.txt"):
            for f in self.config.pdf_dir.rglob(ext):
                domain = f.relative_to(self.config.pdf_dir).parts[0]
                docs.append((f, domain))
        return sorted(docs, key=lambda x: (x[1], x[0].name))

    def _write_chunks(self, domain: str, chunks: list) -> None:
        domain_dir = self.config.chunks_dir
        domain_dir.mkdir(parents=True, exist_ok=True)
        path = domain_dir / f"{domain}_chunks.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "domain": c.domain,
                    "text": c.text, "section_path": c.section_path,
                    "prev_chunk_id": c.prev_chunk_id, "next_chunk_id": c.next_chunk_id,
                }, ensure_ascii=False) + "\n")

    def _load_questions(self) -> list[dict]:
        questions: list[dict] = []
        q_dir = self.config.questions_dir / "group_a"
        if not q_dir.exists():
            logger.warning("Questions directory not found: %s", q_dir)
            return questions
        for qf in sorted(q_dir.glob("*_questions.json")):
            data = json.loads(qf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                questions.extend(data)
        return questions

    def _write_answer_csv(self, answers: dict[str, dict]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "answer.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("qid,answer\n")
            for qid in sorted(answers.keys()):
                a = answers[qid]
                f.write(f"{qid},{a.get('answer', '')}\n")

    def _write_evidence(self, evidence: dict[str, list]) -> None:
        path = self.config.output_dir / "evidence.json"
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_run_log(self, entries: list[dict]) -> None:
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "run_log.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _ensure_dirs(self) -> None:
        for d in [self.config.parsed_dir, self.config.chunks_dir,
                   self.config.index_dir, self.config.output_dir]:
            d.mkdir(parents=True, exist_ok=True)
