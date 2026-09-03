import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import fitz
from bs4 import BeautifulSoup

from rag_system.config import RAGConfig

logger = logging.getLogger(__name__)


class PDFParser:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.error_log_path = config.parsed_dir / "error_log.jsonl"

    def parse_all(self, pdf_paths: list[Path]) -> Iterator[dict]:
        for pdf_path in pdf_paths:
            rel = pdf_path.relative_to(self.config.pdf_dir)
            domain = rel.parts[0]
            doc_id = pdf_path.stem
            output_dir = self.config.parsed_dir / domain
            output_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = output_dir / f"{doc_id}.jsonl"
            done_path = output_dir / f"{doc_id}.done"

            if self._is_done(jsonl_path, done_path):
                yield from self._read_jsonl(jsonl_path, doc_id)
                continue

            try:
                records = self._run_parser(pdf_path, doc_id)
                self._write_jsonl(jsonl_path, records)
                done_path.write_text("ok", encoding="utf-8")
                self._cleanup_md_dir(pdf_path)
                self._check_disk_space()
                yield from records
            except Exception:
                logger.exception("Failed to parse %s", pdf_path)
                self._log_error(pdf_path, sys.exc_info()[1])

    def _run_parser(self, file_path: Path, doc_id: str) -> list[dict]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            if self.config.parser_backend == "pymupdf":
                return self._run_pymupdf(file_path, doc_id)
            return self._run_mineru(file_path, doc_id)
        if suffix in (".html", ".htm"):
            return self._run_html(file_path, doc_id)
        if suffix == ".txt":
            return self._run_txt(file_path, doc_id)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _run_pymupdf(self, pdf_path: Path, doc_id: str) -> list[dict]:
        records: list[dict] = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if not text or len(text) < 2:
                    continue
                block_type = "paragraph"
                heading_level = 1
                if re.match(r"^第[一二三四五六七八九十\d]+[章节条款]", text):
                    block_type = "title"
                elif re.match(r"^\d+[\.\、]", text):
                    block_type = "title"
                    heading_level = 2
                records.append({
                    "text": text,
                    "block_type": block_type,
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "heading_level": heading_level,
                })
        doc.close()
        return records

    def _run_html(self, file_path: Path, doc_id: str) -> list[dict]:
        html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one(".content") or soup.select_one("article") or soup
        for tag in main(["script", "style", "nav", "footer", "header", "link", "meta"]):
            tag.decompose()
        text = main.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        records: list[dict] = []
        for line in lines:
            block_type = "paragraph"
            heading_level = 1
            if re.match(r"^第[一二三四五六七八九十\d]+[章节条款]", line):
                block_type = "title"
            elif re.match(r"^\d+[\.\、]", line):
                block_type = "title"
                heading_level = 2
            records.append({
                "text": line,
                "block_type": block_type,
                "doc_id": doc_id,
                "page_num": 0,
                "heading_level": heading_level,
            })
        return records

    def _run_txt(self, file_path: Path, doc_id: str) -> list[dict]:
        content = file_path.read_text(encoding="utf-8-sig")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        records: list[dict] = []
        for line in lines:
            block_type = "paragraph"
            heading_level = 1
            if re.match(r"^第[一二三四五六七八九十\d]+[章节条款]", line):
                block_type = "title"
            elif re.match(r"^\d+[\.\、]", line):
                block_type = "title"
                heading_level = 2
            records.append({
                "text": line,
                "block_type": block_type,
                "doc_id": doc_id,
                "page_num": 0,
                "heading_level": heading_level,
            })
        return records

    def _run_mineru(self, pdf_path: Path, doc_id: str) -> list[dict]:
        cmd = [
            sys.executable, "-m", "magic_pdf.tools.cli",
            "-p", str(pdf_path),
            "-o", str(self.config.parsed_dir),
            "-m", "auto",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            stderr_snippet = (
                result.stderr[:500] if result.stderr else "unknown error"
            )
            raise RuntimeError(f"MinerU failed: {stderr_snippet}")

        md_dir = self.config.parsed_dir / pdf_path.stem / "auto"
        md_files = sorted(md_dir.rglob("*.md")) if md_dir.exists() else []
        if not md_files:
            raise RuntimeError(
                f"No markdown output from MinerU for {pdf_path}"
            )

        return self._parse_markdown_to_records(md_files, doc_id)

    def _parse_markdown_to_records(
        self, md_files: list[Path], doc_id: str
    ) -> list[dict]:
        records: list[dict] = []
        for md_file in md_files:
            page_num = self._extract_page_num(md_file)
            content = md_file.read_text(encoding="utf-8")
            for block in self._split_blocks(content):
                block["doc_id"] = doc_id
                block["page_num"] = block.get("page_num", page_num)
                records.append(block)
        return records

    def _split_blocks(self, content: str) -> list[dict]:
        blocks: list[dict] = []
        lines = content.split("\n")
        buf: list[str] = []
        current_type = "paragraph"
        current_heading_level = 1

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buf:
                    block = {"text": "\n".join(buf), "block_type": current_type}
                    if current_type == "title":
                        block["heading_level"] = current_heading_level
                    blocks.append(block)
                    buf = []
                current_type = "paragraph"
                continue
            if stripped.startswith("#"):
                if buf:
                    block = {"text": "\n".join(buf), "block_type": current_type}
                    if current_type == "title":
                        block["heading_level"] = current_heading_level
                    blocks.append(block)
                    buf = []
                current_heading_level = len(stripped) - len(stripped.lstrip("#"))
                current_type = "title"
                buf.append(stripped.lstrip("#").strip())
                continue
            if stripped.startswith("|"):
                if all(c in "|-: " for c in stripped):
                    continue
                is_table_row = True
            else:
                is_table_row = False
            if is_table_row and current_type != "table":
                if buf:
                    blocks.append({
                        "text": "\n".join(buf),
                        "block_type": current_type,
                    })
                    buf = []
                current_type = "table"
            elif not is_table_row and current_type == "table":
                if buf:
                    blocks.append({
                        "text": "\n".join(buf),
                        "block_type": "table",
                    })
                    buf = []
                current_type = "paragraph"
            buf.append(stripped)

        if buf:
            block = {"text": "\n".join(buf), "block_type": current_type}
            if current_type == "title":
                block["heading_level"] = current_heading_level
            blocks.append(block)
        return blocks

    def _extract_page_num(self, md_file: Path) -> int:
        name = md_file.stem
        try:
            parts = name.replace("page_", "").split("_")
            nums = [int(p) for p in parts if p.isdigit()]
            return nums[0] if nums else 1
        except ValueError:
            return 1

    def _is_done(self, jsonl_path: Path, done_path: Path) -> bool:
        if not jsonl_path.exists() or not done_path.exists():
            return False
        try:
            with open(jsonl_path, "rb") as f:
                if jsonl_path.stat().st_size == 0:
                    return False
                f.seek(max(0, jsonl_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8")
                lines = tail.strip().split("\n")
                if not lines:
                    return False
                json.loads(lines[-1].strip())
            return True
        except Exception:
            return False

    def _read_jsonl(self, jsonl_path: Path, doc_id: str) -> Iterator[dict]:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["doc_id"] = doc_id
                    yield record

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    def _log_error(self, pdf_path: Path, exc: BaseException | None = None) -> None:
        entry: dict = {"pdf": str(pdf_path), "timestamp": time.time()}
        if exc is not None:
            entry["error_type"] = type(exc).__name__
            entry["error_message"] = str(exc)
        with open(self.error_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _cleanup_md_dir(self, pdf_path: Path) -> None:
        md_dir = self.config.parsed_dir / pdf_path.stem
        if md_dir.exists():
            shutil.rmtree(md_dir, ignore_errors=True)

    def _check_disk_space(self) -> None:
        usage = shutil.disk_usage(self.config.parsed_dir)
        if usage.free < 1_000_000_000:
            raise OSError("Disk space below 1GB, stopping")
