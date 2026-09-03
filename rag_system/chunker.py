from dataclasses import dataclass
from typing import Optional

from rag_system.config import RAGConfig


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    section_path: str
    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]


class Chunker:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_document(self, domain: str, records: list[dict]) -> list[Chunk]:
        if not records:
            return []
        sections = self._build_sections(records)
        chunks: list[Chunk] = []
        chunk_idx = 0
        for section_path, section_text in sections:
            section_chunks = self._split_text(
                section_text, section_path, records[0]["doc_id"], domain, chunk_idx
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
        chunks = self._postprocess_chunks(chunks)
        self._reindex_chunks(chunks, domain, records[0]["doc_id"])
        self._link_chunks(chunks)
        return chunks

    def _build_sections(self, records: list[dict]) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_path: list[str] = []
        current_text: list[str] = []
        for record in records:
            if record["block_type"] == "title":
                if current_text:
                    sections.append((" > ".join(current_path), "\n".join(current_text)))
                    current_text = []
                text = record["text"]
                level = record.get("heading_level", 1)
                heading = text.lstrip("#").strip()
                current_path = current_path[:level - 1] + [heading]
            elif record["block_type"] == "table":
                if current_text:
                    sections.append((" > ".join(current_path), "\n".join(current_text)))
                    current_text = []
                table_path = " > ".join(current_path + ["[表格]"])
                sections.append((table_path, record["text"]))
            else:
                current_text.append(record["text"])
        if current_text:
            sections.append((" > ".join(current_path), "\n".join(current_text)))
        return sections

    def _split_text(
        self, text: str, section_path: str, doc_id: str, domain: str, start_idx: int
    ) -> list[Chunk]:
        if "|" in text and "---" in text:
            return [Chunk(
                chunk_id=f"{domain}_{doc_id}_t{start_idx}",
                doc_id=doc_id, domain=domain, text=text,
                section_path=section_path,
                prev_chunk_id=None, next_chunk_id=None,
            )]

        if len(text) <= self.chunk_size:
            return [Chunk(
                chunk_id=f"{domain}_{doc_id}_{start_idx}",
                doc_id=doc_id, domain=domain, text=text,
                section_path=section_path,
                prev_chunk_id=None, next_chunk_id=None,
            )]

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        idx = start_idx
        pos = 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))
            chunk_text = text[pos:end]
            chunks.append(Chunk(
                chunk_id=f"{domain}_{doc_id}_{idx}",
                doc_id=doc_id, domain=domain, text=chunk_text,
                section_path=section_path,
                prev_chunk_id=None, next_chunk_id=None,
            ))
            idx += 1
            pos += step
        return chunks

    def _link_chunks(self, chunks: list[Chunk]) -> None:
        for i, chunk in enumerate(chunks):
            chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None

    def _postprocess_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        chunks = self._merge_short_chunks(chunks)
        chunks = self._dedup_chunks(chunks)
        return chunks

    def _merge_short_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        min_len = self.config.min_chunk_length
        if not chunks or len(chunks) == 1:
            return list(chunks)
        result: list[Chunk] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if len(chunk.text) >= min_len:
                result.append(chunk)
                i += 1
                continue
            if result:
                result[-1].text = result[-1].text + "\n" + chunk.text
                i += 1
                continue
            if i + 1 < len(chunks):
                chunks[i + 1].text = chunk.text + "\n" + chunks[i + 1].text
                i += 2
                continue
            result.append(chunk)
            i += 1
        return result

    def _dedup_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        seen: set[int] = set()
        result: list[Chunk] = []
        for chunk in chunks:
            h = hash(chunk.text)
            if h not in seen:
                seen.add(h)
                result.append(chunk)
        return result

    def _reindex_chunks(self, chunks: list[Chunk], domain: str, doc_id: str) -> None:
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{domain}_{doc_id}_{i}"
