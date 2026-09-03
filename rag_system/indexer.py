import json
import logging
import pickle
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from rag_system.chunker import Chunk
from rag_system.config import RAGConfig

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self._embedding_model: Optional[object] = None
        self._cloud_embedder: Optional[object] = None

    def build_bm25(self, domain: str, chunks: list[Chunk]) -> None:
        if not chunks:
            logger.warning("build_bm25: empty chunks for domain=%s, skipping", domain)
            return
        domain_dir = self.config.index_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building BM25 index for domain=%s, chunks=%d", domain, len(chunks))
        tokenized = [jieba_lcut(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        bm25_path = domain_dir / "bm25.pkl"
        tmp = bm25_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"model": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, f)
        tmp.replace(bm25_path)
        logger.info("BM25 index written: %s", bm25_path)

    def build_embedding(self, domain: str, chunks: list[Chunk]) -> None:
        if self.config.skip_embedding:
            logger.info("Embedding index skipped (skip_embedding=True) for domain=%s", domain)
            return
        if not chunks:
            logger.warning("build_embedding: empty chunks for domain=%s, skipping", domain)
            return
        domain_dir = self.config.index_dir / domain
        emb_dir = domain_dir / "embedding"
        if self._embedding_done(emb_dir):
            logger.info("Embedding index already built for domain=%s, skipping", domain)
            return
        logger.info("Building embedding index for domain=%s, chunks=%d", domain, len(chunks))
        if emb_dir.exists():
            shutil.rmtree(emb_dir)
        tmp_dir = domain_dir / "embedding.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.config.cloud_embedding:
                embedder = self._get_cloud_embedder()
                texts = [c.text for c in chunks]
                embeddings = embedder.embed(texts)
                if embeddings.shape[1] != embedder.dim:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: "
                        f"got {embeddings.shape[1]}, expected {embedder.dim}"
                    )
            else:
                model = self._get_embedding_model()
                texts = [c.text for c in chunks]
                embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            emb_array = np.array(embeddings, dtype=np.float32)
            import faiss

            index = faiss.IndexFlatIP(emb_array.shape[1])
            index.add(emb_array)
            faiss.write_index(index, str(tmp_dir / "index.faiss"))
            with open(tmp_dir / "id_map.json", "w", encoding="utf-8") as f:
                json.dump([c.chunk_id for c in chunks], f, ensure_ascii=False)
            tmp_dir.rename(emb_dir)
            (emb_dir / "done.semaphore").write_text("ok")
            logger.info("Embedding index written: %s", emb_dir)
        except Exception:
            logger.exception("Failed to build embedding for domain=%s", domain)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def build_field_index(self, domain: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        domain_dir = self.config.index_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        dates: dict[str, list[str]] = {}
        amounts: dict[str, list[str]] = {}
        percentages: dict[str, list[str]] = {}

        date_pattern = re.compile(
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?|'
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        )
        amount_pattern = re.compile(
            r'\d+\.?\d*\s*[亿元万千百]|'
            r'\d+\.?\d*\s*[元美欧港日英]|'
            r'\d{1,3}(?:,\d{3})+\s*[亿元万千百]?'
        )
        pct_pattern = re.compile(r'\d+\.?\d*\s*[%％]')

        for chunk in chunks:
            cid = chunk.chunk_id
            for m in date_pattern.finditer(chunk.text):
                dates.setdefault(m.group(), []).append(cid)
            for m in amount_pattern.finditer(chunk.text):
                amounts.setdefault(m.group(), []).append(cid)
            for m in pct_pattern.finditer(chunk.text):
                percentages.setdefault(m.group(), []).append(cid)

        field_index = {"dates": dates, "amounts": amounts, "percentages": percentages}
        path = domain_dir / "field_index.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(field_index, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        logger.info("Field index written: %s (%d dates, %d amounts, %d pcts)",
                     path, len(dates), len(amounts), len(percentages))

    def build_doc_profiles(self, domain_chunks: dict[str, list[Chunk]]) -> None:
        logger.info("Building doc profiles for %d domains", len(domain_chunks))
        profiles: dict = {}
        for domain, chunks in domain_chunks.items():
            if not chunks:
                logger.warning("build_doc_profiles: empty chunks for domain=%s, skipping", domain)
                continue
            profiles[domain] = {}
            doc_groups: dict[str, list[Chunk]] = {}
            for c in chunks:
                doc_groups.setdefault(c.doc_id, []).append(c)
            for doc_id, doc_chunks in doc_groups.items():
                preview_chunks = doc_chunks[:3]
                preview_text = "\n".join(c.text for c in preview_chunks)[:600]
                keywords = self._extract_keywords(preview_text)
                profiles[domain][doc_id] = {
                    "title": doc_chunks[0].section_path.split(">")[0].strip()
                    if doc_chunks[0].section_path
                    else doc_id,
                    "total_chunks": len(doc_chunks),
                    "total_chars": sum(len(c.text) for c in doc_chunks),
                    "preview": preview_text,
                    "keywords": keywords,
                }
        profiles_path = self.config.index_dir / "doc_profiles.json"
        tmp = profiles_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(profiles_path)
        logger.info("Doc profiles written: %s", profiles_path)

    def _embedding_done(self, emb_dir: Path) -> bool:
        return emb_dir.exists() and (emb_dir / "done.semaphore").exists()

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model

    def _get_cloud_embedder(self):
        if self._cloud_embedder is None:
            from rag_system.cloud_embedder import CloudEmbedder

            self._cloud_embedder = CloudEmbedder(self.config)
        return self._cloud_embedder

    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        import jieba.analyse

        return jieba.analyse.extract_tags(text, topK=top_n)

    def load_bm25(self, domain: str) -> Optional[dict]:
        bm25_path = self.config.index_dir / domain / "bm25.pkl"
        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            logger.exception("Failed to load BM25 index for domain=%s", domain)
            return None

    def load_embedding(self, domain: str) -> Optional[tuple]:
        import faiss

        emb_dir = self.config.index_dir / domain / "embedding"
        try:
            index = faiss.read_index(str(emb_dir / "index.faiss"))
            id_map = json.loads((emb_dir / "id_map.json").read_text(encoding="utf-8"))
            return index, id_map
        except Exception:
            logger.exception("Failed to load embedding index for domain=%s", domain)
            return None


def jieba_lcut(text: str) -> list[str]:
    import jieba

    return jieba.lcut(text)
