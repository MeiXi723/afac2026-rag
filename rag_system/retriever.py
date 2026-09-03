import json
import logging
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from rag_system.chunker import Chunk

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config) -> None:
        self.config = config
        self._bm25_cache: dict[str, dict] = {}
        self._embedding_cache: dict[str, tuple] = {}
        self._chunks_cache: dict[str, list[dict]] = {}
        self._chunk_map_cache: dict[str, dict[str, dict]] = {}
        self._cross_encoder = None
        self._cross_encoder_failed = False
        self._embedding_model = None
        self._cloud_embedder = None
        self._doc_profiles: Optional[dict] = None

    def search(self, question: dict) -> Optional[list[dict]]:
        raw_domain = question.get("domain")
        domain = self._route_domain(raw_domain, question.get("question", ""))
        if not domain:
            return None

        doc_ids: Optional[list[str]] = question.get("doc_ids") or None
        if doc_ids:
            self._check_doc_ids(domain, doc_ids)

        if question.get("answer_format") == "multi":
            return self._search_multi(question, domain, doc_ids)

        query = self._build_query(question)
        return self._search_single(question, domain, query, doc_ids)

    def _search_single(
        self, question: dict, domain: str, query: str, doc_ids: Optional[list[str]] = None
    ) -> Optional[list[dict]]:
        bm25_results = self._bm25_search(domain, query, self.config.top_n_bm25, doc_ids)
        emb_results = self._embedding_search(domain, query, self.config.top_n_embedding, doc_ids)
        merged = self._rrf_fusion(bm25_results, emb_results, domain, query)

        if len(merged) < 3:
            merged = self._fallback_search(question, domain)
        if not merged:
            return None

        ranked = self._cross_encoder_rerank(query, merged)

        filtered = self._quality_filter(ranked, domain, doc_ids)
        supplemented = self._supplement_docs(filtered, domain, doc_ids)
        balanced = self._balance_docs(supplemented)

        difficulty = self._assess_difficulty(question)
        final = self._dynamic_truncate(balanced, difficulty, doc_ids)

        return final

    def _route_domain(self, domain: Optional[str], question: str) -> Optional[str]:
        if domain:
            domain_dir = self.config.index_dir / domain
            if domain_dir.exists() and (domain_dir / "bm25.pkl").exists():
                return domain
        return self._guess_domain(question)

    def _build_query(self, question: dict) -> str:
        parts = [question.get("question", "")]
        entities = self._extract_entities(parts[0])
        if entities:
            parts.append(" ".join(entities))

        answer_format = question.get("answer_format", "")
        options = question.get("options", {})
        if answer_format != "multi" and options:
            option_texts = [f"{k}: {v}" for k, v in options.items()]
            parts.append(" ".join(option_texts))

        return " ".join(parts)

    def _extract_entities(self, text: str) -> list[str]:
        entities: list[str] = []
        entities.extend(re.findall(r'\d+\.?\d*\s*[亿元万千%％]', text))
        entities.extend(re.findall(r'第[一二三四五六七八九十百千]+条', text))
        entities.extend(re.findall(r'《([^》]+)》', text))
        return entities

    def _bm25_search(self, domain: str, query: str, top_n: int, doc_ids: Optional[list[str]] = None) -> list[dict]:
        bm25_data = self._load_bm25(domain)
        if bm25_data is None:
            return []

        from rag_system.indexer import jieba_lcut

        model = bm25_data["model"]
        chunk_ids = bm25_data["chunk_ids"]
        tokenized = jieba_lcut(query)
        scores = model.get_scores(tokenized)
        if len(scores) == 0:
            return []

        doc_id_set: Optional[set[str]] = set(doc_ids) if doc_ids else None
        top_indices = np.argsort(scores)[::-1][:top_n * 3]
        results: list[dict] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            if score < self.config.bm25_score_threshold:
                continue
            chunk_id = chunk_ids[idx]
            chunk_info = self._get_chunk_info(domain, chunk_id)
            if doc_id_set and chunk_info and chunk_info.get("doc_id") not in doc_id_set:
                continue
            results.append({
                "chunk_id": chunk_id,
                "score": score,
                "source": "bm25",
                "text": chunk_info.get("text", "") if chunk_info else "",
                "doc_id": chunk_info.get("doc_id", "") if chunk_info else "",
            })
            if len(results) >= top_n:
                break
        return results

    def _embedding_search(self, domain: str, query: str, top_n: int, doc_ids: Optional[list[str]] = None) -> list[dict]:
        emb_data = self._load_embedding(domain)
        if emb_data is None:
            return []

        index, id_map = emb_data
        doc_id_set: Optional[set[str]] = set(doc_ids) if doc_ids else None
        try:
            model = self._get_embedding_model()
            if model is None:
                return []
            if self.config.cloud_embedding:
                query_vec = model.embed_query(query)
            else:
                query_vec = model.encode([query], normalize_embeddings=True)
                query_vec = np.array(query_vec, dtype=np.float32)
            scores, indices = index.search(query_vec, min(top_n * 3, index.ntotal))
        except Exception:
            logger.exception("Embedding search failed for domain=%s", domain)
            return []

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(id_map):
                continue
            if float(score) < self.config.embedding_score_threshold:
                continue
            chunk_id = id_map[idx]
            chunk_info = self._get_chunk_info(domain, chunk_id)
            if doc_id_set and chunk_info and chunk_info.get("doc_id") not in doc_id_set:
                continue
            results.append({
                "chunk_id": chunk_id,
                "score": float(score),
                "source": "embedding",
                "text": chunk_info.get("text", "") if chunk_info else "",
                "doc_id": chunk_info.get("doc_id", "") if chunk_info else "",
            })
            if len(results) >= top_n:
                break
        return results

    def _rrf_fusion(
        self, bm25_results: list[dict], emb_results: list[dict], domain: str, query: str = ""
    ) -> list[dict]:
        k = self.config.rrf_k
        combined: dict[str, dict] = {}

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if cid in combined:
                combined[cid]["score"] += rrf_score
                combined[cid]["sources"].add(item["source"])
            else:
                combined[cid] = {
                    "chunk_id": cid,
                    "score": rrf_score,
                    "sources": {item["source"]},
                    "text": item.get("text", ""),
                }

        for rank, item in enumerate(emb_results):
            cid = item["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            if cid in combined:
                combined[cid]["score"] += rrf_score
                combined[cid]["sources"].add(item["source"])
            else:
                combined[cid] = {
                    "chunk_id": cid,
                    "score": rrf_score,
                    "sources": {item["source"]},
                    "text": item.get("text", ""),
                }

        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        for item in sorted_results:
            item["source"] = "+".join(sorted(item["sources"]))
            del item["sources"]
            chunk_info = self._get_chunk_info(domain, item["chunk_id"])
            if chunk_info:
                item.setdefault("doc_id", chunk_info.get("doc_id", ""))
                item.setdefault("section_path", chunk_info.get("section_path", ""))
                if not item.get("text"):
                    item["text"] = chunk_info.get("text", "")

        return self._boost_by_fields(sorted_results, domain, query)

    def _boost_by_fields(self, results: list[dict], domain: str, query: str) -> list[dict]:
        field_index = self._load_field_index(domain)
        if not field_index:
            return results

        query_fields: set[str] = set()
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

        for m in date_pattern.finditer(query):
            query_fields.add(m.group())
        for m in amount_pattern.finditer(query):
            query_fields.add(m.group())
        for m in pct_pattern.finditer(query):
            query_fields.add(m.group())

        if not query_fields:
            return results

        all_fields = {**field_index.get("dates", {}), **field_index.get("amounts", {}), **field_index.get("percentages", {})}
        boost = 0.15
        for r in results:
            cid = r["chunk_id"]
            for field_val, chunk_ids in all_fields.items():
                if cid in chunk_ids and field_val in query_fields:
                    r["score"] = r.get("score", 0) + boost
                    boost *= 0.8
        return results

    def _load_field_index(self, domain: str) -> Optional[dict]:
        path = self.config.index_dir / domain / "field_index.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cross_encoder_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return candidates

        try:
            model = self._get_cross_encoder()
            if model is None:
                return candidates
            pairs = [(query, c.get("text", "")) for c in candidates]
            scores = model.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates
        except Exception:
            logger.exception("CrossEncoder rerank failed, returning original order")
            return candidates

    def _fallback_search(self, question: dict, domain: str) -> list[dict]:
        chunks = self._load_domain_chunks(domain)
        if not chunks:
            return []

        doc_ids = question.get("doc_ids", [])
        results: list[dict] = []

        if doc_ids:
            doc_counts: dict[str, int] = {}
            for chunk in chunks:
                if chunk["doc_id"] not in doc_ids:
                    continue
                count = doc_counts.get(chunk["doc_id"], 0)
                if count >= 3:
                    continue
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "section_path": chunk.get("section_path", ""),
                    "score": 0.5,
                    "source": "fallback_head",
                })
                doc_counts[chunk["doc_id"]] = count + 1

        extra_bm25 = self._bm25_search(domain, question.get("question", ""), 2)
        existing_ids = {r["chunk_id"] for r in results}
        for r in extra_bm25:
            if r["chunk_id"] not in existing_ids:
                r["source"] = "fallback_bm25"
                results.append(r)

        return results[:5]

    def _search_multi(self, question: dict, domain: str, doc_ids: Optional[list[str]] = None) -> list[dict]:
        options = question.get("options", {})
        if not options:
            return self.search({**question, "answer_format": ""}) or []

        all_results: dict[str, dict] = {}
        question_text = question.get("question", "")

        with ThreadPoolExecutor(max_workers=min(self.config.retrieval_workers, len(options))) as executor:
            futures = {}
            for letter, option_text in options.items():
                sub_query = f"{question_text} {option_text}"
                futures[executor.submit(self._search_single, question, domain, sub_query, doc_ids)] = letter
            for future in as_completed(futures):
                letter = futures[future]
                try:
                    results = future.result()
                    if results:
                        for r in results:
                            cid = r["chunk_id"]
                            if cid not in all_results or r["score"] > all_results[cid]["score"]:
                                all_results[cid] = r
                except Exception:
                    logger.exception("Multi-search failed for option=%s", letter)

        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        filtered = self._quality_filter(merged, domain, doc_ids)
        supplemented = self._supplement_docs(filtered, domain, doc_ids)
        balanced = self._balance_docs(supplemented)

        difficulty = self._assess_difficulty(question)
        final = self._dynamic_truncate(balanced, difficulty, doc_ids)

        return final

    def _load_domain_chunks(self, domain: str) -> list[dict]:
        if domain in self._chunks_cache:
            return self._chunks_cache[domain]

        chunks_path = self.config.chunks_dir / f"{domain}_chunks.jsonl"
        chunks: list[dict] = []
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
        except Exception:
            logger.exception("Failed to load chunks for domain=%s", domain)
            return []

        self._chunks_cache[domain] = chunks
        self._chunk_map_cache[domain] = {c["chunk_id"]: c for c in chunks}
        return chunks

    def _get_chunk_info(self, domain: str, chunk_id: str) -> Optional[dict]:
        if domain not in self._chunk_map_cache:
            self._load_domain_chunks(domain)
        return self._chunk_map_cache.get(domain, {}).get(chunk_id)

    def _guess_domain(self, question: str) -> Optional[str]:
        profiles = self._load_doc_profiles()
        if not profiles:
            return None

        from rag_system.indexer import jieba_lcut

        question_tokens = set(jieba_lcut(question))
        best_domain: Optional[str] = None
        best_score = 0

        for domain, doc_profiles in profiles.items():
            domain_score = 0
            for doc_id, profile in doc_profiles.items():
                keywords = profile.get("keywords", [])
                for kw in keywords:
                    if kw in question_tokens or kw in question:
                        domain_score += 1
            if domain_score > best_score:
                best_score = domain_score
                best_domain = domain

        return best_domain

    def _load_doc_profiles(self) -> Optional[dict]:
        if self._doc_profiles is not None:
            return self._doc_profiles
        profiles_path = self.config.index_dir / "doc_profiles.json"
        try:
            self._doc_profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load doc_profiles.json")
            self._doc_profiles = {}
        return self._doc_profiles

    def _load_bm25(self, domain: str) -> Optional[dict]:
        if domain in self._bm25_cache:
            return self._bm25_cache[domain]
        bm25_path = self.config.index_dir / domain / "bm25.pkl"
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_cache[domain] = data
            return data
        except Exception:
            logger.exception("Failed to load BM25 for domain=%s", domain)
            return None

    def _load_embedding(self, domain: str) -> Optional[tuple]:
        if domain in self._embedding_cache:
            return self._embedding_cache[domain]
        import faiss

        emb_dir = self.config.index_dir / domain / "embedding"
        try:
            index = faiss.read_index(str(emb_dir / "index.faiss"))
            id_map = json.loads((emb_dir / "id_map.json").read_text(encoding="utf-8"))
            self._embedding_cache[domain] = (index, id_map)
            return (index, id_map)
        except Exception:
            logger.exception("Failed to load embedding for domain=%s", domain)
            return None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            if self.config.skip_embedding:
                return None
            if self.config.cloud_embedding:
                return self._get_cloud_embedder()
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        return self._embedding_model

    def _get_cloud_embedder(self):
        if self._cloud_embedder is None:
            from rag_system.cloud_embedder import CloudEmbedder

            self._cloud_embedder = CloudEmbedder(self.config)
        return self._cloud_embedder

    def _get_cross_encoder(self):
        if self._cross_encoder_failed:
            return None
        if self._cross_encoder is None:
            if self.config.skip_embedding or self.config.cloud_embedding:
                self._cross_encoder_failed = True
                return None
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self.config.reranker_model)
            except Exception:
                logger.warning("CrossEncoder init failed, rerank disabled")
                self._cross_encoder_failed = True
                return None
        return self._cross_encoder

    def _check_doc_ids(self, domain: str, doc_ids: list[str]) -> None:
        chunk_map = self._chunk_map_cache.get(domain)
        if chunk_map is None:
            self._load_domain_chunks(domain)
            chunk_map = self._chunk_map_cache.get(domain, {})
        indexed_docs: set[str] = set()
        for c in chunk_map.values():
            indexed_docs.add(c.get("doc_id", ""))
        missing = [d for d in doc_ids if d not in indexed_docs]
        if missing:
            logger.warning("Target doc_ids not in index domain=%s: %s", domain, missing)

    def _quality_filter(self, results: list[dict], domain: str, doc_ids: Optional[list[str]]) -> list[dict]:
        if doc_ids:
            doc_id_set = set(doc_ids)
            results = [r for r in results if r.get("doc_id") in doc_id_set]

        seen: set[int] = set()
        deduped: list[dict] = []
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            h = hash(r.get("text", ""))
            if h not in seen:
                seen.add(h)
                deduped.append(r)

        min_len = self.config.min_chunk_length
        return [r for r in deduped if len(r.get("text", "")) >= min_len]

    def _supplement_docs(self, results: list[dict], domain: str, doc_ids: Optional[list[str]]) -> list[dict]:
        if not doc_ids:
            return results

        doc_counts: dict[str, int] = {}
        for r in results:
            did = r.get("doc_id", "")
            if did:
                doc_counts[did] = doc_counts.get(did, 0) + 1

        min_per = self.config.min_chunks_per_doc
        chunks = self._load_domain_chunks(domain)
        existing_ids = {r.get("chunk_id") for r in results}

        for did in doc_ids:
            current = doc_counts.get(did, 0)
            if current >= min_per:
                continue
            needed = min_per - current
            added = 0
            for c in chunks:
                if c.get("doc_id") != did:
                    continue
                if c["chunk_id"] in existing_ids:
                    continue
                if len(c.get("text", "")) < self.config.min_chunk_length:
                    continue
                results.append({
                    "chunk_id": c["chunk_id"],
                    "doc_id": c.get("doc_id", did),
                    "text": c.get("text", ""),
                    "section_path": c.get("section_path", ""),
                    "score": 0.35,
                    "source": "supplement",
                })
                existing_ids.add(c["chunk_id"])
                added += 1
                if added >= needed:
                    break

        return results

    def _balance_docs(self, results: list[dict]) -> list[dict]:
        max_per = self.config.max_chunks_per_doc
        doc_buckets: dict[str, list[dict]] = {}
        for r in results:
            did = r.get("doc_id", "") or "__unknown__"
            doc_buckets.setdefault(did, []).append(r)

        balanced: list[dict] = []
        for bucket in doc_buckets.values():
            bucket.sort(key=lambda x: x.get("score", 0), reverse=True)
            balanced.extend(bucket[:max_per])

        balanced.sort(key=lambda x: x.get("score", 0), reverse=True)
        return balanced

    def _assess_difficulty(self, question: dict) -> str:
        doc_ids = question.get("doc_ids", [])
        num_docs = len(doc_ids) if doc_ids else 1
        qtype = question.get("answer_format", "")
        qtext = question.get("question", "")
        options = question.get("options", {})
        num_options = len(options)

        calc_kw = {"计算", "赔付", "金额", "比例", "费率", "应缴", "保费", "赔款", "给付"}
        compare_kw = {"对比", "差异", "区别", "哪些", "分别是", "各自", "分别"}
        hard_kw = {"顺序", "先后", "依次", "流程", "步骤"}

        has_calc = bool(calc_kw & set(qtext))
        has_compare = bool(compare_kw & set(qtext))
        has_hard = bool(hard_kw & set(qtext))

        if num_docs >= 3 or has_hard or (qtype == "multi" and num_options > 3):
            return "hard"
        if num_docs == 2 or qtype == "multi" or has_compare or has_calc:
            return "medium"
        return "easy"

    def _dynamic_truncate(self, results: list[dict], difficulty: str, doc_ids: Optional[list[str]]) -> list[dict]:
        max_map = {
            "easy": self.config.max_chunks_easy,
            "medium": self.config.max_chunks_medium,
            "hard": self.config.max_chunks_hard,
        }
        max_chunks = max_map.get(difficulty, self.config.max_chunks_medium)

        if len(results) <= max_chunks:
            return results

        doc_ids_list = doc_ids or []
        doc_quota: dict[str, int] = {did: 2 for did in doc_ids_list}

        selected: list[dict] = []
        remaining: list[dict] = []
        doc_count: dict[str, int] = {}

        for r in results:
            did = r.get("doc_id", "") or "__unknown__"
            cur = doc_count.get(did, 0)
            quota = doc_quota.get(did, 2)
            if cur < quota:
                selected.append(r)
                doc_count[did] = cur + 1
            else:
                remaining.append(r)

        slots_left = max_chunks - len(selected)
        if slots_left > 0:
            remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
            selected.extend(remaining[:slots_left])

        selected.sort(key=lambda x: x.get("score", 0), reverse=True)
        return selected[:max_chunks]
