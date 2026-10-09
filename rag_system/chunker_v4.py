"""Chunker v4 —— 语义单元打包（域自适应 + 条文/自然段完整 + 合并至目标块幅）。

设计原则（v2/v3 两轮实验教训 + 需求"完整条款为一块/自然段为一块"）：
1. 语义单元不可拆：法规/保险/合同以"条"为最小单元（第X条边界聚合），
   研报/财报以自然段为最小单元——单元内绝不切开（保引文完整）
2. 贪心合并至 448 token（≈v1 512 字符的有效块幅）；超大单元内部按行滑窗
3. title 正文进内容（根因修复）；目录点线行行级过滤；pages 页码溯源
4. 归一化哈希去重 + 短块合并 + prev/next 链接（与 v1 一致）

chunk_id 同形（{domain}_{doc_id}_{i}），落盘 data/chunks_v4/，路径隔离。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

MIN_LEN = 40
_CLAUSE_PAT = re.compile(r'第[一二三四五六七八九十百千零\d]+条')
_TOC_LINE = re.compile(r'\.{3,}|…{2,}')

CHUNK_PROFILES = {
    'insurance': {'unit': 'clause', 'tokens': 448, 'overlap_tokens': 64},
    'regulatory': {'unit': 'clause', 'tokens': 448, 'overlap_tokens': 64},
    'financial_contracts': {'unit': 'clause', 'tokens': 448, 'overlap_tokens': 64},
    'financial_reports': {'unit': 'paragraph', 'tokens': 448, 'overlap_tokens': 64},
    'research': {'unit': 'paragraph', 'tokens': 448, 'overlap_tokens': 64},
}
_TOK = None


def _get_tokenizer():
    global _TOK
    if _TOK is None:
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
    return _TOK


def tlen(text: str) -> int:
    return len(_get_tokenizer()(text, add_special_tokens=False)['input_ids'])


@dataclass
class ChunkV4:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    section_path: str
    pages: list = field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


class ChunkerV4:
    def __init__(self, domain: str):
        self.domain = domain
        self.prof = CHUNK_PROFILES[domain]

    def chunk_document(self, doc_id: str, records: list[dict]) -> list[ChunkV4]:
        # 分节（title 进 path + 正文）
        path: list[str] = []
        sections: list[tuple[str, list[str], list[int]]] = []
        cur_path: list[str] = []
        cur_lines: list[str] = []
        cur_pages: list[int] = []

        for rec in records:
            bt = rec.get('block_type', 'paragraph')
            t = rec.get('text', '')
            pn = rec.get('page_num', -1)
            if pn >= 0 and pn not in cur_pages:
                cur_pages.append(pn)
            if bt == 'title':
                if cur_lines:
                    sections.append((' > '.join(cur_path), cur_lines, cur_pages))
                    cur_lines, cur_pages = [], []
                level = rec.get('heading_level', 1) or 1
                heading = t.lstrip('#').strip()
                cur_path = cur_path[:max(level - 1, 0)] + [heading]
                if t.strip():  # 根因修复：title 进正文
                    cur_lines.append(t.strip())
            else:
                cur_lines.append(t.rstrip())
        if cur_lines:
            sections.append((' > '.join(cur_path), cur_lines, cur_pages))

        # 语义单元 → 贪心合并
        chunks: list[ChunkV4] = []
        for sec_path, sec_lines, sec_pages in sections:
            text = '\n'.join(sec_lines)
            units = self._semantic_units(text)
            for group in self._pack(units):
                chunks.append(ChunkV4(
                    chunk_id='', doc_id=doc_id, domain=self.domain,
                    text=group, section_path=sec_path,
                    pages=sorted(set(sec_pages)),
                ))

        # 去重 + 短合并 + 重编号 + 链接
        return self._postprocess(doc_id, chunks)

    def _semantic_units(self, text: str) -> list[str]:
        lines = [l for l in text.split('\n') if l.strip() and not _TOC_LINE.search(l)]
        if self.prof['unit'] == 'clause':
            units, buf = [], []
            for ln in lines:
                if _CLAUSE_PAT.search(ln) and buf:
                    units.append('\n'.join(buf))
                    buf = [ln]
                else:
                    buf.append(ln)
            if buf:
                units.append('\n'.join(buf))
            return units
        return lines

    def _pack(self, units: list[str]) -> list[str]:
        """贪心合并至 448t；超大单元内部按行滑窗；重叠回带。"""
        budget = self.prof['tokens']
        overlap = self.prof['overlap_tokens']
        out, cur, cur_tok = [], [], 0

        def flush():
            nonlocal cur, cur_tok
            if cur:
                out.append('\n'.join(cur))
                tail, t = [], 0
                for x in reversed(cur):
                    xt = tlen(x)
                    if t + xt > overlap:
                        break
                    tail.insert(0, x)
                    t += xt
                cur, cur_tok = tail, t

        for u in units:
            ut = tlen(u)
            if ut > budget:
                if cur:
                    flush()
                # 超大单元按行滑窗
                sls = u.split('\n')
                sub, st = [], 0
                for sl in sls:
                    slt = tlen(sl)
                    if st + slt > budget and sub:
                        out.append('\n'.join(sub))
                        sub, st = [], 0
                    sub.append(sl)
                    st += slt
                if sub:
                    out.append('\n'.join(sub))
                cur, cur_tok = [], 0
                continue
            if cur_tok + ut > budget:
                flush()
            cur.append(u)
            cur_tok += ut
        if cur:
            out.append('\n'.join(cur))
        return out

    def _postprocess(self, doc_id: str, chunks: list[ChunkV4]) -> list[ChunkV4]:
        # 去重
        seen: set = set()
        ded = []
        for c in chunks:
            h = hash(N(c.text))
            if h not in seen:
                seen.add(h)
                ded.append(c)
        # 短块合并
        merged = []
        for c in ded:
            if len(c.text) >= MIN_LEN or not merged:
                merged.append(c)
            else:
                merged[-1].text += '\n' + c.text
                merged[-1].pages = sorted(set(merged[-1].pages) | set(c.pages))
        # 重编号 + 链接
        for i, c in enumerate(merged):
            c.chunk_id = f'{self.domain}_{doc_id}_{i}'
        for i, c in enumerate(merged):
            c.prev_chunk_id = merged[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = merged[i + 1].chunk_id if i < len(merged) - 1 else None
        return merged


# 导入放底部避免循环
from rag_system.eval_norm import norm as N  # noqa: E402
