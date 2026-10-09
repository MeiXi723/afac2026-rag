"""Chunker v3 —— 激进根因重切（M2 主项终版方案）。

v3 = v1 的被验证行为（~512 级大块、行边界、全量内容）+ v2 的工程改进（token 实测、
域 profile、pages 溯源）+ **title 正文进内容**（根因修复：parser 误标 title 的条款块
不再丢失——v1 由此蒸发 1,158 块 9.4 万字符，靠 1,158 个 _rt 补丁事后补救）。

与 v2 的差异（v2 探索失败回退的教训）：
- 不做条文硬边界聚合（v2 实测把决定性引文切碎，引文覆盖 -16.4pp）
- 不做 table_rows 特判（v2 实测微块化 + 覆盖伪影）
- 预算对齐 v1 有效块幅（448 tokens ≈ v1 的 512 字符行为），不缩小

chunk_id 方案与 v1 同形（{domain}_{doc_id}_{i}），落盘 data/chunks_v3/，路径隔离。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rag_system.eval_norm import norm as N

MIN_LEN = 40
TOC = re.compile(r'\.{6,}')

# 域 profile：统一 448 tokens（≈v1 512 字符的 token 当量），重叠 64
CHUNK_PROFILES = {
    'insurance': {'tokens': 448, 'overlap_tokens': 64},
    'regulatory': {'tokens': 448, 'overlap_tokens': 64},
    'financial_contracts': {'tokens': 448, 'overlap_tokens': 64},
    'financial_reports': {'tokens': 448, 'overlap_tokens': 64},
    'research': {'tokens': 448, 'overlap_tokens': 64},
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
class ChunkV3:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    section_path: str
    pages: list = field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


class ChunkerV3:
    """v3：title 正文进内容 + 行贪心 token 打包 + 全量内容保留。"""

    def __init__(self, config, domain: str):
        self.config = config
        self.domain = domain
        self.budget = CHUNK_PROFILES[domain]['tokens']
        self.overlap = CHUNK_PROFILES[domain]['overlap_tokens']

    def chunk_document(self, doc_id: str, records: list[dict]) -> list[ChunkV3]:
        chunks: list[ChunkV3] = []
        path: list[str] = []
        lines: list[str] = []   # 当前节的行缓冲（含 title 行——根因修复）
        pages: list[int] = []

        def flush():
            nonlocal lines, pages
            if not lines:
                return
            text = '\n'.join(lines)
            for group in self._pack_lines(text):
                chunks.append(ChunkV3(
                    chunk_id='',
                    doc_id=doc_id, domain=self.domain,
                    text=group, section_path=' > '.join(path), pages=sorted(set(pages)),
                ))
            lines, pages = [], []

        for rec in records:
            bt = rec.get('block_type', 'paragraph')
            t = rec.get('text', '')
            pn = rec.get('page_num', -1)
            if pn >= 0 and pn not in pages:
                pages.append(pn)
            if bt == 'title':
                flush()
                level = rec.get('heading_level', 1) or 1
                heading = t.lstrip('#').strip()
                path = path[:max(level - 1, 0)] + [heading]
                # 根因修复：title 文本作为正文行保留（条款内容常被误标 title）。
                # P1-1：目录点线块过滤（NFKC 归一后匹配；全角点/中文省略号一并拦）
                tn = N(t)
                if len(tn) < MIN_LEN and not re.search(r'[一-龥]{4,}', tn):
                    continue  # 纯词条目行，无正文
                if re.search(r'\.{3,}|…{2,}', tn) and len(tn) < 80:
                    continue  # 目录条目（点线+短）
                if t.strip():
                    lines.append(t.strip())
                continue
            lines.append(t.rstrip())
        flush()

        # 重编号 + 去重 + 短合并
        ded: list[ChunkV3] = []
        seen: set = set()
        buf: list[ChunkV3] = []
        for c in chunks:
            c.chunk_id = f'{self.domain}_{doc_id}_{len(ded) + len(buf)}'
            if len(N(c.text)) < MIN_LEN and buf:
                buf[-1].text += '\n' + c.text
                buf[-1].pages = sorted(set(buf[-1].pages) | set(c.pages))
                continue
            h = hash(N(c.text))
            if h in seen:
                continue
            seen.add(h)
            buf.append(c)
        for i, c in enumerate(buf):
            c.chunk_id = f'{self.domain}_{doc_id}_{i}'
        ded = buf
        for i, c in enumerate(ded):
            c.prev_chunk_id = ded[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = ded[i + 1].chunk_id if i < len(ded) - 1 else None
        return ded

    def _pack_lines(self, text: str) -> list[str]:
        """行贪心 token 打包：行边界优先，单行超预算才硬切。目录点线行零检索价值，行级过滤。"""
        import re as _re
        budget = self.budget
        lines = [l for l in text.split('\n')
                 if l.strip() and not _re.search(r'\.{3,}|…{2,}', l)]
        out, cur, cur_tok = [], [], 0
        for ln in lines:
            lt = tlen(ln)
            if lt > budget:
                if cur:
                    out.append('\n'.join(cur))
                    cur, cur_tok = [], 0
                # 硬切超长行（字符比例近似）
                step = max(int(budget * 2.2), 1)
                for s in range(0, len(ln), step):
                    out.append(ln[s:s + step])
                continue
            if cur_tok + lt > budget and cur:
                # 重叠：尾部行回带至 overlap 预算
                tail, t = [], 0
                for x in reversed(cur):
                    xt = tlen(x)
                    if t + xt > self.overlap:
                        break
                    tail.insert(0, x)
                    t += xt
                out.append('\n'.join(cur))
                cur, cur_tok = tail, t
            cur.append(ln)
            cur_tok += lt
        if cur:
            out.append('\n'.join(cur))
        return out
