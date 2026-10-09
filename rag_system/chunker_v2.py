"""Chunker v2 —— M2 主项：类型自适应 + token 预算 + 条文边界 + 行打包表格保护 + 页码溯源。

与 v1（chunker.py，512 字符定步长滑窗）的差异：
1. 按域 profile 设定 token 预算（对齐 bge 嵌入模型截断窗口，外部审查 P0-1）
2. 行边界优先打包：块边界永不落在行中间（表格行不再被拦腰切断，外部审查 P0-2 的可行替代——
   parser 层无结构化表格块，表格以"数字密集行"形态存在，行打包即天然保护）
3. 条文边界保护：reg/ins/fc 域在第 X 条处优先分块（外部审查 P1-2）
4. 财报域数字密集行按 table_rows 行数分片，首片携带区段表头
5. chunk 携带 pages（来源页码列表），修复"bundle 证据无页码溯源"缺口
6. 去重沿用归一化哈希；短块合并、prev/next 链接、重编号与 v1 一致

chunk_id 方案与 v1 同形（{domain}_{doc_id}_{i}），但落盘于 data/chunks_v2/（路径隔离，
禁止与 v1 混载——两套 id 文本同形、内容不同，混用即静默错配，见 ARTIFACTS）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rag_system.config import RAGConfig
from rag_system.eval_norm import norm as N

# 域 profile：tokens=块预算（含重叠）；overlap_tokens=重叠预算；clause=条文边界优先；
# table_rows=数字密集行分片行数（None=不特判）
CHUNK_PROFILES = {
    'insurance': {'tokens': 320, 'overlap_tokens': 48, 'clause': True, 'table_rows': None},
    'regulatory': {'tokens': 288, 'overlap_tokens': 48, 'clause': True, 'table_rows': None},
    'financial_contracts': {'tokens': 384, 'overlap_tokens': 64, 'clause': True, 'table_rows': None},
    'financial_reports': {'tokens': 384, 'overlap_tokens': 64, 'clause': False, 'table_rows': None},  # 2026-10-08: table_rows=18 实测产生 23.5k 微块（均值 172 字符）且窗口覆盖伪影 20%，关停；行打包已天然保护表格行
    'research': {'tokens': 448, 'overlap_tokens': 64, 'clause': False, 'table_rows': None},
}

CLAUSE_RE = re.compile(r'(?=第[一二三四五六七八九十百千零\d]+条)')
MIN_LEN = 40  # 短块合并阈值（字符）


def _get_tokenizer():
    global _TOK
    try:
        return _TOK
    except NameError:
        pass
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
    return _TOK


def tlen(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok(text, add_special_tokens=False)['input_ids'])


@dataclass
class ChunkV2:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    section_path: str
    pages: list = field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


def _is_tabley(line: str) -> bool:
    """数字密集短行 ≈ 表格行。"""
    t = line.strip()
    if not t:
        return False
    digits = sum(ch.isdigit() for ch in t)
    return (digits >= 2 and len(t) <= 60) or t.endswith('%') or bool(re.fullmatch(r'[\d\s.,%亿元万千百-]+', t))


def _split_units(section_text: str, clause: bool) -> list[str]:
    """行级单元；clause=True 时按"第 X 条"边界聚合成条文单元。"""
    lines = section_text.split('\n')
    if not clause:
        return lines
    units: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if CLAUSE_RE.match(ln.strip()) and buf:
            units.append('\n'.join(buf))
            buf = [ln]
        else:
            buf.append(ln)
    if buf:
        units.append('\n'.join(buf))
    return units


def _pack(units: list[str], budget: int, overlap: int, table_rows: Optional[int]) -> list[list[str]]:
    """贪心打包单元到 token 预算；带重叠尾部；数字密集连续行按 table_rows 分片。"""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    i = 0
    while i < len(units):
        u = units[i]
        run = []
        if table_rows and _is_tabley(u):
            j = i
            while j < len(units) and _is_tabley(units[j]) and len(run) < table_rows:
                run.append(units[j])
                j += 1
            if len(run) >= 3:  # 成片：整片出块（携带前一行做伪表头由调用方保证）
                if cur:
                    chunks.append(cur)
                    # 重叠：尾部单元回带
                    tail, t = [], 0
                    for x in reversed(cur):
                        tx = tlen(x)
                        if t + tx > overlap:
                            break
                        tail.insert(0, x)
                        t += tx
                    cur, cur_tok = tail, t
                chunks.append(run)
                i = j
                continue
        ut = tlen(u)
        if ut > budget:  # 超长单元：硬切（按行再切不了时按字符比例）
            pieces = max(1, -(-ut // (budget - overlap)))
            k = max(1, len(u) // pieces)
            for s in range(0, len(u), k):
                piece = u[s:s + k]
                pt = tlen(piece)
                if cur_tok + pt > budget and cur:
                    chunks.append(cur)
                    cur, cur_tok = [], 0
                cur.append(piece)
                cur_tok += pt
            i += 1
            continue
        if cur_tok + ut > budget and cur:
            # 重叠：尾部单元回带
            tail, t = [], 0
            for x in reversed(cur):
                tx = tlen(x)
                if t + tx > overlap:
                    break
                tail.insert(0, x)
                t += tx
            chunks.append(cur)
            cur, cur_tok = tail, t
        cur.append(u)
        cur_tok += ut
        i += 1
    if cur:
        chunks.append(cur)
    return chunks


class ChunkerV2:
    def __init__(self, config: RAGConfig, domain: str):
        self.config = config
        self.domain = domain
        self.prof = CHUNK_PROFILES[domain]

    def chunk_document(self, doc_id: str, records: list[dict]) -> list[ChunkV2]:
        sections = self._build_sections(records)
        chunks: list[ChunkV2] = []
        idx = 0
        budget = self.prof['tokens']
        overlap = self.prof['overlap_tokens']
        for section_path, sec_records in sections:
            text = '\n'.join(r['text'] for r in sec_records)
            pages = sorted({r.get('page_num', -1) for r in sec_records})
            units = _split_units(text, self.prof['clause'])
            for group in _pack(units, budget, overlap, self.prof['table_rows']):
                chunks.append(ChunkV2(
                    chunk_id=f'{self.domain}_{doc_id}_{idx}',
                    doc_id=doc_id, domain=self.domain,
                    text='\n'.join(group), section_path=section_path,
                    pages=[p for p in pages if p >= 0],
                ))
                idx += 1
        chunks = self._postprocess(chunks)
        self._link(chunks)
        return chunks

    def _build_sections(self, records: list[dict]) -> list[tuple[str, list[dict]]]:
        """标题分节（与 v1 同逻辑），但保留块引用以便携带页码。"""
        sections: list[tuple[str, list[dict]]] = []
        path: list[str] = []
        buf: list[dict] = []
        for rec in records:
            if rec.get('block_type') == 'title':
                if buf:
                    sections.append((' > '.join(path), buf))
                    buf = []
                level = rec.get('heading_level', 1) or 1
                heading = rec['text'].lstrip('#').strip()
                path = path[:max(level - 1, 0)] + [heading]
            else:
                buf.append(rec)
        if buf:
            sections.append((' > '.join(path), buf))
        return sections

    def _postprocess(self, chunks: list[ChunkV2]) -> list[ChunkV2]:
        # 短块合并
        out: list[ChunkV2] = []
        for c in chunks:
            if len(c.text) >= MIN_LEN or not out:
                out.append(c)
            else:
                out[-1].text += '\n' + c.text
                out[-1].pages = sorted(set(out[-1].pages) | set(c.pages))  # P2-7 provenance 保留
        # 归一化哈希去重（P1-7 同款）
        seen: set[int] = set()
        ded: list[ChunkV2] = []
        for c in out:
            h = hash(N(c.text))
            if h not in seen:
                seen.add(h)
                ded.append(c)
        # 重编号
        for i, c in enumerate(ded):
            c.chunk_id = f'{self.domain}_{c.doc_id}_{i}'
        return ded

    @staticmethod
    def _link(chunks: list[ChunkV2]) -> None:
        for i, c in enumerate(chunks):
            c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
