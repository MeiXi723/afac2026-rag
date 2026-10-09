"""RAG 评测全链路共用归一化函数。

quote 校验、忠实率判分、eval_v2 数值比较必须引用同一实现，防止口径漂移。
"""
import re
import unicodedata

_PUNCT_MAP = str.maketrans({
    '，': ',', '；': ';', '：': ':', '（': '(', '）': ')',
    '％': '%', '　': ' ', '－': '-', '－': '-',
})


def norm(s: str) -> str:
    """评测归一化：NFKC 全角→半角 + 标点统一 + 去全部空白。

    幂等：norm(norm(x)) == norm(x)。
    """
    if not s:
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.translate(_PUNCT_MAP)
    s = re.sub(r'\s+', '', s)
    return s


def find_chunks_with_quote(quote: str, doc_chunks: dict[str, str]) -> list[str]:
    """在指定文档的归一化 chunk 文本里反查 quote，返回命中的 chunk_id 列表（按块号序）。

    doc_chunks: {chunk_id: 归一化文本}
    """
    q = norm(quote)
    if len(q) < 6:
        return []
    return [cid for cid in sorted(doc_chunks) if q in doc_chunks[cid]]
