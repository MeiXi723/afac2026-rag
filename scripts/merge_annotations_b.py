"""B 榜 golden 合并：标注引文 → chunks_v1 反查（多片段指纹匹配，与 A 版同款口径）。

用法: python scripts/merge_annotations_b.py
"""
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_p = os.path.join(BASE, 'scripts')
if sys_p not in sys.path:
    sys.path.insert(0, sys_p)

from rag_system.eval_norm import norm as N  # noqa: E402

DOM = {'fc': 'financial_contracts', 'fin': 'financial_reports', 'ins': 'insurance',
       'reg': 'regulatory', 'res': 'research'}
_CH = {}


def _ch(dom):
    if dom not in _CH:
        _CH[dom] = [(c['chunk_id'], c.get('doc_id', ''), N(c.get('text', '')))
                    for line in open(f'{BASE}/data/chunks/{dom}_chunks.jsonl', encoding='utf-8')
                    for c in [json.loads(line)]]
    return _CH[dom]


def quote_to_chunks(quote, dom, hint):
    qn = N(quote)
    runs = sorted((r for r in re.split(r'…|。|；|，|、', qn) if len(r) >= 8), key=len, reverse=True)
    for r in runs[:3]:
        hits = [cid for cid, did, t in _ch(dom) if (not hint or hint in did) and r in t]
        if hits:
            return hits[:6]
    return []


def main():
    gp = os.path.join(BASE, 'data', 'eval', 'golden_chunks_b.json')
    golden = json.load(open(gp, encoding='utf-8'))
    G = golden['questions']
    stats = {'added': 0, 'quotes_ok': 0, 'quotes_miss': 0}

    for path in sorted(glob.glob(os.path.join(BASE, 'output', 'brain_batch', 'annotate_b_*.json'))):
        batch = json.load(open(path, encoding='utf-8'))
        for qid, rec in batch.items():
            dom = DOM[qid.split('_')[0]]
            dec = []
            for sq in rec.get('support_quotes', []):
                q = sq.get('quote', '').strip()
                if not q:
                    stats['quotes_miss'] += 1
                    continue
                hits = quote_to_chunks(q, dom, sq.get('doc_hint'))
                if hits:
                    stats['quotes_ok'] += 1
                    dec.extend(hits)
                else:
                    stats['quotes_miss'] += 1
            if dec:
                entry = G.setdefault(qid, {'decisive': [], 'exclusion': [], 'ambiguous': [],
                                           'unmatched': [], 'doc_ids': []})
                newd = set(entry['decisive']) | set(dec)
                entry['decisive'] = sorted(newd)
                entry['source'] = 'agent_annotation_b_20261008'
                stats['added'] += 1

    golden['questions'] = G
    golden['_meta']['b_annotated'] = (f"{stats['added']} 题代理标注转入，"
                                      f"引文反查 {stats['quotes_ok']} 条命中 / {stats['quotes_miss']} 条未命中")
    json.dump(golden, open(gp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    cov = sum(1 for v in G.values() if v.get('decisive'))
    print(f"golden_b decisive 覆盖: {cov}/100 | 新增题 {stats['added']} | 引文命中 {stats['quotes_ok']} / 未命中 {stats['quotes_miss']}")


if __name__ == '__main__':
    main()
