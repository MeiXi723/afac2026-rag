"""切块覆盖修复：把 parsed 里有但 chunks 里缺的块补成附加 chunk（append-only）。

判定：文档字符覆盖 <97% 触发；块级 30 字窗覆盖 <0.7 视为缺失；
目录点线条目（'......'）跳过。新 chunk_id = {doc_id}_r{n}，prev/next 置空。
修复后需对该域重建 BM25 + embedding（rebuild_embedding_local.py / 全量 indexer）。

用法: python scripts/repair_chunk_coverage.py [--domain research] [--apply]
  默认 dry-run 只打印；--apply 才写 chunks jsonl。
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from rag_system.eval_norm import norm as N  # noqa: E402

SH = 30
THR = 0.7
TOC = re.compile(r'\.{6,}')

# 2026-10-07 覆盖审计触发的修复范围（data/eval/chunk_coverage_*.json）
TRIGGER_DOCS = {
    'financial_contracts': ['text04'],
    'insurance': ['6', '7', '9', '10', '11', '14', '16'],
    'research': ['pack2_text01'],
}
# regulatory 域无条件全量触发（strict_v3 实测字符覆盖 47~69%，逐块窗覆盖二次把关）
REG_ALWAYS = True
MIN_CHARS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    domains = ([args.domain] if args.domain
               else ['insurance', 'financial_contracts', 'regulatory', 'research'])

    for dom in domains:
        chunk_path = os.path.join(BASE, 'data', 'chunks', f'{dom}_chunks.jsonl')
        chunks = [json.loads(l) for l in open(chunk_path, encoding='utf-8')]
        doc_norm = defaultdict(str)
        for c in chunks:
            doc_norm[c['doc_id']] += N(c.get('text', '')) + '\uffff'

        import glob
        parsed_files = glob.glob(os.path.join(BASE, 'data', 'parsed', dom, '*.jsonl'))
        blocks_by_doc = defaultdict(list)
        for pf in parsed_files:
            for line in open(pf, encoding='utf-8'):
                b = json.loads(line)
                blocks_by_doc[b.get('doc_id', '')].append(b)

        existing_ids = {c['chunk_id'] for c in chunks}
        appended = []
        for doc_id, blocks in sorted(blocks_by_doc.items()):
            triggered = (doc_id in TRIGGER_DOCS.get(dom, [])
                         or (dom == 'regulatory' and REG_ALWAYS))
            if not triggered:
                continue
            k = 0
            for cid in existing_ids:
                if '_r' in cid and cid.rsplit('_r', 1)[0] == doc_id:
                    try:
                        k = max(k, int(cid.rsplit('_r', 1)[1]))
                    except ValueError:
                        pass
            for b in blocks:
                t = b.get('text', '')
                tn = N(t)
                if len(tn) < MIN_CHARS or TOC.search(t):
                    continue
                wins = [tn[i:i + SH] for i in range(0, max(len(tn) - SH + 1, 1), 8)]
                cov = sum(1 for w in wins if w in doc_norm[doc_id]) / len(wins)
                if cov >= THR:
                    continue
                k += 1
                appended.append({
                    'chunk_id': f'{doc_id}_r{k}',
                    'doc_id': doc_id,
                    'domain': dom,
                    'section_path': f"repaired(p{b.get('page_num')})",
                    'text': t,
                    'prev_chunk_id': '',
                    'next_chunk_id': '',
                })
                doc_norm[doc_id] += tn + '\uffff'  # 相邻缺失块不重复补
        print(f'== {dom}: 补块 {len(appended)}'
              + (f'，字符 {sum(len(c["text"]) for c in appended)}' if appended else ''))
        for c in appended[:5]:
            print(f"  [{c['chunk_id']}] {c['text'][:60]!r}")
        if args.apply and appended:
            with open(chunk_path, 'a', encoding='utf-8') as f:
                for c in appended:
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
            print(f'  APPLIED -> {chunk_path} (+{len(appended)} chunks)')


if __name__ == '__main__':
    main()
