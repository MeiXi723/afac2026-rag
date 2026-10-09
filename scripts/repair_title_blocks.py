"""修复 parser title 误标导致的条款内容丢失（M2 主项发现的系统性 bug）。

机制：parser 将部分带正文的条款块误标为 block_type='title'，chunker 分节时
title 只进 section_path 不进正文 → 内容整块丢失。本脚本扫描全部 parsed 块，
把"窗口覆盖 <0.7 且非纯目录"的 title 内容块以 append-only 方式补回
（chunk_id = {domain}_{doc_id}_rt{k}，pages 保留）。

用法: python scripts/repair_title_blocks.py [--domain insurance] [--apply]
"""
import argparse
import glob
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
MIN_CHARS = 40
TOC = re.compile(r'\.{6,}')
DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    domains = [args.domain] if args.domain else DOMAINS

    total = 0
    for dom in domains:
        chunk_path = os.path.join(BASE, 'data', 'chunks', f'{dom}_chunks.jsonl')
        chunks = [json.loads(l) for l in open(chunk_path, encoding='utf-8')]
        doc_norm = defaultdict(str)
        for c in chunks:
            doc_norm[c['doc_id']] += N(c.get('text', '')) + '\uffff'

        blocks_by_doc = defaultdict(list)
        for pf in glob.glob(os.path.join(BASE, 'data', 'parsed', dom, '*.jsonl')):
            for line in open(pf, encoding='utf-8'):
                b = json.loads(line)
                blocks_by_doc[b.get('doc_id', '')].append(b)

        appended = []
        existing_ids = {c['chunk_id'] for c in chunks}
        for doc_id in sorted(blocks_by_doc):
            k = 0
            for cid in existing_ids:
                if f'_rt' in cid and cid.rsplit('_rt',1)[0] == f'{dom}_{doc_id}':
                    try: k = max(k, int(cid.rsplit('_rt',1)[1]))
                    except ValueError: pass
            for b in blocks_by_doc[doc_id]:
                t = b.get('text', '')
                tn = N(t)
                if len(tn) < MIN_CHARS or TOC.search(t):
                    continue
                wins = [tn[i:i + SH] for i in range(0, max(len(tn) - SH + 1, 1), 8)]
                hay = doc_norm[doc_id]
                cov = sum(1 for w in wins if w in hay) / len(wins)
                if cov >= THR:
                    continue
                # 仅补"带正文的条款标题"（title 块含句号/数字/较长），纯词条目已过滤
                k += 1
                assert f'{dom}_{doc_id}_rt{k}' not in {c['chunk_id'] for c in appended}, 'id collision'
                appended.append({
                    'chunk_id': f'{dom}_{doc_id}_rt{k}' if not doc_id.startswith(dom) else f'{doc_id}_rt{k}',
                    'doc_id': doc_id, 'domain': dom,
                    'section_path': f"title_repair(p{b.get('page_num')})",
                    'text': t, 'prev_chunk_id': None, 'next_chunk_id': None,
                    'pages': [b.get('page_num', -1)],
                })
                doc_norm[doc_id] += tn + '￿'  # P1-2: 回写 dict
        if appended:
            print(f'{dom}: 补 title 内容块 {len(appended)}（{sum(len(c["text"]) for c in appended)} 字符）')
            total += len(appended)
            if args.apply:
                with open(chunk_path, 'a', encoding='utf-8') as f:
                    for c in appended:
                        f.write(json.dumps(c, ensure_ascii=False) + '\n')
        else:
            print(f'{dom}: 无需补块')
    if args.apply:
        print('APPLIED（重建受影响域索引后生效）')


if __name__ == '__main__':
    main()
