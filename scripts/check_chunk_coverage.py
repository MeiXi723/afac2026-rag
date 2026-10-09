"""切块覆盖审计 v2：parsed 块文本是否被子串包含于 chunks 全文（归一化后）。

判定：块归一化文本的 30 字滑窗（步长 8）≥70% 能在"该文档全部 chunk 归一化
拼接串"中找到 → 覆盖。对页眉/页脚插入 chunk 中部的情况稳健（窗级判定）。
输出 data/eval/chunk_coverage_{domain}.json。

用法: python scripts/check_chunk_coverage.py [--domain research]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from rag_system.eval_norm import norm as N  # noqa: E402

SH = 30
THR = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='')
    args = ap.parse_args()
    domains = ([args.domain] if args.domain
               else ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research'])

    for dom in domains:
        doc_norm: dict[str, str] = defaultdict(str)
        chunk_path = os.path.join(BASE, 'data', 'chunks', f'{dom}_chunks.jsonl')
        for line in open(chunk_path, encoding='utf-8'):
            c = json.loads(line)
            doc_norm[c['doc_id']] += N(c.get('text', '')) + '\uffff'  # 块间哨兵防跨块假窗

        parsed_files = glob.glob(os.path.join(BASE, 'data', 'parsed', dom, '*.jsonl'))
        dropped, total_blocks = [], 0
        for pf in parsed_files:
            for line in open(pf, encoding='utf-8'):
                blk = json.loads(line)
                t = blk.get('text', '')
                tn = N(t)
                if len(tn) < 40:
                    continue
                total_blocks += 1
                windows = [tn[i:i + SH] for i in range(0, max(len(tn) - SH + 1, 1), 8)]
                hay = doc_norm.get(blk.get('doc_id', ''), '')
                cov = sum(1 for w in windows if w in hay) / len(windows)
                if cov < THR:
                    dropped.append({'doc_id': blk.get('doc_id', ''),
                                    'page': blk.get('page_num'),
                                    'type': blk.get('block_type'),
                                    'cov': round(cov, 2),
                                    'n_chars': len(t),
                                    'text_head': t[:80].replace('\n', ' ')})
        by_doc = defaultdict(int)
        for d in dropped:
            by_doc[d['doc_id']] += 1
        print(f'== {dom}: parsed 块 {total_blocks}，丢失 {len(dropped)} '
              f'({len(dropped) / max(total_blocks, 1):.0%}) ==')
        for doc_id, n in sorted(by_doc.items(), key=lambda x: -x[1])[:8]:
            print(f'  {doc_id}: {n}')
        for d in dropped[:8]:
            print(f"    p{d['page']} [{d['type']}] cov={d['cov']} chars={d['n_chars']} {d['text_head']!r}")
        out = os.path.join(BASE, 'data', 'eval', f'chunk_coverage_{dom}.json')
        json.dump({'dropped': dropped, 'total_blocks': total_blocks},
                  open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('  written:', out)


if __name__ == '__main__':
    main()
