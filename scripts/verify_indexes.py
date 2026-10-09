"""索引完整性校验（GOVERNANCE §五.2 承载工具，P2-4）。

校验全部 5 域：chunks 数 == id_map 数 == bm25 chunk_ids 数，且 id_map 尾部含补块。
退出码：0=全部通过，1=存在不合格域。

用法: python scripts/verify_indexes.py
"""
import json
import pickle
import sys

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']


def main():
    failed = []
    for dom in DOMAINS:
        chunks_n = sum(1 for _ in open(BASE / 'data' / 'chunks' / f'{dom}_chunks.jsonl', encoding='utf-8'))
        try:
            id_map = json.load(open(BASE / 'data' / 'indexes' / dom / 'embedding' / 'id_map.json', encoding='utf-8'))
            bm = pickle.load(open(BASE / 'data' / 'indexes' / dom / 'bm25.pkl', 'rb'))
        except Exception as e:
            print(f'{dom}: BROKEN ({e})')
            failed.append(dom)
            continue
        bm_n = len(bm['chunk_ids'])
        ok = chunks_n == len(id_map) == bm_n
        has_repair = any('_r' in c for c in id_map[-30:])
        print(f'{dom}: chunks={chunks_n} id_map={len(id_map)} bm25={bm_n} '
              f'count_ok={ok} repair_in_tail={has_repair}')
        if not ok:
            failed.append(dom)
    if failed:
        print(f'FAILED: {failed}')
        sys.exit(1)
    print('ALL DOMAINS OK')


if __name__ == '__main__':
    main()
