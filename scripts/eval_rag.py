"""S3: 检索层评测——对 bundles 计算 Recall/Hit@5/MRR/噪声率/冗余度。

用法: python scripts/eval_rag.py --split a [--k 5] [--bundle-dir output/bundles_a_v2]
依赖: data/eval/golden_chunks_{split}.json（build_golden_chunks.py 产出）+ bundle 目录
"""
import argparse
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--bundle-dir', default='',
                    help='bundle 目录（默认 output/bundles_{split}；S6 对比变体时指向变体目录）')
    ap.add_argument('--out', default='', help='结果落盘路径（默认 data/eval/rag_baseline_{split}.json）')
    ap.add_argument('--golden', default='', help='覆盖默认 golden 文件路径（chunkset v2 用 golden_chunks_a_v2.json）')
    args = ap.parse_args()

    golden_path = args.golden or os.path.join(BASE, 'data', 'eval', f'golden_chunks_{args.split}.json')
    golden = json.load(open(golden_path, encoding='utf-8'))
    bundle_dir = args.bundle_dir or os.path.join(BASE, 'output', f'bundles_{args.split}')

    per_q = []
    unlabeled = []
    for path in sorted(glob := __import__('glob').glob(os.path.join(bundle_dir, '*.json'))):
        b = json.load(open(path, encoding='utf-8'))
        qid = b['qid']
        g = golden.get('questions', {}).get(qid)
        if not g or not g.get('decisive'):
            unlabeled.append(qid)
            continue
        gset = set(g['decisive'])
        ev = b['evidence']
        ev_ids = [c['chunk_id'] for c in ev]

        recall = len(gset & set(ev_ids)) / len(gset)
        hit5 = int(any(c['chunk_id'] in gset for c in ev[:args.k]))
        # MRR：只数锚块（neighbor 不占位）
        rank, mrr = None, 0.0
        n_anchor = 0
        for c in ev:
            if c.get('neighbor'):
                continue
            n_anchor += 1
            if c['chunk_id'] in gset and rank is None:
                rank = n_anchor
                mrr = 1.0 / n_anchor
        # 口径 v2（审查 P1-4）：分母=锚点数（全部非邻接块，含 guaranteed）。
        # 注意：分子排除 guaranteed 而分母不排除 → 保底越激进噪声率越好看，
        # 属已知特性；口径戳见输出 noise_version 字段
        n_anchor_q = sum(1 for c in ev if not c.get('neighbor'))
        noise = sum(1 for c in ev if not c.get('neighbor') and not c.get('guaranteed')
                    and c['chunk_id'] not in gset) / max(n_anchor_q, 1)
        chars = sum(len(c.get('text', '')) for c in ev)
        per_q.append({'qid': qid, 'domain': b.get('domain', ''),
                      'recall': recall, 'hit5': hit5, 'mrr': mrr,
                      'noise': noise, 'n_chunks': len(ev), 'n_chars': chars,
                      'golden_n': len(gset)})

    def macro(key):
        vals = [q[key] for q in per_q]
        return sum(vals) / len(vals) if vals else 0.0

    print(f'== {args.split.upper()} 榜检索层基线（golden 已标注 {len(per_q)}/{len(golden.get("questions", {}))} 题，'
          f'未标注 {len(unlabeled)} 题跳过）[{bundle_dir}] ==')
    print(f'Recall@bundle : {macro("recall"):.1%}')
    print(f'Hit@{args.k}          : {sum(q["hit5"] for q in per_q)}/{len(per_q)} = '
          f'{sum(q["hit5"] for q in per_q) / len(per_q):.1%}')
    print(f'MRR(锚块序)    : {macro("mrr"):.3f}')
    print(f'噪声率         : {macro("noise"):.1%}')
    print(f'冗余度         : 块数均值 {macro("n_chunks"):.0f} | 字符均值 {macro("n_chars"):.0f}')

    per_dom = defaultdict(lambda: defaultdict(list))
    for q in per_q:
        per_dom[q['domain']]['recall'].append(q['recall'])
        per_dom[q['domain']]['mrr'].append(q['mrr'])
        per_dom[q['domain']]['hit5'].append(q['hit5'])
    print('分域:')
    for dom in sorted(per_dom):
        d = per_dom[dom]
        r = d['recall']
        print(f"  {dom}: n={len(r)} recall={sum(r)/len(r):.0%} hit@{args.k}="
              f"{sum(d['hit5'])/len(r):.0%} mrr={sum(d['mrr'])/len(d['mrr']):.2f}")

    weak = sorted(per_q, key=lambda q: (q['recall'], q['mrr']))[:10]
    print('最弱 10 题（检索迭代靶点）:')
    for q in weak:
        print(f"  {q['qid']} recall={q['recall']:.0%} mrr={q['mrr']:.2f} golden={q['golden_n']}")

    out = args.out or os.path.join(BASE, 'data', 'eval', f'rag_baseline_{args.split}.json')
    json.dump({'per_question': per_q, 'unlabeled': unlabeled, 'k': args.k,
               'bundle_dir': bundle_dir, 'noise_version': 'v2_anchors_incl_guaranteed'},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written:', out)


if __name__ == '__main__':
    main()
