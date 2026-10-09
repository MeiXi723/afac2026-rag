"""Score pipeline answers against the v2 ground truth (A+B sets).

Usage:
  python scripts/eval_v2.py                       # score output/answer.csv (A set)
  python scripts/eval_v2.py --answers out.csv --split a
  python scripts/eval_v2.py --split b
"""
import argparse
import csv
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from answer_judge import value_ok as b_value_ok_shared
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt(split):
    path = os.path.join(BASE, 'data', f'ground_truth_{split}.json')
    return json.load(open(path, encoding='utf-8'))['answers']


def load_answers(path):
    answers = {}
    if path.endswith('.json'):
        data = json.load(open(path, encoding='utf-8'))
        raw = data.get('answers', data)
        for qid, rec in raw.items():
            answers[qid] = (rec.get('answer') if isinstance(rec, dict) else str(rec) or '').strip()
    else:
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                answers[row['qid']] = (row.get('answer') or '').strip()
    return answers


def option_f1(gt: str, got: str) -> float:
    g, o = set(gt), set(got)
    if not g or not o:
        return 0.0
    p = len(g & o) / len(o)
    r = len(g & o) / len(g)
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', default=os.path.join(BASE, 'output', 'answer.csv'))
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    args = ap.parse_args()

    gt = load_gt(args.split)
    ans = load_answers(args.answers)

    per_dom = defaultdict(lambda: [0, 0])
    f1s = []
    wrong = []
    empty = 0
    num_pat = re.compile(r'-?\d[\d,]*(?:\.\d+)?\s*%?')  # P1-1：% 入捕获（原版 % 检验死代码）

    for qid, rec in sorted(gt.items()):
        got = ans.get(qid, '')
        g, o = rec['answer'].upper(), got.upper()
        if args.split == 'b' and not set(g) <= set('ABCD'):
            ok = b_value_ok_shared(rec['answer'], got)
        else:
            ok = g == o
        per_dom[rec['domain']][1] += 1
        per_dom[rec['domain']][0] += int(ok)
        if not got:
            empty += 1
        if len(g) > 1 and set(g) <= set('ABCD'):
            f1s.append(option_f1(g, o))
        if not ok:
            wrong.append((qid, rec['domain'], g, got))

    n = len(gt)
    acc = sum(v[0] for v in per_dom.values())
    print(f'== {args.split.upper()} 榜评测: {os.path.basename(args.answers)} ==')
    print(f'总准确率(完全匹配): {acc}/{n} = {acc / n:.1%}')
    if f1s:
        print(f'多选题选项级 F1 均值: {sum(f1s) / len(f1s):.3f}  (n={len(f1s)})')
    print(f'空答案: {empty}')
    print('分领域:')
    for dom in sorted(per_dom):
        m, t = per_dom[dom]
        print(f'  {dom}: {m}/{t} = {m / t:.0%}')
    print(f'答错/分歧 {len(wrong)} 题:')
    for qid, dom, g, got in wrong:
        print(f'  {qid} [{dom}] 真值={g} | 管道={got or "(空)"}')


if __name__ == '__main__':
    sys.exit(main())
