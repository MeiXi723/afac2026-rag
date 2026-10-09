"""盲检抽样：从 A 榜 100 题分层抽取 30 题交外部模型独立判读（GT 质量审计）。

分层：域（5 域各 6 题）× 题型（优先保证每域至少 1 题 mcq/tf，其余 multi）
× 置信度尽量覆盖。固定 seed=20261007，可复现。
排除：无（含当年答错的题，盲检本来就该覆盖它们）。
输出: data/eval/blind_sample_qids.json
"""
import json
import os
import random
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

gt = json.load(open(os.path.join(BASE, 'data', 'ground_truth_a.json'), encoding='utf-8'))['answers']
ARBITRATION = {'reg_a_003', 'reg_a_016', 'fc_a_004'}  # 单独仲裁，不进盲检样本

by_dom = defaultdict(list)
for qid, rec in sorted(gt.items()):
    if qid in ARBITRATION:
        continue
    fmt = 'multi' if len(rec.get('answer', '')) > 1 else 'single'
    by_dom[rec['domain']].append({'qid': qid, 'fmt': fmt,
                                  'conf': rec.get('confidence', '?')})

random.seed(20261007)
sample = []
for dom in sorted(by_dom):
    pool = by_dom[dom]
    singles = [x for x in pool if x['fmt'] == 'single']
    multis = [x for x in pool if x['fmt'] == 'multi']
    picks = random.sample(singles, min(2, len(singles)))  # 单选/判断优先抽 2 题
    rest = 6 - len(picks)
    picks += random.sample([m for m in multis if m not in picks], min(rest, len(multis)))
    sample.extend(picks)

out = {'seed': 20261007, 'n': len(sample),
       'by_domain': {d: sum(1 for s in sample if s['qid'] in [x['qid'] for x in by_dom[d]])
                     for d in by_dom},
       'qids': sorted(s['qid'] for s in sample)}
json.dump(out, open(os.path.join(BASE, 'data', 'eval', 'blind_sample_qids.json'),
                    'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
