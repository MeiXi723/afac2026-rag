"""合并 B 榜盲答批次 → output/brain_answers_b_v1.json（含格式校验）。"""
import glob
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
merged = {}
problems = []
for p in sorted(glob.glob(os.path.join(BASE, 'output', 'brain_batch_b', 'b*.json'))):
    for qid, rec in json.load(open(p, encoding='utf-8')).items():
        if qid in merged:
            problems.append(f'dup {qid}')
        a = str(rec.get('answer', '')).strip()
        if not re.fullmatch(r'[A-D]{1,4}|[^A-D]{1,}', a) and a:
            pass  # B 榜允许数值串，仅记录空答案
        merged[qid] = rec
for qid in merged:
    if not str(merged[qid].get('answer', '')).strip():
        problems.append(f'empty {qid}')
out = os.path.join(BASE, 'output', 'brain_answers_b_v1.json')
json.dump({'answers': merged, '_meta': {'n': len(merged), 'source': 'blind batches (v7 chain)'}},
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'merged {len(merged)} | problems: {problems[:5] if problems else "无"}')
print('written:', out)
