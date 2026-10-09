"""S4 协议一：引用核查——evidence_ids 合法性 + 断言支持度（规则化，零 LLM，可复算）。

判分口径：
- invalid_citation: evidence_ids 不全在 bundle chunk_ids 内（硬伤，直接 0 支持分）
- 选项级支持：对答案每个选中字母，取其选项文本，提取"数字实体 + 断言语素"作为
  必查关键词；所引块的归一化文本覆盖全部关键词=1，部分=0.5，全无=0
- faithfulness = 各选中选项支持分的均值
- 与 eval_v2 的对错组合成 2×2 归因矩阵（选项级归因见 --detail）

用法:
  python scripts/eval_faith.py --answers output/brain_answers_a_v2.json --split a
"""
import argparse
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))  # 断言语素表来自 retrieve_dump（P2#4 单一来源）

from rag_system.eval_norm import norm as N  # noqa: E402

# 断言语素表单一来源：retrieve_dump.ASSERTION_MORPHEMES（原两份漂移，P2#4）
from retrieve_dump import ASSERTION_MORPHEMES as MORPHEMES  # noqa: E402
NUM_PAT = re.compile(r'\d+\.?\d*\s*[%％]|[1-9]\d*\.?\d*')  # 数字实体（百分比优先整体取）


def option_keywords(text: str) -> list[str]:
    """选项文本的必查关键词：数字实体 + 命中的断言语素。"""
    kws = [m for m in MORPHEMES if m in text]
    kws += [m.group().replace(' ', '') for m in NUM_PAT.finditer(text)]
    return list(dict.fromkeys(kws))[:8]


def support_score(opt_text: str, cited_texts_norm: list[str]) -> float:
    kws = option_keywords(opt_text)
    if not kws:
        return 1.0  # 无可查关键词的选项视为不可核查，不扣分
    hits = [sum(1 for k in kws if k in ct) for ct in cited_texts_norm]
    best = max(hits) if hits else 0
    if best >= len(kws):
        return 1.0
    if best > 0:
        return 0.5
    return 0.0



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', required=True)
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    ap.add_argument('--bundle-dir', default='')
    ap.add_argument('--detail', action='store_true', help='打印每选项支持明细')
    args = ap.parse_args()

    answers = json.load(open(args.answers, encoding='utf-8'))
    raw = answers.get('answers', answers)
    bundle_dir = args.bundle_dir or os.path.join(BASE, 'output', f'bundles_{args.split}')

    gt = json.load(open(os.path.join(BASE, 'data', f'ground_truth_{args.split}.json'),
                        encoding='utf-8'))['answers']

    from collections import defaultdict
    conf = defaultdict(int)   # 2x2
    per_q, no_bundle, invalid = [], [], []
    for qid, rec in sorted(raw.items()):
        ev_ids = rec.get('evidence_ids') or []
        bpath = os.path.join(bundle_dir, qid + '.json')
        if not os.path.exists(bpath):
            no_bundle.append(qid)
            continue
        b = json.load(open(bpath, encoding='utf-8'))
        chunk_map = {c['chunk_id']: N(c.get('text', '')) for c in b['evidence']}
        cited = [i for i in ev_ids if i in chunk_map]
        if ev_ids and len(cited) < len(ev_ids):
            invalid.append(qid)
        cited_norm = [chunk_map[i] for i in cited]
        got = str(rec.get('answer', '')).strip()
        selected = list(dict.fromkeys(got))  # 逐字母（多选按选项归因）
        scores = []
        detail = []
        for ch in selected:
            opt_text = (b.get('options') or {}).get(ch, '')
            s = support_score(opt_text, cited_norm) if opt_text else 1.0
            scores.append(s)
            detail.append((ch, s))
        faith = sum(scores) / len(scores) if scores else 0.0
        # B 榜数值题用容差判分（与 eval_v2 同口径，否则格式差异被误计为错误）
        # 参数序必须与 eval_v2 一致：(GT, 我方)——token 校验以 GT 为基准，
        # 反序会把我方答案多带的单位（"14.41亿元" vs GT "14.41"）误判为错（2026-10-09 修复）
        gt_ans = str(gt.get(qid, {}).get('answer', ''))
        if args.split == 'b' and not set(gt_ans) <= set('ABCD'):
            correct = _b_value_ok(gt_ans, got)
        else:
            correct = got == gt_ans
        key = ('correct' if correct else 'wrong') + ('_faithful' if faith >= 0.75 else '_unfaithful')
        conf[key] += 1
        per_q.append({'qid': qid, 'answer': got, 'correct': correct,
                      'faith': round(faith, 2), 'quadrant': key,
                      'invalid_citation': ev_ids and len(cited) < len(ev_ids),
                      'detail': detail})

    n = len(per_q) or 1
    print(f'== {args.split.upper()} 忠实率（协议一 引用核查）[{os.path.basename(args.answers)}] ==')
    print(f'Faithfulness  : {sum(q["faith"] for q in per_q) / n:.1%}')
    print(f'引用合法率    : {(n - len(invalid)) / n:.1%}（invalid: {invalid[:8]}）')
    print(f'缺 bundle     : {len(no_bundle)} {no_bundle[:5]}')
    print('2×2 归因矩阵  :')
    print(f"  正确且忠实  : {conf['correct_faithful']:>3}   正确但不忠实(猜中): {conf['correct_unfaithful']:>3}")
    print(f"  错误但忠实  : {conf['wrong_faithful']:>3}   错误且不忠实(幻觉): {conf['wrong_unfaithful']:>3}")
    if args.detail:
        for q in per_q:
            if q['faith'] < 1.0:
                print(f"  {q['qid']} faith={q['faith']} {q['detail']}")

    out = os.path.join(BASE, 'data', 'eval', f'faith_{args.split}.json')
    json.dump({'per_question': per_q, 'invalid': invalid,
               'matrix': dict(conf), 'answers_file': args.answers},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written:', out)


if __name__ == '__main__':
    main()
