"""盲判结果三方对照：外部判读 vs 锁定真值 vs 我方当前答案。

用法: python scripts/eval_blind_adjudication.py [--result data/eval/blind_adjudication_result.json]
"""
import argparse
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', default='data/eval/blind_adjudication_result.json')
    args = ap.parse_args()
    res = json.load(open(os.path.join(BASE, args.result), encoding='utf-8'))
    gt = json.load(open(os.path.join(BASE, 'data', 'ground_truth_a.json'),
                        encoding='utf-8'))['answers']
    ours = json.load(open(os.path.join(BASE, 'output', 'brain_answers_a_v2.json'),
                          encoding='utf-8'))['answers']

    print('== 仲裁题（3 道）==')
    for qid, rec in res.get('arbitration', {}).items():
        g = gt.get(qid, {}).get('answer')
        o = ours.get(qid, {}).get('answer')
        print(f'{qid}: 外部={rec.get("answer")} | 锁定GT={g} | 我方={o} '
              f'| 外部vsGT={"一致" if rec.get("answer") == g else "分歧"}')

    blind = res.get('blind', {})
    n = len(blind) or 1
    agree_gt, agree_ours, rows = 0, 0, []
    for qid, rec in sorted(blind.items()):
        ext = str(rec.get('answer', '')).strip()
        g = gt.get(qid, {}).get('answer', '')
        o = str(ours.get(qid, {}).get('answer', '')).strip()
        hit_gt, hit_ours = ext == g, ext == o
        agree_gt += hit_gt
        agree_ours += hit_ours
        rows.append((qid, ext, g, o, hit_gt, hit_ours, rec.get('confidence', '')))

    print(f'\n== 盲检题（{len(blind)} 道）==')
    print(f'外部 vs 锁定GT 一致率 : {agree_gt}/{n} = {agree_gt / n:.0%}  ← GT 质量核心指标')
    print(f'外部 vs 我方答案一致率: {agree_ours}/{n} = {agree_ours / n:.0%}')
    print('\n分歧明细（外部 vs GT）：')
    for qid, ext, g, o, hg, ho, conf in rows:
        if not hg:
            flag = '（我方=GT）' if g == o else '（我方≠GT，双重分歧）'
            print(f'  {qid}: 外部={ext} GT={g} 我方={o} conf={conf} {flag}')
    print('\n分歧明细（外部 vs 我方，但 GT=我方）：')
    for qid, ext, g, o, hg, ho, conf in rows:
        if hg and not ho:
            print(f'  {qid}: 外部={ext} 但我方记录={o}（我方历史答案曾偏离GT，已被修正）')

    out = os.path.join(BASE, 'data', 'eval', 'blind_adjudication_scored.json')
    json.dump({'rows': [{'qid': r[0], 'external': r[1], 'gt': r[2], 'ours': r[3],
                         'agree_gt': r[4], 'agree_ours': r[5]} for r in rows],
               'arbitration': res.get('arbitration', {}),
               'gt_agreement': agree_gt / n if blind else None},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nwritten:', out)


if __name__ == '__main__':
    main()
