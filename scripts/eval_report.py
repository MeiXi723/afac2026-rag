"""S5: 三层汇总报表——检索层(RAG) + 生成层(Accuracy) + 忠实层(Faith) + delta 追踪。

每次迭代产出一份带版本指纹的报表，与上一份（同 split 最新）对比给绝对 pp delta。
版本指纹：PROFILES 哈希 + 关键脚本 sha256（可复现性，回应外部审查 P2-4）。

用法:
  python scripts/eval_report.py --split a --label hybrid_v2 \
      --answers output/brain_answers_a_v2.json --bundle-dir output/bundles_a_v2
"""
import argparse
import hashlib
import json
import os
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from answer_judge import answer_ok
from collections import defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))


def script_sha(path: str) -> str:
    p = os.path.join(BASE, path)
    if not os.path.exists(p):
        return 'missing'
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]


def accuracy_layer(answers_path: str, split: str):
    from eval_v2 import load_gt, load_answers  # 复用同一口径
    gt = load_gt(split)
    got = load_answers(answers_path)
    acc, f1s, per_dom = 0, [], defaultdict(lambda: [0, 0])
    per_q = {}
    for qid, entry in gt.items():
        g = entry.get('answer', '')
        o = got.get(qid, '').strip()
        hit = answer_ok(g, o, split)  # 2026-10-09：全等→统一判分（B 榜数值容差），修复 85% 假阴性
        acc += hit
        dom = entry.get('domain', '?')
        per_dom[dom][0] += hit
        per_dom[dom][1] += 1
        if len(g) > 1 and len(o) > 1:
            gs, os_ = set(g), set(o)
            f1s.append(2 * len(gs & os_) / (len(gs) + len(os_)) if gs | os_ else 1.0)
        per_q[qid] = hit
    n = len(gt)
    return {'accuracy': acc / n, 'n': n,
            'multi_f1': sum(f1s) / len(f1s) if f1s else None,
            'per_domain': {d: {'acc': v[0] / v[1], 'n': v[1]}
                           for d, v in per_dom.items()}}, per_q


def retrieval_layer(bundle_dir: str, split: str, k: int = 5):
    golden = json.load(open(os.path.join(BASE, 'data', 'eval',
                                         f'golden_chunks_{split}.json'), encoding='utf-8'))
    per_q = []
    for name in os.listdir(bundle_dir):
        if not name.endswith('.json'):
            continue
        b = json.load(open(os.path.join(bundle_dir, name), encoding='utf-8'))
        g = golden.get('questions', {}).get(b['qid'])
        if not g or not g.get('decisive'):
            continue
        gset = set(g['decisive'])
        ev = b['evidence']
        ids = [c['chunk_id'] for c in ev]
        n_anchor, mrr = 0, 0.0
        for c in ev:
            if c.get('neighbor'):
                continue
            n_anchor += 1
            if c['chunk_id'] in gset and mrr == 0.0:
                mrr = 1.0 / n_anchor
        per_q.append({'qid': b['qid'],
                      'recall': len(gset & set(ids)) / len(gset),
                      'mrr': mrr,
                      'noise': sum(1 for c in ev if not c.get('neighbor')
                                   and not c.get('guaranteed')
                                   and c['chunk_id'] not in gset)
                              / max(sum(1 for c in ev if not c.get('neighbor')), 1),
                      'n_chunks': len(ev),
                      'n_chars': sum(len(c.get('text', '')) for c in ev)})
    n = len(per_q) or 1
    return {'n_golden': len(per_q),
            'recall': sum(q['recall'] for q in per_q) / n,
            'mrr': sum(q['mrr'] for q in per_q) / n,
            'avg_chunks': sum(q['n_chunks'] for q in per_q) / n,
            'avg_chars': sum(q['n_chars'] for q in per_q) / n}, per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    ap.add_argument('--label', required=True)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--bundle-dir', default='')
    args = ap.parse_args()
    bundle_dir = args.bundle_dir or os.path.join(BASE, 'output', f'bundles_{args.split}')

    acc_layer, acc_per_q = accuracy_layer(args.answers, args.split)
    ret_layer, ret_per_q = retrieval_layer(bundle_dir, args.split)

    faith_path = os.path.join(BASE, 'data', 'eval', f'faith_{args.split}.json')
    faith_layer = {}
    if os.path.exists(faith_path):
        fdata = json.load(open(faith_path, encoding='utf-8'))
        pq = fdata.get('per_question', [])
        if pq:
            faith_layer = {'faithfulness': sum(q['faith'] for q in pq) / len(pq),
                           'matrix': fdata.get('matrix', {})}

    from retrieve_dump import PROFILES
    fingerprint = {
        'profiles_sha': hashlib.sha256(json.dumps(PROFILES, sort_keys=True).encode()).hexdigest()[:8],
        'retrieve_dump_sha': script_sha('scripts/retrieve_dump.py'),
        'retriever_sha': script_sha('rag_system/retriever.py'),
        'embedding': 'BAAI/bge-small-zh-v1.5(local,512d)',
    }

    report = {'label': args.label, 'split': args.split, 'ts': datetime.now().isoformat(timespec='seconds'),
              'fingerprint': fingerprint,
              'accuracy': acc_layer, 'retrieval': ret_layer, 'faith': faith_layer,
              'answers_file': args.answers, 'bundle_dir': bundle_dir}

    # delta vs 上份报表（同 split 最新）
    hist_dir = os.path.join(BASE, 'data', 'eval', 'report_history')
    os.makedirs(hist_dir, exist_ok=True)
    prevs = sorted((f for f in os.listdir(hist_dir)
                    if f.startswith(f'{args.split}_') and f.endswith('.json')),
                   key=lambda f: os.path.getmtime(os.path.join(hist_dir, f)))  # mtime 序，文件名字典序会倒退（P2#9）
    deltas = {}
    if prevs:
        prev = json.load(open(os.path.join(hist_dir, prevs[-1]), encoding='utf-8'))
        for layer, keys in (('accuracy', ['accuracy']), ('retrieval', ['recall', 'mrr']),
                            ('faith', ['faithfulness'])):
            for key in keys:
                old = prev.get(layer, {}).get(key)
                new = report.get(layer, {}).get(key)
                if old is not None and new is not None:
                    deltas[f'{layer}.{key}'] = round((new - old) * 100, 1)  # 绝对 pp

    out_name = f"{args.split}_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(report, open(os.path.join(hist_dir, out_name), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print(f'== 三层报表 [{args.label}] split={args.split.upper()} ==')
    print(f"指纹: profiles={fingerprint['profiles_sha']} retrieve_dump={fingerprint['retrieve_dump_sha']} retriever={fingerprint['retriever_sha']}")
    print(f"生成层 Accuracy   : {acc_layer['accuracy']:.1%} (n={acc_layer['n']}"
          + (f", 多选F1={acc_layer['multi_f1']:.3f}" if acc_layer['multi_f1'] is not None else "") + ")")
    for d, v in sorted(acc_layer['per_domain'].items()):
        print(f"    {d}: {v['acc']:.0%} (n={v['n']})")
    print(f"检索层 Recall     : {ret_layer['recall']:.1%} (golden n={ret_layer['n_golden']}) MRR={ret_layer['mrr']:.3f}")
    print(f"冗余度            : {ret_layer['avg_chunks']:.0f} 块 | {ret_layer['avg_chars']:.0f} 字符")
    if faith_layer:
        print(f"忠实层            : {faith_layer['faithfulness']:.1%} matrix={faith_layer['matrix']}")
    print(f"vs {prevs[-1] if prevs else '(首份)'} delta(pp): {deltas if deltas else 'n/a'}")
    print('written:', os.path.join(hist_dir, out_name))


if __name__ == '__main__':
    main()
