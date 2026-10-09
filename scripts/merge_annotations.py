"""标注引文 → golden chunks 合并（含回验）。

读取 output/brain_batch/annotate_*.json，把每个 verdict=支持/排除 的逐字引文
反查回 chunk（归一化子串，取最长连续段），回验"引文确实在被标块文本中"，
并入 data/eval/golden_chunks_a.json（source 字段标注来源）。

用法: python scripts/merge_annotations.py
"""
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from rag_system.eval_norm import norm as N  # noqa: E402

DOM = {'fc': 'financial_contracts', 'fin': 'financial_reports', 'ins': 'insurance',
       'reg': 'regulatory', 'res': 'research'}


CHUNKSET = 'v1'
_CH: dict = {}


def _ch(dom):
    if dom not in _CH:
        sub = '' if CHUNKSET == 'v1' else f'_{CHUNKSET}'
        _CH[dom] = [(c['chunk_id'], c.get('doc_id', ''), N(c.get('text', '')))
                    for line in open(f'{BASE}/data/chunks{sub}/{dom}_chunks.jsonl', encoding='utf-8')
                    for c in [json.loads(line)]]
    return _CH[dom]


def quote_to_chunks(quote, dom, hint):
    qn = N(quote)
    runs = sorted((r for r in re.split(r'…|\.\.\.|。|；|，', qn) if len(r) >= 8),
                  key=len, reverse=True)
    for r in runs[:3]:
        hits = [cid for cid, did, t in _ch(dom) if (not hint or hint in did) and r in t]
        if hits:
            return hits[:6]
    return []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunkset', default='v1', choices=['v1', 'v2', 'v3', 'v4'])
    _args, _ = ap.parse_known_args()
    global CHUNKSET
    CHUNKSET = _args.chunkset
    suffix = '' if CHUNKSET == 'v1' else f'_{CHUNKSET}'
    gp = os.path.join(BASE, 'data', 'eval', f'golden_chunks_a{suffix}.json')
    golden = json.load(open(gp, encoding='utf-8'))
    G = golden['questions']
    stats = {'added_q': 0, 'added_chunks': 0, 'verified': 0, 'unverified': [], 'missed': []}

    # 盲判引用转入（与代理标注同一指纹匹配器；P1-1 修复后同源）
    gt_all = json.load(open(os.path.join(BASE, 'data', 'ground_truth_a.json'), encoding='utf-8'))['answers']
    blind_path = os.path.join(BASE, 'data', 'eval', 'blind_adjudication_result.json')
    if os.path.exists(blind_path):
        blind = json.load(open(blind_path, encoding='utf-8')).get('blind', {})
        for qid, rec in blind.items():
            if G.get(qid, {}).get('decisive'):
                continue
            dom = DOM[qid.split('_')[0]]
            gt_ans = set(gt_all.get(qid, {}).get('answer', ''))
            dec, excl = [], []
            for letter, verdict in (rec.get('per_option') or {}).items():
                m = re.search(r'「([^」]+)」', verdict or '')
                if not m:
                    continue
                dm = re.search(r'(text\d+|pack2_text\d+|annual_\w+?_\d{4}_report|csrc_\d+|strict_v3_\d+)', verdict)
                hits = quote_to_chunks(m.group(1), dom, dm.group(1) if dm else None)
                if letter.upper() in gt_ans:
                    dec.extend(hits)
                elif (verdict or '').startswith('排除') and len(gt_ans) > 1:
                    excl.extend(hits)
            if dec:
                entry = G.setdefault(qid, {'decisive': [], 'exclusion': [], 'ambiguous': [], 'unmatched': [], 'doc_ids': []})
                entry['decisive'] = sorted(set(entry['decisive']) | set(dec))
                entry['exclusion'] = sorted(set(entry['exclusion']) | set(excl))
                entry['source'] = 'blind_adjudication_v2_transfer'
    print('blind 引用转入完成（仅补无 decisive 的题）')

    for path in sorted(glob.glob(os.path.join(BASE, 'output', 'brain_batch', 'annotate_*.json'))):
        batch = json.load(open(path, encoding='utf-8'))
        for qid, rec in batch.items():
            dom = DOM[qid.split('_')[0]]
            gt_ans = set(rec.get('gt_answer', ''))
            dec, excl, unver = [], [], []
            for letter, opt in (rec.get('per_option') or {}).items():
                opt = opt or {}
                quote = str(opt.get('quote', '') or '').strip()
                verdict = str(opt.get('verdict', '') or '')
                if not quote or verdict == '无据':
                    continue
                hits = quote_to_chunks(quote, dom, (opt or {}).get('doc_hint'))
                if not hits:
                    missed = True
                    stats['missed'].append(f'{qid}:{letter}')
                    continue
                # 回验：引文归一化后确实在被标块文本中（取最长段）
                qn = N(quote)
                runs = sorted((r for r in re.split(r'…|。|；|，', qn) if len(r) >= 8),
                              key=len, reverse=True)
                for cid in hits:
                    t = next(t for c, _, t in _ch(dom) if c == cid)
                    if any(r in t for r in runs[:3]):
                        stats['verified'] += 1
                    else:
                        unver.append(f'{qid}:{letter}:{cid}')
                if letter.upper() in gt_ans:
                    dec.extend(hits)
                elif verdict == '排除' and len(gt_ans) > 1:
                    excl.extend(hits)
            if dec:
                entry = G.setdefault(qid, {'decisive': [], 'exclusion': [],
                                           'ambiguous': [], 'unmatched': [], 'doc_ids': []})
                new_dec = set(entry['decisive']) | set(dec)
                stats['added_chunks'] += len(new_dec - set(entry['decisive']))
                entry['decisive'] = sorted(new_dec)
                entry['exclusion'] = sorted(set(entry['exclusion']) | set(excl))
                entry['source'] = 'agent_annotation_20261007'
                stats['added_q'] += 1
            stats['unverified'].extend(unver)

    golden['questions'] = G
    golden['_meta']['stats']['q_with_golden'] = sum(1 for v in G.values() if v.get('decisive'))
    golden['_meta']['agent_annotated'] = (f"{stats['added_q']} 题由代理标注转入，"
                                          f"引文回验 {stats['verified']} 条，"
                                          f"未反查到 {len(stats['missed'])} 条")
    json.dump(golden, open(gp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"新增题 {stats['added_q']} | decisive 覆盖 {golden['_meta']['stats']['q_with_golden']}/100")
    print(f"引文回验通过 {stats['verified']} | 反查失败 {len(stats['missed'])} | 回验未过 {len(stats['unverified'])}")
    if stats['missed']:
        print('  反查失败:', stats['missed'][:20])
    if stats['unverified']:
        print('  回验未过:', stats['unverified'][:10])
    print('written:', gp)


if __name__ == '__main__':
    main()
