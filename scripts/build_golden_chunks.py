"""S0: 把真值 evidence 引用自动迁移为 golden chunk_id 标注。

用法:
  python scripts/build_golden_chunks.py --split a            # 自动迁移 + 输出未匹配清单
  python scripts/build_golden_chunks.py --split a --stats    # 打印校准统计（可复算）
输出:
  data/eval/golden_chunks_{split}.json
  {qid: {decisive: [chunk_ids], exclusion: [chunk_ids],
         ambiguous: [chunk_ids], unmatched: [原 evidence 串], doc_ids: [...]}}
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_system.eval_norm import norm  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIN_NAME_TOKENS = ['byd', 'catl', 'midea', 'cscec', 'chinamobile', 'cmb']


CHUNKSET = 'v1'  # 由 main() --chunkset 覆盖

def load_domain_chunks(domain):
    """{doc_id: {chunk_id: 归一化文本}}（按 chunk_id 排序稳定）"""
    sub = '' if CHUNKSET == 'v1' else f'_{CHUNKSET}'
    path = os.path.join(BASE, 'data', 'chunks' + sub, f'{domain}_chunks.jsonl')
    docs = {}
    for line in open(path, encoding='utf-8'):
        c = json.loads(line)
        docs.setdefault(c.get('doc_id', ''), {})[c['chunk_id']] = norm(c.get('text', ''))
    for did in docs:
        docs[did] = dict(sorted(docs[did].items()))
    return docs


def resolve_docs(domain, ev_text, all_doc_ids):
    """从 evidence 串解析被引用的 doc_id 列表。"""
    hits = []
    if domain == 'insurance':
        hits = [f'{int(n)}' for n in re.findall(r'\bdoc\s*0*(\d+)\b', ev_text)]
    elif domain == 'financial_contracts':
        hits = [f'text{int(n):02d}' for n in re.findall(r'\btext\s*0*(\d+)\b', ev_text)]
    elif domain == 'financial_reports':
        toks = [t for t in FIN_NAME_TOKENS if t in ev_text.lower()]
        yrs = re.findall(r'2024|2025', ev_text)  # 无捕获组：findall 须返回全年份（盘上 doc_id 形态 annual_xx_2024_report）
        if toks:
            for t in toks:
                for y in (yrs or ['2024', '2025']):
                    hits.append(f'annual_{t}_{y}_report')
        else:
            # 无名字时退化为全部文档曾引发 golden 爆炸（fc_a_005 595 块，审查
            # P0-3b）：报错暴露而非静默污染评测集——该题转入人工补标队列
            raise ValueError(f'evidence 无公司名 token，无法定位文档: {ev_text[:60]}')
    elif domain == 'regulatory':
        nums = re.findall(r'\b0*(\d{3,4})\b', ev_text)
        for n in nums:
            hits += [d for d in all_doc_ids if re.search(rf'(^|[^0-9]){n}([^0-9]|$)', d)]
    elif domain == 'research':
        nums = re.findall(r'text\s*0*(\d+)', ev_text)
        hits = [f'pack2_text{int(n):02d}' for n in nums]
    return sorted(set(h for h in hits if h in all_doc_ids))


def is_exclusion(item: str) -> bool:
    """排除性证据（证伪某选项）判定。"""
    head = item[:40]
    return bool(re.search(r'[A-Da-d]假|假[:：]|不成立|错置|误判|排除', head))


def match_item(item, doc_texts):
    """对单条 evidence 返回 (matched_chunk_ids, quote_used)。"""
    q = norm(item)
    runs = sorted({r for r in re.findall(r'[一-龥]{10,}', q)}, key=len, reverse=True)
    matched = []
    # 单引文命中 >10 块 = 页眉/文档标题级样板（如"…报告书"每页页眉都有，
    # 曾把 fc_a_005 decisive 爆炸到 595 块），跳过该引文
    def run_hits(run, head_len):
        hits = []
        for did, chunks in doc_texts.items():
            for cid, text in chunks.items():
                if run[:head_len] and run[:head_len] in text:
                    hits.append(cid)
        return hits

    for run in runs[:3]:
        hits = run_hits(run, 40)
        if len(hits) > 10:
            continue
        matched.extend(hits)
        if matched:
            break
    if not matched:
        for run in runs[::-1]:  # 最短片段再试一轮（≥10 字已保证，防长句跨块）
            hits = run_hits(run, 12)
            if len(hits) > 10:
                continue
            matched.extend(hits)
            if matched:
                break
    return sorted(set(matched)), (runs[0][:40] if runs else '')


def build(split):
    gt = json.load(open(os.path.join(BASE, 'data', f'ground_truth_{split}.json'),
                        encoding='utf-8'))['answers']
    chunks_cache = {}
    result = {}
    stats = {'questions': 0, 'items': 0, 'matched_items': 0,
             'q_with_golden': 0, 'ambiguous': 0}
    for qid in sorted(gt):
        rec = gt[qid]
        domain = rec['domain']
        if domain not in chunks_cache:
            chunks_cache[domain] = load_domain_chunks(domain)
        all_doc_ids = sorted(chunks_cache[domain])
        doc_texts_full = chunks_cache[domain]
        entry = {'decisive': [], 'exclusion': [], 'ambiguous': [],
                 'unmatched': [], 'doc_ids': []}
        q_has = False
        for item in rec.get('evidence', []):
            stats['items'] += 1
            try:
                docs = resolve_docs(domain, item, all_doc_ids)
            except ValueError as e:
                # 单条 evidence 无文档锚点：记 unmatched 等人工补标，
                # 禁止退化为"全部文档"（会整域爆炸，审查 P0-3b）
                entry['unmatched'].append(f'!! {e}')
                continue
            entry['doc_ids'] = sorted(set(entry['doc_ids']) | set(docs))
            doc_texts = {d: doc_texts_full.get(d, {}) for d in docs} if docs else {}
            matched, quote = match_item(item, doc_texts)
            role = 'exclusion' if is_exclusion(item) else 'decisive'
            if matched:
                stats['matched_items'] += 1
                q_has = True
                bucket = entry[role]
                for cid in matched:
                    if cid not in bucket:
                        bucket.append(cid)
                if len(matched) > 1:
                    entry['ambiguous'].append({'quote': quote, 'chunks': matched})
                    stats['ambiguous'] += 1
            else:
                entry['unmatched'].append(item[:160])
        if q_has:
            stats['q_with_golden'] += 1
            entry['decisive'] = [c for c in entry['decisive'] if c not in entry['exclusion']]
        else:
            entry['unmatched'].insert(0, '!! 全部 evidence 均未命中 chunk，需人工补标')
        stats['questions'] += 1
        result[qid] = entry
    return result, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    ap.add_argument('--stats', action='store_true', help='打印校准统计后退出（S1 校准）')
    ap.add_argument('--chunkset', default='v1', choices=['v1', 'v2', 'v3', 'v4'])
    args = ap.parse_args()

    global CHUNKSET
    CHUNKSET = args.chunkset
    result, stats = build(args.split)
    if args.stats:
        print(json.dumps(stats, ensure_ascii=False, indent=1))
        return
    suffix = '' if args.chunkset == 'v1' else f'_{args.chunkset}'
    out = os.path.join(BASE, 'data', 'eval', f'golden_chunks_{args.split}{suffix}.json')
    # P2#8 护栏：现文件含 source 标注（盲判/代理成果）时重定向，防全量重建静默清空
    if os.path.exists(out):
        _prev = json.load(open(out, encoding='utf-8'))
        _tagged = [q for q, v in _prev.get('questions', {}).items() if v.get('source')]
        if _tagged:
            out = out.replace('.json', '.rebuilt.json')
            print(f'!! 现有 golden 含 {len(_tagged)} 条 source 标注 → 写入 {out}；'
                  f'核对后手动替换原文件，并重跑 merge_annotations.py 补回标注')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {'_meta': {'split': args.split.upper(), 'stats': stats,
                         'norm': 'rag_system.eval_norm.norm（NFKC+去空白+标点统一）'},
               'questions': result}
    json.dump(payload, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    print('written:', out)


if __name__ == '__main__':
    main()
