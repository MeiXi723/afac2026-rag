"""T3-T6 参数化切块扫参器——一次写好，CLI 切配置跑全部实验。

用法:
  python scripts/sweep_chunker.py --tokens 672 --sentence-break --out chunks_t3_672sb
  python scripts/sweep_chunker.py --tokens 448 --overlap 128 --out chunks_t6_ov128
  python scripts/sweep_chunker.py --tokens 896 --sentence-break --recursive --out chunks_t5_rec

每个配置独立落盘 data/chunks_{out}/，路径隔离，不污染现役。
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

MIN_LEN = 40
DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']

_tok = None
def tlen(text):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
    return len(_tok(text, add_special_tokens=False)['input_ids'])

def N(text):
    import unicodedata
    text = __import__('unicodedata').normalize('NFKC', text)
    table = str.maketrans({'，':',','；':';','：':':','（':'(','）':')','％':'%','　':' ','－':'-'})
    text = text.translate(table)
    return __import__('re').sub(r'\s+', '', text)


def build(domain, cfg, out_dir):
    doc_blocks = defaultdict(list)
    for pf in glob.glob(os.path.join(BASE, 'data', 'parsed', domain, '*.jsonl')):
        for line in open(pf, encoding='utf-8'):
            b = json.loads(line)
            doc_blocks[b.get('doc_id', '')].append(b)

    all_chunks = []
    for doc_id in sorted(doc_blocks):
        records = doc_blocks[doc_id]
        # 分节（title 进 path + 正文）
        path, buf, pages = [], [], []
        sections = []
        for rec in records:
            bt = rec.get('block_type', 'paragraph')
            t = rec.get('text', '')
            pn = rec.get('page_num', -1)
            if pn >= 0 and pn not in pages:
                pages.append(pn)
            if bt == 'title':
                if buf:
                    sections.append((' > '.join(path), list(buf), list(pages)))
                    buf, pages = [], []
                level = rec.get('heading_level', 1) or 1
                heading = t.lstrip('#').strip()
                path = path[:max(level - 1, 0)] + [heading]
                if t.strip():
                    buf.append(t.strip())  # title 进正文
            else:
                buf.append(t.rstrip())
        if buf:
            sections.append((' > '.join(path), buf, pages))

        # 逐节：行提取 → 语义切分 → 打包
        unit = cfg.get('unit', 'line')
        sent_break = cfg.get('sentence_break', False)
        chunk_tok = cfg['tokens']
        ov_tok = cfg.get('overlap_tokens', 64)
        rec_pat = re.compile(r'第[一二三四五六七八九十百千零\d]+条')

        for sec_path, sec_lines, sec_pages in sections:
            text = '\n'.join(sec_lines)
            raw_lines = [l for l in text.split('\n')
                         if l.strip() and not re.search(r'\.{3,}|…{2,}', l)]
            if not raw_lines:
                continue

            # 语义切分
            if unit == 'clause' and domain_needs_clause(domain):
                units = []
                buf2 = []
                for ln in raw_lines:
                    if rec_pat.search(ln) and buf2:
                        units.append('\n'.join(buf2))
                        buf2 = [ln]
                    else:
                        buf2.append(ln)
                if buf2:
                    units.append('\n'.join(buf2))
            elif sent_break:
                units = []
                buf2 = []
                for ln in raw_lines:
                    parts = re.split(r'(?<=[。！？])\s*', ln)
                    for p in parts:
                        if p.strip():
                            buf2.append(p.strip())
                    if buf2:
                        units.append(' '.join(buf2))
                        buf2 = []
                if buf2:
                    units.append(' '.join(buf2))
            else:
                units = raw_lines

            # 贪心打包至 token 预算（句子边界优先：如果在句号处截断则优先）
            doc_out = []
        out, cur, cur_tok = [], [], 0
            for u in units:
                ut = tlen(u)
                if ut > chunk_tok:
                    if cur:
                        out.append('\n'.join(cur)); cur, cur_tok = [], 0
                    step = max(int(chunk_tok * 1.8), 1)
                    for s in range(0, len(u), step):
                        out.append(u[s:s + step])
                    continue
                if cur_tok + ut > chunk_tok and cur:
                    # 句子边界回退：在 cur 中找最后一个句号/分号
                    full = '\n'.join(cur)
                    sent_pos = [m.end() for m in re.finditer(r'[。；！？]', full)]
                    if sent_pos and sent_pos[-1] > len(full) * 0.5:
                        out.append(full[:sent_pos[-1]])
                        remainder = full[sent_pos[-1]:].strip()
                        cur = [remainder] if remainder else []
                        cur_tok = tlen(remainder)
                    else:
                        # 重叠回带
                        tail, t = [], 0
                        for x in reversed(cur):
                            xt = tlen(x)
                            if t + xt > ov_tok: break
                            tail.insert(0, x); t += xt
                        out.append('\n'.join(cur))
                        cur, cur_tok = tail, t
                cur.append(u)
                cur_tok += ut
            if cur:
                out.append('\n'.join(cur))

        # 去重 + 短合并 + 重编号 + 链接
        seen, ded = set(), []
        for i, t in enumerate(out):
            h = hash(N(t))
            if h in seen: continue
            seen.add(h)
            tn = N(t)
            if len(tn) < MIN_LEN and ded:
                ded[-1]['text'] += '\n' + t
                continue
            ded.append({'chunk_id': f'{domain}_{doc_id}_{len(ded)}', 'doc_id': doc_id,
                        'domain': domain, 'text': t,
                        'section_path': '', 'pages': [],
                        'prev_chunk_id': None, 'next_chunk_id': None})
        for i, c in enumerate(ded):
            c['chunk_id'] = f'{domain}_{doc_id}_{i}'
        for i, c in enumerate(ded):
            c['prev_chunk_id'] = ded[i-1]['chunk_id'] if i > 0 else None
            c['next_chunk_id'] = ded[i+1]['chunk_id'] if i < len(ded)-1 else None
        all_chunks.extend(ded)

    out = os.path.join(out_dir, f'{domain}_chunks.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    lens = sorted(len(c['text']) for c in all_chunks)
    print(f'{domain}: {len(all_chunks)} chunks p50={lens[len(lens)//2]}')


def domain_needs_clause(domain):
    return domain in ('insurance', 'regulatory', 'financial_contracts')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=448)
    ap.add_argument('--overlap-tokens', type=int, default=64)
    ap.add_argument('--sentence-break', action='store_true')
    ap.add_argument('--unit', default='line', choices=['line', 'clause'])
    ap.add_argument('--out', required=True, help='输出子目录名 chunks_{out}')
    ap.add_argument('--domain', default='')
    args = ap.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    out_dir = os.path.join(BASE, 'data', f'chunks_{args.out}')
    os.makedirs(out_dir, exist_ok=True)

    cfg = {'tokens': args.tokens, 'overlap_tokens': args.overlap_tokens,
           'unit': args.unit, 'sentence_break': args.sentence_break}

    for dom in domains:
        build(dom, cfg, out_dir)


if __name__ == '__main__':
    main()
