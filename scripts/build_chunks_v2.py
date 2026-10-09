"""构建 v2 切块集（data/chunks_v2/，M2 主项）——chunker_v2 + 域 profile。

产出与 v1 同构 jsonl（多 pages 字段），chunk_id 同形但**内容不同**，
严禁与 v1 混载（路径隔离即隔离手段，见 chunker_v2.py 头注）。

用法: python scripts/build_chunks_v2.py [--domain insurance]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from rag_system.chunker_v2 import ChunkerV2, tlen  # noqa: E402

DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='')
    args = ap.parse_args()
    domains = [args.domain] if args.domain else DOMAINS

    out_dir = os.path.join(BASE, 'data', 'chunks_v2')
    os.makedirs(out_dir, exist_ok=True)
    for dom in domains:
        # 按文档聚合 parsed 块（保持顺序）
        docs = defaultdict(list)
        for pf in glob.glob(os.path.join(BASE, 'data', 'parsed', dom, '*.jsonl')):
            for line in open(pf, encoding='utf-8'):
                r = json.loads(line)
                docs[r.get('doc_id', '')].append(r)

        ck = ChunkerV2(type('C', (), {'chunk_size': 0, 'chunk_overlap': 0})(), dom)
        all_chunks, lens = [], []
        for doc_id in sorted(docs):
            all_chunks.extend(ck.chunk_document(doc_id, docs[doc_id]))

        out = os.path.join(out_dir, f'{dom}_chunks.jsonl')
        with open(out, 'w', encoding='utf-8') as f:
            for c in all_chunks:
                f.write(json.dumps({
                    'chunk_id': c.chunk_id, 'doc_id': c.doc_id, 'domain': dom,
                    'text': c.text, 'section_path': c.section_path,
                    'pages': c.pages,
                    'prev_chunk_id': c.prev_chunk_id, 'next_chunk_id': c.next_chunk_id,
                }, ensure_ascii=False) + '\n')
        lens = sorted(len(c.text) for c in all_chunks)
        tlens = sorted(tlen(c.text) for c in all_chunks[::max(1, len(all_chunks) // 500)])
        print(f'{dom}: docs={len(docs)} chunks={len(all_chunks)} '
              f'char_p50={lens[len(lens)//2]} mean={sum(lens)/len(lens):.0f} '
              f'token_p50(抽样)={tlens[len(tlens)//2]}')


if __name__ == '__main__':
    main()
