"""构建 v3 切块集（data/chunks_v3/，chunker_v3 根因重切）。"""
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

from rag_system.chunker_v3 import ChunkerV3  # noqa: E402

DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='')
    args = ap.parse_args()
    domains = [args.domain] if args.domain else DOMAINS

    out_dir = os.path.join(BASE, 'data', 'chunks_v3')
    os.makedirs(out_dir, exist_ok=True)
    for dom in domains:
        docs = defaultdict(list)
        for pf in glob.glob(os.path.join(BASE, 'data', 'parsed', dom, '*.jsonl')):
            for line in open(pf, encoding='utf-8'):
                r = json.loads(line)
                docs[r.get('doc_id', '')].append(r)

        ck = ChunkerV3(None, dom)
        all_chunks, lens = [], []
        for doc_id in sorted(docs):
            all_chunks.extend(ck.chunk_document(doc_id, docs[doc_id]))

        out = os.path.join(out_dir, f'{dom}_chunks.jsonl')
        with open(out, 'w', encoding='utf-8') as f:
            for c in all_chunks:
                f.write(json.dumps({
                    'chunk_id': c.chunk_id, 'doc_id': c.doc_id, 'domain': dom,
                    'text': c.text, 'section_path': c.section_path, 'pages': c.pages,
                    'prev_chunk_id': c.prev_chunk_id, 'next_chunk_id': c.next_chunk_id,
                }, ensure_ascii=False) + '\n')
        lens = sorted(len(c.text) for c in all_chunks)
        print(f'{dom}: docs={len(docs)} chunks={len(all_chunks)} char_p50={lens[len(lens)//2]} mean={sum(lens)/len(lens):.0f}')


if __name__ == '__main__':
    main()
