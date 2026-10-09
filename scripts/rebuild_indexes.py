"""补块后重建受影响域的 BM25 + 本地 embedding 索引（offline bge-small）。

用法: python scripts/rebuild_indexes.py [--domain research]
"""
import argparse
import json
import os
import shutil
import sys
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.indexer import Indexer  # noqa: E402
from rag_system.config import RAGConfig  # noqa: E402

DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']
LOCAL_MODEL = 'BAAI/bge-small-zh-v1.5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='', choices=[''] + DOMAINS)
    ap.add_argument('--chunkset', default='v1', choices=['v1', 'v2', 'v3', 'v5', 'mix'],
                    help='v1=data/chunks+indexes；v2=data/chunks_v2+indexes_v2（M2 主项新切块）')
    args = ap.parse_args()
    domains = [args.domain] if args.domain else DOMAINS

    config = RAGConfig()
    config.cloud_embedding = False
    config.embedding_model = LOCAL_MODEL
    config.skip_embedding = False
    if args.chunkset != 'v1':
        config.chunks_dir = config.chunks_dir.parent / f'chunks_{args.chunkset}'
        config.index_dir = config.index_dir.parent / f'indexes_{args.chunkset}'
        print(f'chunkset={args.chunkset} → {config.chunks_dir} / {config.index_dir}')

    indexer = Indexer(config)
    for domain in domains:
        emb_dir = os.path.join(config.index_dir, domain, 'embedding')
        if os.path.exists(emb_dir):
            shutil.rmtree(emb_dir)
        chunks = [SimpleNamespace(**json.loads(l))
                  for l in open(os.path.join(config.chunks_dir, f'{domain}_chunks.jsonl'),
                                encoding='utf-8')]
        print(f'{domain}: {len(chunks)} chunks')
        indexer.build_bm25(domain, chunks)
        indexer.build_embedding(domain, chunks)
        indexer.build_field_index(domain, chunks)  # P2#7：索引族三件套同步，防新鲜度漂移
        print(f'{domain}: done')


if __name__ == '__main__':
    main()
