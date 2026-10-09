"""本地重建全 5 域 embedding 索引（bge-small-zh-v1.5，零 API）。

用法: python scripts/rebuild_embedding_local.py [--domain insurance]
"""
import argparse
import json
import os
import shutil
import sys
from types import SimpleNamespace

# 模型已在本地 HF 缓存，禁止联网校验（直连 huggingface.co 会 SSL 失败）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.indexer import Indexer  # noqa: E402
from rag_system.config import RAGConfig  # noqa: E402

DOMAINS = ['insurance', 'financial_contracts', 'financial_reports', 'regulatory', 'research']
LOCAL_MODEL = 'BAAI/bge-small-zh-v1.5'  # 已缓存，512 维


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='', choices=[''] + DOMAINS)
    args = ap.parse_args()
    domains = [args.domain] if args.domain else DOMAINS

    config = RAGConfig()
    config.cloud_embedding = False          # 本地向量，零 API
    config.embedding_model = LOCAL_MODEL
    config.skip_embedding = False

    indexer = Indexer(config)
    for domain in domains:
        emb_dir = os.path.join(config.index_dir, domain, 'embedding')
        if os.path.exists(emb_dir):
            shutil.rmtree(emb_dir)  # 旧索引是云模型向量空间，必须重建
            print(f'{domain}: removed old cloud embedding index')

        chunks = []
        path = os.path.join(config.chunks_dir, f'{domain}_chunks.jsonl')
        for line in open(path, encoding='utf-8'):
            c = json.loads(line)
            chunks.append(SimpleNamespace(**c))
        print(f'{domain}: {len(chunks)} chunks, building local embedding ({LOCAL_MODEL})...')
        indexer.build_embedding(domain, chunks)
        print(f'{domain}: done -> {emb_dir}')


if __name__ == '__main__':
    main()
