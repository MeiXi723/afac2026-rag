import logging
import time
from typing import Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class CloudEmbedder:
    def __init__(self, config) -> None:
        self.config = config
        self._client: Optional[OpenAI] = None
        self._dim: Optional[int] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key_qwen,
                base_url=self.config.api_base_qwen,
                timeout=self.config.request_timeout,
            )
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.config.cloud_embedding_dim
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        batch_size = self.config.cloud_embedding_batch_size
        all_vectors: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self._embed_batch(client, batch)
            all_vectors.append(vectors)

        return np.concatenate(all_vectors, axis=0)

    def embed_query(self, query: str) -> np.ndarray:
        client = self._get_client()
        return self._embed_batch(client, [query])

    def _embed_batch(self, client: OpenAI, texts: list[str]) -> np.ndarray:
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                resp = client.embeddings.create(
                    model=self.config.cloud_embedding_model,
                    input=texts,
                    dimensions=self.config.cloud_embedding_dim,
                )
                vectors = np.array(
                    [d.embedding for d in resp.data], dtype=np.float32
                )
                return vectors
            except Exception as e:
                # 鉴权失败重试无意义，直接失败让上层降级为 BM25-only
                msg = str(e)
                if '401' in msg or 'api_key' in msg.lower():
                    raise RuntimeError(f'Embedding auth failed: {e}') from e
                last_exc = e
                delay = self.config.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Embedding API attempt %d/%d failed: %s, retrying in %.1fs",
                    attempt + 1, self.config.max_retries, e, delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Embedding API failed after {self.config.max_retries} attempts"
        ) from last_exc
