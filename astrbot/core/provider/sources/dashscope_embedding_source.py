import asyncio
import os
from functools import partial

import dashscope

from astrbot import logger

from ..entities import ProviderType
from ..provider import EmbeddingProvider
from ..register import register_provider_adapter


@register_provider_adapter(
    "dashscope_embedding",
    "阿里云百炼 Embedding 提供商适配器",
    provider_type=ProviderType.EMBEDDING,
)
class DashScopeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.api_key = provider_config.get("embedding_api_key") or os.getenv(
            "DASHSCOPE_API_KEY", ""
        )
        if not self.api_key:
            raise ValueError("DashScope Embedding API Key 不能为空。")
        self.model = provider_config.get("embedding_model", "text-embedding-v3")
        self._dim = int(provider_config.get("embedding_dimensions", 1024))
        logger.info(
            f"[DashScope Embedding] 使用模型: {self.model}, 向量维度: {self._dim}"
        )

    def _call_embedding(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {
            "api_key": self.api_key,
            "model": self.model,
            "input": texts,
        }
        if self._dim > 0:
            kwargs["dimension"] = self._dim
        resp = dashscope.TextEmbedding.call(**kwargs)
        if resp.status_code != 200:
            raise Exception(
                f"DashScope Embedding API 请求失败: {resp.code} - {resp.message}"
            )
        embeddings_sorted = sorted(
            resp.output["embeddings"], key=lambda x: x["text_index"]
        )
        return [item["embedding"] for item in embeddings_sorted]

    async def get_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, partial(self._call_embedding, [text])
        )
        return results[0]

    async def get_embeddings(self, text: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._call_embedding, text))

    def get_dim(self) -> int:
        return self._dim

    async def terminate(self) -> None:
        pass
