"""
SkillRetriever：用 embedding 做语义召回

Embedding 走 OpenAI 兼容协议（OpenAI / 通义千问 DashScope / DeepSeek 等）。
余弦相似度 top-k；分数低于阈值返回空（让上层放弃复用，回到 LLM 重规划）。
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

from mobilerun.schema import Skill
from mobilerun.skills.store import SkillStore


class BaseEmbedder:
    dim: int = 0

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class OpenAICompatibleEmbedder(BaseEmbedder):
    """走 OpenAI /v1/embeddings 协议的所有后端。

    用法：
        OpenAICompatibleEmbedder(
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small",
        )
        # 通义千问 DashScope（OpenAI 兼容端点）：
        OpenAICompatibleEmbedder(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model="text-embedding-v3",
        )
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        import requests

        url = f"{self.base_url}/embeddings"
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        vec = data["data"][0]["embedding"]
        if self.dim == 0:
            self.dim = len(vec)
        return vec


def cosine(a: List[float], b: List[float]) -> float:
    """真正的余弦相似度：点积 / (|a|·|b|)。

    不依赖向量是否已归一化 —— DashScope text-embedding-v3 等后端
    不保证返回单位向量，若只算点积会让 score_threshold 失去意义。
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SkillRetriever:
    def __init__(self, store: SkillStore, embedder: BaseEmbedder,
                 score_threshold: float = 0.35):
        self.store = store
        self.embedder = embedder
        self.score_threshold = score_threshold

    def index_skill(self, skill: Skill) -> Skill:
        """计算并写入 skill.embedding。put 回 store 由调用方决定。"""
        text = skill.description or skill.name
        skill.embedding = self.embedder.embed(text)
        return skill

    def reindex_all(self):
        for s in self.store.list_all():
            self.index_skill(s)
            self.store.put(s, save=False)
        self.store.save()

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Skill, float]]:
        skills = self.store.list_all()
        if not skills:
            return []
        q_vec = self.embedder.embed(query)
        scored: List[Tuple[Skill, float]] = []
        for s in skills:
            if not s.embedding:
                # 没索引过的退化到关键词匹配（store 自带）
                ks = self.store.keyword_search(query, limit=1)
                if ks and ks[0].id == s.id:
                    scored.append((s, self.score_threshold + 0.01))
                continue
            sim = cosine(q_vec, s.embedding)
            if sim >= self.score_threshold:
                scored.append((s, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def best(self, query: str) -> Optional[Tuple[Skill, float]]:
        r = self.retrieve(query, top_k=1)
        return r[0] if r else None


def build_embedder_from_env(prefer: str = "auto") -> BaseEmbedder:
    """根据环境变量构造 Embedder：
       prefer = 'openai' | 'qwen' | 'auto'
       auto: OPENAI_API_KEY > DASHSCOPE_API_KEY
       缺失时抛错。
    """
    if prefer in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleEmbedder(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )
    if prefer in ("auto", "qwen") and os.environ.get("DASHSCOPE_API_KEY"):
        return OpenAICompatibleEmbedder(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model=os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v3"),
        )
    raise RuntimeError(
        "未配置 Embedding 后端。请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。"
    )
