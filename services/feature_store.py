"""
实时特征存储服务
- Redis Sorted Set 存储用户行为序列 (score=timestamp)
- 滑动窗口计算实时特征 (1h/24h/7d)
- 离线+在线特征合并
- RFM模型计算
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

logger = structlog.get_logger()


class FeatureStore:
    """Redis-backed real-time feature store for user behavior and profiles."""

    def __init__(self, redis_client: Any = None, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    # 行为写入
    async def record_behavior(
            self, user_id: str, behavior_type: str, item_id: str, metadata: dict | None = None
    ):
        """Append a behavior event to user's sorted set (score = timestamp)."""
        if not self.redis:
            return
        key = f"behavior:{user_id}:{behavior_type}"  # 用户：行为类型
        payload = json.dumps({"item_id": item_id, "ts": time.time(), **(metadata or {})})  # 行为数据
        await self.redis.zadd(key, {payload: time.time()})  # 有序添加
        await self.redis.expire(key, self.ttl)  # 设置过期时间

    # 行为查询
    async def get_recent_behaviors(
            self, user_id: str, behavior_type: str, window_seconds: int = 3600
    ) -> list[dict]:
        """Retrieve behaviors within a sliding time window."""
        if not self.redis:
            return []
        key = f"behavior:{user_id}:{behavior_type}"
        cutoff = time.time() - window_seconds  # 窗口下限
        raw_items = await self.redis.zrangebyscore(key, cutoff, "+inf")  # 滑动窗口查询
        return [json.loads(item) for item in raw_items]

    # 用户特征获取
    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Build aggregated feature vector from recent behaviors."""
        views_1h = await self.get_recent_behaviors(user_id, "view", 3600)  # 最近1小时浏览
        views_24h = await self.get_recent_behaviors(user_id, "view", 86400)  # 最近24小时浏览
        clicks_1h = await self.get_recent_behaviors(user_id, "click", 3600)  # 最近1小时点击
        purchases_7d = await self.get_recent_behaviors(user_id, "purchase", 604800)  # 最近7天购买

        recent_view_items = [v.get("item_id", "") for v in views_24h[-20:]]  # 取20个最近浏览
        recent_purchase_items = [p.get("item_id", "") for p in purchases_7d[-10:]]  # 取10个最近购买

        # 从行为事件 metadata 中聚合品类（按出现次数降序去重），
        # 让下游 LLM 能从 item_id 之外读到"用户在看什么类目"的信号
        category_counter: dict[str, int] = {}
        for event in views_24h[-20:] + purchases_7d[-10:]:
            cat = event.get("category")
            if cat:
                category_counter[cat] = category_counter.get(cat, 0) + 1
        recent_categories = [
            c for c, _ in sorted(category_counter.items(), key=lambda x: x[1], reverse=True)
        ]

        # 根据用户最近7天的购买行为，计算RFM价值分：用户在线特征
        rfm = await self._compute_rfm(user_id, purchases_7d)

        # 获取用户离线特征
        profile_key = f"profile:{user_id}"
        offline_tags = {}
        if self.redis:
            raw = await self.redis.get(profile_key)  # 从redis中获取用户离线画像json数据
            if raw:
                offline_tags = json.loads(raw)  # 解析为字典

        return {
            "user_id": user_id,  # 用户ID
            "view_count_1h": len(views_1h),  # 最近1小时浏览
            "view_count_24h": len(views_24h),  # 最近24小时浏览
            "click_count_1h": len(clicks_1h),  # 最近1小时点击
            "purchase_count_7d": len(purchases_7d),  # 最近7天购买
            "recent_views": recent_view_items,  # 最近浏览
            "recent_purchases": recent_purchase_items,  # 最近购买
            "recent_categories": recent_categories,  # 最近浏览/购买涉及的品类（供画像 LLM 推断偏好类目）
            "rfm": rfm,  # RFM模型
            "offline_tags": offline_tags,  # 离线标签
        }

    # ---------- RFM model ----------

    async def _compute_rfm(self, user_id: str, purchases: list[dict]) -> dict[str, float]:
        """
        Recency / Frequency / Monetary scoring (normalised 0-1).
        Without full data we use heuristics.
        """
        if not purchases:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}

        now = time.time()
        latest_ts = max(p.get("ts", 0) for p in purchases)
        days_since = (now - latest_ts) / 86400

        recency = max(0.0, 1.0 - days_since / 30.0)
        frequency = min(1.0, len(purchases) / 10.0)
        avg_amount = sum(p.get("amount", 100) for p in purchases) / len(purchases)
        monetary = min(1.0, avg_amount / 1000.0)

        return {
            "recency": round(recency, 3),
            "frequency": round(frequency, 3),
            "monetary": round(monetary, 3),
        }

    # ---------- offline merge ----------

    async def merge_offline_tags(self, user_id: str, tags: dict[str, Any]):
        """Write offline (batch-computed) tags so the profile agent can read them."""
        if not self.redis:
            return
        key = f"profile:{user_id}"
        await self.redis.set(key, json.dumps(tags), ex=self.ttl)
