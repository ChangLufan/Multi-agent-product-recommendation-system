"""
库存决策Agent
- 实时库存查询：MCP协议同步WMS
- 库存预警：安全库存阈值 + 补货建议
- 限购策略：基于库存深度 + 促销热度动态调整
"""

from __future__ import annotations

from typing import Any

from models.schemas import InventoryResult, Product

from .base_agent import BaseAgent

SAFETY_STOCK_THRESHOLD = 50  # 极低库存阈值
LOW_STOCK_THRESHOLD = 100  # 低库存预警阈值
HOT_ITEM_PURCHASE_LIMIT = 2  # 热销商品限购数量


class InventoryAgent(BaseAgent):
    def __init__(self):
        from config import get_settings

        settings = get_settings()
        super().__init__(   # 调用父类构造函数初始化
            name="inventory",
            timeout=settings.agent_timeout_inventory,
        )
        self.db: Any = None  # injected in Phase 2

    async def _execute(self, **kwargs: Any) -> InventoryResult:
        products: list[Product] = kwargs.get("products", [])

        available = []
        low_stock_alerts = []
        purchase_limits: dict[str, int] = {}

        for product in products:  # 遍历商品取库存
            stock = await self._check_stock(product.product_id, product.stock)

            if stock <= 0:  # 如果库存小于等于0则跳过
                continue

            available.append(product.product_id)  # 将商品ID添加到可用商品列表

            if stock <= SAFETY_STOCK_THRESHOLD:
                low_stock_alerts.append({  # 添加库存警告信息
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "critical",
                    "action": "urgent_restock",
                })
            elif stock <= LOW_STOCK_THRESHOLD:
                low_stock_alerts.append({  # 添加库存警告信息
                    "product_id": product.product_id,
                    "name": product.name,
                    "current_stock": stock,
                    "level": "warning",
                    "action": "plan_restock",
                })

            limit = self._calc_purchase_limit(product, stock)
            if limit is not None:
                purchase_limits[product.product_id] = limit

        return InventoryResult(
            success=True,
            available_products=available,
            low_stock_alerts=low_stock_alerts,
            purchase_limits=purchase_limits,
            data={
                "total_checked": len(products),
                "available_count": len(available),
                "alert_count": len(low_stock_alerts),
            },
            confidence=0.95,
        )

    async def _check_stock(self, product_id: str, fallback_stock: int) -> int:
        if self.db:
            pass  # Phase 2: real DB query via MCP
        return fallback_stock

    def _calc_purchase_limit(self, product: Product, stock: int) -> int | None:
        """Dynamic purchase limit based on stock depth and product heat."""
        is_hot = "新品" in product.tags or "旗舰" in product.tags
        if stock <= SAFETY_STOCK_THRESHOLD:
            return 1
        if stock <= LOW_STOCK_THRESHOLD and is_hot:
            return HOT_ITEM_PURCHASE_LIMIT
        if is_hot and stock <= 300:
            return 3
        return None
