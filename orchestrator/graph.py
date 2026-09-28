"""
LangGraph state graph for the multi-agent recommendation pipeline.

Visualises the DAG of agent execution:

  [start] -> fan_out -> {user_profile, product_recall}  (parallel)
          -> merge_phase1 -> {rerank, inventory}         (parallel)
          -> merge_phase2 -> marketing_copy
          -> aggregate -> [end]
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import Product, UserProfile
from services.ab_test import ABTestEngine

logger = structlog.get_logger()


# 状态类型定义
class PipelineState(TypedDict, total=False):
    request_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str

    user_profile: UserProfile | None
    raw_products: list[Product]
    ranked_products: list[Product]
    available_ids: set[str]
    final_products: list[Product]
    marketing_copies: list[dict[str, str]]

    agent_results: dict[str, Any]  # 存储各个agent的返回结果
    total_latency_ms: float  # 总耗时
    _start_time: float  # 开始时间


# agent实例化对象初始化（feature_store 在 build_recommendation_graph 时注入）
user_profile_agent = UserProfileAgent()
product_rec_agent = ProductRecAgent()
marketing_copy_agent = MarketingCopyAgent()
inventory_agent = InventoryAgent()
ab_engine = ABTestEngine()


# 初始化节点
async def init_node(state: PipelineState) -> PipelineState:
    state["request_id"] = str(uuid.uuid4())  # 请求id
    state["_start_time"] = time.perf_counter()  # 开始时间为当前时间
    state["agent_results"] = {}  # 用于存储各个agent的返回结果，先初始化
    exp = ab_engine.assign(state["user_id"]) # 根据用户ID进行A/B测试分组
    state["experiment_group"] = exp.get("group", "control") # 获取分组结果
    return state


# 用户画像节点，调用用户画像agent获取返回结果实体（profile）填充到state中
async def user_profile_node(state: PipelineState) -> PipelineState:
    result = await user_profile_agent.run(
        user_id=state["user_id"],
        context=state.get("context", {}),
    )
    state["user_profile"] = getattr(result, "profile", None)  # 获取用户画像agent中返回的profile字典类型数据
    state["agent_results"]["user_profile"] = result
    return state

# 商品推荐节点，调用产品推荐agent获取实体（products）填充到state中
async def product_recall_node(state: PipelineState) -> PipelineState:
    result = await product_rec_agent.run(
        user_profile=None,  # 第一次没有用户画像调用商品推荐agent
        num_items=state.get("num_items", 10) * 2,
    )
    state["raw_products"] = getattr(result, "products", [])  # 获取产品推荐agent中返回的products原始列表类型数据
    state["agent_results"]["product_recall"] = result
    return state

# 并行阶段1，同时运行用户画像和产品推荐节点
async def parallel_phase1(state: PipelineState) -> PipelineState:
    """Run user_profile and product_recall in parallel."""
    profile_state, recall_state = await asyncio.gather(
        user_profile_node(dict(state)),
        product_recall_node(dict(state)),
    )
    state.update(profile_state)
    state.update(recall_state)
    return state

# 重排序节点
async def rerank_node(state: PipelineState) -> PipelineState:
    result = await product_rec_agent.run(
        user_profile=state.get("user_profile"),  # 第二次携带用户画像调用商品推荐agent
        num_items=state.get("num_items", 10),
    )
    state["ranked_products"] = getattr(result, "products", state.get("raw_products", []))  # 根据用户画像和商品推荐结果重排
    state["agent_results"]["rerank"] = result
    return state

# 库存检查节点
async def inventory_node(state: PipelineState) -> PipelineState:
    result = await inventory_agent.run(
        products=state.get("raw_products", []),  # 对原始列表中的商品进行库存检查
    )
    state["available_ids"] = set(getattr(result, "available_products", []))  # 获取可用商品的ID集合
    state["agent_results"]["inventory"] = result
    return state

# 并行阶段2，同时运行重排序和库存检查节点
async def parallel_phase2(state: PipelineState) -> PipelineState:
    """Run rerank and inventory in parallel."""
    rerank_state, inv_state = await asyncio.gather(
        rerank_node(dict(state)),
        inventory_node(dict(state)),
    )
    state.update(rerank_state)
    state.update(inv_state)
    return state

# 过滤节点，根据可用库存过滤商品列表
async def filter_node(state: PipelineState) -> PipelineState:
    ranked = state.get("ranked_products", [])   # 重排后的商品列表
    avail = state.get("available_ids", set())  # 可用商品的ID集合
    num = state.get("num_items", 10)  # 需要的商品数量
    final = [p for p in ranked if p.product_id in avail]  # 过滤出可用的商品
    if not final:
        final = ranked  # 如果没有可用商品，则返回重排后的商品列表
    state["final_products"] = final[:num]  # 选择前num个商品作为最终结果
    return state

# 营销文案节点，调用营销文案agent获取实体（copies）填充到state中
async def marketing_copy_node(state: PipelineState) -> PipelineState:
    result = await marketing_copy_agent.run(
        user_profile=state.get("user_profile"),
        products=state.get("final_products", []),
    )
    state["marketing_copies"] = getattr(result, "copies", [])  # copies为营销文案字典列表，每对字典代表一个商品的营销文案
    state["agent_results"]["marketing_copy"] = result
    return state

# 聚合节点，计算总耗时
async def aggregate_node(state: PipelineState) -> PipelineState:
    state["total_latency_ms"] = (time.perf_counter() - state.get("_start_time", 0)) * 1000  # 当前时间-开始时间，计算总耗时
    return state


def build_recommendation_graph(feature_store: Any = None) -> StateGraph:
    """Build and compile the LangGraph state graph.

    Args:
        feature_store: Optional FeatureStore instance. When provided, it is
            injected into the module-level user_profile_agent so the profile
            node reads real Redis features instead of the context fallback.
    """
    if feature_store is not None:
        user_profile_agent.feature_store = feature_store
        logger.info(
            "graph.feature_store_injected",
            agent_id=id(user_profile_agent),
            store_id=id(feature_store),
        )
    else:
        logger.info(
            "graph.feature_store_not_injected",
            agent_id=id(user_profile_agent),
            mode="context_fallback",
        )

    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("init", init_node)
    graph.add_node("parallel_phase1", parallel_phase1)  # 并行：用户画像节点+商品推荐节点
    graph.add_node("parallel_phase2", parallel_phase2)  # 并行：重排序节点+库存检查节点
    graph.add_node("filter", filter_node)  # 过滤节点，根据可用库存过滤商品列表
    graph.add_node("marketing_copy", marketing_copy_node)  # 营销文案节点，调用营销文案agent获取实体（copies）填充到state中
    graph.add_node("aggregate", aggregate_node)  # 聚合节点，计算总耗时

    # 编排节点
    graph.set_entry_point("init")
    graph.add_edge("init", "parallel_phase1")
    graph.add_edge("parallel_phase1", "parallel_phase2")
    graph.add_edge("parallel_phase2", "filter")
    graph.add_edge("filter", "marketing_copy")
    graph.add_edge("marketing_copy", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
