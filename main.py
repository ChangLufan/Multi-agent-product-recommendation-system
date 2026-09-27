"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - 获取个性化推荐
  POST /api/v1/recommend/graph    - 通过LangGraph pipeline推荐
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models.schemas import RecommendationRequest, RecommendationResponse
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services.ab_test import ABTestEngine
from services.metrics import MetricsCollector

logger = structlog.get_logger()
settings = get_settings()


ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
supervisor = SupervisorOrchestrator(ab_engine=ab_engine)
rec_graph = None

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时构建状态图
    global rec_graph
    rec_graph = build_recommendation_graph()  # 构建langgraph状态图
    logger.info("app.startup", model=settings.llm_model)
    yield
    # 关闭时做的
    logger.info("app.shutdown")


# 创建FastAPI应用实例
app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 营销文案Agent + 库存决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,  # 传入生命周期管理
)

# 添加CORS中间件，允许所有前端来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


# 通过LangGraph状态图进行推荐
@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """使用LangGraph状态图进行推荐 (展示LangGraph能力)"""
    if not rec_graph:
        return {"error": "Graph not initialized"}

    # 从请求中提取状态
    state = {
        "user_id": request.user_id,
        "scene": request.scene,  # TODO 不同场景的推荐
        "num_items": request.num_items,  # 推荐商品数量
        "context": request.context,  # 上下文信息
    }

    # 将状态传递给LangGraph状态图，并启动状态图异步执行，获取执行后推荐结果
    result = await rec_graph.ainvoke(state)

    # 返回推荐结果
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],  # 最终推荐的商品列表
        "marketing_copies": result.get("marketing_copies", []),  # 营销文案列表
        "experiment_group": result.get("experiment_group", "control"),  # A/B实验分组
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),  # 总延迟时间
    }


# 查看A/B实验状态
@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


# 查看系统监控指标
@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


# 记录A/B测试结果,更新Thompson Sampling
@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


# 收集推荐系统指标
def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
