from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import AgentResult

logger = structlog.get_logger()

'''
基类agent
'''
class BaseAgent(ABC):
    """All agents inherit from this base class with retry, timeout, and fallback."""

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        self.name = name  # agent名称
        self.timeout = timeout  # 超时时间
        self.max_retries = max_retries # 最大重试次数
        self._call_count = 0  # 调用计数
        self._error_count = 0  # 错误计数

    # 所有子类agent都必须实现的执行方法
    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Core logic implemented by each concrete agent."""

    # 外部调用入口，会进行重试和回退
    async def run(self, **kwargs: Any) -> AgentResult:     # **kwargs为字典，包含接收到的各种参数，后续透传给各agent
        """Public entry: wraps _execute with timing, retries, and fallback."""
        start = time.perf_counter()   # 开始时间
        self._call_count += 1

        try:
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000   # 计算耗时
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except Exception as exc:
            # 异常兜底处理
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000   # 计算耗时
            logger.error("agent.failed", agent=self.name, error=str(exc))
            return self._fallback(latency_ms, exc)

    # 执行方法，会进行重试
    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        @retry(
            stop=stop_after_attempt(self.max_retries),  # 重试次数
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),  # 等待时间
            reraise=True,
        )
        # 内部执行方法
        async def _inner():
            return await self._execute(**kwargs)

        return await _inner()
    # 回退方法，当执行失败时调用

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        """Return a degraded but valid result when the agent fails."""
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    # 错误率计算
    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
