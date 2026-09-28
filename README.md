# 多智能体电商推荐系统 — 学习指南

> 项目路径：`D:/Users/小小凡/Desktop/开发/Agent/project/Mult-agent`
> 技术栈：Python 3.12 + FastAPI + LangChain + LangGraph + Pydantic v2 + asyncio

---

## 一、项目是什么

这是一个**多智能体电商推荐系统**：用户发起一次推荐请求，系统由一个「Supervisor 编排器」调度 4 个分工明确的 Agent，通过**并行 + 聚合**模式完成「画像 → 召回 → 重排 → 库存过滤 → 文案生成」的完整推荐流水线，并附带 A/B 测试与监控能力。

| 模块 | 位置 | 职责 |
|------|------|------|
| FastAPI 入口 | `main.py` | 6 个 HTTP 接口，两条推荐链路 |
| 4 个 Agent | `agents/` | 画像 / 商品推荐 / 营销文案 / 库存决策 |
| 编排器 | `orchestrator/` | Supervisor 模式 + LangGraph 状态图，两套实现 |
| 数据模型 | `models/schemas.py` | Pydantic 请求/响应/结果契约 |
| 配置 | `config/settings.py` | pydantic-settings，`ECOM_` 前缀环境变量 |
| 服务层 | `services/` | A/B 测试引擎 / Redis 特征存储 / 指标收集 |
| 测试 | `tests/test_ab_test.py` | A/B 引擎的 5 个单测 |

### 核心设计思想（先记住这 4 点）

1. **Agent 不是聊天机器人，是"带护栏的异步函数"**：每个 Agent 只做一件专项任务，输入输出都是结构化数据（Pydantic 模型），由 LLM 完成其中的"智能"环节。
2. **模板方法模式**：`BaseAgent.run()` 统一封装计时、重试（tenacity）、降级兜底；子类只需实现 `_execute()`。这是整个项目最值得抄的设计。
3. **并行编排**：能并行的阶段用 `asyncio.gather()` 同时跑（画像‖召回、重排‖库存），把总延迟从"串行相加"压成"取最大值"。
4. **两条等价链路**：`SupervisorOrchestrator`（手写编排，生产推荐用法）和 `graph.py`（LangGraph 状态图，展示声明式编排能力），业务逻辑相同，学会对比着读。

---

## 二、架构与数据流

```
POST /api/v1/recommend
        │
        ▼
┌─────────────────┐
│  Supervisor 编排器 │◄── A/B 引擎：按 user_id 哈希分桶定实验组
└────────┬────────┘
         │ 阶段1（并行）
         ├──────────────► 用户画像Agent（LLM 分析行为数据）
         └──────────────► 商品召回Agent（无画像，粗召回 2N 个）
         │ 阶段2（并行）
         ├──────────────► LLM 重排Agent（画像×候选 → 精排 N 个）
         └──────────────► 库存决策Agent（过滤缺货、预警、限购）
         │ 合并：重排名单 ∩ 有货名单 → 截取 N 个
         │ 阶段3（串行）
         └──────────────► 营销文案Agent（按用户分群选 Prompt 模板）
         ▼
RecommendationResponse（商品 + 文案 + 各 Agent 耗时/成败 + 总延迟）
```

关键数据流细节（`orchestrator/supervisor.py`）：

- 阶段 1 召回时**故意不传画像**（`user_profile=None`），因为画像还没生成完——这就是"先粗后精"的两段式召回。
- 库存检查用的是**原始候选集**（`raw_products`），而不是重排后的，两者并行所以只能用阶段 1 的产出。
- 库存过滤后若为空，会**降级为直接返回重排名单前 N 个**（宁可推缺货也不返回空列表）。
- 阶段 3 文案生成必须等最终商品列表确定后才能跑，所以只能串行收尾。

---

## 三、逐文件精读（建议按此顺序）

### 1. `models/schemas.py` — 先看契约，再看逻辑

所有 Agent 的输入输出都是这里的 Pydantic 模型，看懂它等于看懂系统接口：

- `UserProfile`：画像（分群枚举 `UserSegment`、偏好类目、价格区间、RFM 分数、实时标签）
- `Product`：商品（含 `score` 排序分）
- `AgentResult`：**所有 Agent 结果的基类**（agent_name / success / latency_ms / error / confidence）
- 4 个子类 `UserProfileResult` / `ProductRecResult` / `MarketingCopyResult` / `InventoryResult` 各自扩展业务字段
- `RecommendationRequest` / `RecommendationResponse`：API 出入口


### 2. `agents/base_agent.py`

父类agent:

父类先初始化Agent相关公共参数，子类在初始化时传入，执行父类run方法过程中调用各自重写后的execute抽象方法并返回执行结果实体，父类做统一计时、重试、降级兜底做统一处理。

### 3. `agents/` 四个具体 Agent

| Agent | LLM 用法 | Prompt 设计 | 兜底策略 |
|-------|---------|------------|---------|
| `UserProfileAgent` | temperature=0.3（求稳） | System Prompt 规定只输出 JSON；解析时剥离 \`\`\` 代码围栏，解析失败给默认画像 | JSON 解析失败 → `UserSegment.ACTIVE` 默认分群 |
| `ProductRecAgent` | 召回不用 LLM（本地 MOCK 排序），仅重排用 LLM | RERANK_PROMPT 用 `{user_profile}` `{candidates}` 占位符注入数据，要求输出商品 ID 数组 | LLM 输出解析失败 → 取前 N 个候选 |
| `MarketingCopyAgent` | temperature=0.9（求创意） | **按用户分群选模板**：新客/高价值/价格敏感/流失风险 各有专属文案风格；输出后过敏感词正则替换（广告法合规） | 无商品直接返回空；解析失败返回空列表 |
| `InventoryAgent` | **不用 LLM**，纯规则 | — | `self.db` 为空时直接用商品自带 stock 字段 |

用户画像agent:

根据客户端http请求获取到的用户id信息，经过中间层透传，对应用户购买行为在redis中存储的行为数据(用户偏好类目、价格区间等)，拼入HumanMessage中并利用chain调用llm生成原始画像描述json（提示词限制输出json），最后将画像描述解析为json格式组装为实体返回。

商品推荐agent:

将用户画像agent得到的画像摘要（偏好类别、价格区间等）和商品候选列表（商品名、类别、价格等）透传后拼接成到prompt中并作为HumanMessage，利用chain调用llm生成推荐商品列表json（提示词限制输出json），最后将推荐商品列表解析为json格式组装为实体添加到待返回列表，如果不满足推荐商品数量则从候选列表中补充，最后将待返回列表组装为实体返回。

营销文案agent:

根据用户画像agent得到的画像摘要和推荐商品列表透传，根据用户画像中用户标签（新用户、活跃用户、高价值用户等）选择对应prompt作为SystemPrompt，商品json列表组装为文本作为HumanMessage，为每个用户可能喜欢的商品生成营销文案，并整体解析为字典列表，最后将列表中每个字典数据做违禁词消除处理，最后将处理后的列表组装为实体返回。

库存决策agent:

从编排曾获取到商品对象列表，遍历列表中每个商品对象利用并利用MCP查库存，将大于0的商品组装到**可用列表**，与将库存告急的商品添加的**警告信息**实体和热卖商品**限购次数**一并组装为实体返回。


### 4. `orchestrator/supervisor.py` — 手写编排

- 构造函数组装 4 个 Agent + A/B 引擎（依赖注入）
- `recommend()` 三阶段调度，见上文数据流
- 用 `getattr(profile_result, "profile", None)` 防御式取值——即使画像 Agent 降级失败，流程也不断

### 5. `orchestrator/graph.py` — LangGraph 声明式编排

1 实例化所有agent对象

2 用户画像节点（调用用户画像agent）与商品推荐节点（无用户画像调用商品推荐agent-简单召回）并行运行

3 重排序节点（携带用户画像再次调用商品推荐agent）与库存检查（库存检查agent）并行运行

4 营销文案节点（调用营销文案agent）

5 聚合节点（计算总耗时）

### 6. `services/ab_test.py` — A/B 测试引擎

- **分桶**：`md5(user_id:experiment_id) % 100` → 按权重映射到实验组。哈希保证同一用户永远进同一组（一致性）
- **Thompson Sampling**：每组维护 Beta(successes, failures) 后验，`assign_thompson()` 每次采样取最大者分配流量，`record_outcome()` 更新后验——赢了自动多吃流量
- 内置两个实验：`rec_strategy`（规则重排 vs LLM 重排）、`copy_style`（正式 vs 口语文案）

### 7. `services/feature_store.py` / `metrics.py`

- FeatureStore：Redis Sorted Set 存行为流（score=时间戳），滑动窗口（1h/24h/7d）聚合出实时特征，RFM 启发式打分，离线标签（T+1）与在线标签合并。**注意：当前 Phase 1 没接 Redis，`redis_client=None` 时全部空转**，画像 Agent 走 context 兜底数据
- MetricsCollector：内存版指标收集（Agent 成功率/延迟 + 业务事件），注释写明生产换 Prometheus

### 8. `main.py` — FastAPI 入口

- `lifespan` 异步上下文管理器：启动时编译 LangGraph，优雅关闭时打日志
- 两条链路接口：`POST /api/v1/recommend`（Supervisor）和 `POST /api/v1/recommend/graph`（LangGraph）
- 观测接口：`/api/v1/experiments`（实验状态）、`/api/v1/metrics`（Agent 指标）
- 回报接口：`POST /api/v1/experiments/{id}/outcome` 把线上转化结果喂回 Thompson Sampling
---

启动时基于生命周期管理在启动前构建状态图（添加节点+编排节点）rec_graph

调用graph推荐接口时，从请求中提取用户ID、场景、推荐商品数量、上下文等信息填入state对应字段，再rec_graph.ainvoke(state)启动状态图异步执行，获取执行后推荐结果。

## 四、学习步骤

### 第 0 步：环境跑通（半天）

```bash
cd Mult-agent
pip install -r requirements.txt

# 配置 .env（config/settings.py 读取，前缀 ECOM_）
# ECOM_LLM_API_KEY=你的key
# ECOM_LLM_BASE_URL=https://api.minimax.chat/v1
# ECOM_LLM_MODEL=MiniMax-M1

python main.py          # 或 uvicorn main:app --reload
# 浏览器打开 http://localhost:8000/docs 用 Swagger 自测接口
```

不配 LLM key 时：LLM 调用会失败 → BaseAgent 兜底返回 success=False → 接口仍能返回（商品列表为空/降级）。**这本身就是一次观察降级机制的实验。**

### 第 1 步：先跑接口，建立体感（1 天）

- 用 `/docs` 发一个推荐请求，观察返回 JSON 里 `agent_results` 各 Agent 的 `success` / `latency_ms`
- 对比两次请求的 `experiment_group`：改 `user_id` 看是否换组，同一个 user_id 是否稳定同组
- 调 `/api/v1/experiments/{id}/outcome` 上报几次成功/失败，再看 `/api/v1/experiments` 的统计变化

### 第 2 步：按"契约 → 模板 → 业务 → 编排"顺序读代码（3~4 天）

1. `models/schemas.py`（半天）——画出类继承图
2. `agents/base_agent.py`（半天）——重点吃透 run/_retry_execute/_fallback 三层
3. `agents/` 四个实现（1.5 天）——每天 2~3 个，重点对比它们 LLM 用与不用、温度设置、Prompt 模板、解析兜底的差异
4. `orchestrator/supervisor.py` + `graph.py`（1 天）——对照着读，画出两份执行时序

### 第 3 步：补齐依赖知识（穿插进行）

| 知识点 | 在项目中的位置 | 需要掌握到什么程度 |
|--------|--------------|------------------|
| `asyncio.gather` | supervisor.py 两处 | 理解并发 vs 串行的延迟差异 |
| Pydantic v2 | schemas.py 全文件 | 会写 BaseModel、字段默认值、`model_dump()` |
| tenacity | base_agent.py | 看懂指数退避重试装饰器 |
| LangChain 消息 | 三个 LLM Agent | SystemMessage/HumanMessage、`ainvoke` 异步调用 |
| LangGraph StateGraph | graph.py | add_node/add_edge/compile/ainvoke |
| pydantic-settings | config/settings.py | 环境变量 → 配置对象的映射 |
| structlog | 各文件 | 结构化日志（key-value 而非字符串拼接） |
| Thompson Sampling | ab_test.py | Beta 分布采样 + 后验更新的直觉即可 |

### 第 4 步：做改造实验（巩固，1~2 周）

由易到难：

1. **加日志**：在 supervisor 各阶段打印耗时，验证并行确实生效（总延迟 ≈ 各阶段最大值之和，而非全部相加）
2. **加 Agent**：写一个 `PriceAgent`（比价/券后价），继承 BaseAgent，接入 Supervisor 阶段 2 并行组——检验你是否真的理解了模板方法
3. **换 Prompt**：给文案 Agent 加一种新分群模板；把敏感词表扩充
4. **接真数据**：给 `InventoryAgent.db` 注入一个 SQLite/MySQL 连接实现 `_check_stock`（Phase 2 预留的口子），体验依赖注入
5. **LangGraph 进阶**：给 graph.py 加条件边——当画像 Agent 失败时跳过重排直接用粗排结果
6. **跑测试**：`python tests/test_ab_test.py`，然后给 `BaseAgent` 的重试/兜底逻辑补单测

### 第 5 步：进阶方向

- 把 MetricsCollector 换成 prometheus-client + Grafana 看板
- FeatureStore 接真 Redis，画像 Agent 注入 `feature_store` 字段，打通实时特征链路
- 召回层接 Milvus 向量检索（`vector_store` 口子已留）
- 学习 LangGraph 的 checkpointer（状态持久化/断点续跑）与 human-in-the-loop

---

## 五、常见疑问预答

**Q1：为什么召回要 2N 个、候选要 3N 个？**
漏斗思维：粗召回量大 → 精排裁剪 → 库存过滤再裁。每一层都可能淘汰商品，入口不多备货最后就不够 N 个。

**Q2：Agent 失败了系统会挂吗？**
不会。BaseAgent.run() 捕获一切异常返回 `success=False` 的合法结果；编排器用 `getattr` 防御式取值，缺哪个就降级（如库存结果为空 → 不过滤直接截前 N）。

**Q3：为什么 supervisor 和 graph 各写一份？**
前者展示手写编排（生产可控），后者展示声明式图编排（LangGraph 生态）。这是教学项目故意做的双实现对照。

**Q4：MOCK_PRODUCTS 是干嘛的？**
Phase 1 的模拟商品池（15 个 3C 商品），等 Phase 2 接入真实向量检索/数据库后替换。

**Q5：temperature 为什么画像 0.3、文案 0.9？**
分析类任务要稳定低温度；创意类任务要发散高温度。这是 LLM 工程的常规调参直觉。

---

## 六、一页速查表

```
入口        main.py  →  /api/v1/recommend (Supervisor)  |  /api/v1/recommend/graph (LangGraph)
编排        supervisor.recommend()  →  3 阶段: gather(画像,召回) → gather(重排,库存) → 文案
Agent 基类   BaseAgent.run() = 计时 + tenacity 重试 + 异常兜底 → 子类只写 _execute()
4 个 Agent   画像(LLM) / 商品召回+重排(半LLM) / 营销文案(LLM+合规) / 库存(纯规则)
数据契约     AgentResult 基类 → 4 个子类；Pydantic 全链路类型安全
A/B 引擎     md5 哈希分桶(一致性) + Thompson Sampling(动态流量)
观测        MetricsCollector(内存) + structlog(结构化日志) + /experiments /metrics 接口
预留口子     feature_store / vector_store / self.db —— Phase 2 依赖注入真实数据源
```
