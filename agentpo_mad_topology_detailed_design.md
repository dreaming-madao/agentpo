# AgentPO MAD Topology 详细实现书

> 文档目标：给出一份可直接指导开发的、**非侵入式**的 AgentPO MAD topology 设计方案。  
> 本文档只描述设计与实现方案，**不直接落代码**。  
> 设计原则：
>
> - 不破坏现有 `assistant / critic` 逻辑
> - 通过 `args / config` 分流启用 `mad`
> - MAD 逻辑尽量独立存放在 `agentpo/debate/`
> - 支持本地 `vLLM` 和 OpenAI-compatible API 两种 debater backend
> - 优先做“**MAD 内并行，MAD 间保守**”
> - 参考 `/home/ly/debate-or-vote` 的 **vote-style MAD**，不使用新的四阶段 `debate solver`

---

## 1. 目标与边界

### 1.1 目标

在现有 AgentPO 基础上新增：

```text
cooperation_mode = mad
```

使训练过程变成：

1. Collaborator 对题目生成一段 `debate guide`
2. reward manager 内部调用冻结 actor，模拟多智能体 MAD
3. 多个 debater 按给定 topology 进行多轮讨论
4. 最终通过 majority vote 得到最终答案
5. 用最终答案和 ground truth 计算 reward
6. reward 仍然只回传给 Collaborator

### 1.2 不做什么

本文档明确不做：

- 不改 GRPO / DAPO 主训练逻辑
- 不把 MAD 写进 actor rollout 主流程
- 不替换现有 `assistant / critic` 路径
- 不引入 `/home/ly/debate-or-vote/src/debate_solver.py` 的四阶段新 solver
- 不在第一版默认启用 summarizer
- 不在第一版默认启用 `K rollout -> K MAD` 大规模并行

---

## 2. 当前代码结构与真实切入点

### 2.1 AgentPO 当前已有协作模式

当前 AgentPO 显式支持：

- `assistant`
- `critic`

对应位置：

- [agentpo/rl_dataset.py](/home/ly/agentpo/agentpo/rl_dataset.py:76)
- [agentpo/reward_manager_agentpo.py](/home/ly/agentpo/agentpo/reward_manager_agentpo.py:23)
- [scripts/train.sh](/home/ly/agentpo/scripts/train.sh:78)

### 2.2 真实训练数据流

当前 reward manager 的模式是：

1. rollout 产生 Collaborator response
2. reward manager decode 出该 response
3. 用该 response 作为协作信号，调用冻结 actor
4. actor 产出最终答案
5. `compute_score()` 评分
6. reward 打到 Collaborator response 的最后一个 token

这个结构非常适合插入 MAD。

### 2.3 为什么 MAD 应该放在 reward manager 侧

因为从 PPO / GRPO 视角看：

- policy 仍然只是 Collaborator
- actor / debater / vote 只是 reward environment 的一部分

所以：

- MAD 不应成为训练图的一部分
- MAD 只应成为 reward manager 内部的环境 rollout

---

## 3. 应该参考 debate-or-vote 的哪一部分

### 3.1 正确参考对象

应该参考：

- `/home/ly/debate-or-vote/src/main.py`
- 尤其是 `vote` 路径中的：
  - round 0 初始回答
  - `get_new_message()`
  - full / sparse / centralized topology
  - 每轮 revise
  - 最终投票

关键位置：

- [src/main.py](/home/ly/debate-or-vote/src/main.py:135)
- [src/main.py](/home/ly/debate-or-vote/src/main.py:495)

### 3.2 不参考的部分

不参考：

- `/home/ly/debate-or-vote/src/debate_solver.py`

原因：

- 它是另一套四阶段新 solver
- 你的需求是 MAD topology，不是换 solver
- 你已明确说“普通投票即可”

---

## 4. 总体设计原则

### 4.1 非侵入式

实现方式必须满足：

- 原来的 `assistant` 路径不变
- 原来的 `critic` 路径不变
- 新增 `mad` 只在分流命中时启用

### 4.2 分层

MAD 相关代码建议全部收拢在：

```text
agentpo/debate/
```

不要把 topology、backend、vote、prompt 都直接揉进 `reward_manager_agentpo.py`。

### 4.3 并行分层

并行策略分两层：

1. **单个 MAD 实例内部**
2. **多个 rollout responses 对应的多个 MAD 实例之间**

建议：

- `N agents` 并行：做
- `K rollouts` 并行：第一版保守处理

### 4.4 backend 解耦

MAD 的上层逻辑不应直接依赖：

- OpenAI API
- vLLM `generate`

而应抽象为统一的 backend 接口。

---

## 5. 目标目录结构

推荐新增：

```text
agentpo/
  debate/
    __init__.py
    config.py
    prompts.py
    topology.py
    aggregator.py
    backend.py
    runner.py
```

### 5.1 各文件职责

`config.py`

- MADConfig
- 默认参数
- 参数校验

`prompts.py`

- debater 初始轮 prompt
- debater revise 轮 prompt
- persona 生成
- 文本裁剪

`topology.py`

- full / sparse / centralized 邻居选择

`aggregator.py`

- 从 response 提取答案
- 答案标准化
- majority vote

`backend.py`

- API backend
- vLLM backend
- 统一 `generate_batch()` 接口

`runner.py`

- `run_mad(...)`
- 控制 round 0 / round r / aggregation / history

---

## 6. 配置设计

### 6.1 分流主开关

主开关保持现有风格：

```text
algorithm.cooperation_mode = assistant | critic | mad
```

### 6.2 MAD 专属配置建议

建议增加：

```text
algorithm.mad.backend = api | vllm
algorithm.mad.num_agents = 3
algorithm.mad.debate_rounds = 2
algorithm.mad.topology = full
algorithm.mad.multi_persona = false
algorithm.mad.temperature = 0.0
algorithm.mad.top_p = 1.0
algorithm.mad.max_tokens = 2048
algorithm.mad.timeout = 6000
algorithm.mad.max_peer_chars = 2000
algorithm.mad.parallel_agents = true
algorithm.mad.parallel_rollouts = false
algorithm.mad.max_concurrency = 3
algorithm.mad.api_batch_size = 8
algorithm.mad.use_majority_vote = true
```

### 6.3 推荐默认值

第一版建议默认：

```text
num_agents = 3
debate_rounds = 2
topology = full
multi_persona = false
parallel_agents = true
parallel_rollouts = false
backend = api 或 vllm 二选一
temperature = 0
use_majority_vote = true
```

理由：

- 最贴近你要的普通投票 MAD
- 成本可控
- 容易验证

---

## 7. 数据流设计

### 7.1 单样本数据流

对单条 rollout sample：

```text
problem q
  ->
Collaborator rollout
  ->
guide z
  ->
MAD environment rollout
  ->
final solution y_hat
  ->
compute_score(y_hat, y*)
  ->
reward only to collaborator
```

### 7.2 MAD 内部数据流

```text
Round 0:
  Agent_1 ... Agent_N 独立回答

Round 1..R:
  每个 Agent 读取上一轮自己的答案和 peer 答案
  根据 topology 修正答案

Aggregation:
  从最后一轮答案中提取 boxed answer
  做 majority vote
```

### 7.3 对 K rollout 的理解

对于同一个原始 prompt，rollout 可能采样出 K 个 Collaborator responses。

每个 Collaborator response 都应被视为一个独立 guide，因此：

```text
1 prompt group
-> K collaborator responses
-> K 次独立 MAD
-> K 个 reward
```

不是：

```text
1 prompt group
-> 只做 1 次 MAD
-> 全组共享 reward
```

原因：

- 当前 PPO/GRPO 奖励粒度是“单条 response”
- 如果全组只做一次 MAD，会削弱 Collaborator response 的区分度

---

## 8. 并行策略设计

这是整个实现里最需要提前说清楚的部分。

### 8.1 原则

建议采用：

```text
MAD 内并行，MAD 间保守
```

即：

- 单次 MAD 中的 N 个 agents 并行
- 多个 rollout response 对应的多个 MAD 实例不默认全部并行

### 8.2 单个 MAD 实例内部并行

#### Round 0

各 agent 彼此独立，可并行。

#### Debate Round r

同一轮的每个 agent 只依赖：

- 自己上一轮回答
- peers 上一轮回答

因此可采用：

```text
轮间同步
轮内并行
```

也就是：

```text
Round 0: Agent1..AgentN 并行
等待全部完成
Round 1: Agent1..AgentN 并行
等待全部完成
Round 2: Agent1..AgentN 并行
...
```

### 8.3 多个 MAD 实例之间并行

对于同一个 batch 中的 K 条 collaborator responses：

- 理论上可以并行
- 实践上第一版不建议默认做满

原因：

- 单次 MAD 已经是 `N * (R+1)` 次生成
- 若同时对 K 个 rollout 全并行，负载会迅速升高
- API 容易打爆，vLLM 容易造成调度和显存压力

### 8.4 默认建议

```text
parallel_agents = true
parallel_rollouts = false
```

### 8.5 后续可选增强

如未来要支持 rollout-level 并行，建议只做成：

```text
parallel_rollouts = true
max_concurrent_mad = 2 或 4
```

不要直接让 K 次 MAD 全部放开。

---

## 9. backend 设计

### 9.1 为什么必须做 backend 层

如果 backend 不单独抽象，`runner.py` 会混入：

- OpenAI client 细节
- vLLM prompt batch 细节
- tokenizer / chat template 细节

这样后面会非常难维护。

所以建议 `runner.py` 只处理：

- prompt 组织
- topology
- round 控制
- history

backend 只处理：

- 给一批 messages
- 返回一批 responses

### 9.2 统一接口建议

不论是 API 还是 vLLM，都统一暴露：

```python
generate_batch(message_batches: list[list[dict]]) -> list[str]
```

其中：

- `message_batches[i]` 是第 i 个 agent 的聊天消息
- 返回值 `responses[i]` 与其一一对应

### 9.3 API backend 设计

#### 输入

- `list[list[message]]`

#### 处理

- 每个 agent 一次 `chat.completions.create(...)`
- 用 `ThreadPoolExecutor` 做 agent-level 并发
- 使用固定 `max_workers`

#### 不建议的做法

不建议像 benchmark 一样默认：

```python
max_workers = len(tasks)
```

训练时更稳妥的是：

```text
max_workers = min(num_agents, max_concurrency)
```

#### 适用场景

- 本地 OpenAI-compatible endpoint
- SiliconFlow / DashScope 一类 API

### 9.4 vLLM backend 设计

#### 输入

- `list[list[message]]`

#### 处理

1. 把每个 message batch 转成单条 prompt string
2. 用 tokenizer 应用 chat template
3. 把 N 个 prompts 拼成一个 batch
4. 一次 `llm.generate(prompts, sampling_params)`

#### 重要建议

对于本地 vLLM，不建议：

- 每个 agent 独立开线程各自 `generate`

建议：

- 每一轮 agent prompts 打成一个 batch，一次 generate

这是最自然也最稳的 agent-level 并行方式。

### 9.5 backend 选择方式

建议由配置分流：

```text
algorithm.mad.backend = api | vllm
```

如果要更细，可以把 actor model 是否在 `local_model_list` 中作为辅助判断，但主逻辑最好仍显式配置。

---

## 10. topology 设计

### 10.1 full

每个 agent 看所有其他 agent。

#### 优点

- 信息最完整

#### 缺点

- prompt 最长
- 成本最高

#### 建议

第一版默认采用。

### 10.2 sparse

每个 agent 只看左右邻居。

#### 优点

- 更短 prompt
- 更贴近拓扑研究

#### 缺点

- 信息传播慢

### 10.3 centralized

- 0 号 agent 看所有其他 agent
- 其他 agent 只看 0 号 agent

#### 优点

- 与 debate-or-vote 保持一致

#### 缺点

- 中心 agent 偏置更强

### 10.4 与并行的关系

三种 topology 都满足：

- 同一轮只依赖上一轮结果
- 不依赖当前轮即时结果

所以都支持：

```text
轮内并行
轮间同步
```

---

## 11. prompt 设计

### 11.1 Collaborator 的职责

在 `mad` 模式下，Collaborator 不直接解题，而是生成：

```text
debate guide / coordination signal
```

建议内容：

- 关键子问题
- 常见陷阱
- 推荐交叉检查点
- 最终答案格式要求

### 11.2 Debater 初始轮 prompt

目标：

- 基于 `problem + guide`
- 独立解题
- 输出完整推理
- 最终答案放在 `\boxed{}`

### 11.3 Debater revise 轮 prompt

目标：

- 提供：
  - 当前 problem
  - collaborator guide
  - 自己上一轮回答
  - peers 上一轮回答
- 要求 agent：
  - 参考 peers
  - 但不要盲从
  - 如果发现错误则修正
  - 最终答案继续用 `\boxed{}`

### 11.4 Persona

第一版建议：

- `multi_persona = false`

也就是说所有 debater 用同一种默认 persona。

后续可选：

- Solver
- Skeptic
- Checker

但这不是第一版必要项。

---

## 12. Aggregation 设计

### 12.1 第一版默认 aggregation

第一版建议只做：

```text
majority vote
```

### 12.2 具体流程

1. 从每个最后一轮 response 中提取最后一个 `\boxed{}`
2. 对提取出的答案做轻量标准化
3. 用 `Counter` 统计频次
4. 选择最高票答案
5. 平票时采用固定 deterministic tie-break

### 12.3 tie-break 建议

建议固定为：

- 优先按出现顺序选第一个最高票答案

不要用随机 tie-break，因为：

- 会增加训练噪声
- 不利于复现

### 12.4 fallback

如果全部提取失败：

- fallback 到第一个完整 response
- 让 `compute_score()` 尽量解析

### 12.5 暂不默认启用 summarizer

原因：

- 你要求“普通投票即可”
- summarizer 会额外增加一次模型调用
- 增大不稳定性和成本

---

## 13. 代码级实现方案

### 13.1 `agentpo/rl_dataset.py`

#### 目标

新增 `mad` prompt 分支。

#### 要改的点

1. 新增 `mad_system_prompt`
2. 在 `_build_messages()` 中新增：

```python
elif self.config["cooperation_mode"] == "mad":
    system_prompt = mad_system_prompt
```

3. 最好加一个兜底：

```python
else:
    raise RuntimeError(...)
```

#### 输出形态

Collaborator 生成的是 `debate guide`，不是最终答案。

### 13.2 `agentpo/reward_manager_agentpo.py`

#### 目标

保留 `assistant / critic`，新增 `mad` 分支。

#### 推荐改法

1. 顶部 import：

```python
from .debate import MADConfig, run_mad
```

2. `AgentPORewardManager.__init__()` 增加：

```python
mad_config=None
```

3. `self.mad_config = MADConfig(**(mad_config or {}))`

4. `__call__()` 改成三段式：

- 第一段：收集 items
- 第二段：按 mode 生成 final solutions
- 第三段：统一 compute_score

#### 核心原则

这一句不变：

```python
reward_tensor[idx, valid_response_length - 1] = score
```

### 13.3 `agentpo/main_dapo.py`

#### 第一版建议

主逻辑尽量不动。

#### 可选增强

把：

```text
algorithm.mad
```

透传给 reward manager。

### 13.4 `scripts/train.sh`

#### 至少要做

允许：

```bash
cooperation_mode=mad
```

#### 可选增强

增加：

```bash
mad_backend=api
mad_num_agents=3
mad_debate_rounds=2
mad_topology=full
mad_parallel_agents=True
mad_parallel_rollouts=False
mad_max_concurrency=3
```

---

## 14. `agentpo/debate/` 模块详细设计

### 14.1 `config.py`

负责：

- `MADConfig`
- 参数校验

建议字段：

- `backend`
- `num_agents`
- `debate_rounds`
- `topology`
- `multi_persona`
- `temperature`
- `top_p`
- `max_tokens`
- `timeout`
- `max_peer_chars`
- `parallel_agents`
- `parallel_rollouts`
- `max_concurrency`
- `use_majority_vote`

### 14.2 `prompts.py`

负责：

- persona 生成
- prompt 构造
- peer response 裁剪

建议函数：

- `get_personas(...)`
- `truncate_text(...)`
- `build_initial_debater_messages(...)`
- `build_debate_messages(...)`

### 14.3 `topology.py`

负责：

- 根据 topology 选 peer names

建议函数：

- `select_peers(agent_idx, agent_names, topology)`

### 14.4 `aggregator.py`

负责：

- 提取 boxed answer
- 标准化
- vote

建议函数：

- `extract_boxed_answer(text)`
- `normalize_vote_answer(answer)`
- `majority_vote(responses)`

### 14.5 `backend.py`

负责：

- backend 抽象
- API backend
- vLLM backend

建议结构：

- `build_backend(cfg, actor_model, tokenizer=None, llm=None)`
- `ApiDebateBackend.generate_batch(...)`
- `VLLMDebateBackend.generate_batch(...)`

### 14.6 `runner.py`

负责：

- `run_mad(...)`
- round 控制
- history 收集

建议返回：

```python
final_solution, history
```

其中 `history` 至少包含：

- config
- agent_names
- 每轮 responses
- aggregation meta
- final_solution

---

## 15. API backend 详细方案

### 15.1 client 初始化

复用当前 `reward_manager_agentpo.py` 的模型路由逻辑：

- 本地 OpenAI-compatible endpoint
- SiliconFlow
- DashScope / Aliyun

### 15.2 单轮并行

对于当前轮的 `N` 个 agent messages：

- 用线程池并发请求
- `max_workers = min(N, cfg.max_concurrency)`

### 15.3 为什么用线程池

因为 API 请求本质是 I/O bound：

- 网络等待为主
- 线程池足够
- 不必引入复杂 async 改造

### 15.4 稳定性建议

建议在 backend 内部支持：

- timeout
- exception catch
- 空响应 fallback
- 简单重试策略

第一版即使不加重试，也要保证失败不会直接把整个 batch 弄崩。

---

## 16. vLLM backend 详细方案

### 16.1 初始化

参考：

- [agentpo/evaluation/math_eval_promptpo.py](/home/ly/agentpo/agentpo/evaluation/math_eval_promptpo.py:278)

需要：

- `LLM(...)`
- `SamplingParams(...)`
- optional tokenizer for chat template

### 16.2 单轮生成方式

对当前轮全部 `N` 个 agent prompts：

1. 转为 prompt string
2. 组成 prompt list
3. 一次 `llm.generate(...)`

### 16.3 不建议做什么

不建议：

- 为每个 agent 独立开线程分别 `generate()`

原因：

- vLLM 对 batch generate 最友好
- 多次 generate 并发不一定更快，且更不稳

### 16.4 与 rollout-level 并行的关系

对于 vLLM backend，第一版更建议：

- 单个 MAD 内部批生成
- 多个 MAD 实例串行处理

避免多个 `generate()` 实例互相争抢资源。

---

## 17. rollout-level 并行策略

### 17.1 默认策略

默认：

```text
parallel_rollouts = false
```

### 17.2 为什么默认关闭

假设：

- `K = 16`
- `N = 3`
- `R = 2`

那么一次 batch 可能触发：

```text
K * N * (R + 1) = 16 * 3 * 3 = 144 次 agent generation
```

这还没算训练 batch size 带来的放大。

### 17.3 后续可选策略

如未来开启 rollout 并行，建议：

- 只允许少量 `max_concurrent_mad`
- 优先仅在 API backend 尝试

不建议 vLLM backend 默认开启。

---

## 18. history / 日志设计

### 18.1 为什么要记录

MAD 比单轮 actor 推理复杂得多，如果没有 history，很难 debug：

- peer 消息是否构造正确
- topology 是否生效
- aggregation 是否合理

### 18.2 建议记录的内容

每个 MAD 实例记录：

```python
{
    "config": {...},
    "agent_names": [...],
    "rounds": [
        {"round_idx": 0, "responses": {...}},
        {"round_idx": 1, "responses": {...}},
    ],
    "aggregation": {
        "method": "majority_vote",
        "answers": [...],
        "counts": {...},
        "winner": "...",
    },
    "final_solution": "...",
}
```

### 18.3 是否写入 reward_extra_info

可以，但第一版要注意：

- history 可能较大
- 训练时批量保存可能造成额外内存开销

建议：

- 默认只在 debug 模式记录完整 history
- 正常训练时可只记录聚合结果摘要

---

## 19. 风险点与应对

### 19.1 风险：训练变慢

MAD 比单次 actor 生成复杂很多。

#### 应对

- `num_agents=3`
- `debate_rounds=2`
- `temperature=0`
- 默认不启用 rollout-level 并行

### 19.2 风险：API 不稳定

#### 应对

- timeout
- exception catch
- 保守并发上限
- 可选重试

### 19.3 风险：prompt 过长

特别是 full topology 下，多轮后 peer responses 会很长。

#### 应对

- `max_peer_chars`
- 对 peer response 做截断

### 19.4 风险：训练噪声变大

如果：

- debater temperature 高
- vote tie-break 随机

则 reward 方差会变大。

#### 应对

- 默认 `temperature=0`
- tie-break deterministic

### 19.5 风险：影响原模式

#### 应对

- 所有新逻辑只在 `cooperation_mode == "mad"` 命中
- 原 `assistant / critic` 分支尽量不动

---

## 20. 分阶段实施计划

### Phase 1：最小可跑通版

目标：

- 只支持 `mad + majority_vote`
- backend 先做 `api`
- `N agents` 并行
- `K rollouts` 串行

包含：

- `rl_dataset.py` 新增 `mad`
- `agentpo/debate/` 新增最小模块
- `reward_manager_agentpo.py` 接入 `mad`

### Phase 2：本地 vLLM backend

目标：

- `backend = vllm`
- 每轮 agent prompts batch generate

### Phase 3：配置完善

目标：

- 让 `algorithm.mad.*` 可从 Hydra / train.sh 透传

### Phase 4：受控 rollout-level 并行

目标：

- 可选 `parallel_rollouts = true`
- 带全局限流

---

## 21. 推荐的第一版开发顺序

建议按这个顺序做：

1. 修改 [agentpo/rl_dataset.py](/home/ly/agentpo/agentpo/rl_dataset.py:76)，增加 `mad` prompt
2. 新建 `agentpo/debate/config.py`
3. 新建 `agentpo/debate/prompts.py`
4. 新建 `agentpo/debate/topology.py`
5. 新建 `agentpo/debate/aggregator.py`
6. 新建 `agentpo/debate/backend.py`，先做 API 版
7. 新建 `agentpo/debate/runner.py`
8. 修改 [agentpo/reward_manager_agentpo.py](/home/ly/agentpo/agentpo/reward_manager_agentpo.py:23)，接入 `mad`
9. 修改 [scripts/train.sh](/home/ly/agentpo/scripts/train.sh:78)，允许 `cooperation_mode=mad`
10. 做 smoke test

---

## 22. smoke test 建议

在完整训练前，建议先做以下验证：

1. `cooperation_mode=mad` 时 `rl_dataset.py` 能正常构造 Collaborator prompt
2. reward manager 能识别 `mad`
3. 单条样本下 `run_mad()` 能跑通：
   - round 0
   - 至少 1 轮 debate
   - majority vote
4. reward 能正确回填到最后一个 token
5. 原 `assistant / critic` 路径完全不受影响

---

## 23. 最终建议

最适合当前项目的实现方式是：

```text
通过 cooperation_mode=mad 分流；
把 MAD 拆到 agentpo/debate/；
引入独立 backend 层支持 api / vllm；
默认只做单个 MAD 内部 agent-level 并行；
多个 rollout response 对应的多个 MAD 先保守串行；
最终采用 majority vote；
reward 仍然只训练 Collaborator。
```

一句话概括：

```text
先做“拓扑正确、结构清晰、原逻辑不坏、负载可控”的 MAD，
再逐步扩展 backend 和并行度。
```

