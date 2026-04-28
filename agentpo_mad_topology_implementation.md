# 在 AgentPO 上非侵入式实现 MAD Topology

> 目标：基于现有 `agentpo` 代码，在**不改动 GRPO / DAPO 主训练逻辑**的前提下，为 AgentPO 新增一个 `cooperation_mode=mad`。  
> 这里的 MAD 参考 `/home/ly/debate-or-vote` 的**原始 vote-style 多智能体拓扑流程**，**不使用**后来新增的 `solver=debate` 四阶段新方法。

---

## 1. 先说结论

当前 AgentPO 的训练对象只有 Collaborator，小模型 rollout 生成的 response 会被 reward manager 当作“协作信号”，再交给冻结的 actor 模型生成最终解答，然后只把最终 reward 回传给 Collaborator。

因此实现 MAD topology 的**正确切入点**是：

1. 保持 `main_dapo.py`、`dapo_ray_trainer.py`、verl PPO/GRPO 主流程不动；
2. 给 `rl_dataset.py` 增加一个 `cooperation_mode="mad"`，让 Collaborator 学会输出“debate guide / coordination signal”；
3. 在 `reward_manager_agentpo.py` 中新增 `mad` 分支；
4. 把 MAD 的多 agent 讨论逻辑单独封装到 `./agentpo/debate/` 下的新模块；
5. actor / debater / 聚合器全部冻结，不参与反向传播；
6. reward 仍然只打到 Collaborator response 的最后一个有效 token 上。

这就是最符合你“**非侵入式**”要求的实现方式。

---

## 2. 对现有代码的真实结构判断

### 2.1 AgentPO 当前已经实现了什么

从代码看，当前 AgentPO 只显式支持两种协作模式：

- `assistant`
- `critic`

对应位置：

- [agentpo/rl_dataset.py](/home/ly/agentpo/agentpo/rl_dataset.py:83)
- [agentpo/reward_manager_agentpo.py](/home/ly/agentpo/agentpo/reward_manager_agentpo.py:144)
- [scripts/train.sh](/home/ly/agentpo/scripts/train.sh:78)

其中：

- `assistant`：Collaborator 生成 hint / question rewrite 风格的辅助信号；
- `critic`：Collaborator 对给定 solution 做评论 / 指错；
- reward manager 再调用冻结 actor，通过 API 生成最终答案；
- 最终 score 只回传给 Collaborator。

### 2.2 现有 reward 流程

[agentpo/reward_manager_agentpo.py](/home/ly/agentpo/agentpo/reward_manager_agentpo.py:97) 的核心流程是：

1. 从 rollout batch decode 出 prompt 和 Collaborator response；
2. 根据 `cooperation_mode` 组装 actor prompt；
3. 调用 `get_solution()` 用冻结 actor 生成最终答案；
4. `compute_score()` 对最终答案打分；
5. 把 reward 放到 Collaborator response 的最后一个 token 上。

这说明：

- MAD 应该被视为 reward manager 内部的“环境模拟”；
- 不应该把 MAD 写进 actor rollout 主图里；
- 也不应该去改 PPO / GRPO 的 advantage、loss、sampling 主流程。

---

## 3. 参考 debate-or-vote 时，应该参考哪一部分

这个地方是原文档里最需要纠正的点。

### 3.1 正确参考对象

应该参考 `/home/ly/debate-or-vote/src/main.py` 的 **vote solver / MAD communication topology**，也就是：

- round 0 各 agent 独立回答；
- 后续 round 中，每个 agent 读取其他 agent 的上一轮回答；
- 支持：
  - `full` / decentralized
  - `sparse`
  - `centralized`
- 每轮用 evaluator 抽取 answer；
- 最终用 vote / aggregation 得到群体答案。

对应关键逻辑：

- [src/main.py](/home/ly/debate-or-vote/src/main.py:135) `get_new_message()`
- [src/main.py](/home/ly/debate-or-vote/src/main.py:495) vote-style debate loop

### 3.2 不应该参考哪一部分

**不要把** `/home/ly/debate-or-vote/src/debate_solver.py` 的 4-stage 新 debate solver 当成目标实现。

原因：

1. 你已经明确说了“不要 solver 使用新的 debate 方法，普通投票即可”；
2. `debate_solver.py` 是后加的另一套流程，不是原始 MAD topology；
3. 它是“逐轮缩小候选空间”的结构化 solver，不是 AgentPO 现在最自然能接上的多轮互看-修正式 topology；
4. 如果照它实现，会把你的需求从“topology”变成“新 solver”，偏题了。

因此本项目里提到的 MAD，应该严格理解为：

```text
multi-agent vote-style debate topology
```

而不是：

```text
new 4-stage debate solver
```

---

## 4. 推荐的实现形态

### 4.1 目录组织

为了非侵入式，建议新增目录：

```text
agentpo/
  debate/
    __init__.py
    config.py
    prompts.py
    topology.py
    aggregator.py
    runner.py
```

最小版本也可以先只做：

```text
agentpo/
  debate/
    __init__.py
    mad_runner.py
```

但从后续可维护性考虑，我更推荐前一种拆法。

### 4.2 模块职责建议

`config.py`

- 定义 `MADConfig`
- 包含：
  - `num_agents`
  - `debate_rounds`
  - `topology`
  - `multi_persona`
  - `use_summarizer`
  - `temperature`
  - `top_p`
  - `max_tokens`
  - `max_peer_chars`

`prompts.py`

- 初始回答 prompt
- 多轮 debate prompt
- optional summarizer prompt
- persona prompt 模板

`topology.py`

- `full` 邻居选择
- `sparse` 邻居选择
- `centralized` 邻居选择

`aggregator.py`

- 从每个 debater response 提取 final answer
- majority vote
- tie-break 策略
- optional summarizer aggregation

`runner.py`

- 给定 `problem + collaborator_signal + actor_model + MADConfig`
- 执行完整 MAD rollout
- 返回：
  - `final_solution`
  - `history`

---

## 5. 正确的 MAD 训练数据流

对单个训练样本 `(q, y*)`，数据流应该是：

```text
Collaborator rollout:
  z ~ pi_theta(. | q)

Reward manager 内部环境:
  round 0:
    a_1^(0), ..., a_N^(0) = FrozenActor(q, z)

  round 1..R:
    a_j^(r) = FrozenActor(q, z, own_prev, peer_prev)

Aggregation:
  y_hat = VoteOrSummarize(a_1^(R), ..., a_N^(R))

Scoring:
  reward = compute_score(y_hat, y*)

Training:
  只把 reward 回传给 z 的最后一个 token
```

关键点：

- debater 是冻结 actor 的多次调用，不是独立可训练参数；
- topology 发生在 reward manager 里；
- Collaborator 只学会“如何写出更有用的 debate guide”；
- MAD rollout 的中间文本不参与梯度。

---

## 6. 需要改哪些文件

### 6.1 `agentpo/rl_dataset.py`

这是原文档遗漏的重要改动。

当前 `RLHFCustomDataset._build_messages()` 只支持：

- `critic`
- `assistant`

对应代码在 [agentpo/rl_dataset.py](/home/ly/agentpo/agentpo/rl_dataset.py:83)。

所以必须新增：

```python
elif self.config["cooperation_mode"] == "mad":
    system_prompt = mad_system_prompt
```

建议 `mad_system_prompt` 的职责不是“解题”，而是“给辩论系统写引导”：

```text
You are a debate coordinator for a multi-agent reasoning system.
Given a math problem, produce a concise debate guide that helps debaters:
- identify key subproblems,
- avoid common pitfalls,
- check each other's reasoning,
- keep the final answer format consistent.
Do not fully solve the problem.
```

也就是说，Collaborator 输出内容建议是：

- 关键思路
- 易错点
- 校验重点
- 最终答案格式要求

而不是直接给出完整标准解。

### 6.2 `agentpo/reward_manager_agentpo.py`

这是主改动点。

当前文件里：

- `get_solution()` 只支持 `assistant` / `critic`
- `AgentPORewardManager.__call__()` 里也只支持这两种分支

所以应该改成：

1. 保留现有 `assistant` / `critic` 逻辑不动；
2. 为 `mad` 单独增加分支；
3. `mad` 分支调用 `agentpo.debate.runner.run_mad(...)`。

推荐结构：

```python
if self.cooperation_mode == "assistant":
    ...
elif self.cooperation_mode == "critic":
    ...
elif self.cooperation_mode == "mad":
    final_solution, history = run_mad(...)
else:
    raise RuntimeError(...)
```

最终依然执行：

```python
result = self.compute_score(...)
reward_tensor[i, valid_response_length - 1] = score
```

这一点不能变。

### 6.3 `agentpo/main_dapo.py`

这里基本**不用改主逻辑**。

它已经会把：

- `config.algorithm.cooperation_mode`
- `config.reward_model.actor_model`

传给 `AgentPORewardManager`，见：

- [agentpo/main_dapo.py](/home/ly/agentpo/agentpo/main_dapo.py:190)
- [agentpo/main_dapo.py](/home/ly/agentpo/agentpo/main_dapo.py:202)

所以只需要确保：

- `cooperation_mode=mad` 能被下游识别；
- 如需额外 MAD 参数，可以通过 Hydra config 再补一组 `algorithm.mad.*` 或 `reward_model.mad.*`。

### 6.4 `scripts/train.sh`

这里也要补最小配置入口。当前脚本只写了：

```bash
cooperation_mode=assistant # critic assistant
```

建议改成支持：

```bash
cooperation_mode=mad
```

以及后续可选加入：

```bash
mad_num_agents=3
mad_debate_rounds=2
mad_topology=full
mad_multi_persona=False
mad_use_summarizer=False
```

---

## 7. 具体实现步骤

这是我建议的实际落地顺序。

### Step 1. 先加 `mad` 数据集 prompt

目标：

- 让 Collaborator rollout 生成的是“debate guide”，而不是 hint 或 critic comment。

修改：

- [agentpo/rl_dataset.py](/home/ly/agentpo/agentpo/rl_dataset.py:76)

完成标准：

- `cooperation_mode=mad` 时可以正常构造训练 prompt。

### Step 2. 新建 `agentpo/debate/` 模块

目标：

- 把 MAD 逻辑从 reward manager 中剥离出来；
- 避免把多轮 debate 细节塞进一个文件。

建议最小接口：

```python
final_solution, history = run_mad(
    problem=problem,
    collaborator_signal=signal,
    actor_model=actor_model,
    cfg=mad_cfg,
)
```

完成标准：

- 单文件或子模块内可以独立完成：
  - round 0 initial answers
  - round 1..R debate
  - topology peer selection
  - aggregation

### Step 3. 在 reward manager 接 `mad` 分支

目标：

- 不破坏 `assistant/critic`；
- 只在 `mad` 时走新的 debate runner。

修改：

- [agentpo/reward_manager_agentpo.py](/home/ly/agentpo/agentpo/reward_manager_agentpo.py:23)

完成标准：

- `cooperation_mode=mad` 时，reward manager 能：
  - decode collaborator response
  - 调用 MAD rollout
  - 拿到 final solution
  - compute score
  - 正确回填 reward tensor

### Step 4. 加最小配置项

目标：

- 不把 debate 超参数硬编码死。

建议配置项：

- `num_agents`
- `debate_rounds`
- `topology`
- `multi_persona`
- `use_summarizer`
- `temperature`

完成标准：

- 可以在 `train.sh` 或 Hydra override 中直接切换 topology。

### Step 5. 做一个离线 smoke test

建议先不跑完整训练，先人工构造一个 batch/sample 验证：

1. `cooperation_mode=mad`
2. Collaborator response 是一段 guide
3. reward manager 成功调用 3-agent / 2-round MAD
4. 最终答案被 `compute_score()` 正常评分

完成标准：

- 不报错
- reward tensor shape 正常
- `valid_response_length - 1` 位置被成功赋值

---

## 8. topology 设计建议

### 8.1 full

每个 agent 看其他所有 agent 的上一轮回答。

优点：

- 信息最充分

缺点：

- prompt 最长
- API 成本最高

建议：

- 先作为默认实现

### 8.2 sparse

每个 agent 只看左右邻居。

优点：

- prompt 更短
- 更接近拓扑约束实验

缺点：

- 收敛可能更慢

### 8.3 centralized

一个中心 agent 看所有人，其他 agent 只看中心 agent。

优点：

- 与 `debate-or-vote` 保持一致

缺点：

- 中心 agent 偏置更强

### 8.4 推荐起步方案

第一版建议：

- `num_agents=3`
- `debate_rounds=2`
- `topology=full`
- `multi_persona=False`
- `use_summarizer=False`
- `temperature=0`

原因：

- 先把系统跑通；
- majority vote 比 summarizer 更稳定、也更贴近你说的“普通投票即可”。

---

## 9. aggregation 建议

你的需求里已经强调“普通投票即可”，所以第一版建议：

1. 从每个 debater response 中提取 `\boxed{}`；
2. 做 majority vote；
3. 若平票：
   - 先选最先出现的候选，保证 deterministic；
   - 或者选第一个 agent 的答案作为 fallback；
4. 仅当全部解析失败时，再 fallback 到某个完整 response。

因此：

- 第一版**不建议默认启用 summarizer**
- summarizer 可以作为后续可选实验开关

---

## 10. 对原文档的主要纠错结论

下面是这份文档里原先不准确、现在已经纠正的点。

### 10.1 正确点

原文档这些判断是对的：

- “不需要改 GRPO/DAPO 主训练逻辑”
- “MAD 应该放在 reward manager 侧”
- “只有 Collaborator 参与训练，debater / summarizer 全冻结”
- “最好把 debate 单独写到独立 `.py` 模块里”

### 10.2 需要修正的点

原文档这些地方不够准确：

1. 把 `debate-or-vote` 的“参考流程”写成了泛化的 MAD，但没有明确区分 `vote solver` 和 `debate solver`
2. 文档示例里把 `use_summarizer=True` 当成默认，更偏离你“普通投票即可”的要求
3. 原文档低估了 `rl_dataset.py` 的必要改动，这不是可选项，而是必须项
4. 原文档默认把新模块写成单个 `mad_topology.py`，可行，但从维护性上更推荐放入 `agentpo/debate/`
5. 如果直接照文档把 `extract_boxed` 和 aggregation 写死在 reward manager 里，会破坏你要的“非侵入式”

---

## 11. 我建议的最终落地方案

如果现在开始实现，我会按下面的最小闭环来做：

1. 在 `rl_dataset.py` 新增 `mad` prompt
2. 新建 `agentpo/debate/`，先实现：
   - peer selection
   - round 0 / round r prompt builder
   - majority vote aggregator
   - `run_mad()`
3. 在 `reward_manager_agentpo.py` 接入 `mad`
4. 在 `scripts/train.sh` 暴露最小 MAD 参数
5. 先做 smoke test，再考虑 persona / summarizer / history logging

这个方案和你当前需求是一致的：

- 非侵入式
- 拓扑独立封装
- 基于 AgentPO 现有训练框架
- 参考 debate-or-vote 的 vote-style MAD，而不是新 solver

---

## 12. 当前结论

这份文档的核心方向是对的，但原来把 `debate-or-vote` 的两套 debate 机制混在了一起。现在修正后，正确实现目标应当是：

```text
在 AgentPO 中新增 cooperation_mode=mad；
Collaborator 生成 debate guide；
reward manager 内部调用冻结 actor 进行 vote-style MAD rollout；
支持 full / sparse / centralized topology；
最终用 majority vote 聚合；
reward 仍然只训练 Collaborator。
```

如果你愿意，下一步我可以基于这份修正后的文档，继续给你拆成更具体的代码改动清单，或者直接开始实现 `agentpo/debate/` 的第一版骨架。
