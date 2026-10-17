# Token 用量分析：Social Simulation with Moral Agents

## 1. 配置参数

| 符号 | 配置路径 | 描述 | 示例值 |
|------|----------|------|--------|
| **N** | `agent.initial_count` | Agent 数量 | 2, 8, 16 |
| **S** | `world.max_life_steps` | 模拟步数 | 2, 40, 80, 100 |
| **W** | `agent.view.visible_steps` | 观测窗口 | 3, 15 |
| **K** | `llm.two_stage_model` | 反思开/关 (0 或 1) | true/false |
| **f** | *(经验值)* | 重试失败率 | ~0.05–0.15 |

---

## 2. 推导

**(1) 总成本**

```
Cost = Pᵢ · Tᵢ + Pₒ · Tₒ
```

where Pᵢ, Pₒ = input/output 单价 ($/token), Tᵢ, Tₒ = 总 input/output tokens。

**(2) 对 agents × steps 求和**

```
Tᵢ = M · Σ_{t=1}^{S} N(t) · tᵢ(t)

Tₒ = M · Σ_{t=1}^{S} N(t) · tₒ(t)
```

where S = 总步数, N(t) = step t 存活 agent 数 (无繁殖时 N(t) ≡ N),
tᵢ(t), tₒ(t) = 单个 agent 在 step t 的 input/output tokens,
M = 1 + f·Rₑ 为重试乘数 (f = 失败率 ∈ [0.05, 0.15], Rₑ ≈ 1.2 = 重试/原始 token 比)。

**(3) 单 agent 单 step：(1+K) 次调用**

每个 agent 每步做 1+K 次 LLM 调用，K ∈ {0, 1} 为反思开关 (`llm.two_stage_model`)：

```
Call 1:  input = Sₚ + O(t),                    output = R₁
Call 2:  input = Sₚ + O(t) + R₁ + Pᵣ,          output = R₂       (× K)
```

where Sₚ ≈ 7,900 = system prompt, Pᵣ ≈ 1,750 = reflect prompt,
R₁ ≈ 1,500, R₂ ≈ 2,000 = 有效平均输出长度 (与 T₀ 联合校准，见 §3),
O(t) = user observation (随 t, N, W 增长，见 (5))。

用 K 作 0/1 乘子合并：

```
tᵢ(t) = (1+K)·[Sₚ + O(t)] + K·(R₁ + Pᵣ)

tₒ(t) = R₁ + K·R₂
```

**(4) 合并 input + output**

定义 T₀(t) ≡ Sₚ + O(t) + R₁ (单阶段总量)，则：

```
T(t) ≡ tᵢ(t) + tₒ(t) = (1+K)·T₀(t) + K·(Pᵣ + R₂)
```

注意 K=1 时 T = 2T₀ + 3750，**非** 2 倍（反思重发了 Sₚ + O(t)）。

**(5) O(t) 的增长模型**

```
O(t, N, W) = (α₀ + α₁N) + β·min(t,2) + (γ₀ + γ₁N)·min(t,W) + δ·max(t−W, 0)
```

where W = 观测窗口 (`agent.view.visible_steps`),
α₀ + α₁N = 固定观测 (status + env, 每个 agent 贡献 α₁ 到 social env),
β·min(t,2) = 记忆初始化 (step 1–2 从空到填充),
(γ₀ + γ₁N)·min(t,W) = 活动窗口填充 (在 step W 饱和),
δ·max(t−W,0) = 饱和后长期记忆漂移。

---

## 3. 校准

将常量吸收进系数，T₀(t) 的数值形式（拟合自 K=1/N=8 和 K=0/N=2 两组数据）：

```
T₀(t, N, W) = (9400 + 400N) + 500·min(t,2) + (110 + 45N)·min(t,W) + 55·max(t−W, 0)
```

where 9400 = Sₚ + α₀ + E[R₁], 400N = α₁N (每 agent ~500 chars social env ÷ 4),
500 = β, 110 + 45N = γ₀ + γ₁N, 55 = δ, 3750 = Pᵣ + E[R₂]。

验证 (K=1, N=8, W=15): err < 5% 全部 9 点。验证 (K=0, N=2, W=15): err < 8%（噪声较大，step 6 繁殖改变 N）。

| t | K=1,N=8 公式 / 实测 | K=0,N=2 公式 / 实测 |
|---|---|---|
| 1 | 30,890 / 31,000 (−0.4%) | 10,900 / 10,912 (−0.1%) |
| 5 | 35,650 / 37,000 (−3.6%) | 12,200 / 11,691 (+4.4%) |
| 15 | 45,050 / 46,000 (−2.1%) | — |
| 80 | 52,200 / 53,000 (−1.5%) | — |

---

## 4. 缩放性质

由公式结构直接得出：

- ∂Cost/∂N ∝ N → **2N ⇒ ~2× cost**
- ∂Cost/∂S ≈ 线性 (t > W 后 O(t) 增长 ≈ 0) → **2S ⇒ ~2× cost**
- K: 1→0 ⇒ ×(T₀)/(2T₀+3750) ≈ **×0.37** (at N=8)
- K: 0→1 ⇒ 逆运算 ≈ **×2.7** (at N=8)
- W: 15→3 ⇒ O_window 积分面积减小 → **−8%**

---

## 5. 实现

`scr/utils/token_estimator.py`，CLI:

```bash
python main.py estimate-cost --config_dir <name>
python main.py estimate-cost --config_dir <name> --config.world.max_life_steps 80
```

---

## 6. 限制

1. 仅在 N∈{2,8} 校准；N≫8 时线性 α₁N 可能低估（二次交互效应）。
2. M = 1+fRₑ 为一阶近似，f > 0.2 时失效。
3. 系数基于 OpenAI tokenizer (~4 chars/tok)，其他模型偏差 10–20%。
4. 繁殖场景用 N̄ = (1/S)Σ N(t) 近似。

---
---

## Appendix A: 实测数据（N=8, K=1, W=15, gpt-5-mini）

基于 run `0219-000839`（2 life steps, 8 agents）。

### A.1 架构：每个 Agent 每步 2 次 LLM 调用

| | Call 1（初始响应） | Call 2（反思修正） |
|---|---|---|
| **Input** | system_prompt + user_observation | system_prompt + user_observation + response_1 + reflect_prompt |
| **Output** | 完整 JSON（thinking + memory + plan + action） | 修正后的 JSON |

### A.2 实测数据（8 agent 平均值）

**Step 1:**

| 组件 | 字符数 | ~Token (tiktoken) |
|---|---|---|
| System prompt（固定） | ~38,800 | ~7,900 |
| User observation | 9,171 | 2,292 |
| Reflect prompt（固定） | 7,000 | 1,750 |
| Response 1（输出） | 7,095 | 1,773 |
| Response 2（输出） | 8,345 | 2,086 |

| | Input tokens | Output tokens | 合计 |
|---|---|---|---|
| **Call 1** | 11,995 | 1,773 | 13,768 |
| **Call 2** | 15,519 | 2,086 | 17,605 |
| **每 Agent** | **27,514** | **3,860** | **31,374** |
| **每 Step (×8 agents)** | **220,114** | **30,880** | **~251K** |

**Step 2:**

| | Input tokens | Output tokens | 合计 |
|---|---|---|---|
| **每 Agent** | **33,353** | **4,761** | **38,114** |
| **每 Step (×8 agents)** | **266,826** | **38,094** | **~305K** |

**2-Step Run 总计（全部 8 agents）: ~556K tokens**

### A.3 增长模式分析

三类增长模式：

**1. 完全固定（不增长）：**
- System prompt: ~38,800 chars / ~7,900 tokens
- Reflect prompt: 7,000 chars
- Basic Status + Social Env: ~5,000 chars

**2. 有硬上限（visibility window = W 步）：**
- Activity observations：每步 ~1,200 chars 新增，超过 W 步后旧的被丢弃
- General Activities：~600 chars（旧的替换新的）

**3. 缓慢增长（Agent 自己写的 memory）：**
- `long_term_memory` 总计 step 2 已填充后：~5,500 chars，之后每步仅增长 ~100-200 chars

### A.4 增长曲线预测（N=8, K=1, W=15, 无繁殖）

| Step | 每 Agent 每步 | 每 Step 全部 8 agents | 累计总 tokens |
|---|---|---|---|
| 1 | 31K | 251K | 251K |
| 2 | 34K | 274K | 525K |
| 5 | 37K | 297K | 1.4M |
| 10 | 42K | 337K | 2.9M |
| **15** | **46K** | **369K** | **4.8M** |
| 20 | 47K | 373K | 6.7M |
| 40 | 49K | 389K | 14.3M |
| 80 | 53K | 421K | 30.5M |

关键拐点在 Step W=15：observation window 饱和后，每步增量从 ~12K tokens/step 降到 ~2K tokens/step。

### A.5 Token 去向分布

每个 Agent 每步 ~28K tokens：

| 组件 | 发送次数 | Token 数 | 占比 |
|---|---|---|---|
| **System prompt** | ×2 | **~15,800t** | **57%** |
| **User observation** | ×2 | ~4,600t | 17% |
| **Response 1** | input ×1 + output ×1 | ~3,550t | 13% |
| **Response 2** | output ×1 | ~2,100t | 8% |
| **Reflect prompt** | ×1 | ~1,750t | 6% |

### A.6 优化方向

| 优化 | 节省比例 | 说明 |
|---|---|---|
| **OpenAI Prompt Caching** | Input 成本 -90% | System prompt 可用 cached 读取（$0.025/M vs $0.25/M） |
| 压缩 system prompt | -10~15% 总量 | 规则部分有重复，可精简 ~3-4K chars |
| 早期步骤跳过 reflection | -40% 前 3 步 | Step 1 无交互历史，reflection 修正价值低 |
| Memory 裁剪 | 长期 -5~10% | 已完成的 hunt 可以压缩总结 |
