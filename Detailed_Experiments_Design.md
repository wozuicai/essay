# 多语言大模型高级路由与正交约束实验详细设计文档 (2026-06-25)

本报告详细记录了为了在多语言微调场景下（高资源英语 + 多个低资源语言 yo, so, ha）超越传统的“硬标签路由（Tag Routing）”，而正在并行推进的三个重量级创新实验：**Exp C (逐层渐进式路由)**、**Exp D (软路由 MoE)** 和 **Exp E (正交子空间 LoRA)**。

这三个实验分别从 **层级深度解耦 (Depth-wise Disentanglement)**、**Token级别动态软路由 (Token-level Soft Routing)** 和 **参数空间数学正交 (Mathematical Subspace Orthogonality)** 三个不同维度切入，旨在彻底解决多语言微调中的“灾难性遗忘”与“语言混淆”问题，并实现低资源语言间的正向迁移。

---

## 🚀 Exp C：逐层渐进式路由 (Layer-wise Progressive Language Routing)

### 1. 核心动机
大语言模型的不同 Transformer 层具有不同的分工：浅层（Bottom layers）往往负责提取语言无关的底层逻辑和抽象语义；而深层（Top layers）则负责表层句法构建和特定语言的词汇生成。
因此，我们可以让所有语言共享底层的参数（最大化推理能力的正向迁移），并在顶层为每种语言分配专属参数（精准隔离语言，防止生成时发生语言混淆）。

### 2. 架构设计
- **基础模型**：Qwen3.5-9B-Base（共 32 层 Transformer）。
- **层级划分**：
  - **Bottom Layers (0-15层)**：挂载一个名为 `shared` 的 LoRA Adapter（秩 `r=16`, `alpha=32.0`）。
  - **Top Layers (16-31层)**：挂载针对具体目标语言（yo, so, ha）的专属 LoRA Adapter（秩 `r=16`, `alpha=32.0`）。
- **挂载位置**：注意力层及前馈层全覆盖（`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`）。

### 3. 两阶段训练流程
代码实现见：[train_layerwise.py](file:///root/project/scripts/train_layerwise.py) 和 [layerwise_lora.py](file:///root/project/src/models/layerwise_lora.py)

- **Stage 1 (共享空间预热)**：
  - **数据**：四语言 (en, yo, so, ha) 全量数据等比例混合打乱（共约 48K 条），训练 2 epochs。
  - **训练动作**：只将 `shared` Adapter 挂载到 0-15 层，模型其余部分冻结。通过四语言的联合训练，让底层充分学习多语言的通用语义。
- **Stage 2 (顶层专属对齐)**：
  - **数据**：单目标语言数据（例如全量约鲁巴语 yo），训练 1 epoch。
  - **训练动作**：加载 Stage 1 训练好的 `shared` Adapter **并将其冻结**。在 16-31 层挂载新的语言专属 Adapter（如 `yo`），并仅对这部分进行梯度更新。
- **Merge 阶段**：测试前，将底层的 `shared` 和顶层的语言专属 Adapter 同时激活，并通过 `merge_and_unload()` 物理合并到基础模型中，产出标准的 HuggingFace 模型用于评测。

---

## 🚀 Exp D：软路由 MoE (Soft Mixture of LoRA Experts, LA-MoA)

### 1. 核心动机
硬标签路由（如输入 `<|tgt_lang:yo|>`）具有强制性且粗粒度，无法处理代码夹杂（Code-Switching）或需要借用英语知识回答非英语问题的情况。
通过构建一个小型的 LoRA 专家池，引入一个内部的轻量级 Router，让模型自己去学习在每个 Token 上该分配多少权重给哪个专家。

### 2. 架构设计
代码实现见：[moe_lora.py](file:///root/project/src/models/moe_lora.py)
模型原本的每一个 Target Linear Layer 都被替换为了自定义的 `MoELoRALinear`，包含：
- **K 个 LoRA 专家**：设定 `n_experts = 4`，每个专家拥有自己独立的 `lora_A` 和 `lora_B`（秩 `r=8`, `alpha=16.0`）。参数量加起来约等于 2 倍的传统 r=16 LoRA。
- **轻量级路由网络 (Router)**：一个没有偏置的单层线性层 `Linear(d_in, n_experts)`。

**前向传播数学公式**：
1. 计算当前 Token 对 4 个专家的激活概率：`gate = softmax(Router(x))`
2. 聚合专家输出：`lora_out = sum_{i=1}^{4} (gate_i * lora_B_i(lora_A_i(x))) * (alpha / r)`
3. 最终输出：`Base(x) + lora_out`

### 3. 训练与评估流程
代码实现见：[train_moe_lora.py](file:///root/project/scripts/train_moe_lora.py)
- **训练数据**：四语言 (en, yo, so, ha) 等比例全量联合训练，2 epochs。
- **参数解冻**：Base 模型完全冻结，只允许所有层的所有专家的 `lora_A`, `lora_B` 以及 `Router` 的权重进行更新（可训练参数约 87M）。
- **特异性评估**：由于这改变了底层网络结构，无法使用常规的 `PeftModel` 合并，因此需要使用专属评测脚本 `scripts/eval_moe_lora.py` 进行在线动态推理评测。

---

## 🚀 Exp E：正交子空间 LoRA (Shared-Specific Orthogonal LoRA, SSO-LoRA)

### 1. 核心动机
在多语言联合微调中，高资源语言（英语）和低资源语言经常会在同一个参数空间中发生“争抢”，导致低资源语言不仅学不好，还会破坏原有的英语能力。
如果能通过数学约束，在全层都分配一个“共享空间”和一个“专属空间”，并强制它们**在几何空间中相互垂直（正交）**，那么模型就能实现物理意义上的干扰解耦。

### 2. 架构设计
代码实现见：[sso_lora.py](file:///root/project/src/models/sso_lora.py)
在模型的 0-31 全层中，并联挂载两套 LoRA Adapter：
- **Shared Adapter** (`r=16`, `alpha=32.0`)：负责通用跨语言表征。
- **Language-Specific Adapter** (`r=8`, `alpha=16.0`)：负责特定语言生成。

### 3. 核心创新：正交惩罚损失函数 (Orthogonal Penalty Loss)
在 Stage 2 的 SFTTrainer 中，我们重写了 `compute_loss` 方法，加入了一个正交惩罚项：
$$ L_{orth} = \frac{1}{N} \sum_{layers} \left( \| A_{shared} \cdot A_{lang}^T \|_F^2 + \| B_{shared}^T \cdot B_{lang} \|_F^2 \right) $$
**物理意义**：该损失函数计算了两个 Adapter 的输入投影矩阵 $A$ 和输出投影矩阵 $B$ 之间的内积。损失最小化将强制使得 $A_{lang}$ 的行向量和 $A_{shared}$ 的行向量相互正交。这意味着特定语言学到的特征向量，在通用空间的投影严格为 0，从而实现完美解耦。

### 4. 两阶段训练流程
代码实现见：[train_sso_lora.py](file:///root/project/scripts/train_sso_lora.py)
- **Stage 1 (共享表征学习)**：
  - 用四语言全量数据混合（约 48K 条），训练 2 epochs。
  - **仅激活并更新** `shared` Adapter。
- **Stage 2 (专属空间正交分离)**：
  - 用单目标语言数据，训练 1 epoch。
  - 冻结 `shared` Adapter，激活并更新目标语言（如 `yo`）Adapter。
  - 在计算交叉熵损失的基础上，附加权重为 `0.1` 的正交损失：`Loss = L_CE + 0.1 * L_orth`。
- **Merge 阶段**：将训练好的正交双 Adapter 同时加载，并使用 `merge_and_unload()` 物理合并，产出最终模型。

---

## 🎯 实验预期与论文 Storyline 对照

1. **Exp C (Layer-wise)** 证明了语言表征在深度神经网络中的空间分布，为高效路由提供了架构层面的优化方向。
2. **Exp D (Soft MoE)** 证明了基于上下文的内部隐式路由，优于外部人为施加的硬性 Tag 路由，在复杂多语言环境中具备更高的鲁棒性。
3. **Exp E (SSO-LoRA)** 提供了坚实的数学理论支持，证明正交约束能完美化解多任务/多语言微调中的负向干扰（Negative Interference），并在英语保留率和低资源任务双线上取得 SOTA。