# 模型训练与评估详细总结

## 1. 训练方法 (Training Methods)

根据代码库中的训练脚本分析，`results/` 目录下的模型主要使用了以下五种训练方法：

### 1.1 标准 LoRA 微调 (Standard LoRA)
- **代表模型**: `lis_Qwen3.5-9B-Base_train_en`, `lis_Qwen3.5-9B-Base_train_ha` 等 (位于 `phase2_v2/` 和 `phase2_lis_matrix/` 目录下)。
- **核心代码**: `scripts/train.py`
- **机制**: 对 Qwen3.5-9B-Base 基础模型应用标准的 LoRA（Low-Rank Adaptation）微调。配置（见 `configs/experiments/phase2_v2.yaml`）在注意力机制和前馈网络的关键投影层 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) 上注入秩为 16 的低秩矩阵。
- **用途**: 作为各单语种微调的基线，并用于计算不同语言之间的干扰分数矩阵（LIS Matrix）。

### 1.2 混合数据微调 (Mixed LoRA)
- **代表模型**: `mix_Qwen3.5-9B-Base_en_ha`, `mix_Qwen3.5-9B-Base_en_so`, `mix_Qwen3.5-9B-Base_en_yo` (位于 `mix_en/` 目录下)。
- **核心代码**: `scripts/train.py` (通过 `--mix_all` 参数触发)。
- **机制**: 将全量英语数据与全量目标语言（如 Hausa, Somali, Yoruba）数据进行拼接（Concatenate）和打乱后，进行标准 LoRA 微调。
- **用途**: 测试高资源语言（英语）和低资源语言混合训练时，是否能利用跨语言迁移能力并减少英语能力的遗忘。

### 1.3 语言标签路由微调 (Tag Routing)
- **代表模型**: `tag_routing_Qwen3.5-9B-Base` (位于 `tag_routing/` 目录下)。
- **核心代码**: `scripts/train.py` (通过 `--mix_all_langs` 参数触发)。
- **机制**: 在混合了英语(en)、约鲁巴语(yo)、索马里语(so)、豪萨语(ha)四种语言的联合数据集上进行 LoRA 微调，且在训练数据中显式注入了语言标签 `<|tgt_lang:xx|>`。
- **用途**: 让模型在生成时通过指定的语言标签显式路由到对应的语言参数空间，从而缓解语言混淆现象。

### 1.4 机制接口蒸馏 (MID: Mechanistic Interface Distillation)
- **代表模型**: `mid_Qwen3.5-9B-Base_ha`, `mid_Qwen3.5-9B-Base_so`, `mid_Qwen3.5-9B-Base_yo` (位于 `mid/` 目录下)。
- **核心代码**: `scripts/train_mid.py`
- **机制**: 
  - **Teacher模型**: 冻结的“Base模型 + 英语LoRA”融合后的模型。
  - **Student模型**: Base模型 + 仅在纯目标语言上微调的可训练 LoRA (`LoRA_spec`)。
  - **损失函数**: 交叉熵损失 (CE) + **余弦距离损失 (CosDist)**。在指令结束 token（Pos1）和回答的前几个 token（Pos2）位置，计算 Student 和 Teacher 模型 top-K 隐藏层的余弦距离。
- **用途**: 试图在不使用任何英语训练数据的情况下，让 Student 模型“借用” Teacher 模型在英语中学会的通用“指令控制方向”（Instruction-control direction）。

### 1.5 双空间约束微调 (DSCT: Dual-Space Constrained Tuning)
- **代表模型**: `dsct_Qwen3.5-9B-Base_ha`, `dsct_Qwen3.5-9B-Base_so`, `dsct_Qwen3.5-9B-Base_yo` (位于 `dsct/` 目录下)。
- **核心代码**: `scripts/train_dsct.py`
- **机制**: 
  - **Teacher模型**: 与 MID 相同，是冻结的“Base模型 + 英语LoRA”融合模型。
  - **Student模型**: 在已经合并了英语 LoRA (`LoRA_donor`) 的基础上，再额外添加一层可训练的特定目标语言 LoRA (`LoRA_spec`)。
  - **损失函数**: 交叉熵损失 (CE) + MID 余弦距离损失 (CosDist) + **正交正则化损失 (Orthogonal Loss)**。正交损失强制 `LoRA_spec` 的权重矩阵与原英语 `LoRA_donor` 的矩阵占据相互正交的子空间（$\cos^2(A_{donor}, A_{spec}) + \cos^2(B_{donor}, B_{spec})$）。
- **用途**: 通过正交约束防止新学习的目标语言能力覆盖或破坏已有的英语参数空间，在学习低资源语言的同时最大限度缓解灾难性遗忘。


重点：注意其实不止tag_routing_Qwen3.5-9B-Base注入了`<|tgt_lang:xx|>`，其实全部都注入了。

---

## 2. 评估方法 (Evaluation Methods)

评估流程由 `scripts/evaluate.py` 统筹，评估结果统一保存在各个 `*_eval.json` 文件中。评估分为以下三个主要维度：

### 2.1 英语能力基准 (English Evaluation)
- **代码**: `src/evaluation/english_eval.py`
- **包含指标**: 
  - **MMLU** (大规模多任务语言理解)
  - **HellaSwag** (常识推理)
  - **ARC-Challenge** (科学问答)
  - **TruthfulQA (MC1)** (真实性与幻觉检测)
- **评估目的**: 衡量微调目标低资源语言后，模型原有的英语核心能力是否发生了退化（Eng Leak / Retention）。

### 2.2 多语言能力基准 (Multilingual Evaluation)
- **代码**: `src/evaluation/multilingual_eval.py`
- **包含指标**:
  - **SIB-200**: 多语言主题分类任务（涵盖 en, yo, so, ha 等）。
  - **Belebele**: 多语言并行阅读理解数据集。
  - **Irokobench / AfriQA / Aya**: 专门针对非洲语言等低资源语言的问答和多项选择评测。
- **评估目的**: 评估模型在目标低资源语言（Target Language）上的实际理解与生成能力。

### 2.3 LCB (Language Capability Benchmark)
- **代码**: `scripts/eval_lcb*.py` (如 `eval_lcb_matrix.py`)
- **包含指标**: 
  - **Language Consistency Rate (`lc_rate`)**: 模型实际回答的语言与 Prompt 期望回答的语言一致的比例。
  - **English Leakage (`en_leak`)**: 回答中“泄露”成英语（即本该用目标语言回答却用了英语）的比例。
- **评估目的**: 定量测试大语言模型在多语言微调后经常出现的“语言混淆”现象（Language Confusion）。


---

## 3. 模型结果汇总 (Model Results Summary)

以下是从各个 `*_eval.json` 提取并整理的各模型核心能力指标汇总（部分缺失值为空）：

| 模型名称 (Model) | 英语平均分 (Eng Avg) | SIB200 (en) | SIB200 (yo) | SIB200 (so) | SIB200 (ha) | Belebele (en) | Belebele (yo) | Belebele (so) | Belebele (ha) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DSCT (双空间约束)** |
| `dsct_Qwen3.5-9B-Base_ha` | 0.6239 | 0.550 | 0.380 | 0.350 | 0.325 | 0.9378 | 0.4300 | 0.5044 | 0.5022 |
| `dsct_Qwen3.5-9B-Base_so` | 0.6255 | 0.520 | 0.330 | 0.325 | 0.335 | 0.9389 | 0.4011 | 0.4900 | 0.4933 |
| `dsct_Qwen3.5-9B-Base_yo` | 0.6237 | 0.580 | 0.395 | 0.350 | 0.375 | 0.9322 | 0.4556 | 0.4789 | 0.4967 |
| **标准 LoRA 基线** |
| `lis_Qwen3.5-9B-Base_train_en` | 0.6217 | 0.560 | 0.390 | 0.395 | 0.410 | 0.9220 | 0.4567 | 0.4833 | 0.4922 |
| `lis_Qwen3.5-9B-Base_train_ha` | 0.5975 | 0.485 | 0.375 | 0.310 | 0.230 | 0.9211 | 0.4344 | 0.4789 | 0.5067 |
| `lis_Qwen3.5-9B-Base_train_so` | 0.5953 | 0.550 | 0.325 | 0.370 | 0.330 | 0.9144 | 0.3889 | 0.4822 | 0.4633 |
| `lis_Qwen3.5-9B-Base_train_yo` | 0.6007 | 0.370 | 0.255 | 0.360 | 0.280 | 0.9178 | 0.4444 | 0.4622 | 0.4789 |
| **MID (机制蒸馏)** |
| `mid_Qwen3.5-9B-Base_ha` | 0.3317 | 0.485 | 0.345 | 0.305 | 0.255 | 0.9333 | 0.4289 | 0.5178 | 0.5100 |
| `mid_Qwen3.5-9B-Base_so` | 0.6200 | 0.540 | 0.270 | 0.315 | 0.300 | 0.9400 | 0.4189 | 0.4933 | 0.4911 |
| `mid_Qwen3.5-9B-Base_yo` | 0.6253 | 0.500 | 0.320 | 0.375 | 0.350 | 0.9300 | 0.4689 | 0.4822 | 0.4833 |
| **混合微调 (Mixed)** |
| `mix_Qwen3.5-9B-Base_en_ha` | 0.3599 | 0.530 | 0.330 | 0.355 | 0.275 | 0.9222 | 0.4200 | 0.4500 | 0.4767 |
| `mix_Qwen3.5-9B-Base_en_so` | 0.3647 | 0.525 | 0.325 | 0.335 | 0.280 | 0.9211 | 0.4211 | 0.4622 | 0.4633 |
| `mix_Qwen3.5-9B-Base_en_yo` | 0.3660 | 0.525 | 0.350 | 0.350 | 0.355 | 0.9122 | 0.4578 | 0.4411 | 0.4478 |
| **语言路由 (Tag Routing)** |
| `tag_routing_Qwen3.5-9B-Base`| 0.6208 | 0.430 | 0.220 | 0.195 | 0.250 | 0.9356 | 0.4856 | 0.4689 | 0.5000 |

### 结果总结分析：
1. **纯目标语言微调（Standard LoRA）**会导致英语能力的显著退化（灾难性遗忘）。例如 `train_ha` 和 `train_so` 的英语平均分均跌破了 0.60。
2. **DSCT 方法**展现出了最好的平衡性。它不仅使目标语言的性能有所提升（或保持在合理水平），更关键的是极好地保留了英语能力（Eng Avg 达到 ~0.624），与直接在纯英语数据上微调的基线模型（0.621）相当。这说明通过正交损失可以有效防止语言之间的干扰。
3. **混合微调（Mixed）**反而遭遇了严重的性能崩溃，其英语平均分大幅跌至 ~0.36。
4. **MID** 在部分语言上（如 `ha`）出现了崩溃（0.33），但在其他语言上（如 `so`, `yo`）取得了接近 DSCT 的较好成绩。