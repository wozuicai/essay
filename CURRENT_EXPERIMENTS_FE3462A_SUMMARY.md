# 指定实验流程总结（commit `fe3462a`）

当前 checkout：

```text
branch: initial-pulled-fe3462a
commit: fe3462a Refactor training scripts for flash attention and liger
```

本文只总结这些训练脚本产生的实验结果：

```text
scripts/train.py
scripts/train_mid.py
scripts/train_dsct.py
scripts/train_moe_lora.py
scripts/train_layerwise.py
scripts/train_sso_lora.py
```

目标 results 目录：

```text
results/phase2_lis_matrix
results/phase2_v2
results/mix_en
results/mid
results/dsct
results/moe_lora
results/layerwise
results/sso_lora
results/tag_routing
```

## 0. 公共训练口径

这些训练脚本共用 `src/training/trainer.py` 里的 SFT 配置。

当前版本的核心口径：

| 项 | 当前行为 |
|---|---|
| 基础模型 | `/root/project/models/Qwen3.5-9B-Base` |
| 数据字段 | `dataset_text_field="text"` |
| loss | 整条 `text` 都算 causal LM loss，即 prompt + response 都算 |
| completion-only loss | 未设置，不是 completion-only SFT |
| max length | 主要来自 YAML，通常是 `2048` |
| packing | 默认关，只有 `PACKING=1` 才开 |
| FlashAttention | 默认 `attn_implementation="flash_attention_2"` |
| Liger | 代码尝试传 `use_liger_kernel`，但是否生效取决于 TRL 版本 |

样本通常长这样：

```text
### Instruction:
<|tgt_lang:{language}|> {instruction}

### Response:
{response}
```

## 1. 结果目录到训练脚本的对应关系

| results 目录 | 训练脚本 | 主要 launcher | 训练内容 |
|---|---|---|---|
| `results/phase2_lis_matrix` | `scripts/train.py` | `launch_phase2_h100.sh`, `launch_phase2_newh100.sh`, `run_phase2_remaining.sh` | 多语言单语言 LoRA，用于 LIS 矩阵 |
| `results/phase2_v2` | `scripts/train.py` | `launch_phase2_v2.sh` | en/yo/so/ha 单语言 LoRA + 4x4 LIS |
| `results/mix_en` | `scripts/train.py` | `launch_mix_en.sh` | English + target language 混训 |
| `results/tag_routing` | `scripts/train.py` | `launch_tag_routing.sh` | en/yo/so/ha 全语言混训，评测时注入语言 tag |
| `results/mid` | `scripts/train_mid.py` | `launch_mid.sh` | English LoRA teacher 的 hidden-state distillation |
| `results/dsct` | `scripts/train_dsct.py` | `launch_dsct.sh` | donor English LoRA + target spec LoRA + orthogonal loss |
| `results/moe_lora` | `scripts/train_moe_lora.py` | `launch_moe_lora.sh` | token-level soft MoE-LoRA |
| `results/layerwise` | `scripts/train_layerwise.py` | `launch_layerwise.sh` | bottom shared LoRA + top lang-specific LoRA |
| `results/sso_lora` | `scripts/train_sso_lora.py` | `launch_sso_lora.sh` | shared LoRA + lang LoRA + orthogonal penalty |

## 2. `scripts/train.py` 相关实验

`train.py` 是普通 LoRA 实验的主入口。这里涉及 4 个结果目录。

### 2.1 `results/phase2_lis_matrix`

目的：训练多个单语言 LoRA adapter，做 LIS / 跨语言干扰分析。

典型训练：

```bash
scripts/train.py \
  --method standard_lora \
  --train_lang {LANG} \
  --train_samples 500 \
  --config configs/experiments/lis_matrix.yaml
```

常见语言：

```text
en, fr, zh, sw, th, bn, yo
```

输出形态：

```text
results/phase2_lis_matrix/lis_Qwen3.5-9B-Base_train_{LANG}
results/phase2_lis_matrix/lis_Qwen3.5-9B-Base_train_{LANG}_eval.json
```

### 2.2 `results/phase2_v2`

目的：只针对 `en/yo/so/ha` 做单语言 LoRA，并计算 4x4 LIS。

流程：

```text
1. 评测 base model -> baseline json
2. 依次训练 en/yo/so/ha 单语言 LoRA
3. 每个 adapter 训练后评测
4. 跑 compute_lis.py 得到 LIS matrix
```

训练：

```bash
scripts/train.py \
  --method standard_lora \
  --train_lang {en|yo|so|ha} \
  --config configs/experiments/lis_matrix.yaml
```

输出：

```text
results/phase2_v2/Qwen3.5-9B-Base_baseline.json
results/phase2_v2/lis_Qwen3.5-9B-Base_train_{LANG}
results/phase2_v2/lis_Qwen3.5-9B-Base_train_{LANG}_eval.json
```

### 2.3 `results/mix_en`

目的：看 English + target language 混训能否缓解跨语言退化。

训练组合：

```text
en + yo
en + so
en + ha
```

训练：

```bash
scripts/train.py \
  --method standard_lora \
  --train_lang {yo|so|ha} \
  --mix_all \
  --config configs/experiments/lis_matrix.yaml
```

`--mix_all` 会把 English 全量数据和目标语言全量数据 concat 后 shuffle。

输出：

```text
results/mix_en/mix_Qwen3.5-9B-Base_en_{LANG}
results/mix_en/mix_Qwen3.5-9B-Base_en_{LANG}_eval.json
```

### 2.4 `results/tag_routing`

目的：把 en/yo/so/ha 全量数据混到一个 LoRA 里，评测时通过 `<|tgt_lang:xx|>` tag 控制目标语言。

训练：

```bash
scripts/train.py \
  --method standard_lora \
  --train_lang en \
  --mix_all_langs \
  --config configs/experiments/lis_matrix.yaml
```

`--mix_all_langs` 会把 `en, yo, so, ha` 四种语言 concat 后 shuffle。

输出：

```text
results/tag_routing/tag_routing_Qwen3.5-9B-Base
results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json
results/tag_routing/tag_routing_Qwen3.5-9B-Base_lcb_chat.json
```

## 3. `scripts/train_mid.py` -> `results/mid`

目的：不用 English 数据训练 target adapter，而是用 English LoRA teacher 的 hidden-state 约束保持 instruction interface。

结构：

```text
Teacher = Base + English LoRA merge 后冻结
Student = Base + target-language LoRA
```

依赖：

```text
results/phase2_v2/lis_Qwen3.5-9B-Base_train_en
```

训练语言：

```text
yo, so, ha
```

训练：

```bash
scripts/train_mid.py \
  --teacher_adapter results/phase2_v2/lis_Qwen3.5-9B-Base_train_en \
  --train_lang {yo|so|ha} \
  --alpha 0.1 \
  --beta 0.05 \
  --top_n_layers 4 \
  --n_pos2 3
```

loss：

```text
CE(student) + hidden-state cosine distance at Pos1/Pos2
```

位置：

```text
Pos1 = "### Response:\n" 前一个 token
Pos2 = response 开头的前 n_pos2 个 token
```

输出：

```text
results/mid/mid_Qwen3.5-9B-Base_{LANG}
results/mid/mid_Qwen3.5-9B-Base_{LANG}_eval.json
results/mid/mid_Qwen3.5-9B-Base_{LANG}_lcb_matrix.json
```

注意：MID 不应该开 `PACKING=1`。

## 4. `scripts/train_dsct.py` -> `results/dsct`

目的：在 donor English LoRA 已 merge 的 base 上训练 target spec LoRA，并用 orthogonal loss 避免覆盖 donor 子空间。

结构：

```text
Teacher = Base + donor English LoRA merge 后冻结
Student = Base + donor English LoRA merge 后，再挂 target LoRA spec
```

依赖：

```text
results/phase2_v2/lis_Qwen3.5-9B-Base_train_en
```

训练语言：

```text
yo, so, ha
```

训练：

```bash
scripts/train_dsct.py \
  --donor_adapter results/phase2_v2/lis_Qwen3.5-9B-Base_train_en \
  --train_lang {yo|so|ha} \
  --alpha 0.1 \
  --beta 0.05 \
  --lambda_ortho 0.01 \
  --top_n_layers 4 \
  --n_pos2 3
```

loss：

```text
CE(student)
+ MID-style hidden-state cosine loss
+ orthogonal_loss(spec, donor)
```

输出：

```text
results/dsct/dsct_Qwen3.5-9B-Base_{LANG}
results/dsct/dsct_Qwen3.5-9B-Base_{LANG}_eval.json
```

注意：DSCT 不应该开 `PACKING=1`。

## 5. `scripts/train_moe_lora.py` -> `results/moe_lora`

目的：把普通 LoRA 变成 token-level soft MoE-LoRA。

结构：

```text
每个目标 Linear 层替换成 MoELoRALinear
每层 K=4 个 LoRA expert
每个 token 通过 router 得到 expert 权重
base model 冻结
只训练 lora_A / lora_B / router
```

训练数据：

```text
en + yo + so + ha 全量 concat shuffle
```

训练：

```bash
scripts/train_moe_lora.py \
  --n_experts 4 \
  --r 8 \
  --lora_alpha 16.0 \
  --config configs/experiments/lis_matrix.yaml
```

输出：

```text
results/moe_lora/moe_lora_Qwen3.5-9B-Base/moe_weights.safetensors
results/moe_lora/moe_lora_Qwen3.5-9B-Base/moe_config.json
results/moe_lora/moe_lora_Qwen3.5-9B-Base_eval.json
```

MoE-LoRA 保存的是自定义 MoE 权重，不是标准 PEFT adapter，也不保存完整 base model。

## 6. `scripts/train_layerwise.py` -> `results/layerwise`

目的：底层共享跨语言能力，顶层做语言特化。

结构：

```text
layers 0-15: shared bottom LoRA
layers 16-31: per-language top LoRA
```

Stage 1：

```text
mode = stage1
数据 = en + yo + so + ha 全量 concat shuffle
训练 shared bottom LoRA
输出 results/layerwise/stage1_shared
```

Stage 2：

```text
mode = stage2
语言 = yo, so, ha
加载 stage1 shared adapter
冻结 shared
训练对应语言的 top adapter
输出 results/layerwise/stage2_{LANG}
```

评测：

```text
mode = merge_eval
内存中 merge shared + lang adapter 后评测
不保存 merged full model
```

输出：

```text
results/layerwise/stage1_shared
results/layerwise/stage2_{LANG}
results/layerwise/layerwise_Qwen3.5-9B-Base_{LANG}_eval.json
```

## 7. `scripts/train_sso_lora.py` -> `results/sso_lora`

目的：训练 shared LoRA 和 lang-specific LoRA，并用 orthogonal penalty 分离二者子空间。

结构：

```text
shared adapter: 全语言共享
lang adapter: 每个目标语言一个
orthogonal_loss: 约束 shared 与 lang adapter 的 A/B 矩阵正交
```

Stage 1：

```text
mode = stage1
数据 = en + yo + so + ha 全量 concat shuffle
训练 shared LoRA
输出 results/sso_lora/stage1_shared
```

Stage 2：

```text
mode = stage2
语言 = yo, so, ha
加载 shared adapter
添加 lang adapter
forward 中激活 shared + lang
只训练 lang adapter
额外加 orthogonal_loss
输出 results/sso_lora/stage2_{LANG}
```

评测：

```text
mode = merge_eval
内存中 merge shared + lang adapter 后评测
不保存 merged full model
```

输出：

```text
results/sso_lora/stage1_shared
results/sso_lora/stage2_{LANG}
results/sso_lora/sso_Qwen3.5-9B-Base_{LANG}_eval.json
```

## 8. 评测概览

这些实验主要使用以下评测入口：

| 入口 | 用途 |
|---|---|
| `scripts/evaluate.py` | English + multilingual 主评测，常配 `--skip_flores` |
| `scripts/eval_extended.py --only_iroko_mcq` | 补跑 IrokoBench MCQ |
| `scripts/eval_lcb_chat.py` | LCB-chat |
| `scripts/eval_lcb_matrix.py` | LCB 4x4 matrix，主要用于 MID |
| `scripts/eval_moe_lora.py` | MoE-LoRA 专用评测 |

## 9. 当前版本和这些实验相关的主要风险

1. `trl==0.9.6` 与 `SFTTrainer(..., processing_class=tokenizer)` 可能不兼容，按 requirements 跑可能直接 `TypeError`。

2. MID / DSCT 依赖 `### Response:\n` 的样本内位置，不能手动打开 `PACKING=1`。

3. Layerwise / SSO 的 stage1 skip 判断可能漏看 `stage1_shared/shared/adapter_config.json`，已有 stage1 adapter 时仍可能重复训练。

4. `train_layerwise.py --mode merge` 和 `train_sso_lora.py --mode merge` 会保存完整 HF 模型；默认 launcher 用的是 `merge_eval`，不会落完整 merged model。
