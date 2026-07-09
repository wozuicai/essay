# Refactor Summary

## 这次做了什么

本次没有直接在当前机器启动训练或评测，而是在本地创建了一个可迁移的 `/root/project` 镜像：

```text
root/project
```

后续把这个目录内容放到真实 GPU 机器的 `/root/project` 后即可运行。

核心目标是：参考 `world20k_lora_sft/scripts/trl_lora_sft.py`，把项目里多个实验的训练入口统一重构为 TRL + PEFT LoRA SFT 路径，同时只保留本轮需要的评测集合。当前训练口径已切回旧版可比设置：`MAX_SEQ_LENGTH=2048`，并对整条 `text` 序列计算 LM loss。

## 训练侧改动

已重构这些训练入口：

```text
scripts/train.py
scripts/train_mid.py
scripts/train_dsct.py
scripts/train_layerwise.py
scripts/train_moe_lora.py
scripts/train_sso_lora.py
src/training/trainer.py
src/data/trl_dataset_utils.py
src/data/dataset_loader.py
```

主要变化：

- 统一把数据转成 TRL `text` 格式，保留 `### Instruction ... ### Response ...` 整条训练样本。
- 默认关闭 `completion_only_loss`，prompt 和 response 都参与 LM loss。
- 默认使用 `truncation_mode=keep_end`，对长样本尽量保留回答侧。
- 默认 `save_strategy=no`，避免训练中间 checkpoint 产生大文件。
- 支持 `MAX_SEQ_LENGTH` 环境变量覆盖训练长度。
- 支持 `MAX_TRAIN_CHARS` 环境变量过滤异常长样本。
- MID/DSCT 的 hidden-state 位置约束优先用 `### Response:\n` separator 定位回答起点，避免 full-sequence loss 下把第 0 个 token 误判为 response start。
- MID/DSCT 默认关闭 packing，避免 packed sequence 中多个样本混在一起影响位置约束。

## 评测侧改动

新增统一评测入口：

```text
scripts/eval_required.py
src/evaluation/irokobench_eval.py
src/evaluation/batched_scoring.py
```

现在主实验评测只跑用户指定的数据集：

```text
English: mmlu, hellaswag, arc_challenge, truthfulqa_mc1, english_avg
Multilingual: belebele(en/yo/so/ha)
IrokoBench: AfriMMLU MCQ + AfriXNLI + AfriMGSM
```

已把主 launcher 的旧 eval 调用替换为 `eval_required.py`，避免继续跑 SIB200、FLORES、AfriQA、Aya、LCB 等这次没要求的数据集。

评测速度优化：

```text
English: lm-eval harness
Belebele: lm-eval harness task definitions
AfriMMLU: 批量 next-token MCQ scoring
AfriXNLI: 批量 next-token MCQ scoring
AfriMGSM: 批量 greedy generate
```

Belebele 已改为 lm-eval 口径，普通 HF/PEFT、DSCT donor+spec、Layerwise/SSO 内存 merge、MoE-LoRA live model 都通过 lm-eval HFLM 路径评测。`--inject_lang_tag` 不会改写 Belebele 的 lm-eval 模板。

AfriMMLU/AfriXNLI/AfriMGSM 当前没有走 lm-eval；它们使用项目内 open/local evaluator，普通 Transformers forward/generate，对 PEFT LoRA 和自定义 MoE-LoRA 都可用，不依赖 vLLM。

## 数据和长度检查

新增：

```text
scripts/preflight_required.py
scripts/audit_sft_data.py
```

`preflight_required.py` 会在 launcher 开始前检查：

- base model 文件是否齐全；
- `data/processed/{en,yo,so,ha}.jsonl` 是否存在；
- JSONL schema 是否支持；
- 字符长度分布和超过 `MAX_TRAIN_CHARS` 的样本数量。

当前本地镜像里缺少：

```text
data/processed/en.jsonl
```

真实 `/root/project` 里必须补齐，否则含 English 的实验会被 preflight 直接拦住。

当前目标语言的字符长度分布：

```text
yo: p95=5768, p99=11432, max=135759, >200000 共 0 条
so: p95=3266, p99=6253, max=6592963, >200000 共 1 条
ha: p95=1685, p99=5235, max=12749, >200000 共 0 条
```

因此默认策略是：

```bash
export MAX_TRAIN_CHARS=200000
export MAX_SEQ_LENGTH=2048
```

所有主实验 launcher 都已对齐到同一组长度默认值。`MAX_TRAIN_CHARS=200000` 只过滤当前 so 数据中 659 万字符的严重异常词典行，尽量保留其他长样本；`MAX_SEQ_LENGTH=2048` 用于恢复旧版训练速度和 token 截断口径。

## Launcher 改动

主实验 launcher 已更新：

```text
scripts/launch_phase2_v2.sh
scripts/launch_mix_en.sh
scripts/launch_mid.sh
scripts/launch_dsct.sh
scripts/launch_moe_lora.sh
scripts/launch_tag_routing.sh
scripts/launch_sso_lora.sh
scripts/launch_layerwise.sh
```

变化：

- 开头增加 preflight。
- 默认允许 HF 数据集在线下载，避免 eval 数据集没缓存时失败。
- 统一调用 `eval_required.py`。
- eval 后调用清理脚本删除 checkpoint 和临时大文件。

## 大文件保护

已增强 `.gitignore`，避免误上传：

```text
root/
world20k_lora_sft/
models/
data/**/*.jsonl
data/**/*.arrow
*.safetensors
*.bin
*.pt
*.ckpt
results/**/checkpoint-*/
```

新增清理脚本：

```text
scripts/cleanup_large_artifacts.sh
```

默认只删 checkpoint、optimizer、scheduler、临时模型权重等，不删 `adapter_model.safetensors` / `moe_weights.safetensors`，因为 MID/DSCT 后续依赖 English donor adapter。

训练产物保存策略：

```text
phase2_v2 / mix_en / tag_routing: 只保存 PEFT LoRA adapter
mid / dsct: donor/base merge 只在内存中发生，落盘只保存新 LoRA adapter
layerwise stage1: 只保存 shared LoRA adapter
layerwise stage2: 只保存 lang LoRA adapter
layerwise merge_eval: shared+lang merge 只在内存中发生，不保存 full model
sso_lora stage1: 只保存 shared LoRA adapter
sso_lora stage2: 只保存 lang LoRA adapter
sso_lora merge_eval: shared+lang merge 只在内存中发生，不保存 full model
moe_lora: 只保存 moe_weights.safetensors，不保存 base model
```

以下会写 full model 的路径已经默认禁用，必须显式设置 `ALLOW_FULL_MODEL_SAVE=1` 才能运行：

```text
scripts/train.py --method full_ft
scripts/train.py --method isolated_lora
scripts/train_layerwise.py --mode merge
scripts/train_sso_lora.py --mode merge
```

全部实验跑完后，如果确认不再需要权重，可以执行：

```bash
DELETE_TRAINED_WEIGHTS=1 bash scripts/cleanup_large_artifacts.sh results/phase2_v2 results/dsct results/mid results/mix_en results/moe_lora results/tag_routing results/sso_lora
```

## 怎么跑

主要运行说明在：

```text
TRAIN_EVAL_RUNBOOK.md
```

建议先跑：

```bash
python scripts/audit_sft_data.py --data_dir data/processed --langs en,yo,so,ha --model /root/project/models/Qwen3.5-9B-Base --max_length 2048 --max_train_chars 200000
```

再按 runbook 里的实验顺序启动 launcher。

## 已验证

在当前 macOS 环境中做了静态检查：

```text
Python py_compile: 通过
主 launcher bash -n: 通过
旧 eval 调用残留检查: 通过
```

未执行训练和评测，因为当前机器没有 `/root/project`、CUDA、`nvidia-smi`。
