# Train/Eval Runbook

## 路径对齐

本地镜像目录是 `root/project`，放到机器上运行时对齐为：

```bash
cd /root/project
```

确认大文件只放本地，不提交：

```bash
git status --short
```

## 运行前检查

需要准备：

```bash
ls /root/project/models/Qwen3.5-9B-Base
ls /root/project/data/processed/{en,yo,so,ha}.jsonl
python -c "import torch, transformers, trl, peft, datasets; print(torch.cuda.is_available())"
```

如果 TRL/Liger 或 FlashAttention 环境不完整，请先安装

先做数据长度审计：

```bash
python scripts/audit_sft_data.py --data_dir data/processed --langs en,yo,so,ha --model /root/project/models/Qwen3.5-9B-Base --max_length 2048 --max_train_chars 200000
```

默认运行策略：

```bash
export MAX_TRAIN_CHARS=200000
export MAX_SEQ_LENGTH=2048
```

所有主实验 launcher 都已对齐到同一组长度默认值。`MAX_SEQ_LENGTH=2048` 恢复旧版训练速度和截断口径；`MAX_TRAIN_CHARS=200000` 会保留当前 yo/so/ha 中除严重异常外的长样本，本地 so 的 659 万字符词典行仍会被过滤。当前训练默认关闭 `completion_only_loss`，整条 `text` 序列都会参与 LM loss。

所有实验配置的 `training.num_epochs` 已统一为 2；Layerwise、SSO-LoRA、Isolated LoRA 这类多阶段实验的 stage 训练也对齐为 2 epochs。

## 主实验顺序

```bash
nohup bash scripts/launch_phase2_v2.sh > logs/phase2_v2_master.log 2>&1 &
```

```bash
nohup bash scripts/launch_mix_en.sh > logs/mix_en_master.log 2>&1 &
```

```bash
nohup bash scripts/launch_mid.sh > logs/mid_master.log 2>&1 &
```

```bash
nohup bash scripts/launch_dsct.sh > logs/dsct_master.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/launch_moe_lora.sh > logs/moe_lora_master.log 2>&1 &
```

```bash
nohup bash scripts/launch_tag_routing.sh > logs/tag_routing_master.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_sso_lora.sh stage1
CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_sso_lora.sh stage2
```

## 可选 Layerwise

`train_layerwise.py` 也已重构；如果需要单独补跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_layerwise.sh stage1
CUDA_VISIBLE_DEVICES=0,1 bash scripts/launch_layerwise.sh stage2
```

## 单模型补评测

普通 HF model 或 PEFT adapter：

```bash
python scripts/eval_required.py --model_path results/phase2_v2/lis_Qwen3.5-9B-Base_train_en --languages en,yo,so,ha --batch_size 32 --generation_batch_size 4 --output /tmp/eval.json
```

MoE-LoRA：

```bash
python scripts/eval_required.py --moe_dir results/moe_lora/moe_lora_Qwen3.5-9B-Base --languages en,yo,so,ha --batch_size 16 --generation_batch_size 4 --output /tmp/moe_eval.json
```

`--batch_size` 会用于 English lm-eval、Belebele lm-eval、AfriMMLU、AfriXNLI 的批量 forward；`--generation_batch_size` 只用于 AfriMGSM 批量生成。Belebele 使用 lm-eval 官方 task 模板；即使传入 `--inject_lang_tag`，Belebele 也不会改写 lm-eval 模板。

当前 eval 口径：

```text
English: lm-eval harness
Belebele: lm-eval harness
AfriMMLU/AfriXNLI/AfriMGSM: 项目内 open/local evaluator，不是 lm-eval
```

## 产物清理

各 launcher 已在 eval 后调用清理脚本。手动清理某个实验目录：

```bash
bash scripts/cleanup_large_artifacts.sh results/phase2_v2/lis_Qwen3.5-9B-Base_train_en
```

清理会删除 checkpoint、`.bin`、`.pt`、大模型 safetensors 等，只保留 adapter/config/tokenizer/metadata/eval JSON。

默认不会删除 `adapter_model.safetensors` / `moe_weights.safetensors`，因为 MID/DSCT 依赖 `phase2_v2` 的 English donor adapter。所有实验都跑完并确认不再需要权重后再执行：

```bash
DELETE_TRAINED_WEIGHTS=1 bash scripts/cleanup_large_artifacts.sh results/phase2_v2 results/dsct results/mid results/mix_en results/moe_lora results/tag_routing results/sso_lora
```

默认禁止会写 full model 的路径：

```bash
scripts/train.py --method full_ft
scripts/train.py --method isolated_lora
scripts/train_layerwise.py --mode merge
scripts/train_sso_lora.py --mode merge
```

主 launcher 使用的是 LoRA adapter 保存和 `merge_eval` 内存合并，不会把中间 stage 的 full model 落盘。
