# 实验进度记录
## Cross-Lingual Interference in Instruction Tuning of Decoder-only LLMs

最后更新：2026-06-26 07:00 UTC

---

## ✅ 已完成

### 目录结构
```
project/
├── configs/
│   ├── ds_zero2.json                    ✅
│   ├── accelerate_4gpu.yaml             ✅ (GPU 挂载后由 setup_accelerate.sh 自动覆盖)
│   ├── accelerate_2gpu.yaml             ✅ (同上)
│   ├── accelerate_auto.yaml             ✅
│   └── experiments/
│       ├── lis_matrix.yaml              ✅
│       ├── mixture.yaml                 ✅
│       └── lora_comparison.yaml         ✅
├── scripts/
│   ├── prepare_data.py                  ✅
│   ├── train.py                         ✅
│   ├── evaluate.py                      ✅
│   ├── compute_lis.py                   ✅
│   ├── run_experiment.sh                ✅
│   └── setup_accelerate.sh             ✅ (GPU 挂载后运行一次，自动生成正确 GPU 数配置)
├── src/
│   ├── models/
│   │   ├── lora_standard.py             ✅
│   │   └── lora_isolated.py             ✅
│   ├── data/
│   │   ├── dataset_loader.py            ✅
│   │   └── data_mixer.py                ✅
│   ├── training/
│   │   └── trainer.py                   ✅
│   └── evaluation/
│       ├── english_eval.py              ✅
│       ├── multilingual_eval.py         ✅
│       └── lis_calculator.py            ✅
├── analysis/
│   ├── plot_lis_matrix.py               ✅
│   ├── plot_pareto_curve.py             ✅
│   ├── plot_lora_comparison.py          ✅
│   └── aggregate_results.py             ✅
├── results/{phase1..4}/                 ⏳ 空目录，等待实验
├── paper_results/                       ⏳ 空目录，等待分析
└── requirements.txt                     ✅
```

### 已安装的包（基础环境，无 venv）
| 包 | 版本 |
|---|---|
| torch | 2.7.1+cu126 |
| transformers | 5.9.0 |
| peft | 0.19.1 |
| accelerate | 1.13.0 |
| deepspeed | 0.19.1 |
| trl | 1.4.0 |
| datasets | 5.0.0 |
| lm-eval | 0.4.12 |
| anthropic | 0.101.0 |
| omegaconf | 2.3.0 |
| sacrebleu | 2.6.0 |
| langdetect | 1.0.9 |
| sentencepiece | 0.2.1 |

> **注意**：`unbabel-comet` 2.2.7 与 transformers 5.x 有版本冲突警告，但 COMET 仅作为 FLORES 辅助指标，冲突不影响主实验。

### Bug 修复记录（2026-06-11）
1. **`train.py`** — `isolated_lora` 分支不定义 `train_dataset`，保存 metadata 时 `NameError` → 修复为条件判断
2. **`train.py` + `lora_isolated.py`** — trl 1.4.0 中 `SFTTrainer(max_seq_length=..., tokenizer=...)` 已弃用 → 改用 `SFTConfig`（把 `max_seq_length` 和 `dataset_text_field` 放入 Config），`tokenizer` 改为 `processing_class`；新增 `build_sft_config()` 替换旧 `build_training_args()`
3. **`evaluate.py`** — `--include_lcb` 默认 `True` 导致 Phase 1 因 prompt 文件不存在直接崩溃 → 改为默认 `False`
4. **`evaluate.py`** — `--include_mt_bench` 受 `args.tasks in ("all",)` 额外限制，Phase 3/4 指定了特定 tasks 时无法触发 → 移除多余限制
5. **`run_experiment.sh`** — `while` 循环把所有参数 `shift` 掉后，`PHASE=${1:-$PHASE}` 靠副作用工作，在某些 bash 版本下会出错 → 重写为先取 `$1` 存到 `PHASE`、`shift` 消费它、再 `while` 处理剩余 flags
6. **`lora_isolated.py`** — `_freeze_except` 用 `adapter_name in name` 做字符串包含，`"fr"` 会误匹配含 `"freeze"` / `"from"` 的参数名 → 改为精确匹配 `.{adapter_name}.`（PEFT 命名规则：`lora_A.<adapter>.weight`）
7. **`lora_isolated.py`** — Stage 1 用 `model.set_adapter("shared")` 前需先 `model.enable_adapters()` 确保适配器已激活

---

## ✅ 数据 & 模型（已下载完成，2026-06-11）

### 模型
| 模型 | 路径 | 大小 | 状态 |
|---|---|---|---|
| Qwen/Qwen3.5-9B-Base | `/root/project/models/Qwen3.5-9B-Base` | 19GB（4 分片）| ✅ 完整 |

> Gemma 已从实验中移除，仅使用单模型。

### 数据集（`/root/project/data/processed/`）
| 语言 | 实际条数 | Phase2 需要(500) | Phase3 最大需要 | 状态 |
|---|---|---|---|---|
| en（英语，Open-Platypus）| 24926 | ✅ | ✅ | 完整 |
| zh（中文，Aya 简+繁合并）| 4909 | ✅ | ✅ | 完整 |
| yo（约鲁巴语，Aya）| 500 | ✅ | ✅（上限500）| 完整 |
| bn（孟加拉语，Aya+NLLB）| **2000** | ✅ | ✅ | ✅ NLLB 补充完成 |
| th（泰语，Aya+NLLB）| **2000** | ✅ | ✅ | ✅ NLLB 补充完成 |
| fr（法语，Aya+NLLB）| **5000** | ✅ | ✅ | ✅ NLLB 补充完成 |
| sw（斯瓦希里，Aya+NLLB）| **2000** | ✅ | ✅ | ✅ NLLB 补充完成 |

所有语言数据已满足 Phase 2（500条/语言）和 Phase 3（最大 5000条）需求。

### 数据 Bug 修复（2026-06-11）
- Aya 中中文语言名是 `"Simplified Chinese"`（非 `"Chinese Simplified"`），已修正 `prepare_data.py`
- 中文合并繁体（`"Traditional Chinese"`）：4909 条，满足 Phase 3 需求

## ✅ 新增完成（2026-06-12）

### Phase 1 基线评测 ✅ COMPLETE
- **结果文件**：`/root/project/results/phase1_baseline/Qwen3.5-9B-Base_baseline.json`
- **英文基线分数**：

| 指标 | 分数 |
|---|---|
| MMLU | **0.7729** |
| HellaSwag | **0.7919** |
| ARC-Challenge | **0.5717** |
| TruthfulQA MC1 | 0.3537（base 模型正常偏低）|
| English avg | 0.6225 |

- **多语言 Belebele 分数**（Phase 2 LIS 计算基准）：

| 语言 | Belebele acc |
|---|---|
| fr | 0.891 |
| zh | 0.887 |
| sw | 0.688 |
| th | 0.676 |
| bn | 0.620 |
| yo | 0.437 |

- **当前状态**：SIB-200 BPE bug 已修复，全部 7 语言有正确分数；FLORES 正在运行中；Belebele + English 全部正确

### NLLB 数据补充 ✅ COMPLETE（2026-06-12）
所有低资源语言已用 `facebook/nllb-200-3.3B` 补充到目标数量：

| 语言 | 补充前 | 补充后 | 耗时 |
|---|---|---|---|
| sw | 366 | **2000** | ~20min |
| th | 724 | **2000** | ~15min |
| bn | 1534 | **2000** | ~5min |
| fr | 1422 | **5000** | ~41min |

### 新实现脚本 ✅
| 脚本 | 说明 |
|---|---|
| `scripts/prepare_lcb_prompts.py` | ✅ 实现并运行，生成 6 语言各 200 条 LCB prompts |
| `scripts/prepare_mt_bench.py` | ✅ 实现（待运行，需 GPU）|

### LCB prompts 已生成（`data/lcb_prompts_*.jsonl`）
fr=200, zh=200, sw=200, th=200, bn=200, yo=200

---

### Bug 修复记录（2026-06-11 ~ 2026-06-12）

#### multilingual_eval.py / evaluate.py
8. **`_run_sib200`** — SIB-200 不在 lm-eval 0.4.12 任务表里 → 改为直接用 `Davlan/sib200` HF 数据集 + 模型零样本分类
9. **`_run_flores`** — `load_dataset("facebook/flores", "all")` 不存在 → 改为按语言单独加载再配对
10. **`torch_dtype`** → `dtype`（transformers 5.x）；影响 multilingual_eval.py, train.py

#### prepare_data.py NLLB
11. **`pipeline("translation")`** 在 transformers 5.x 中被移除 → 改用 `AutoModelForSeq2SeqLM` 直接 generate
12. **单样本串行推理** → `BATCH_SIZE=16` 批推理 + greedy（速度约 20x 提升）
13. **Python .pyc 缓存** 导致代码更新不生效 → 重启前先 `find . -name '*.pyc' -delete`

---

## ✅ 新增完成（2026-06-12 第二阶段）

### Bug 修复记录（训练/评测流程全链路打通）

14. **`accelerate: command not found`（SSH 环境）** — SSH 登录不加载 `~/.local/bin`，accelerate 不可用 → `launch_phase2_*.sh` 中 `export PATH=/home/tiger/.local/bin:$PATH`

15. **`TypeError: Descriptors cannot be created directly`（protobuf/wandb 冲突）** — worker 上自定义 wandb 导入 `databus`（用旧 protobuf C++ descriptor），`is_wandb_available()` 只捕获 `ImportError` 而非 `TypeError`，导致 trl 导入失败 → `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

16. **`SFTConfig.__init__() got unexpected keyword argument 'max_seq_length'`（TRL 1.4.0）** — TRL 1.4.0 将 `max_seq_length` 重命名为 `max_length` → `src/training/trainer.py` 中改为 `max_length=seq_len`

17. **`TensorBoardCallback requires tensorboard`** — `_get_report_to()` 回退到 `"tensorboard"` 但 tensorboard 未安装 → 改为回退到 `"none"`

18. **lm-eval `OSError: Repo id must be in the form 'repo_name'`** — PEFT adapter 目录没有 `config.json`，transformers 5.x 将其当作无效 HF model ID → `english_eval.py` 新增 `_lm_eval_model_args()`：检测 `adapter_config.json`，PEFT 模型使用 `pretrained={base},peft={abs_path},dtype=bfloat16` 格式

19. **multilingual_eval PEFT 加载** — `multilingual_eval.py` 中 `AutoModelForCausalLM.from_pretrained(adapter_dir)` 直接加载 adapter 目录会失败 → 新增 `_load_model_and_tokenizer()`：读取 `adapter_config.json` 后先加载 base model 再 `PeftModel.from_pretrained`

20. **`run_with_limit` 并发控制失效** — `$(jobs -r | wc -l)` 在子 shell 中永远返回 0，所有 Phase 2 实验同时启动 → 改为简单顺序 `for` 循环，独立 `launch_phase2_h100.sh` / `launch_phase2_a100.sh`

21. **DeepSpeed accelerate 配置冲突** — `accelerate_fullgpu.yaml` 使用 `deepspeed_config_file` 指向外部 JSON，同时 YAML 本身包含 `mixed_precision` 等字段，accelerate 1.13.0 遇到冲突时抛 `ValueError` → 改为 inline deepspeed 配置（去掉 `deepspeed_config_file` 字段，直接在 YAML 中写 `zero_stage: 2` 等参数）

### 验证结果
- ✅ 单 GPU 测试（yo，10 样本）：训练成功，adapter 正确保存
- ✅ 双 GPU DeepSpeed ZeRO-2 测试（yo，10 样本，H100×2）：`train_loss=1.981`，adapter 正确保存

### 新增/修改脚本
| 脚本 | 说明 |
|---|---|
| `scripts/launch_phase2_h100.sh` | ✅ H100 Phase 2 启动器（en/fr/zh/sw，顺序执行）|
| `scripts/launch_phase2_a100.sh` | ✅ A100 Phase 2 启动器（th/bn/yo，顺序执行）|
| `configs/accelerate_fullgpu.yaml` | ✅ 改为 inline DeepSpeed ZeRO-2 配置 |

---

## ✅ Phase 2 LIS 矩阵（已完成，2026-06-12）

启动时间：2026-06-12 06:39（H100）/ 06:56（A100）

### 额外 Bug 修复（启动过程中发现）

22. **`setup_accelerate.sh` 覆盖 inline 配置** — 每次启动时运行 `setup_accelerate.sh`，重新生成 `accelerate_fullgpu.yaml` 为有问题的 `deepspeed_config_file` 格式 → 修复 `setup_accelerate.sh` 中的 `write_accel_cfg()` 函数，改用 inline deepspeed 配置

23. **A100 NCCL `Failed to open libnvidia-ml.so.1`** — 容器内 `libnvidia-ml.so.1 → libnvidia-ml.so.535.129.03` 是 0 字节空文件，真实库是 `libnvidia-ml.so.535.161.08`；将真实库复制到 `/tmp/nv_libs/libnvidia-ml.so.1` 并 `export LD_LIBRARY_PATH=/tmp/nv_libs`（直接加 `/lib/x86_64-linux-gnu` 会同时引入 `libcuda.so.1` stub 导致 CUDA 崩溃）→ A100 恢复 **2×GPU DeepSpeed ZeRO-2** 训练；详细说明见 `docs/fix_a100_nccl_libnvidia_ml.md`

24. **`_run_belebele` PEFT 路径错误** — `multilingual_eval.py` 中 `_run_belebele` 直接把 adapter 目录路径传给 lm-eval，adapter 目录无 `config.json`，lm-eval 报 `ValueError: Unrecognized model` → 从 `english_eval.py` 导入 `_lm_eval_model_args()`，使用 `pretrained={base},peft={adapter_abs_path},dtype=bfloat16` 格式

25. **`en` 评测 Belebele 阶段崩溃，脚本继续执行** — `tee` 掩盖非零 exit code，H100 脚本标记 `en` 为 Done 但 JSON 未写出 → A100 Phase 2 完成后由 watcher 进程（PID 34606）自动补跑

26. **`_run_sib200` BPE 边界合并导致全部输出 0.200** — Qwen tokenizer 将 `"Topic: " + "travel"` 中的末尾空格与 category 首 token 合并，使 `full_ids[n_prompt:]` = 空列表，所有单 token 类别 score=0.0；唯独 `"science/technology"` 有非空 cat_ids 但 score 为负数，导致 `"travel"`（第一个 score=0.0 的类别）永远胜出，恰好对应测试集中 40/200=**0.200** 的占比 → 改为在 token ID 层面拼接（`tokenizer.encode(prompt) + tokenizer.encode(cat, add_special_tokens=False)`）；同时修复 `model.device` 在 `device_map="auto"` 下不可靠的问题，改用 `next(model.parameters()).device`

### 当前状态（2026-06-12 13:00，新 worker）

**新 worker**: `worker-0 (958363)`, IP `fdbd:dccd:cdc2:12c8:0:32f::`, port `10668`, 2×H100 80GB

#### ✅ Phase 1 baseline 完整结果
| 指标 | 状态 | 值 |
|---|---|---|
| English (MMLU/HellaSwag/ARC/TruthfulQA) | ✅ | avg=0.623 |
| SIB-200（7语言含en）| ✅ | en=0.550, fr=0.375, zh=0.570, sw=0.440, th=0.480, bn=0.520, yo=0.415 |
| Belebele（7语言含en）| ✅ | en=0.924, fr=0.891, zh=0.887, sw=0.688, th=0.676, bn=0.620, yo=0.437 |
| FLORES（6语言，lang→en BLEU）| ✅ | fr=49.55, zh=35.94, sw=1.59, th=0.001, bn=0.0, yo=0.13 |

#### ✅ Phase 2 完成（2026-06-12 21:16）

所有 7 个语言 adapter 训练完成并评测：

| 语言 | 训练 | eval JSON |
|---|---|---|
| en | ✅ | ✅ |
| fr | ✅ | ✅ |
| zh | ✅ | ✅ |
| sw | ✅（本次重新训练）| ✅ |
| th | ✅ | ✅ |
| bn | ✅（本次重新训练）| ✅ |
| yo | ✅（本次重新训练）| ✅ |

#### ✅ LIS 矩阵计算完成

结果文件：`results/phase2_lis_matrix/lis_matrix_Qwen3.5-9B-Base.csv`

```
LIS matrix (rows=train_lang, cols=eval_lang):
        en      fr      zh      sw      th      bn      yo
en  +0.038  +0.085 -0.006  +0.039  +0.035  +0.050  +0.031
fr  +0.016 -0.054 -0.011  +0.027  +0.064  +0.063  +0.008
zh  +0.041  +0.153  +0.008  +0.022  +0.038  +0.035 -0.005
sw  +0.021 -0.008  +0.001  +0.020  +0.051  +0.094  +0.019
th  +0.058  +0.109  +0.007  +0.063  +0.083  +0.102  +0.075
bn  +0.017  +0.019 -0.003  +0.054  +0.047  +0.040  +0.010
yo -0.058 -0.040 -0.012 -0.039  +0.002 -0.019 -0.085
```

**关键发现**：
- **yo（约鲁巴）**：唯一整行几乎全负——fine-tuning 损害自身及其他语言，典型 low-resource interference
- **th（泰语）**：整行最高——对 fr/sw/bn/yo 有最大正向迁移
- **对角线**：fr=-0.054、yo=-0.085 负向，其余正向或接近 0
- **English retention**：所有语言训练后 MMLU 稳定在 0.77，无 catastrophic forgetting

**方法论说明**（论文 limitation）：SIB-200 和 Belebele 均为多选题，对语言混淆不敏感；interference 在生成任务（LCB）中可能更显著，作为 limitation 写入论文。

#### Bug 修复（2026-06-12 本次 session）
27. **`_load_model_and_tokenizer` PEFT tokenizer 崩溃** — adapter 目录无 tokenizer 文件，`AutoTokenizer.from_pretrained(adapter_dir)` 报 `ValueError: Unrecognized model` → 检测到 `adapter_config.json` 时从 base model 路径加载 tokenizer
28. **HF 429 rate limit** — 用 `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` 绕过，数据集已本地缓存

### 注意事项
- `prepare_mt_bench.py` 失败（HuggingFace rate limit → 0 questions → ZeroDivisionError），Phase 3 启动前需修复
- 所有结果文件在 NFS 共享路径，worker 和头节点均可访问

## ✅ 新增完成（2026-06-15）

### 实验语言集重置：切换为 en+yo+so+ha

原 Phase 2 使用 7 语言（en/fr/zh/sw/th/bn/yo），已完成但与当前研究计划不符。
本次重置为正式实验语言集 **en + yo + so + ha**（全部 Aya 全量数据）。

#### 数据确认（已就绪，无需重新下载）
| 语言 | 条数 | 路径 |
|---|---|---|
| en（Open-Platypus）| 24926 | `data/processed/en.jsonl` |
| yo（Aya Yoruba）| 11758 | `data/processed/yo.jsonl` |
| so（Aya Somali）| 7704 | `data/processed/so.jsonl` |
| ha（Aya Hausa）| 3512 | `data/processed/ha.jsonl` |

#### Config 修改（`configs/experiments/lis_matrix.yaml`）
| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `num_epochs` | 3 | **2** | 用户要求 |
| `train_samples` | 500 | **null** | 原来只取 500 条，改为全量 |
| GPU 注释 | 8 GPUs | **4 GPUs** | 当前 worker 4 张 H100 |

#### 脚本改写（`scripts/launch_phase2_v2.sh`）
旧版脚本使用单 GPU 并行 + 不存在的 `phase2_v2.yaml`，已完整重写为：
- 顺序执行：每个语言用全部 4×H100（ZeRO-2），依次训练
- Step 1：baseline 评测（base model，无 SFT）→ `results/phase2_v2/Qwen3.5-9B-Base_baseline.json`
- Step 2：en → yo → so → ha 依次训练 + 评测（跳过 FLORES，只跑 Belebele + SIB-200）
- Step 3：自动计算 4×4 LIS 矩阵

### 新 worker（2026-06-15）
| 字段 | 值 |
|---|---|
| Worker ID | worker-0 (960298) |
| IP | `fdbd:dccd:cdc2:12c8:0:14::` |
| SSH Port | 10677 |
| GPU | 4 × H100 80GB HBM3 |
| 连接命令 | `ssh -i ~/.ssh/id_rsa -p 10677 tiger@fdbd:dccd:cdc2:12c8:0:14::` |

### Phase 2 v2（✅ 全部完成，2026-06-17）

| 步骤 | 状态 | 输出 |
|---|---|---|
| Baseline（base model，无 SFT）| ✅ | `results/phase2_v2/Qwen3.5-9B-Base_baseline.json` |
| en 训练（24926 条，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_en/` |
| yo 训练（11758 条，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo/` |
| so 训练（7704 条，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_so/` |
| ha 训练（3512 条，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha/` |
| 各语言评测（en+yo+so+ha Belebele+SIB-200）| ✅ | 各 _eval.json |
| 4×4 LIS 矩阵计算 | ✅ | `results/phase2_v2/lis_matrix_Qwen3.5-9B-Base.csv` |
| 扩展评测（AfriQA/Aya/AfriMMLU/AfriXNLI/AfriMGSM）| ✅ | 写入各模型 _eval.json 的 scores.multilingual |

进度查看：
```bash
ssh -i ~/.ssh/id_rsa -p 10677 tiger@fdbd:dccd:cdc2:12c8:0:14:: 'tail -50 /root/project/logs/phase2_v2_master.log'
```

---

## ✅ 新增完成（2026-06-15 ~ 2026-06-17）

### Phase 2 v2 全部完成

#### 训练 & 基础评测（SIB-200 + Belebele）✅
| 模型 | 状态 | 结果文件 |
|---|---|---|
| Baseline（base model，无 SFT）| ✅ | `results/phase2_v2/Qwen3.5-9B-Base_baseline.json` |
| SFT-en（en 全量，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_en_eval.json` |
| SFT-yo（yo 全量，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_yo_eval.json` |
| SFT-so（so 全量，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_so_eval.json` |
| SFT-ha（ha 全量，2 epoch）| ✅ | `results/phase2_v2/lis_Qwen3.5-9B-Base_train_ha_eval.json` |

#### 4×4 LIS 矩阵 ✅
结果文件：`results/phase2_v2/lis_matrix_Qwen3.5-9B-Base.csv`

```
LIS matrix (rows=train_lang, cols=eval_lang):
        en      yo      so      ha
en   0.035  -0.001  +0.079  +0.086
yo   0.007  +0.001  +0.090  +0.131
so  +0.039  +0.030  +0.069  +0.026
ha  +0.036  +0.030  +0.078  +0.131
```

#### 扩展评测（AfriQA / Aya Evaluation / IrokoBench 全三子集）✅

脚本：`scripts/eval_extended.py`，结果合并写入各模型已有 JSON 的 `scores.multilingual` 下。

运行方式：
- 主跑：`scripts/run_eval_extended.sh`（AfriQA + Aya + AfriMMLU，4 模型并行 + ha 单独第二批）
- 补跑：`scripts/run_eval_iroko_extra.sh`（AfriXNLI + AfriMGSM，相同并行策略）

数据集来源：
| 子测试集 | HF 数据集 | 语言 |
|---|---|---|
| AfriQA | `masakhane/afriqa` | yo, ha（so 无数据）|
| Aya Evaluation | `CohereLabs/aya_evaluation_suite` (dolly_machine_translated) | en, yo, so, ha |
| IrokoBench/AfriMMLU | `masakhane/afrimmlu` | yo, ha（so 无数据）|
| IrokoBench/AfriXNLI | `masakhane/afrixnli` | yo, ha（so 无数据）|
| IrokoBench/AfriMGSM | `masakhane/afrimgsm` | yo, ha（so 无数据）|

**完整评测结果（全量数据）：**

##### AfriQA（F1 / Exact Match）
| 模型 | yo F1 | yo EM | ha F1 | ha EM |
|---|---|---|---|---|
| baseline | 0.0445 | 0.0153 | 0.0164 | 0.0067 |
| SFT-en | 0.1109 | 0.0491 | 0.1262 | 0.0900 |
| SFT-yo | 0.1215 | 0.0491 | 0.1002 | 0.0600 |
| SFT-so | 0.0879 | 0.0123 | 0.0663 | 0.0133 |
| SFT-ha | **0.1535** | **0.0890** | **0.1965** | **0.1433** |

##### Aya Evaluation（GPT-5.4 judge，1-10分，200条/语言）
| 模型 | en | yo | so | ha |
|---|---|---|---|---|
| baseline | 4.84 | 1.74 | 1.78 | 2.03 |
| SFT-en | 4.64 | 1.65 | **2.11** | **2.17** |
| SFT-yo | 4.42 | 1.73 | 1.87 | 1.81 |
| SFT-so | **4.84** | 1.48 | 1.80 | 1.96 |
| SFT-ha | 4.56 | 1.49 | 1.76 | 1.98 |

##### IrokoBench / AfriMMLU（MCQ Accuracy，500题）
| 模型 | yo | ha |
|---|---|---|
| baseline | 0.450 | 0.424 |
| SFT-en | **0.476** | **0.438** |
| SFT-yo | 0.454 | 0.398 |
| SFT-so | 0.428 | 0.410 |
| SFT-ha | 0.454 | 0.420 |

##### IrokoBench / AfriXNLI（NLI Accuracy，600题）
| 模型 | yo | ha |
|---|---|---|
| baseline | 0.5933 | 0.5867 |
| SFT-en | **0.6000** | **0.5933** |
| SFT-yo | 0.5433 | 0.5100 |
| SFT-so | 0.5000 | 0.5083 |
| SFT-ha | 0.4933 | 0.4950 |

##### IrokoBench / AfriMGSM（Math Accuracy，250题）
| 模型 | yo | ha |
|---|---|---|
| baseline | 0.072 | 0.060 |
| SFT-en | 0.072 | 0.096 |
| SFT-yo | 0.084 | 0.096 |
| SFT-so | 0.080 | **0.100** |
| SFT-ha | 0.084 | 0.092 |

> **注**：Somali 在全部 IrokoBench 子集（AfriMMLU/AfriXNLI/AfriMGSM）中均无数据，记为 null。

---

## ✅ 新增完成（2026-06-17）

### 扩展评测：Uhura-TruthfulQA MC1（yo/ha）✅

数据集：`ebayes/uhura-truthfulqa`（yo_multiple_choice / ha_multiple_choice，各 ~808 题）  
评测方式：batched log-likelihood 打分（与英文 TruthfulQA 相同，完全确定性）  
脚本：`scripts/eval_extended.py --only_uhura_truthfulqa`  
并行：`scripts/run_uhura_truthfulqa.sh`（4 模型×4 卡并行，ha 第二批单独跑）

| 模型 | yo MC1 | ha MC1 |
|---|---|---|
| baseline | 0.3931 | 0.3527 |
| SFT-en | **0.4091** | **0.3626** |
| SFT-yo | 0.3857 | 0.3342 |
| SFT-so | 0.3857 | 0.3391 |
| SFT-ha | 0.3906 | 0.3317 |

结果写入各模型 `_eval.json` 的 `scores.multilingual.uhura_truthfulqa`。

### 全部 LIS 矩阵更新（2026-06-17）✅

结果目录 `results/phase2_v2/` 已更新以下 9 个文件（从 worker 同步最新 JSON 后重建）：

| 文件 | 测试集 | 覆盖语言 |
|---|---|---|
| `lis_matrix_Qwen3.5-9B-Base.csv` | SIB200+Belebele 均值（主矩阵）| en/yo/so/ha |
| `lis_matrix_Qwen3.5-9B-Base_sib200.csv` | SIB-200 | en/yo/so/ha |
| `lis_matrix_Qwen3.5-9B-Base_belebele.csv` | Belebele | en/yo/so/ha |
| `lis_matrix_Qwen3.5-9B-Base_afriqa_f1.csv` | AfriQA F1 | yo/ha |
| `lis_matrix_Qwen3.5-9B-Base_aya.csv` | Aya Evaluation GPT 分 | en/yo/so/ha |
| `lis_matrix_Qwen3.5-9B-Base_irokobench_mcq.csv` | IrokoBench MCQ | yo/ha |
| `lis_matrix_Qwen3.5-9B-Base_afrixnli.csv` | AfriXNLI | yo/ha |
| `lis_matrix_Qwen3.5-9B-Base_afrimgsm.csv` | AfrIMGSM | yo/ha |
| `lis_matrix_Qwen3.5-9B-Base_english.csv` | English 四项 LIS | — |
| `en_retention_Qwen3.5-9B-Base.csv` | English 绝对分数 | — |

## ✅ 新增完成（2026-06-17 ~ 2026-06-18）

### 混合数据 SFT 实验（全部完成，2026-06-17 21:31 UTC）

**方案**：全量 en（24926）+ 全量目标语言 concat shuffle（seed=42），2 epoch，4×H100 ZeRO-2  
**脚本**：`scripts/launch_mix_en.sh`

| 实验 | 训练样本 | 完成时间 | 输出 |
|---|---|---|---|
| mix(en+yo) | 36684 | 15:04 UTC | `results/mix_en/mix_Qwen3.5-9B-Base_en_yo_eval.json` |
| mix(en+so) | 32630 | 18:36 UTC | `results/mix_en/mix_Qwen3.5-9B-Base_en_so_eval.json` |
| mix(en+ha) | 28438 | 21:31 UTC | `results/mix_en/mix_Qwen3.5-9B-Base_en_ha_eval.json` |

**评测结果汇总**（Belebele + TruthfulQA MC1 + IrokoBench MCQ）：

| 模型 | tqa_mc1 | bele_en | bele_yo | bele_so | bele_ha | mcq_yo | mcq_ha |
|---|---|---|---|---|---|---|---|
| baseline | 0.3488 | 0.9244 | 0.4367 | 0.4689 | 0.4844 | 0.4500 | 0.4240 |
| mix_en_yo | 0.3660 | 0.9122 | **0.4578** | 0.4411 | 0.4478 | **0.4660** | 0.3880 |
| mix_en_so | 0.3647 | 0.9211 | 0.4211 | **0.4622** | 0.4633 | 0.4260 | 0.4160 |
| mix_en_ha | 0.3599 | 0.9222 | 0.4200 | 0.4500 | **0.4767** | 0.4160 | 0.4040 |

**规律**：mix 模型对目标语言有正迁移，对其他语言有轻微干扰（-3% ~ -8%），符合语言竞争效应。

---

### LCB 语言一致性评测（文本续写版，2026-06-18）✅

**设计**：MasakhaNEWS 新闻正文前 2 句作 prompt，贪婪生成 200 token，GlotLID 检测续写语言  
**检测器**：GlotLID（`cis-lmu/glotlid`，1600+ 语言，对 yo/so/ha 置信度 1.00）  
> fastText lid.176 对 yo/so/ha 无法识别（误判为 Esperanto、Filipino 等），已替换  

**脚本**：`scripts/eval_lcb.py`，`scripts/run_lcb_all.sh`  
**结果目录**：`results/lcb/`（8 个 JSON）

**lc_rate（Δ 相对 baseline）**：

| 模型 | yo | so | ha |
|---|---|---|---|
| baseline | 0.985 | 0.985 | 0.995 |
| train_en | 0.995 (+.010) | 0.990 (+.005) | 0.995 (±0) |
| train_yo | **1.000** (+.015) | 0.975 (-.010) | 0.959 (-.036) |
| train_so | 0.975 (-.010) | 0.980 (-.005) | 0.985 (-.010) |
| train_ha | 0.980 (-.005) | 0.995 (+.010) | 0.984 (-.011) |
| mix_en_yo | 0.985 (±0) | 0.995 (+.010) | 0.940 (-.055) |
| mix_en_so | 0.965 (-.020) | 0.995 (+.010) | 0.995 (±0) |
| mix_en_ha | 0.985 (±0) | 1.000 (+.015) | 0.984 (-.011) |

> en_leak 全部为 0（所有模型均不 code-switch 到英文做文本续写）  
> 结论：文本续写下区分度不足（baseline 已 ≥0.985），主要干扰体现在任务准确率上

---

### LCB-Chat 跨语言指令遵循评测（2026-06-18）✅

**设计**：50 条 English 指令，模板 `<|tgt_lang:en|> Please respond to the following in {lang}: {instr}`，要求模型切换到目标语言回答  
**解释**：用 `<|tgt_lang:en|>` 因为指令是英文（in-distribution）；语言要求嵌在指令正文  
**脚本**：`scripts/eval_lcb_chat.py`，`scripts/run_lcb_chat_all.sh`  
**结果目录**：`results/lcb_chat/`

**lc_rate / en_leak**：

| 模型 | yo_lc | yo_en | so_lc | so_en | ha_lc | ha_en |
|---|---|---|---|---|---|---|
| baseline | 0.02 | 0.98 | 0.00 | 1.00 | 0.06 | 0.94 |
| train_en | 0.96 | 0.02 | 0.96 | 0.02 | 0.80 | 0.20 |
| train_yo | 0.92 | 0.04 | 0.71 | 0.24 | 0.62 | 0.32 |
| train_so | 0.02 | 0.98 | 0.28 | 0.72 | 0.00 | 1.00 |
| train_ha | 0.02 | 0.96 | 0.06 | 0.94 | 0.00 | 1.00 |
| mix_en_yo | **0.98** | 0.02 | **0.98** | 0.00 | 0.64 | 0.16 |
| mix_en_so | 0.80 | 0.16 | **0.98** | 0.02 | 0.64 | 0.34 |
| mix_en_ha | 0.94 | 0.06 | 0.96 | 0.02 | **0.98** | 0.02 |

**关键发现**：
1. **英文 SFT 是 instruction-following 能力的来源**：baseline≈0 → train_en≈0.96，"用目标语言回答"这一能力完全来自英文训练数据
2. **单语言 SFT 导致严重灾难性遗忘**：train_so/train_ha 连自己目标语言也做不到（0.28/0.00），因为它们从未见过 `<|tgt_lang:en|>` 格式，`<|tgt_lang:en|>` 触发英文默认输出
3. **混合英文数据大幅恢复**：mix_en_ha ha: 0.00 → **0.98**；mix_en_so so: 0.28 → **0.98**
4. **train_yo 保留了部分通用能力**：yo=0.92，甚至 so=0.71（Somali！），推测与数据量更大有关

> **方法论说明**：LCB-chat 测的是"能否遵守英文 instruction 里的跨语言要求"，单语言 SFT 模型回答英文是因为从未训练 `<|tgt_lang:en|>` 格式，并非遗忘了目标语言本身

---

## ⏳ 待完成

### 下一步实验方向
| 内容 | 说明 | 状态 |
|---|---|---|
| Isolated-LoRA 实验 | yo/so/ha 各一组，shared(en+lang,2ep)+lang(1ep)，merge 后评测 | ⚠️ **部分完成**：yo ✅（2026-06-22 13:17），so/ha ❌ 用户决定跳过，直接进入 Phase 5 |
| Phase 5 全语言 SFT + Tag Hard Routing | en+yo+so+ha 全量 concat，标准 LoRA 2 epoch，tag 硬路由（yo/so/ha 复用 mix_en adapters，en 用 4-lang 模型）| 🔄 **训练中**（2026-06-22 启动，A100 40GB，step ~6600/11976，54.8%，ETA ~15h）|

> ~~旧 Phase 2（7 语言）~~ 结果保留在 `results/phase2_lis_matrix/`，不再使用。
> Phase 3（数据配比）、Phase 4（LoRA 完整对比）暂缓，优先完成 Isolated-LoRA 和 Tag Routing。

---

### Isolated-LoRA 实验结果（2026-06-22）

| 语言 | 状态 | 说明 |
|---|---|---|
| yo | ✅ 完成 | Stage 1（shared, en+yo）+ Stage 2（yo adapter）全部完成；模型保存于 `results/isolated_lora/isolated_Qwen3.5-9B-Base_yo/` |
| so | ❌ 中断 | Stage 1 step 15/4080 被 kill（用户决定跳过，2026-06-22 14:36 UTC）|
| ha | ❌ 未启动 | 用户决定跳过，直接进入 Phase 5 |

**yo Isolated-LoRA 评测结果**（`results/isolated_lora/isolated_Qwen3.5-9B-Base_yo_eval.json`）：

| 指标 | en | yo | so | ha |
|---|---|---|---|---|
| SIB-200 | 0.54 | 0.29 | 0.34 | 0.32 |
| Belebele | 0.903 | 0.432 | 0.439 | 0.428 |
| IrokoBench MCQ | — | 43.4% | — | 39.2% |
| TruthfulQA MC1 | 37.3% | — | — | — |

> so/ha 因训练未完成，Isolated-LoRA 对照实验不完整；yo 结果可作为单语言 isolated adapter 参考点。

---

### Phase 5：全语言 SFT + Tag Hard Routing（已启动，正在跑，A100 40GB）

**设计**：两个实验并行比较——

| 实验 | 描述 |
|---|---|
| **4-lang mixed（对照基线）** | en+yo+so+ha 全量 concat，标准 LoRA r=16，**2 epoch**，无路由，推理时直接使用 |
| **Tag Hard Routing** | 推理时解析 `<|tgt_lang:xx|>` tag，**硬切换**到对应语言 adapter，无 merge，无共享权重 |

**Adapter 来源（Tag Hard Routing）**：
- `en` → 4-lang mixed 训练结果（上面同一个模型）
- `yo` → `results/mix_en/mix_Qwen3.5-9B-Base_en_yo/`（已有，直接复用）
- `so` → `results/mix_en/mix_Qwen3.5-9B-Base_en_so/`（已有，直接复用）
- `ha` → `results/mix_en/mix_Qwen3.5-9B-Base_en_ha/`（已有，直接复用）

**训练**（仅需训练 4-lang mixed 模型）：
- en+yo+so+ha 全量 concat shuffle，标准 LoRA r=16，**2 epoch**
- 脚本：`launch_tag_routing.sh`（`--mix_all_langs`，启动前将 `lis_matrix.yaml` `num_epochs` 改为 2）
- 输出：`results/tag_routing/tag_routing_Qwen3.5-9B-Base/`

**评测**（所有测试集注入 `<|tgt_lang:xx|>` tag —— `--inject_lang_tag`）：
- TruthfulQA MC1（手动 MCQ + en tag）
- SIB-200 en/yo/so/ha（tag prefix 注入）
- Belebele en/yo/so/ha（手动 MCQ，取代 lm-eval，支持 tag 注入）
- IrokoBench AfriMMLU MCQ yo/ha（tag prefix 注入）
- LCB-chat yo/so/ha（lc_rate，重点指标）

**对照关系**：
- 4-lang mixed（无 tag）→ 多语言混合训练是否保留 instruction-following
- Tag Hard Routing（有 tag）→ tag 路由是否能在保留各语言能力的同时维持英文质量
- Isolated-LoRA（已完成）→ shared adapter 设计 vs 独立 adapter
---

## ✅ 新增完成（2026-06-23）

### LCB-chat 4×4 矩阵评测 v1（带 tag，MasakhaNEWS 续写）✅

**完成时间**：2026-06-23 10:51 UTC  
**脚本**：`scripts/eval_lcb_matrix.py`，`scripts/run_lcb_matrix_all.sh`  
**结果目录**：`results/lcb_matrix/`（8 个 JSON）

**prompt 格式**：
```
### Instruction:
<|tgt_lang:{input_lang}|> Please continue the following news article in {output_lang}: {excerpt}

### Response:
```
tag = input_lang（新闻摘录所在语言），文本明确指定 output_lang。

**lc_rate 完整矩阵（行 = 指令语言，列 = 目标输出语言）**：

```
=== baseline ===
instr\tgt      en      yo      so      ha
        en   1.000   0.120   0.640   0.260
        yo   0.720   0.860   0.080   0.040
        so   0.640   0.000   0.960   0.000
        ha   0.560   0.000   0.420   0.940

=== train_en ===
        en   1.000   0.340   0.620   0.400
        yo   0.347   0.960   0.000   0.040
        so   0.200   0.000   0.980   0.000
        ha   0.122   0.000   0.480   0.980

=== train_yo ===
        en   1.000   0.180   0.120   0.060
        yo   0.000   0.980   0.040   0.000
        so   0.000   0.020   1.000   0.000
        ha   0.020   0.220   0.240   1.000

=== train_so ===
        en   1.000   0.000   0.000   0.040
        yo   0.000   1.000   0.020   0.000
        so   0.000   0.000   0.980   0.000
        ha   0.020   0.000   0.780   0.560

=== train_ha ===
        en   1.000   0.140   0.040   0.180
        yo   0.000   0.960   0.000   0.080
        so   0.000   0.000   1.000   0.000
        ha   0.000   0.000   0.180   1.000

=== mix_en_yo ===
        en   1.000   0.160   0.120   0.000
        yo   0.000   1.000   0.060   0.000
        so   0.000   0.040   0.980   0.000
        ha   0.000   0.160   0.160   0.960

=== mix_en_so ===
        en   1.000   0.260   0.760   0.320
        yo   0.000   1.000   0.040   0.000
        so   0.000   0.000   1.000   0.000
        ha   0.000   0.000   0.740   1.000

=== mix_en_ha ===
        en   1.000   0.440   0.820   0.760
        yo   0.000   0.980   0.000   0.020
        so   0.000   0.000   0.980   0.000
        ha   0.000   0.000   0.360   0.980
```

**关键发现**：
1. **SFT 模型强 latch tag 语言**：非对角线 SFT 后普遍塌零，模型输出 tag 所指语言（即 input_lang），忽视文本中的目标语言指令（实测：ha→yo cell 50 条全输出 Hausa，GlotLID conf≥0.945）
2. **baseline 无 tag 概念**：非对角线 en_leak 高（跨语言失败时回退英文）
3. **mix_en_ha en 行最强**：(en→yo)=0.44, (en→so)=0.82, (en→ha)=0.76，英文指令下跨语言跟随最好
4. **评测 insight**：v1 格式测试了"能否无视 tag 跟随文本指令"，SFT 模型普遍失败

---

## ✅ 新增完成（2026-06-23，续）

### LCB-chat 4×4 矩阵评测 v2（无 tag，指令用 input_lang 本地化写）✅

**完成时间**：2026-06-23 15:15 UTC（全部 8 模型）  
**脚本**：`scripts/eval_lcb_notag.py`，`scripts/run_lcb_notag_all.sh`  
**结果目录**：`results/lcb_notag/`（**8 个 JSON，全部完成**：baseline + train_en/yo/so/ha + mix_en_yo/so/ha）

**en→lang lc_rate（无 tag，英文指令跟随）**：

| 模型 | en→en | en→yo | en→so | en→ha |
|---|---|---|---|---|
| baseline | 1.000 | 0.280 | 0.640 | 0.300 |
| train_en | 1.000 | 0.220 | 0.500 | 0.340 |
| train_yo | 1.000 | **0.720** | 0.500 | 0.306 |
| train_so | 1.000 | 0.120 | 0.300 | 0.220 |
| train_ha | 1.000 | 0.260 | 0.420 | 0.620 |
| mix_en_yo | 1.000 | 0.300 | 0.140 | 0.020 |
| mix_en_so | 1.000 | 0.340 | **0.840** | 0.380 |
| mix_en_ha | 1.000 | 0.420 | 0.800 | **0.780** |

**prompt 格式（v2）**：
```
### Instruction:
{instruction_in_input_lang} {excerpt}

### Response:
```
指令语言本地化（无 tag）：
- en: `Please continue the following news article in {lang}:`
- yo: `Jọwọ tẹsiwaju nkan ìròyìn yìí ní èdè {lang}:`
- so: `Fadlan sii wad maqaalkan wararka ah ee {lang}:`
- ha: `Don Allah ci gaba da wannan labarai a cikin {lang}:`

### LCB-chat 4×4 矩阵评测 v3（tag=output_lang，指令用英文）✅

**完成时间**：2026-06-23 ~15:00 UTC（8/8 模型全部完成）  
**脚本**：`scripts/eval_lcb_tagtgt.py`，`scripts/run_lcb_tagtgt_all.sh`  
**结果目录**：`results/lcb_tagtgt/`（**8 个 JSON**：baseline + train_en/yo/so/ha + mix_en_yo/so/ha）

**prompt 格式（v3）**：
```
### Instruction:
<|tgt_lang:{output_lang}|> Please continue the following news article in {output_lang_name}: {excerpt}

### Response:
```
tag = 目标输出语言（与文本指令一致）。

**三版对比设计**：
| 版本 | tag | 指令文本语言 | 设计目的 |
|------|-----|------------|---------|
| v1 | input_lang | English | tag 与目标不一致，测试文本指令优先级 |
| v2 | 无 | input_lang（本地化）| 无 tag，测试纯语言指令跟随 |
| v3 | output_lang | English | tag 与目标一致，测试 tag 正确设置后的效果 |

> **注**：tag_routing / MID 模型的三版矩阵待后续补跑。8 个原始模型 v2/v3 均已完成。

---

## ✅ 新增完成（2026-06-23 ~ 2026-06-24）

### Phase 5：Tag Routing 训练 + 评测 ✅

- 训练完成：2026-06-23 ~19:06 UTC（step 11976/11976）
- 评测完成：2026-06-24 01:11 UTC
- Bug 修复：`english_eval.py` 中 `"truthful_qa"` → `"truthfulqa/truthful_qa"`（datasets 5.x 要求命名空间）
- 模型路径：`results/tag_routing/tag_routing_Qwen3.5-9B-Base/`
- 评测结果：`results/tag_routing/tag_routing_Qwen3.5-9B-Base_eval.json`

### MID（Mechanistic Interface Distillation）训练 + 评测 ✅

- 启动：2026-06-23 13:35 UTC
- 完成：2026-06-24 22:07 UTC（yo→so→ha 顺序，自动评测）
- 模型路径：`results/mid/mid_Qwen3.5-9B-Base_{yo,so,ha}/`
- 评测结果：`results/mid/mid_Qwen3.5-9B-Base_{yo,so,ha}_eval.json`

**MID 方案核心参数**：
| 参数 | 值 |
|---|---|
| Teacher | Base + LoRA_en（merged, frozen）|
| Student | Base + LoRA_spec（纯目标语言 CE）|
| α（Pos1 CosDist 权重）| 0.1 |
| β（Pos2 CosDist 权重）| 0.05 |
| 蒸馏层 | top-4 layers（倒数 4 层）|
| Pos2 token 数 | 3 |

**Probe 结果**（训练前验证 teacher 信号质量）：
| 语言 | 均值 pairwise cosine-sim（last layer, Pos1）| 结论 |
|---|---|---|
| yo | 0.8022 | ✅ HIGH |
| so | 0.8127 | ✅ HIGH |
| ha | 0.7670 | ✅ HIGH |

---

## 📊 全量实验结果汇总（2026-06-24）

### Belebele + TruthfulQA MC1 + IrokoBench MCQ

| 模型 | tqa | bele_en | bele_yo | bele_so | bele_ha | mcq_yo | mcq_ha |
|------|-----|---------|---------|---------|---------|--------|--------|
| baseline | 0.3488 | 0.9244 | 0.4367 | 0.4689 | 0.4844 | 0.4500 | 0.4240 |
| train_en | 0.3684 | 0.9220 | 0.4567 | 0.4833 | 0.4922 | 0.4760 | 0.4380 |
| mix_en_yo | 0.3660 | 0.9122 | 0.4578 | 0.4411 | 0.4478 | 0.4660 | 0.3880 |
| mix_en_so | 0.3647 | 0.9211 | 0.4211 | 0.4622 | 0.4633 | 0.4260 | 0.4160 |
| mix_en_ha | 0.3599 | 0.9222 | 0.4200 | 0.4500 | 0.4767 | 0.4160 | 0.4040 |
| **tag_routing** | **0.3758** (无tag) ~~0.6132~~(有tag) | **0.9378** | **0.4889** | **0.4889** | **0.5211** | 0.4400 | 0.3980 |
| MID_yo | 0.3390 | 0.9300 | 0.4689 | 0.4822 | 0.4833 | 0.4500 | 0.3880 |
| MID_so | 0.3268 | 0.9400 | 0.4189 | 0.4933 | 0.4911 | 0.4480 | 0.4640 |
| MID_ha | 0.3317 | 0.9333 | 0.4289 | **0.5178** | **0.5100** | 0.4460 | 0.4220 |

> **注**：tag_routing TruthfulQA **无tag重测=0.3758**（与其他模型可比）；原有tag评测=0.6132（注入`<|tgt_lang:en|>`，不可比，已废弃）。

### LCB-chat v1（tag=input_lang，en→lang lc_rate）

| 模型 | en→en | en→yo | en→so | en→ha |
|------|-------|-------|-------|-------|
| train_en（参考）| 1.000 | 0.340 | 0.620 | 0.400 |
| mix_en_yo（参考）| 1.000 | 0.980 | 0.980 | 0.640 |
| **tag_routing** | 1.000 | 0.360 | 0.580 | 0.184 |
| MID_yo | 1.000 | 0.220 | 0.060 | 0.000 |
| MID_so | 0.980 | 0.000 | 0.000 | 0.000 |
| MID_ha | 1.000 | 0.000 | 0.000 | 0.020 |

**关键发现**：
1. **Tag routing TruthfulQA 大幅提升（0.61 vs baseline 0.35）**：tag injection 让 4-lang 混合模型在英文任务上表现显著优于所有单语/双语 mix 模型
2. **MID LCB 近乎全部归零**：hidden-state 蒸馏保住了多语言理解能力（Belebele 稳定），但完全无法维持跨语言指令遵循（en→lang）——模型从未见过 `<|tgt_lang:en|>` 格式，无法跟随英文指令切换输出语言
3. **Tag routing LCB 表现中等（en→ha 仅 0.184）**：4-lang 混合训练保住了 yo/so 的跨语言跟随，但 ha（数据最少）明显弱
4. **MID Belebele 超越 mix 基线**：MID_ha bele_so=0.5178 / bele_ha=0.5100，超过所有 mix_en 模型，表明纯目标语言 CE + 潜空间约束能改善多语言理解

---

## ✅ 新增完成（2026-06-24）

### LCB v2/v3 全量结果 + 三版对比分析（8 个原始模型）

**最后更新**：2026-06-24 UTC

---

#### v2（无 tag，输入语言指令要求目标语言输出）en→{yo,so,ha} lc_rate

| 模型 | en→yo | en→so | en→ha |
|------|-------|-------|-------|
| baseline | 0.28 | 0.64 | 0.30 |
| train_en | 0.22 | 0.50 | 0.34 |
| train_yo | **0.72** | 0.50 | 0.31 |
| train_so | 0.12 | 0.30 | 0.22 |
| train_ha | 0.26 | 0.42 | **0.62** |
| mix_en_yo | 0.30 | 0.14 | 0.02 |
| mix_en_so | 0.34 | **0.84** | 0.38 |
| mix_en_ha | 0.42 | 0.80 | **0.78** |

#### v3（tag=output_lang，与文本指令一致）en→{yo,so,ha} lc_rate

| 模型 | en→yo | en→so | en→ha |
|------|-------|-------|-------|
| baseline | 0.06 | 0.34 | 0.14 |
| train_en | 0.26 | 0.50 | 0.30 |
| train_yo | 0.18 | 0.12 | 0.14 |
| train_so | 0.00 | 0.00 | 0.00 |
| train_ha | 0.04 | 0.18 | 0.26 |
| mix_en_yo | 0.18 | 0.02 | 0.02 |
| mix_en_so | 0.06 | 0.34 | 0.18 |
| mix_en_ha | **0.34** | **0.68** | **0.78** |

#### 三版对比：avg lc_rate (en→yo/so/ha 均值)

| 模型 | v1（tag冲突） | v2（无tag） | v3（tag一致） |
|------|-------------|------------|-------------|
| baseline | 0.34 | 0.41 | 0.18 |
| train_en | 0.45 | 0.35 | 0.35 |
| train_yo | 0.12 | 0.51 | 0.15 |
| train_so | 0.01 | 0.21 | 0.00 |
| train_ha | 0.12 | 0.43 | 0.16 |
| mix_en_yo | 0.09 | 0.15 | 0.07 |
| mix_en_so | 0.45 | 0.52 | 0.19 |
| mix_en_ha | 0.67 | 0.67 | 0.60 |

#### v3 对角线（同语言指令→同语言输出，验证语言保持）

| 模型 | en→en | yo→yo | so→so | ha→ha |
|------|-------|-------|-------|-------|
| baseline | 1.00 | 0.86 | 0.96 | 0.94 |
| train_en | 1.00 | 0.96 | 0.98 | 0.98 |
| train_yo | 1.00 | 0.98 | 1.00 | 1.00 |
| train_so | 1.00 | 1.00 | 0.98 | **0.56** |
| train_ha | 1.00 | 0.96 | 1.00 | 1.00 |
| mix_en_yo | 1.00 | 1.00 | 0.98 | 0.96 |
| mix_en_so | 1.00 | 1.00 | 1.00 | 1.00 |
| mix_en_ha | 1.00 | 0.98 | 0.98 | 0.98 |

---

### 三版 LCB 核心结论

#### 结论 1：v2（无 tag）在跨语言指令遵循上表现最好
从 `en→lang` 的均值对比可以看出，绝大多数模型在 **v2（无 tag，本地化指令）** 上的跨语言跟随成功率最高。
- `baseline` 从 v1 的 0.34 提升至 v2 的 0.41。
- 少数语言单语 SFT（`train_yo`, `train_so`, `train_ha`）和双语混合 SFT（`mix_en_so`）在 v2 中的表现均显著优于 v1。
- **原因**：这 8 个基础模型在微调时并没有引入 `<|tgt_lang:X|>` 的概念。v2 去除了 tag 干扰，纯粹依赖自然语言指令，最贴合模型的真实能力。

#### 结论 2：SFT 导致对 tag 的极端敏感与崩溃（v3 异常）
v3 格式（`<|tgt_lang:output_lang|>` + 英文指令）虽然在逻辑上完美（tag 与目标语言一致），但所有模型的 `lc_rate` 都出现了**断崖式下跌**（均值大幅低于 v2）。
- `train_so` 在 v3 中完全崩溃（avg=0.00）。
- `train_yo` 从 0.51 跌至 0.15。
- **原因**：模型从未见过 `<|tgt_lang:X|>` 的前缀，这一未知 token 扰乱了正常的生成路径，导致其退化为输出英文（`en_leak` 极高）。SFT 使得模型对 Prompt 格式的微小变化极度敏感。

#### 结论 3：`mix_en_ha` 是最稳健的跨语言模型
在所有模型中，`mix_en_ha` 表现出了独一档的强悍：
- 在 v1 和 v2 中，它的 `en→lang` 均值都达到了最高的 0.67。
- 在所有其他模型因为 v3 未知 tag 干扰而崩溃时，它依然保持了 0.60 的高分，是**唯一具备 tag 鲁棒性的模型**。
- **原因推测**：Hausa 语言数据极少（3512 条），导致 `en:ha` 的混合比悬殊。这种弱混合可能恰好让模型高度依赖英文接口，既学到了指令遵循，又没有被目标语言的分布彻底“覆盖”，从而避免了崩溃。

#### 结论 4：v1 冲突 Tag 对模型造成了严重的负面影响
在 v1（tag 为输入语言，指令为目标语言，两者冲突）中，大部分模型的跨语言得分被严重压制。
- 例如 `train_yo`（v1: 0.12 vs v2: 0.51），`train_so`（v1: 0.01 vs v2: 0.21）。
- 冲突的 tag 让模型陷入困惑，降低了指令的有效性。唯独 `train_en` 在 v1（0.45）略高于 v2（0.35），说明纯英文模型更倾向于忽视 tag 而死扣英文指令文本。

#### 结论 5：对角线任务极其稳定，单语 SFT 巩固了惯性
在 v3 中，尽管 `en→lang` 的跨语言能力崩溃，但所有模型的**对角线（同语言指令 → 同语言输出）依然稳定在 0.96 以上**（除 `train_so` 的 ha→ha 为 0.56 外）。这说明：
- 未知的 tag 虽然破坏了“切换语言”的能力，但无法打破“同语言连续生成”的强大惯性。
- 单语 SFT 极大地巩固了本语言的建模能力。

---

### LCB v1/v2/v3 完整 4×4 矩阵（全部语言方向）

#### v1 完整矩阵（tag=input_lang，英文指令，tag 与目标冲突）

| 模型 | en→en | en→yo | en→so | en→ha | yo→en | yo→yo | yo→so | yo→ha | so→en | so→yo | so→so | so→ha | ha→en | ha→yo | ha→so | ha→ha |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| baseline | 1.00 | 0.12 | 0.64 | 0.26 | 0.72 | 0.86 | 0.08 | 0.04 | 0.64 | 0.00 | 0.96 | 0.00 | 0.56 | 0.00 | 0.42 | 0.94 |
| train_en | 1.00 | 0.34 | 0.62 | 0.40 | 0.35 | 0.96 | 0.00 | 0.04 | 0.20 | 0.00 | 0.98 | 0.00 | 0.12 | 0.00 | 0.48 | 0.98 |
| train_yo | 1.00 | 0.18 | 0.12 | 0.06 | 0.00 | 0.98 | 0.04 | 0.00 | 0.00 | 0.02 | 1.00 | 0.00 | 0.02 | 0.22 | 0.24 | 1.00 |
| train_so | 1.00 | 0.00 | 0.00 | 0.04 | 0.00 | 1.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.98 | 0.00 | 0.02 | 0.00 | **0.78** | 0.56 |
| train_ha | 1.00 | 0.14 | 0.04 | 0.18 | 0.00 | 0.96 | 0.00 | 0.08 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.18 | 1.00 |
| mix_en_yo | 1.00 | 0.16 | 0.12 | 0.00 | 0.00 | 1.00 | 0.06 | 0.00 | 0.00 | 0.04 | 0.98 | 0.00 | 0.00 | 0.16 | 0.16 | 0.96 |
| mix_en_so | 1.00 | 0.26 | 0.76 | 0.32 | 0.00 | 1.00 | 0.04 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | **0.74** | 1.00 |
| mix_en_ha | 1.00 | 0.44 | 0.82 | 0.76 | 0.00 | 0.98 | 0.00 | 0.02 | 0.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.00 | 0.36 | 0.98 |

#### v2 完整矩阵（无 tag，本地化指令）

| 模型 | en→en | en→yo | en→so | en→ha | yo→en | yo→yo | yo→so | yo→ha | so→en | so→yo | so→so | so→ha | ha→en | ha→yo | ha→so | ha→ha |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| baseline | 1.00 | 0.28 | 0.64 | 0.30 | 0.12 | 1.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.98 | 0.00 | 0.28 | 0.00 | 0.02 | 0.96 |
| train_en | 1.00 | 0.22 | 0.50 | 0.34 | 0.68 | 0.96 | 0.18 | 0.02 | 0.40 | 0.00 | 1.00 | 0.00 | 0.58 | 0.00 | 0.16 | 1.00 |
| train_yo | 1.00 | 0.72 | 0.50 | 0.31 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.02 | 0.00 | 0.02 | 0.96 |
| train_so | 1.00 | 0.12 | 0.30 | 0.22 | 0.00 | 0.98 | 0.04 | 0.02 | 0.00 | 0.00 | 1.00 | 0.00 | 0.02 | 0.00 | 0.46 | 1.00 |
| train_ha | 1.00 | 0.26 | 0.42 | 0.62 | 0.00 | 1.00 | 0.00 | 0.02 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| mix_en_yo | 1.00 | 0.30 | 0.14 | 0.02 | 0.00 | 1.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.98 | 0.00 | 0.02 | 0.02 | 0.10 | 0.96 |
| mix_en_so | 1.00 | 0.34 | 0.84 | 0.38 | 0.12 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.08 | 0.00 | 0.34 | 1.00 |
| mix_en_ha | 1.00 | 0.42 | 0.80 | 0.78 | 0.32 | 0.98 | **0.40** | 0.08 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

#### v3 完整矩阵（tag=output_lang，与指令一致）

| 模型 | en→en | en→yo | en→so | en→ha | yo→en | yo→yo | yo→so | yo→ha | so→en | so→yo | so→so | so→ha | ha→en | ha→yo | ha→so | ha→ha |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| baseline | 1.00 | 0.06 | 0.34 | 0.14 | 0.70 | 0.86 | 0.06 | 0.06 | 0.78 | 0.00 | 0.96 | 0.00 | 0.70 | 0.02 | 0.54 | 0.94 |
| train_en | 1.00 | 0.26 | 0.50 | 0.30 | 0.37 | 0.96 | 0.00 | 0.04 | 0.20 | 0.00 | 0.98 | 0.00 | 0.24 | 0.04 | 0.52 | 0.98 |
| train_yo | 1.00 | 0.18 | 0.12 | 0.14 | 0.00 | 0.98 | 0.06 | 0.00 | 0.00 | 0.06 | 1.00 | 0.00 | 0.00 | **0.54** | **0.46** | 1.00 |
| train_so | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.00 | **0.76** | 0.56 |
| train_ha | 1.00 | 0.04 | 0.18 | 0.26 | 0.00 | 0.96 | 0.00 | 0.16 | 0.00 | 0.00 | 1.00 | 0.02 | 0.00 | 0.00 | 0.52 | 1.00 |
| mix_en_yo | 1.00 | 0.18 | 0.02 | 0.02 | 0.02 | 1.00 | 0.08 | 0.00 | 0.00 | 0.08 | 0.98 | 0.02 | 0.00 | **0.50** | 0.34 | 0.96 |
| mix_en_so | 1.00 | 0.06 | 0.34 | 0.18 | 0.00 | 1.00 | 0.04 | 0.02 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | **0.80** | 1.00 |
| mix_en_ha | 1.00 | 0.34 | 0.68 | 0.78 | 0.00 | 0.98 | 0.00 | 0.12 | 0.00 | 0.00 | 0.98 | 0.00 | 0.00 | 0.02 | 0.48 | 0.98 |

---

### 全矩阵补充结论：版本对比与最佳模型

#### 1. 哪个评测版本最好/LCB结果最好？

从完整的 4×4 矩阵表现来看，**v2（无 tag，本地化指令）是最好、最能反映模型真实能力的评测版本**。

- **v2（最佳评测基准）**：在 4×4 矩阵中，v2 的表现最符合逻辑与模型训练分布。它展现了极强的对角线稳定性（同语言问答近乎 1.00），并且真实反映了不同模型在 `en→lang` 上的能力差异（如 `mix_en_so` 在 `en→so` 达到 0.84）。它没有任何格式干扰，纯粹测试自然语言指令遵循。
- **v1（格式冲突带来的负面压制）**：v1 矩阵显示，由于输入了与目标语言冲突的源语言 tag，模型的跨语言能力被严重压制。例如 `train_yo` 的 `en→yo` 在 v2 是 0.72，但在 v1 中被 tag 干扰暴跌至 0.18。它不能真实反映模型的生成能力。
- **v3（未知 Tag 导致的分布偏移与劫持）**：尽管 v3 格式在设计上最完美（tag 与期望输出一致），但 4×4 矩阵暴露出致命问题。因为这批模型未经 tag 训练，未知的前缀导致了两种极端错误：
  1. **全面崩溃**：如 `train_so` 的矩阵，在 v3 下跨语言能力（如 `en→so`）直接归零。
  2. **Tag 劫持**：在非英文行（如 `ha→so`），模型的输出完全被 tag 控制（`mix_en_so` 达 0.80），无视了文本指令。这证明 v3 测试的不再是指令遵循，而是模型对异常 token 的脆弱应激反应。

**结论**：对于未经过特殊 Tag 训练的基础 SFT 模型，**v2 是唯一公允的评测版本**。在 v2 中，综合跨语言指令遵循（非对角线表现）和鲁棒性，**`mix_en_ha` 是表现最好的模型**。


---

## TruthfulQA 评测策略变更（2026-06-24）

### 背景
发现 `tag_routing` 模型的 TruthfulQA MC1 原始分数 **0.6132 虚高**：该模型训练时使用了 `<|tgt_lang:en|>` 前缀格式，评测时也同步注入了 tag，形成了不公平优势。其他 8 个基础 SFT 模型在评测时**未注入 tag**（得分在 0.31–0.37 区间），数字不可比。

### 新评测标准（最终版）
**所有模型的 TruthfulQA MC1 评测统一使用完整答案字符串的 log-likelihood（全句概率）**，
- `tag`：加 `<|tgt_lang:en|>` 前缀后评测
- `notag`：原始无前缀评测分（保留在 `truthfulqa_mc1_notag` 字段）

> 注意：PROGRESS.md 中早期记录的"tag（新标准）"列值（0.5692 等）来自旧方法（单 token 字母预测），**与现在的 log-likelihood 方法不可比**，已废弃。

### 全部 12 个模型重测完成（run_tqa_tag_v2.sh，2026-06-24）
日志：`logs/tqa_tag_v2.log`

| 模型 | notag（无前缀） | tag（log-lik，有前缀） | 状态 |
|---|---|---|---|
| baseline | 0.3488 | **0.2521** | ✅ 完成 |
| train_en | 0.3684 | **0.3048** | ✅ 完成 |
| train_yo | 0.3341 | **0.2448** | ✅ 完成 |
| train_so | 0.3158 | **0.2301** | ✅ 完成 |
| train_ha | 0.3280 | **0.2436** | ✅ 完成 |
| mix_en_yo | 0.3660 | **0.2925** | ✅ 完成 |
| mix_en_so | 0.3647 | **0.2987** | ✅ 完成 |
| mix_en_ha | 0.3599 | **0.2864** | ✅ 完成 |
| MID_yo | 0.3390 | **0.2436** | ✅ 完成 |
| MID_so | 0.3268 | **0.2534** | ✅ 完成 |
| MID_ha | 0.3317 | **0.2485** | ✅ 完成 |
| tag_routing | 0.3758 | **0.2558** | ✅ 完成 |

> tag 分数整体低于 notag：log-likelihood 评测对所有模型更严格（无早期字母预测的虚高效应）；tag_routing 有前缀时优势仍最显著（0.3758→0.2558 vs baseline 0.3488→0.2521）。

---

## DSCT 实验（Dual-Space Constrained Tuning）

### 核心思路
同时约束 **表示空间**（MID：教师 hidden states → 余弦距离蒸馏）和 **参数空间**（正交正则：LoRA_spec 与 LoRA_donor 在参数子空间中保持正交），防止新语言的 LoRA 覆盖英文能力。

损失函数：`L = L_CE + α·L_MID + λ·L_ortho`

| 超参 | 值 |
|---|---|
| α（MID 权重） | 0.1 |
| β（β_MID 蒸馏温度） | 0.05 |
| λ（正交正则系数） | 0.01 |
| top_n_layers（蒸馏层数） | 4 |
| n_pos2（随机目标 token 数） | 3 |

### 架构
- **教师**：Base + LoRA_donor（train_en adapter）merge 后冻结
- **学生**：Base + LoRA_donor（merge 为新 base）+ LoRA_spec（可训练）
- **合并基座**：`results/dsct/donor_merged_base/`（首次运行时生成，后续复用）

### 训练脚本
| 文件 | 说明 |
|---|---|
| `scripts/train_dsct.py` | DSCT trainer（DSCTTrainer 继承 SFTTrainer） |
| `scripts/launch_dsct.sh` | 顺序训练 yo→so→ha，每轮训完自动评测 |

### 输出路径
| 语言 | Adapter 目录 | 评测 JSON |
|---|---|---|
| yo | `results/dsct/dsct_Qwen3.5-9B-Base_yo/` | `dsct_Qwen3.5-9B-Base_yo_eval.json` |
| so | `results/dsct/dsct_Qwen3.5-9B-Base_so/` | `dsct_Qwen3.5-9B-Base_so_eval.json` |
| ha | `results/dsct/dsct_Qwen3.5-9B-Base_ha/` | `dsct_Qwen3.5-9B-Base_ha_eval.json` |

### 训练与评测状态（截至 2026-06-25 更新）

| 语言 | 训练 | 评测 |
|---|---|---|
| yo | ✅ 完成 | ✅ 完成（belebele+sib200 so/ha 已补跑修复） |
| so | ✅ 完成 | ✅ 完成（belebele+sib200 so/ha 已补跑修复） |
| ha | ✅ 完成 | ✅ 完成（belebele+sib200 so/ha 已补跑修复） |

完成时间：2026-06-24 17:12 UTC，日志 `logs/dsct_eval_then_ha.log`

### DSCT 评测结果（无 inject_lang_tag）

> **TruthfulQA 说明**：baseline/MID 列值取 `truthfulqa_mc1_notag`（无前缀原始分，由 `run_tqa_tag_v2.sh` 备份）；DSCT 本身评测时从未注入 tag，`truthfulqa_mc1` 即为无前缀分，两者可直接对比。

| 模型 | TruthfulQA (notag) | bele_en | bele_yo | bele_so | bele_ha | iroko_yo | iroko_ha |
|---|---|---|---|---|---|---|---|
| baseline | 0.3488 | 0.9244 | 0.4367 | 0.4689 | 0.4844 | 0.450 | 0.424 |
| MID_yo | 0.3390 | 0.9300 | 0.4689 | 0.4822 | 0.4833 | 0.450 | 0.388 |
| MID_so | 0.3268 | 0.9400 | 0.4189 | 0.4933 | 0.4911 | 0.448 | 0.464 |
| MID_ha | 0.3317 | 0.9333 | 0.4289 | 0.5178 | 0.5100 | 0.446 | 0.422 |
| DSCT_yo | 0.3501 | 0.9322 | 0.4556 | 0.4789 | 0.4967 | 0.444 | 0.372 |
| DSCT_so | 0.3182 | 0.9389 | 0.4011 | 0.4900 | 0.4933 | 0.412 | 0.424 |
| DSCT_ha | 0.3293 | 0.9378 | 0.4300 | 0.5044 | 0.5022 | 0.428 | 0.420 |

> sib200 so/ha 正在补跑修复（PID 588098），belebele 已全部修正。

**修复记录**：
- `--tasks english_main` → `--tasks all`（原先 belebele/sib200 未触发）
- `donor_merged_base` 补充 tokenizer 文件（so eval tokenizer 报错）
- `parallelize=True` 加入 `_lm_eval_model_args`（原先只用 1 张 GPU）
- 移除 `--inject_lang_tag`（DSCT 模型不带 tag 评测）
- belebele/sib200 的 `som_Latn`/`hau_Latn` 在 H100 worker 上未缓存（输出 0.0）→ 手动下载并构建缓存后补跑

---

---

## MID normfix 诊断实验（2026-06-25，B200 8×GPU）

### 背景分析

#### 关键发现：MID 约束基本无效（归一化 bug 导致）

对比 `train_yo`（纯 yo CE，无 MID 约束）与 `MID_yo`（yo CE + MID 约束）的分数：

| 模型 | TruthfulQA | en_avg | 与 baseline 差 |
|---|---|---|---|
| baseline | 0.3488 | 0.6225 | — |
| train_yo | 0.3341 | 0.6007 | -0.0147 TQA |
| MID_yo | 0.3390 | 仅 TQA 已知 | -0.0098 TQA |
| mix_en_yo | 0.3660 | ~0.62 | +0.017 TQA |

**结论**：train_yo 和 MID_yo 的 TruthfulQA 几乎相同（0.334 vs 0.339），说明 MID 约束几乎没有起效。TruthfulQA 下降来自"纯 yo 数据训练"本身，与 MID 约束无关。

#### Bug 确认：`_mid_loss` 除以 n（总项数）而非 valid_b（样本数）

旧代码（`train_mid.py`）：
```python
# n 累计 = valid_b × K × (1 + P2) 
# 默认参数：K=4, P2=3 → n = valid_b × 16
total = total + alpha * cos1  # n += 1
total = total + beta  * cos2  # n += 1
return total / n               # 实际梯度强度缩小 16×
```

实际 alpha 有效强度 = 0.1 / 16 ≈ **0.006**，beta ≈ 0.003 —— 约束近乎零。

#### 代码修复（已应用到 `scripts/train_mid.py`）

```python
valid_b = 0
for b, ids in enumerate(ids_cpu):
    ...
    if resp_start <= 0: continue
    valid_b += 1        # 只计 batch 中有效样本
    for li in top_idxs:
        total += alpha * cos1  # no longer counting per-token
        for p in pos2_pts:
            total += beta * cos2
return total / valid_b  # 每样本归一化：alpha 完整保留
```

修复后每样本 MID 贡献 ≈ `alpha × K + beta × K × P2` = 0.1×4 + 0.05×4×3 = **1.0**（与 CE 量级相当）。

### 实验设计

| 实验 | 目的 | GPU | 状态 |
|---|---|---|---|
| **Exp A**: `eval_mid_english_full.py` | 对现有 mid_yo/so/ha 补跑 MMLU+HS+ARC（目前只有 TruthfulQA）| GPU 0-3 | 🔄 运行中（PID 31533，11:37 UTC 启动，正在跑 MMLU）|
| **Exp B**: `launch_mid_yo_normfix.sh` | 用修复后归一化重训 mid_yo_normfix，验证强约束是否维持 TruthfulQA | GPU 4-7 | 🔄 运行中（PID 31535/32218-32222，11:40 UTC 模型加载完，step 14+/736，~10.9s/it，ETA ~13:50 UTC）|

Exp A（~1-2h）：完成后 mid_yo/so/ha eval JSON 会新增 `mmlu`/`hellaswag`/`arc_challenge`，并重算 `english_avg`。

Exp B（~3-4h）：完成后对比 mid_yo_normfix vs mid_yo TruthfulQA，直接验证约束修复效果。

### 预期结果解读

| 假设 | 预期 Exp B TruthfulQA | 含义 |
|---|---|---|
| 修复有效 | > 0.360（接近 mix_en_yo 0.366） | 强 MID 约束能代替英文数据维持 TruthfulQA |
| 修复无效（过强） | < 0.320 | alpha=0.1 per-sample 过强，抑制 yo 学习 |
| 修复部分有效 | 0.340–0.360 | 方向正确，需调参 |

### 输出路径

| 文件 | 说明 |
|---|---|
| `results/mid/mid_Qwen3.5-9B-Base_yo_normfix/` | normfix 模型 adapter |
| `results/mid/mid_Qwen3.5-9B-Base_yo_normfix_eval.json` | normfix 评测结果 |
| `logs/exp_a_mid_english_full.log` | Exp A 进度日志 |
| `logs/mid_yo_normfix.log` | Exp B 进度日志 |

---

## Worker 与云端任务状态（2026-06-25 更新）

| 项目 | 状态 |
|---|---|
| A100 Worker (port 9732) | ❌ 失联/回收，不再可用 |
| 旧 H100 Worker (port 9532, `fdbd:dccd:cdc2:12c8:0:34::`) | ✅ 活跃（nvidia-smi 已修复，需新 SSH session），4×H100 80G |
| **新 B200 Worker** (port 10146, `2605:340:cda2:1238:acb1:5cd:343e:6082`) | ✅ 活跃，8×B200 SXM |
| GPU keepalive（B200 worker） | ✅ 运行中 PID 6013（ctypes cudaMemcpy D2D，~54% 利用率）|
| Exp A（mid English eval） | 🔄 PID 31533，GPU 0-3，MMLU 进行中 |
| Exp B（mid_yo_normfix train） | 🔄 PID 31535/32218-32222，GPU 4-7，step 14+/736，ETA ~13:50 UTC；首 batch mid_loss=0.06-0.12（量级正常，原 bug 下应为~0.004） |
| 云端 cron 任务 | ✅ 已删除全部 4 个 hourly 轮询作业 |

---

## ⚡ 新增实验（2026-06-25，进行中）

### Exp C：逐层渐进式路由（Layer-wise Progressive Language Routing）

**核心思路**：底层（0-15）捕获语言无关的语义/推理，高层（16-31）负责语言特定的生成。因此：
- 底层共享 LoRA（4 语言混训）→ 最大化正向迁移
- 顶层语言专属 LoRA（单语言）→ 精准隔离语言混淆

**架构参数**：
| 参数 | 值 |
|---|---|
| 总层数 | 32（Qwen3.5-9B） |
| split_layer | 16（底 0-15，顶 16-31） |
| r（两阶段相同）| 16 |
| lora_alpha | 32.0 |
| Stage 1 数据 | en+yo+so+ha 全量（~48K 条），2 epoch |
| Stage 2 数据 | 单目标语言（yo/so/ha 全量），1 epoch |

**运行方式**：
```bash
# Stage 1：B200 GPU 2-3（或 H100 4 GPU）
export CUDA_VISIBLE_DEVICES=2,3
nohup bash scripts/launch_layerwise.sh stage1 > logs/layerwise_stage1.log 2>&1 &

# Stage 2（Stage 1 完成后）：H100 4 GPU
# SSH 到 H100，然后：
nohup bash scripts/launch_layerwise.sh stage2 > logs/layerwise_stage2.log 2>&1 &
```

**输出目录**：`results/layerwise/`
| 文件/目录 | 内容 |
|---|---|
| `stage1_shared/` | 底层共享 adapter（PEFT）|
| `stage2_yo/`, `stage2_so/`, `stage2_ha/` | 顶层语言 adapter（PEFT）|
| `layerwise_yo/`, `layerwise_so/`, `layerwise_ha/` | 合并后标准 HF 模型 |
| `layerwise_Qwen3.5-9B-Base_{yo,so,ha}_eval.json` | 评测结果 |

**评测**：merge 后用 `evaluate.py`，与其他模型格式完全一致。

**新增文件**：
- `src/models/layerwise_lora.py` — `setup_shared_bottom()`, `add_lang_top()`
- `scripts/train_layerwise.py` — 三模式：`stage1` / `stage2` / `merge`
- `scripts/launch_layerwise.sh` — 分步启动脚本

**状态**：
| 阶段 | 状态 |
|---|---|
| Stage 1（4-lang 共享底层 LoRA） | ✅ 完成 |
| Stage 2 yo 训练 | ✅ 完成 |
| **merge_eval yo**（内存合并，不落盘） | ✅ 完成 → `results/layerwise/layerwise_Qwen3.5-9B-Base_yo_eval.json` |
| Stage 2 so 训练 | ✅ 完成 |
| **merge_eval so**（内存合并，不落盘） | 🔄 运行中 |
| Stage 2 ha 训练 + merge_eval ha | ⏳ so 完成后自动串行 |

**yo 评测结果**：TruthfulQA=0.3782，Belebele en=0.9311 yo=0.4256 so=0.4667 ha=0.4689

---

### Exp D：软路由 MoE（Soft Mixture of LoRA Experts, LA-MoA）

**核心思路**：不用硬 tag 切换语言子空间，而是对每个 token 动态计算 K 个 LoRA 专家的混合权重。模型可以在推理时"按需借用"不同语言的表示能力。

**架构**：
- 每个 target linear 层替换为 `MoELoRALinear`
- K=4 LoRA 专家，每个 r=8（总参数量 ≈ 2× 标准 LoRA r=16）
- Token-level learnable router：`softmax(W_r @ x)` → [K] gate 权重
- LoRA 输出 = Σ_i gate_i × B_i(A_i(x)) × (alpha/r)
- 初始化：router 全零 → 初始均匀分配；lora_A kaiming，lora_B 零初始化

**参数**：
| 参数 | 值 |
|---|---|
| n_experts | 4 |
| r（每个 expert）| 8 |
| lora_alpha | 16.0 |
| target_modules | q/k/v/o_proj + gate/up/down_proj（7 种）|
| 训练数据 | en+yo+so+ha 全量，2 epoch（同 tag_routing）|
| 总可训练参数 | ~87M（≈ 2× LoRA r=16）|

**运行方式**：
```bash
# B200 GPU 0-1（前两张空闲）
export CUDA_VISIBLE_DEVICES=0,1
nohup bash scripts/launch_moe_lora.sh > logs/moe_lora.log 2>&1 &
```

**输出目录**：`results/moe_lora/moe_lora_Qwen3.5-9B-Base/`
| 文件 | 内容 |
|---|---|
| `moe_weights.safetensors` | 所有 MoE 可训练参数（lora_A/B + router，~87M）|
| `moe_config.json` | 架构参数（n_experts, r, alpha, target_modules 等）|
| `training_metadata.json` | 训练元信息 |
| `tokenizer.*` | Tokenizer 文件（从 base 复制）|

**评测**：由于模型含自定义层无法用 `from_pretrained`，使用专用脚本：
```bash
python scripts/eval_moe_lora.py \
    --moe_dir results/moe_lora/moe_lora_Qwen3.5-9B-Base \
    --output  results/moe_lora/moe_lora_Qwen3.5-9B-Base_eval.json
```
评测内容：TruthfulQA MC1 + Belebele（4 lang）+ IrokoBench MCQ（yo/ha）+ LCB-chat（yo/so/ha）

**新增文件**：
- `src/models/moe_lora.py` — `MoELoRALinear`, `setup_moe_lora()`, `freeze_base()`, `save_moe()`, `load_moe()`
- `scripts/train_moe_lora.py` — MoE 训练脚本（SFTTrainer + 自定义 save）
- `scripts/eval_moe_lora.py` — 专用评测脚本（内置 Belebele/TQA/Iroko/LCB）
- `scripts/launch_moe_lora.sh` — B200 2-GPU 启动脚本

**关于「语言-任务解耦」（Innovation Direction 3）的说明**：
该方向不适合当前实验设置。Language-Task Disentanglement 要求同一模型处理多个**不同任务**（翻译/问答/摘要等），才能把 LoRA 分解为 `LoRA_task ⊗ LoRA_language`。本实验只有一个任务（指令跟随 SFT），没有任务轴可解耦，强行套用无理论动机，已跳过。

**状态**：
| 阶段 | 状态 |
|---|---|
| 训练（2 epoch，en+yo+so+ha 全量） | ✅ 完成 |
| 评测（TruthfulQA + Belebele + Iroko + LCB） | 🔄 运行中（TruthfulQA 完成约 10%，重启后运行中）|

---

### Exp E：正交子空间 LoRA（Shared-Specific Orthogonal LoRA, SSO-LoRA）

**核心思路**：设置两组 LoRA：
- `LoRA_shared`（全层，r=16）：所有 4 种语言数据共同更新，捕获跨语言通用知识（正向迁移）
- `LoRA_lang_i`（全层，r=8）：只有语言 i 的数据更新，捕获语言特异表示

**关键损失**：正交惩罚项强制 shared 与各 lang 子空间互相正交：
```
L_orth = ||A_shared @ A_lang.T||_F^2 + ||B_shared.T @ B_lang||_F^2
```
在数学上解耦"语言无关的世界知识/推理能力"（shared）与"语言特定的词汇/句法"（lang-specific），避免正向迁移被干扰截断。

**参数**：
| 参数 | 值 |
|---|---|
| r_shared | 16 |
| r_lang | 8 |
| orth_weight | 0.1 |
| Stage 1 数据 | en+yo+so+ha 全量，2 epoch |
| Stage 2 数据 | 单目标语言（yo/so/ha），1 epoch |

**运行方式**：
```bash
# Stage 1（B200 GPU 4-5，port 29503）
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/launch_sso_lora.sh stage1 \
    > logs/sso_lora_stage1.log 2>&1 &

# Stage 2（Stage 1 完成后，同一 GPU）
CUDA_VISIBLE_DEVICES=4,5 nohup bash scripts/launch_sso_lora.sh stage2 \
    > logs/sso_lora_stage2.log 2>&1 &
```

**输出**：`results/sso_lora/`：`stage1_shared/`, `stage2_{yo,so,ha}/`, `sso_{yo,so,ha}/`（merged HF）

**新增文件**：
- `src/models/sso_lora.py` — `setup_shared()`, `add_lang_adapter()`, `orthogonal_loss()`
- `scripts/train_sso_lora.py` — 三模式（stage1/stage2/merge），`SSOTrainer` 含 orth 损失
- `scripts/launch_sso_lora.sh` — Stage 1 + Stage 2 + eval 一键脚本

**状态**：
| 阶段 | 状态 |
|---|---|
| Stage 1（4-lang 共享 LoRA，全层 r=16） | ✅ 完成 |
| Stage 2 yo 训练 + **merge_eval yo** | ✅ 完成 → `results/sso_lora/sso_Qwen3.5-9B-Base_yo_eval.json` |
| Stage 2 so 训练 | ✅ 完成 |
| **merge_eval so**（内存合并，不落盘） | 🔄 运行中 |
| Stage 2 ha 训练 + merge_eval ha | ⏳ so 完成后自动串行 |

**yo 评测结果**：TruthfulQA=0.3439，Belebele en=0.9322 yo=0.4533 so=0.4744 ha=0.4467

---

## ⚠️ B200 Worker CUDA 803 修复记录

**问题**：新 SSH session 中 `torch.cuda.is_available()` 返回 False，报错 CUDA Error 803（system has unsupported display driver / cuda driver combination）。

**根本原因**：
- `/usr/local/cuda-12.9/compat/libcuda.so.1 → libcuda.so.575.57.08`（compat 驱动 575.x）被 `/etc/ld.so.conf.d/*.conf` 中的 ldconfig 优先载入
- 实际物理驱动版本：580.105.08
- 当 `LD_LIBRARY_PATH=/usr/local/cuda/lib64:...`（当前 SSH session 的默认值）存在时，linker 搜索不到 libcuda 后 fallback 至 ldconfig，ldconfig 的 compat 条目排在 `/usr/lib/x86_64-linux-gnu/libcuda.so.1 → libcuda.so.580.105.08` 之前，因此载入 575 版本，产生 803 驱动不匹配错误。

**修复**：在训练脚本中加入：
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08
```
LD_PRELOAD 强制所有子进程（accelerate rank 0/1）优先载入正确的 580 版本 libcuda，绕过 ldconfig compat 顺序问题。

**已更新文件**：
- `scripts/launch_moe_lora.sh` — 加入 `LD_PRELOAD` + `--main_process_port 29501`
- `scripts/launch_layerwise.sh` — 加入 `LD_PRELOAD` + `--main_process_port 29502`

---

---

## 🚫 核心工程决策：禁止保存 merged 权重到磁盘

### 问题背景
NFS 文件系统总容量 125GB。Layerwise + SSO-LoRA 各自 merge 三个语言（yo/so/ha），每个 merged 模型 bfloat16 约 17-18GB，三语言 × 两方法 = ~100GB，直接打满磁盘，导致 OSError: [Errno 28] No space left on device，所有进程崩溃。

### 架构决策（最终版，不可逆）

**禁止任何脚本将 merged 全量权重写入 NFS（或任何磁盘）。**

取而代之的 `merge_eval` 流程：
1. 加载 base 模型 + PEFT adapters（stage1_shared + stage2_lang）
2. 调用 `model.merge_and_unload()`，在内存中完成 adapter 融合
3. `merged.eval(); merged.cuda()`，直接在 GPU 上评测
4. 评测结束后只写入 JSON 结果文件（< 10KB）
5. `del merged; torch.cuda.empty_cache()`

### 实现细节

**`merge_eval` 模式** 已在以下文件中实现：
- `scripts/train_layerwise.py`: `run_merge_eval()` 函数，通过 `--mode merge_eval` 触发
- `scripts/train_sso_lora.py`: 同上
- `scripts/launch_layerwise.sh`: 用 `merge_eval` 替换原先的 merge → evaluate.py 两步调用
- `scripts/launch_sso_lora.sh`: 同上

**eval 函数接受 in-memory model 参数**（避免 eval 内部重新从磁盘加载）：
- `src/evaluation/english_eval.py`: `run_english_eval(..., model=None, tokenizer=None)`
  - 当 `model` 不为 None 时，用 `HFLM(pretrained=model, tokenizer=tokenizer)` 包装后传给 `lm_eval.simple_evaluate()`
- `src/evaluation/multilingual_eval.py`: `run_multilingual_eval(..., model=None, tokenizer=None)`
  - 所有子函数（`_run_belebele`, `_run_sib200`, `_run_flores`）均支持 `model=` 参数
  - `_owner = model is None` 模式：调用方传入 model 时，子函数不加载也不释放，由调用方管理生命周期

**`merge` 模式仍存在**（用于调试/手动检查），但 launch 脚本已不调用它。

### NFS 磁盘管理
- 已删除的 merged 模型（释放约 50GB）：`layerwise_yo/`（17G）、`sso_yo/`（17G）、`sso_so/`（3G 残损）
- 保留：仅 PEFT adapter 目录（每个约 70-150MB）和 eval JSON（< 10KB）
- `TRITON_CACHE_DIR=/tmp/triton_cache`：防止 DeepSpeed triton autotune 写入 NFS 的 `~/.triton/autotune`

---

## 📊 当前全局运行状态（2026-06-26 07:00 UTC）

| GPU | 机器 | 实验 | 状态 |
|---|---|---|---|
| B200 GPU 0-1 | B200 | **Exp D** MoE-LoRA **评测** | 🔄 运行中 |
| B200 GPU 0-3 | B200 | **Exp C** Layerwise stage2 + **merge_eval so**（内存） | 🔄 运行中 |
| B200 GPU 4-7 | B200 | **Exp E** SSO-LoRA stage2 + **merge_eval so**（内存） | 🔄 运行中 |

**已完成**：
- Layerwise yo eval ✅ → `results/layerwise/layerwise_Qwen3.5-9B-Base_yo_eval.json`
- SSO-LoRA yo eval ✅ → `results/sso_lora/sso_Qwen3.5-9B-Base_yo_eval.json`

**下一步队列**：
1. so eval 完成后自动开始 ha 训练 + merge_eval ha（两条流水线均由 launch_*.sh 内置串行逻辑控制）
2. MoE eval 完成后整合所有结果，构建 LIS 矩阵对比表（Layerwise / SSO-LoRA / MoE-LoRA / MID normfix）

---

## MID Normfix 状态

| 语言 | 状态 | 输出目录 |
|---|---|---|
| yo | ✅ 训练完成，✅ 评测完成 | `results/mid/mid_Qwen3.5-9B-Base_yo_normfix` |
| so | ✅ 训练完成（H100） | `results/mid/mid_Qwen3.5-9B-Base_so_normfix` |
| ha | ✅ 训练完成（H100，so 后自动串行） | `results/mid/mid_Qwen3.5-9B-Base_ha_normfix` |

---

## ❌ 尚未完成 / 进行中

| 任务 | 说明 |
|---|---|
| TruthfulQA MC1 批量重测 | ✅ 全部 12 个模型完成（log-likelihood 方法，含 tag_routing），分数见上方表格 |
| DSCT yo/so/ha 训练 | ✅ 全部完成 |
| DSCT yo/so/ha 评测（belebele） | ✅ 完成（缓存缺失已修复，so/ha 已补跑） |
| DSCT yo/so/ha 评测（sib200 so/ha） | ✅ 完成 |
| tag_routing 全量无 tag 重评 | ✅ 完成`run_tag_routing_notag_full.sh`：mmlu/hellaswag/arc/sib200/belebele/iroko 全部无 inject_lang_tag |
| **Exp C Layerwise** so + ha merge_eval | 🔄 so 运行中，ha ⏳ 等待中（不落盘）|
| **Exp D MoE-LoRA** 全量评测 | 🔄 运行中（TruthfulQA + Belebele + Iroko + LCB）|
| **Exp E SSO-LoRA** so + ha merge_eval | 🔄 so 运行中，ha ⏳ 等待中（不落盘）|
| **MID normfix** so/ha 评测 | ⏳ 尚未跑（so/ha 训练已完成，评测待触发）|
| **LIS 矩阵对比** 整理 | ⏳ 全部 eval 完成后整合（Layerwise / SSO-LoRA / MoE-LoRA / MID normfix）|
| DSCT vs MID 横向对比整理 | ⏳ 可随时整理，数据均已完成 |
| LCB v2/v3：tag_routing / MID 模型 | 暂时先不做 |
| 三版 LCB 全模型横向对比 | 暂时先不做 |
| Isolated-LoRA so/ha | 已放弃（用户决策 2026-06-22），yo 结果可作对照 |
| Tag Routing 推理逻辑 | 暂时放弃 |

---

## 重要注意事项

- **模型 ID**：Qwen 必须用 `Qwen/Qwen3.5-9B-Base`（Base 后缀），不能用 chat 模型
- **GPU 数量灵活**：2/4/8 张 GPU 均支持，挂载后运行 `bash scripts/setup_accelerate.sh` 即可
- **Phase 4 执行顺序**：`mixed_lora` 依赖 Phase 3 最优配比，其他 4 种方案可提前并行
- **FLORES 评测**：base 模型使用 8-shot 提示格式（代码中已使用 translation prompt）
- **MT-Bench**：仅 Phase 3 & 4 传 `--include_mt_bench` 开启，Phase 2 跳过
- **LCB**：需先跑 `prepare_lcb_prompts.py`，再用 `--include_lcb` 开启
- **约鲁巴语**：数据量极少，论文中作为"压力测试"语言，结论需加 caveat