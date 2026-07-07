# B200 Worker CUDA Error 803 修复记录

**问题日期**：2026-06-25  
**机器**：B200×8 worker（Arnold job 971652，IP `2605:340:cda2:1238:acb1:5cd:343e:6082`，port 10146）

---

## 症状

新建 SSH session 后，在 B200 worker 上运行任何 PyTorch 代码都报：

```
/home/tiger/.local/lib/python3.11/site-packages/torch/cuda/__init__.py:174: UserWarning:
CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda
functions before calling NumCudaDevices() that might have already set an error?
Error 803: system has unsupported display driver / cuda driver combination
  return torch._C._cuda_getDeviceCount() > 0
```

`torch.cuda.is_available()` 返回 `False`，但 `torch.cuda.device_count()` 仍返回 8（设备存在，但无法初始化）。

用 `accelerate launch` + DeepSpeed 启动训练时，SFTTrainer 检测到"无 GPU"后抛出：

```
ValueError: Your setup doesn't support bf16/gpu. You need to assign use_cpu
if you want to train the model on CPU.
```

**奇怪之处**：同一台机器上已运行的训练进程（Exp A/B，更早启动）完全正常，GPU 利用率 100%，无任何报错。

---

## 根本原因

### 环境变量

当前 SSH session 默认带有：

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/install/lib:
```

其中 `/usr/local/cuda/lib64`（= `/usr/local/cuda-12.9/lib64`）是 CUDA runtime 目录，**不含 `libcuda.so`**（驱动库）。

### ldconfig 配置问题

`/etc/ld.so.conf.d/` 中包含：

```
/usr/local/cuda-12.9/compat
```

该 compat 目录内有：

```
/usr/local/cuda-12.9/compat/libcuda.so.1 → libcuda.so.575.57.08   # 575.x compat 版本
```

而系统正确的驱动库位于：

```
/usr/lib/x86_64-linux-gnu/libcuda.so.1 → libcuda.so.580.105.08    # 实际驱动 580.x
```

### 冲突链

动态链接器查找 `libcuda.so.1` 时的顺序：

1. 先查 `LD_LIBRARY_PATH` 中各目录 → `/usr/local/cuda/lib64` 中**无** `libcuda.so.1`
2. Fallback 到 `ldconfig` 缓存 → `compat` 目录的配置条目**排在前面**
3. 载入 **575.x** 版本的 `libcuda.so.1`
4. 实际物理驱动版本为 **580.105.08**，两者不匹配 → CUDA Error 803

### 为何早期进程不受影响

推测早期进程（Exp A/B，PID 31533/31535）启动时环境不同（不同 tmux/SSH session，无 `LD_LIBRARY_PATH` 或其初始值不含 cuda 路径），或在 ldconfig compat 条目加入前启动。其 `/proc/PID/environ` 中确认无 `LD_LIBRARY_PATH`，因此 ldconfig 的搜索行为与新 session 不同。

---

## 验证

```bash
# 无修复：CUDA 不可用
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 输出：False 8

# 有修复：CUDA 正常
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08 \
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 输出：True 8
```

---

## 修复方法

在所有训练 launch 脚本的环境变量设置区块加入：

```bash
# Fix CUDA 803: compat dir (575.x) in ldconfig overrides the real driver (580.x)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08
```

`LD_PRELOAD` 在动态链接器搜索任何目录之前强制载入指定库，覆盖 ldconfig 的 compat 条目，确保所有子进程（accelerate rank 0/1、DeepSpeed worker）使用正确的 580.x 驱动。

### 同步修复：端口冲突

Exp B（mid_yo_normfix）已占用 port 29500（accelerate rendezvous 默认端口），新实验需指定不同端口：

```bash
accelerate launch --config_file "$ACCEL_CFG" --main_process_port 29501 ...
```

### 已修改文件

| 文件 | 修改内容 |
|---|---|
| `scripts/launch_moe_lora.sh` | 加 `LD_PRELOAD` + `--main_process_port 29501` |
| `scripts/launch_layerwise.sh` | 加 `LD_PRELOAD` + `--main_process_port 29502` |

---

## 不适用的方案

| 方案 | 原因 |
|---|---|
| `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` | 该目录同时含 stub 版 `libcuda.so`，可能引入其他库冲突（参见 A100 教训） |
| 修改 ldconfig conf 文件（删除 compat 条目） | 需要 root 权限；影响全局，可能破坏其他进程 |
| 将 `/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08` 复制到 `/tmp/` 再加入 LD_LIBRARY_PATH | 可行，但不如 `LD_PRELOAD` 精准 |
| 单 GPU 训练（无 accelerate）| 绕过问题而非修复；速度损失 2× |

---

## 相关文档

- `docs/fix_a100_nccl_libnvidia_ml.md` — A100 worker 上类似的 libnvidia-ml stub 问题
