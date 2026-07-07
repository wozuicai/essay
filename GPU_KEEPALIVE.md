# GPU 利用率 Keepalive 使用说明

这个项目里已经添加了脚本：

```bash
/root/project/scripts/ensure_worker_gpu_util.sh
```

它会在 worker 上启动一个常驻 Python 进程，对指定 GPU 反复做矩阵乘法，并根据 `nvidia-smi` 看到的 GPU 利用率自适应调节负载。默认目标是让 GPU 利用率保持在 **40% 以上**，控制目标约 **45%**。

## 当前已启动的 worker

本次已经在下面这个 worker 上启动：

```text
worker-0
IP:   fdbd:dccd:cdc2:12c8:0:14::
Port: 10677
GPU:  4 x H100-SXM-80G
PID:  9086
Log:  /tmp/gpu_keepalive_tiger/keepalive.log
```

启动命令是：

```bash
cd /root/project
bash scripts/ensure_worker_gpu_util.sh start \
  --host fdbd:dccd:cdc2:12c8:0:14:: \
  --port 10677 \
  --gpus 0,1,2,3 \
  --min-util 40 \
  --target 45 \
  --high-util 70 \
  --max-sleep 10
```

检查时看到 4 张 H100 都在跑，最近一次 `nvidia-smi` 利用率为：

```text
GPU0: 94%
GPU1: 52%
GPU2: 57%
GPU3: 54%
```

注意：GPU 利用率是瞬时采样，可能会有轻微波动；脚本会持续自适应调节。


## 自适应退让逻辑

脚本看的指标是 `nvidia-smi` 的 **总 GPU utilization**，它不能区分 utilization 是训练造成的还是 keepalive 自己造成的。因此脚本采用保守策略：

- 如果某张 GPU 低于 `--min-util 40`，keepalive 会增加矩阵乘法频率。
- 如果某张 GPU 高于 `--target 45` 较多，keepalive 会逐步减少频率。
- 如果某张 GPU 达到 `--high-util 70` 或更高，脚本认为正式训练/评测大概率已经在占用 GPU，会强制退让到非常稀疏的 heartbeat 负载，sleep 最多到 `--max-sleep 10` 秒。

所以你开训练后，如果训练本身把 GPU 打到 70% 以上，keepalive 会自动降到很低；如果训练本身只有 45%-60%，keepalive 仍可能补一点负载，让总利用率别掉到 40% 以下。

## 查看状态

```bash
cd /root/project
bash scripts/ensure_worker_gpu_util.sh status \
  --host fdbd:dccd:cdc2:12c8:0:14:: \
  --port 10677
```

这个命令会显示：

1. 当前 `nvidia-smi` 的 GPU 利用率和显存占用。
2. keepalive 进程是否还活着。
3. `/tmp/gpu_keepalive_tiger/keepalive.log` 的最近日志。

## 停止 keepalive

```bash
cd /root/project
bash scripts/ensure_worker_gpu_util.sh stop \
  --host fdbd:dccd:cdc2:12c8:0:14:: \
  --port 10677
```

## 重启 keepalive

```bash
cd /root/project
bash scripts/ensure_worker_gpu_util.sh restart \
  --host fdbd:dccd:cdc2:12c8:0:14:: \
  --port 10677 \
  --gpus 0,1,2,3 \
  --min-util 40 \
  --target 45
```

## 换新 worker 怎么跑

如果平台分配了新 worker，先拿到新 worker 的 IPv6，例如：

```text
fdbd:dccd:cdc2:12c8:0:xxx::
```

然后从 known_hosts 找端口：

```bash
rg 'fdbd:dccd:cdc2:12c8:0:xxx' /home/tiger/.ssh/known_hosts
```

你会看到类似：

```text
[fdbd:dccd:cdc2:12c8:0:xxx::]:12345 ssh-rsa ...
```

这里的 `12345` 就是 SSH 端口。然后启动：

```bash
cd /root/project
bash scripts/ensure_worker_gpu_util.sh start \
  --host fdbd:dccd:cdc2:12c8:0:xxx:: \
  --port 12345 \
  --gpus 0,1,2,3 \
  --min-util 40 \
  --target 45
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--gpus` | `0,1,2,3` | 要施加负载的 GPU ID |
| `--min-util` | `40` | 希望至少达到的 GPU 利用率 |
| `--target` | `45` | 空闲/低负载时的控制目标，通常比 min-util 高一点更稳 |
| `--high-util` | `70` | 看到 GPU 总利用率达到这个值时，认为训练/评测可能已在跑，会大幅退让 |
| `--max-sleep` | `10` | 退让时两次 keepalive 矩阵乘法之间最多 sleep 秒数 |
| `--matrix-size` | `8192` | 矩阵乘法大小；越大负载越重、显存占用越高 |
| `--interval` | `3` | 每隔多少秒检查一次利用率 |
| `--remote-dir` | `/tmp/gpu_keepalive_<user>` | worker 上保存脚本、PID 和 log 的目录 |

如果某些 GPU 偶尔低于 40%，可以把 target 提高一点，例如：

```bash
bash scripts/ensure_worker_gpu_util.sh restart \
  --host fdbd:dccd:cdc2:12c8:0:14:: \
  --port 10677 \
  --gpus 0,1,2,3 \
  --min-util 40 \
  --target 55
```

## 已经 SSH 到 worker 里面时怎么跑

如果你已经登录到 worker 内部，也可以直接本地运行：

```bash
bash /root/project/scripts/ensure_worker_gpu_util.sh local \
  --gpus 0,1,2,3 \
  --min-util 40 \
  --target 45
```

不过这种方式会占用当前终端；推荐在 master/workspace 上使用 `start`，脚本会自动用 `nohup` 在 worker 后台常驻运行。

## 重要提醒

- 这个脚本会真实占用 GPU 算力。如果要跑正式训练/评测，最好先 `stop`，避免干扰性能。
- 它不会大量占显存，默认矩阵大小下每张 GPU 只占几 GB 左右；但如果 GPU 上已有其他进程，显存占用会叠加。
- 如果 worker 被释放，进程会随 worker 一起消失；新 worker 需要重新 `start`。
