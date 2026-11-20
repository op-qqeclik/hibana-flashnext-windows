# 01 · 移植全流程：把 Linux/aarch64 引擎搬到 Windows/x86 独立显卡

> **English abstract** — How to take an engine that only declares `POSIX / Linux / aarch64 / GB10 unified
> memory` and make it run natively on Windows 10 x64 with a discrete Blackwell GPU: toolchain pinning,
> redirecting flashinfer's JIT through `tvm_ffi`'s cross-platform builder, the `_plat.py` POSIX shim, the
> unified→discrete VRAM model change, and the acceptance gates. Every step below was executed on the
> machine in `docs/04-benchmarks.md`; the exact patch code lives in `port/winpatch.py`.

本文是**施工图**，不是概述。照着做，能在同一张卡上得到同一个结果。
`port/winpatch.py` 是全部引擎改动的可执行版本（幂等、带标签、失败即停），`port/setup_windows.py` 是工具链安装器。

---

## 0. 目标与源机

| 项 | 源机（DGX Spark "102"） | 目标机（"107"） |
|---|---|---|
| CPU/OS | aarch64 / DGX OS (Linux) | x86_64 / **Windows 10.0.19045** |
| GPU | GB10，**统一内存** 128 GB | RTX PRO 6000 Blackwell，**独立显存 96 GB**，SM **12.0**（`compute_120a`） |
| 内存 | 与 GPU 同池 | 191 GB DDR5（PLE 表 47.7 GB 靠它 page-cache） |
| 显存带宽 | 273 GB/s | 1792 GB/s（实测 copy 1282、read 1173、MoE 内核 1536–1608） |
| 驱动 | — | 616.92 |
| 引擎 | hibana 0.1.0：torch 2.13.0+cu130 / flashinfer 0.6.17 / triton 3.7.1 / xgrammar 0.2.8 | **同一份代码**，Windows 原生（不走 WSL、不装 Docker） |
| 权重 | `Qwen3.8-Flash-Next-NVFP4` 135 GB + W3 3-bit 48 GB | 同一份，局域网直传（275 MB/s，4 + 3 分钟） |

**一句话目标**：`hibana serve flashnext` 在 107 上原样跑起来，OpenAI 兼容接口 `:30004` 能出中文、代码、看图。

> 为什么不能用 WSL：本移植的目的就是**吃 Windows 主机的 191 GB RAM 给 PLE 表做页缓存**，并把
> WDDM 的握手成本量化出来。WSL 会把这两件事都糊掉。（另外 107 的显卡驱动在 WSL 下要另配一套，
> 且 `D:\` 的模型目录跨文件系统访问会让 47.7 GB 的 PLE 随机读更慢。）

---

## 1. 五个硬卡点（全部实测，不是推测）

| # | 卡点 | 证据 / 表现 | 解法（章节） |
|---|---|---|---|
| 1 | 引擎自报平台 | `pyproject`: `Operating System :: POSIX :: Linux`；`environments = linux & aarch64`；`hardware/gb10.py` 写死 `GPU_NAME="NVIDIA GB10"`、`COMPUTE_CAPABILITY=(12,1)`、`MIN_TOTAL_MEMORY_KIB≈120 GB` | §4 平台垫片 + 型号白名单 |
| 2 | 三处 Linux-only 依赖 | flashinfer 的 ninja 生成器（`c++` / `-fPIC` / `-MMD` / `.so`）；`nccl4py` 无 win 轮子；引擎里 `/proc/meminfo`、`os.statvfs`、`fcntl.flock`、`os.preadv`、`os.posix_fadvise`、`mmap.madvise`、`os.uname`、`os.kill` | §3 / §4 |
| 3 | 显存模型不同 | 引擎按「统一内存里还剩多少（`MemAvailable`）」规划 KV/state 池；独立卡必须按**显存**算，且 WDDM 下驱动的 `free` 会假报 0 | §5 |
| 4 | 编译器版本 | 107 默认 MSVC 2026（VS 18），CUDA 13 只支持 2019–2022 → `cudafe++` 直接 `0xC0000005` 崩 | §3.3 |
| 5 | FP4 指令的 arch 名 | `-gencode` 必须写 family 架构 `compute_120f/sm_120f`；写普通 `sm_120` 时 `ptxas` 报 `Instruction 'cvt with .e2m1x2' not supported on .target 'sm_120'` | §3.4 |

---

## 2. 开工前的 107 探路（15 分钟，省掉后面几小时的瞎猜）

```powershell
# 系统 / GPU / 盘 / CUDA / MSVC / 网络，一次问完
systeminfo | Select-String "OS Name","OS Version","Total Physical Memory"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv
Get-PSDrive -PSProvider FileSystem | Select Name,Used,Free
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" | Select Name
& "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -all -products * -property displayName,installationVersion
```

要记下来的四件事：

1. **卡上有多少显存已经被桌面吃了**：本机 `memory.used 13308 MiB` → 池预算必须留余量（见 §5）。
2. **哪块盘是 NVMe**：C/B 是 NVMe（200 MB/s 级持续），M:/K:/L: 是 USB 盘，只有 4–20 MB/s，**永远不要放模型**。
3. **CUDA 有几套**：107 有 12.8 / 13.0 / 13.1 —— 用 13.0（trition-windows 与 torch cu130 的口径）。
4. **VS 有几套**：107 有 VS2026 + VS2022 BuildTools 17.14（MSVC 14.44）—— **钉住 2022 那套**。

---

## 3. 工具链（`port/setup_windows.py` 做完全部这些）

### 3.1 版本矩阵（不要"升级一下试试"，这套是互相咬合的）

```
Python 3.12.11 (uv 托管)
torch           2.13.0+cu130        # download.pytorch.org/whl/cu130
triton-windows  3.7.1.post27        # PyPI，cp312-win_amd64，与 pyproject 的 triton==3.7.1 对齐
flashinfer      0.6.17 (--no-deps)  # flashinfer_python-0.6.17-py3-none-any.whl，纯 Python + 运行时编译
nccl4py         --no-deps           # 只有 manylinux 轮子；Windows 上根本不加载，只为满足 import
xgrammar        0.2.8
apache-tvm-ffi / cuda-tile / cudnn-frontend
ninja           1.13.1              # 包内自带那份也行，盘符坑一样
```

### 3.2 两个路径技巧

- **`C:\cuda13` 目录联接**（`New-Item -ItemType Junction`）指向真正的 CUDA 13.0 安装目录。
  理由：`Program Files` 里的空格会让 ninja 的 build 行、`-I` 路径、`vcvars` 的引用全部需要转义，
  而这条链路上有三个不同工具各自用不同方式转义 —— 一个字都不要赌。
- **`.venv` 放在代码盘**（`B:\hibana-win\.venv`），不放在 `%USERPROFILE%`：Windows 的长路径限制与中文用户名
  会在 `torch/lib` 的深层 DLL 上翻车。

### 3.3 MSVC 钉死（CUDA 13 的红线）

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set DISTUTILS_USE_SDK=1
set HIBANA_MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
```

不钉的话，`vswhere -latest` 会挑到 VS2026，然后：

```
fatal error C1189: unsupported Microsoft Visual Studio version
cudafe++ died with 0xC0000005
```

`win_compat.h` 通过 `--pre-include` / `/FI` 强制包含，补上 CUDA 头在 MSVC 下缺的东西：

```c
// port/win_compat.h —— ushort/uint/ulong、M_PI 系列、__restrict__
```

### 3.4 flashinfer 的 JIT：钥匙在 `tvm_ffi`，不在 flashinfer 自己

这是整个移植**最有价值的一个发现**，值得单独一节。

flashinfer 的 JIT 生成器是 Linux 形状的：`c++ -fPIC -shared -MMD`、产物 `.so`、`ninja` 规则里写死 `c++`。
硬改它的 flags 体系是死路（散落在 `jit/core.py`、`jit/cpp_ext.py`、多个 `prebuilt` 路径里）。

但它编译时依赖的 **`apache-tvm-ffi` 自带一个跨平台构建器**：`tvm_ffi/cpp/extension.py`
本来就是 `cl.exe` + `nvcc` + `deps=msvc` + `.dll` 的分支，且轮子里直接带 `core.cp312-win_amd64.pyd`
和 `lib/tvm_ffi.lib`。

于是改法只有三处（`setup_windows.py::FLASHINFER_JIT_PATCH`）：

1. `JitSpecNvcc.build()` → 调 `tvm_ffi.cpp.extension.build(...)`，把 `nvcc` 路径、arch、`cudart.lib`/`cuda.lib`、
   `--pre-include win_compat.h` 一起传进去；
2. `JitSpecNvcc.load()` → `tvm_ffi.load_module(dll_path)`；
3. 库路径解析 → 从 `.so` 改 `.dll`（以及 `.pyd`）。

**收益**：不用碰 flashinfer 的任何 flag 逻辑；它以后升级，只要 `JitSpecNvcc` 这三个方法还在，补丁就还成立。

### 3.5 tvm_ffi 构建器的三个 Windows 坑

| 坑 | 现象 | 补丁 |
|---|---|---|
| ninja 不认未转义盘符 | `build C:\t\out.o: cc C:\t\in.c` → `expected build command name`（官方 ninja 1.13.1 与包内 ninja 都一样；**错误信息完全不指向盘符**，极易误判成规则名写错） | 盘符转义成 `C$:`：`build C$:/t/out.o: cc C$:/t/in.c` |
| `-Xcompiler` 多 flag | `-Xcompiler -O2 -Xcompiler -MD` 在 MSVC 下要合并成 `-Xcompiler=/O2,/MD` 单参数 | 构建器里合并 |
| `vcvars` 挑错 VS | 见 §3.3 | `HIBANA_MSVC_PATH` 环境变量钉住 |

### 3.6 头文件级修补（flashinfer 的 CUDA 源码）

| 修补 | 原因 |
|---|---|
| `__attribute__((always_inline))` → `__forceinline__` | MSVC 不认 GNU attribute |
| `int64 % IntFastDiv` → 显式 `static_cast<int>` | MSVC 的整型提升规则与 gcc 不同，会选错重载 |
| `std::optional::value_or` → `has_value() ? *x : 0` | device 代码里 MSVC 拒绝 `value_or`（配合 `--expt-relaxed-constexpr`） |
| workspace 128/256 MB → **512 MB / 1 GiB** | fp8 KV + head_dim 256 的 prefill 临时区实测要 282 MiB |

### 3.7 装完的验收（30 秒）

```powershell
B:\hibana-win\.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 2.13.0+cu130 True (12, 0)
B:\hibana-win\.venv\Scripts\python.exe -c "import triton;print(triton.__version__)"
B:\hibana-win\.venv\Scripts\python.exe -c "import flashinfer, hibana.cli, hibana.kernels.ops;print('engine import OK')"
```

`import hibana.kernels.ops` 会触发 flashinfer + triton 的 kernel 入口 —— 这一步过了，编译链就通了。

---

## 4. 引擎侧：一个垫片 + 26 处定点补丁

### 4.1 `_plat.py`：把 POSIX 语义翻译成 Windows 语义

不要在每个调用点写 `if win32`，会散。新增一个 288 行的 `hibana/_plat.py`，然后**把调用点改指过去**。
`winpatch.py` 会先把垫片文件写进 `hibana/_plat.py`，再做替换。

| POSIX 原语 | Windows 实现 | 语义陷阱 |
|---|---|---|
| `/proc/meminfo` | `GlobalMemoryStatusEx` + `psapi` | 独立卡上 `MemAvailable` 无意义，改为报告显存（§5） |
| `os.statvfs` | `shutil.disk_usage` | 块大小语义不同，只用于"盘够不够"判断 |
| `fcntl.flock` | `msvcrt.locking` | 锁是**强制**的且**按字节区间**；要锁 1 字节并处理 `EACCES` 轮询 |
| `os.preadv` | `ReadFile` + `OVERLAPPED`（或 `seek+read` 加锁） | `preadv` 的 iovec 分散读没有对等物；PLE 路径改成线程池 + 每线程一个句柄 |
| `os.posix_fadvise(WILLNEED/SEQUENTIAL/DONTNEED)` | `PrefetchVirtualMemory` / **no-op** | `DONTNEED` 在 Windows 上没有可靠的逐段丢弃，降级为 no-op 是正确的 |
| `mmap.madvise(MADV_RANDOM)` | `PrefetchVirtualMemory` 提示 或 no-op | Windows 无 per-mapping random hint |
| `os.uname` | `platform.uname()` | 只用于日志 |
| `os.kill(pid, 9)` | `taskkill /F /T` → `TerminateProcess` | 必须杀进程树，否则留僵尸子进程占显存 |
| RSS 统计 | `psapi.GetProcessMemoryInfo` | — |
| `PATH` 分隔符 / `CUDA_HOME` | `;`，且回退 `CUDA_PATH`、再回退扫 `v13*` | 引擎里写死了 `:` 与 `/usr/local/cuda` |

### 4.2 26 处定点补丁（`winpatch.py` 的全部标签）

全部是**字符串锚点替换** + 幂等（已经改过就跳过并打印 `kept`，不会叠加）。按文件分组：

**`hardware/gb10.py` / `target.py` — 型号白名单**
1. 平台垫片安装
2. 接受 GPU 表 + 独立显存标志（`ACCEPTED_GPUS = {GPU_NAME: (12,1), "NVIDIA RTX PRO 6000 Blackwell...": (12,0)}`）
3. `MIN_DEVICE_MEMORY_KIB = 60_000_000`（3-bit profile 常驻 53 GiB，卡上还要给桌面留余量）
4. `host memory only on a Spark`（统一内存机才需要 120 GB 主机内存检查）
5. `accepted GPUs`（错误信息改成列出白名单）

**`cache/ledger.py` / `cache/ssd_store.py` — 缓存与锁**
6. `kill_tree` on Windows
7. `tasklist` 判活（原为 `os.kill(pid, 0)`）
8. 去掉 `fcntl` 的 import
9. `shutil.disk_usage` 替代 `statvfs`

**`pools` / `capacity` — 显存模型**
10. arch 检查只在非 Windows 生效
11. `/proc/meminfo` → 垫片
12/13. 见 §5

**`model/flashnext/ple.py` — 47.7 GB PLE 表（改动最多的地方）**
14. `mmap.PAGESIZE` 在 Windows 上不存在 → `getattr(mmap, "PAGESIZE", 4096)`
15. `madvise_random` 走垫片
16. 保留分片路径（Windows 分支要按路径重开句柄，`os.open` 的 fd 不能反复 dup）
17. `paths` 字段
18. Windows 上走 `_gather_threads` 分支（`preadv2` 没有对等物，靠线程池并发读）
19. **线程池 + 每线程句柄缓存**（这一条是单流性能的关键，见 `docs/02`）
20. `fadvise WILLNEED` 走垫片
21. `fadvise SEQUENTIAL` 走垫片
22. `pread` 走垫片
23. `fadvise DONTNEED` 走垫片

**`launch/run.py` — 环境变量**
24/25. `/usr/bin:/bin` 的两处默认值
26. Windows 的 `CUDA_HOME` 与 `PATH`（`;` 分隔 + 自动发现 `v13*`）

> **纪律**：每处替换都带唯一锚点，替换不上直接抛异常并列出全部未命中的锚点。
> 引擎升级后锚点变了，你会立刻知道，而不是"跑起来行为怪怪的"。

### 4.3 SM 12.1 硬门

引擎里一批图融合（`flashnext_fused` / `mlp_fusion` / `mamba2` / `qsa`）写的是 `if cap == (12, 1)`。
107 是 `(12, 0)`，于是一批"66 个 kernel 融成一个 graph replay"的优化全部被跳过。
改成 `cap in {(12, 0), (12, 1)}` 之后：code 162 → 176 tok/s、zh_prose 86 → 93、接受率 0.517 → 0.563。

**这是移植里性价比最高的一行改动**，也是"性能优化"的第一步 —— 先确认引擎没在按平台名静默降级。

---

## 5. 显存模型：从"统一内存"到"独立显存 + WDDM"

### 5.1 引擎原来的假设

```
可用池 = MemAvailable - 主机保留        # 统一内存：host 与 GPU 同池
```

独立卡上这句必须换成：

```
池预算 = 总显存 - torch 已 reserved - HIBANA_VRAM_RESERVE_GIB(默认 12，本机用 2~4)
```

### 5.2 WDDM 的两个陷阱（都会让你误判）

1. **驱动的 `free` 会假报 0，且允许超额提交。**
   实测：显存只剩 20 GiB 时，一直分块到 200 GiB「reserved」才 OOM，而 `mem_get_info().free` 恒为 0。
   ⇒ **绝对不要用驱动 free 做准入判断**，要用 `torch.cuda.memory_reserved()` 做池预算。
2. **`torch.cuda.memory_reserved()` 是 O(n_segments) 的遍历，本机要几秒。**
   而 `can_admit()` 是**每步都调**的准入函数 —— 把它放进热路径的结果是：引擎线程 100% CPU 空转、
   请求永远进不来（表现为"服务起来了但永远不回话"）。
   ⇒ 改成驱动 `mem_get_info()` + **250 ms 缓存**。

> 这两个坑的症状都长得像"引擎没起来"，但其实是准入逻辑在自锁。移植时优先怀疑它。

### 5.3 容量规划（96 GiB 卡上的实际配比）

| 项 | 值 | 备注 |
|---|---|---|
| 桌面/其他进程占用 | 13.3 GiB | `nvidia-smi` 起始值，必须先扣掉 |
| trunk FP8 + 3-bit 专家 | 46.1 GiB | `C:\hibana-models\w3` |
| PLE 表 | 47.7 GB | **在主机内存/页缓存，不占显存** |
| KV 池 | `HIBANA_KV_POOL_TOKENS=524288` (fp8) | 1M 会触发 headroom 保护（计划只给 574087 tokens） |
| state slots | `HIBANA_STATE_SLOTS=36` | 48 配合 `max_batch 32` 装不下 |
| 保留 | `HIBANA_VRAM_RESERVE_GIB=2` | 4 也行；2 是压测过的下限 |
| **合计 torch reserved** | **~72.4 GiB** | 卡上总计 ~78 GiB |

`--exact` 档（NVFP4 专家 63.3 GiB）在 96 GiB 上装不下 KV 池（实测 `only 7.3 GiB available`），
只适合空卡或直接走 3-bit 档。

---

## 6. 权重落地

- 快照按 HF 缓存布局落盘（不要用 `--local-dir` 摊平，引擎认的是 `snapshots/<rev>` 结构）：
  `B:\hibana-models\hub\models--RadixArk--Qwen3.8-Flash-Next-NVFP4\snapshots\7b719225…` —— 420 文件 / 135 GB。
- W3 3-bit 副本 333 文件 / 48 GB 落 `C:\hibana-models\w3`（B 盘只剩 78 GB，必须分盘）。
- 局域网直接 `scp`：102 → 107 实测 **275 MB/s**，比重新下载快且省流量。
- **传完必须校验**：`B:` 盘写满时 `scp` 会写出**半截文件且不明显报错** —— 我们差点用 2244 MB 的截断
  safetensors 启动引擎。核对字节数 + `md5`，再启动。

---

## 7. 权重之外的启动前置：PLE 预热

PLE 表 47.7 GB 在 NVMe 上，每 token 要 16 行随机读。**Windows 的页缓存是懒的**：
不预热的话，冷启动后前几分钟的随机读会持续打穿性能（实测 c8 只有 129 tok/s）。

`launch/preload_ple.ps1` 做一次顺序全读（3.3 GB/s 顺序读，约 15 s），把页缓存填满：

```powershell
# 打点方式：跑 3 遍，取第 3 遍的耗时，稳定在 15 s 左右说明已经热了
(Measure-Command { Get-Content $ple -Raw -ReadCount 0 > $null }).TotalSeconds
```

收益：c8 129.5 → 171.8（+33%）、c16 198.7 → 218.8。已固化进 `run-flashnext.bat`。

---

## 8. 首次启动与验收

```bat
launch\run-flashnext.bat        :: 日志：B:\hibana-win\results\flashnext.log
```

冷启动约 **4 分钟**（PLE 进页缓存 + 图捕获），日志出现 `listening on 0.0.0.0:30004` 才算好。

**六项验收（缺一不可）**

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | `GET /health` | 200 |
| 2 | `GET /v1/models` | 列出 flashnext |
| 3 | `GET /get_server_info` | profile、context、KV tokens、slots 与预期一致 |
| 4 | 中文问答 | 通顺、不重复、不截断 |
| 5 | 代码生成 | 完整可运行 |
| 6 | 看图（若启用视觉） | 能描述图片内容；首轮 ~6 s，之后 0.08–0.16 s/轮 |

```powershell
python port\verify.py             # 1–5 自动跑一遍
python bench\refcheck.py cmp base # 位一致闸门，必须 IDENTICAL
```

**回滚**：删掉 `B:\hibana-win`、`B:\hibana-models`、`C:\hibana-models`、`C:\cuda13` 四个目录即可 ——
本移植没有改动 107 上任何原有文件。

---

## 9. 明确没做的部分（写清楚，省得别人踩）

| 没做 | 原因 | 想做得从哪里开始 |
|---|---|---|
| 张量并行 `--tp 2` | 走 NCCL，Windows 没有可用轮子 | 单卡多流（`--max-batch`）已经够用；真要 TP 得上 Linux |
| `b12x`（Bolt/Nemotron 后端） | 钉 `nvidia-cutlass-dsl==4.7.1`，只有 manylinux 轮子 | 自己编 cutlass-dsl |
| SSD 前缀层 | 引擎那套走 `fcntl`/`pread` 垫片，语义对了但没实测收益 | 在有 2 块 NVMe 的机器上重测 |
| NVFP4 KV 缓存 | flashinfer 的 FP4-KV 模块在本移植里编不过 | 修 `sm120a` 的 FP4 KV 内核；fp8 已够用 |
| 视觉塔默认加载 | Flash-Next 默认 `vision: False` | `HIBANA_VISION=1` + JPEG 解码（`libturbojpeg`）；不装也能跑文本 |

---

## 10. 一页纸清单（打印出来贴机箱上）

```
[ ] 107 探路：GPU/SM/VRAM/盘速/CUDA/MSVC 记下来
[ ] C:\cuda13 junction；B:\hibana-win\.venv（uv + 3.12.11）
[ ] torch 2.13.0+cu130 / triton-windows 3.7.1.post27 / flashinfer 0.6.17 --no-deps / xgrammar 0.2.8
[ ] 钉 MSVC 2022（HIBANA_MSVC_PATH）；HIBANA_CUDA_ARCH=12.0a
[ ] flashinfer JIT → tvm_ffi 构建器（3 处）；ninja C$:；workspace 512M/1G
[ ] winpatch.py <engine_root>（26 处，幂等）
[ ] 权重按 HF 缓存布局落盘 + md5 校验（别信 scp 的退出码）
[ ] preload_ple.ps1 预热 47.7 GB
[ ] run-flashnext.bat → /health 200 → verify.py → refcheck IDENTICAL
```
