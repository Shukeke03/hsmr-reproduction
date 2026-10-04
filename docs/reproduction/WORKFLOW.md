# HSMR 复现与微调开发记录

原项目：[HSMR](https://github.com/IsshikiHugh/HSMR)，论文 CVPR 2025 Oral。
上游固定提交 `88486dc14329456615a9eef2f8e362122bc72589`；SKEL 子模块固定提交
`c32cf16581295bff19399379efe5b776d707cd95`。保留原项目 MIT LICENSE；人体模型和数据遵循各自条款。

## 开发布局

- Mac 代码：`/Users/shuxing/AIlearning/hsmr-reproduction`
- Legion 代码：`/home/shuxing/workspace/projects/hsmr-reproduction`
- Legion 数据及原始权重：`/home/shuxing/workspace/datasets/HSMR`
- Legion 训练输出：`/home/shuxing/workspace/checkpoints/HSMR`
- 两端环境：Miniforge `hsmr`，Python 3.10，PyTorch 2.3.1；服务器 CUDA wheel 为 cu121。

本地改代码、Git 提交并推送；服务器拉取代码后训练。快速同步未提交修改：

```bash
bash scripts/sync_legion.sh
ssh legion
cd ~/workspace/projects/hsmr-reproduction
conda activate hsmr
export HSMR_INPUTS_ROOT=$HOME/workspace/datasets/HSMR
export HSMR_OUTPUTS_ROOT=$HOME/workspace/checkpoints/HSMR
```

安装或重新检查两端环境：`bash scripts/setup_env.sh`。
训练依赖已包含 SKEL、Lightning、WebDataset；Detectron2 是完整图片 demo 的额外依赖，
不用于已裁剪训练数据。若需要自动检测 demo，可在已激活环境中安装
`MAX_JOBS=1 python -m pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git`。
不修改现有 `model-reproduction` 和 EEG 环境。

## 数据与权重

2026-10-04 实测 Hugging Face 文件元数据，明细见 `asset-sizes.json`：

| 项目 | 大小 |
|---|---:|
| 官方 Dropbox 7 个训练压缩包，标称加总 | 264 G |
| Hugging Face 未压缩训练 tar，共 4452 个 | 342,581,780,480 bytes = 342.58 GB / 319.05 GiB |
| HSMR 发布模型压缩包 | 2,509,409,979 bytes = 2.51 GB |
| ViTPose 初始化骨干 | 2,523,785,910 bytes = 2.52 GB |
| HMR2 评测标签压缩包 | 77,556,233 bytes |
| HSMR 评测标签压缩包 | 105,085,074 bytes |

训练 tar 已包含图像和伪标签，可直接由 WebDataset 读取，无需解成图片。
264 G 是官方近似压缩包标称值；342.58 GB 是另一分发形式的精确文件总和。
评测标签不包含全部基准图像。完整评测图像需另外准备，因此不能把标签大小当作评测数据集总大小。

当前遵循用户要求：**不下载训练数据**，只准备官方权重和人体模型，用仓库自带图片跑通推理。
以下命令均在 legion `hsmr` 环境中运行：

```bash
python scripts/prepare_data.py --assets-only --body-models --dry-run
HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_data.py --assets-only --body-models
# 后续用户选择微调数据后再准备分片：
python scripts/prepare_data.py --dataset coco14 --shards 1
# 完整训练集，直接下载 tar，不保留另外一套 gzip：
python scripts/prepare_data.py --dataset all --shards 0
# 从头训练才需要 ViTPose 初始化骨干：
python scripts/prepare_data.py --backbone --dataset coco14 --shards 1
```

Legion 直连 Hugging Face 当前超时，已确认 `hf-mirror.com` 可访问；下载命令可设置
`HF_ENDPOINT=https://hf-mirror.com`。下载版本固定在 `download-lock.json`，发布模型、回归器
与初始 COCO 分片的 SHA256 来自官方 API。脚本支持缓存恢复、检查 LFS SHA256 并记录到服务器
`download-manifest.jsonl`。模型归档会保留压缩包和解压结果，准备约 6 GB 初始空间。
训练数据、权重、环境缓存和结果不进入公开 GitHub。

**SKEL 模型必需**：已确认作者公开的 `body_models.tar.gz` 包含 v1.1.1 模型，
可用上述 `--body-models` 获取（315 MB 压缩包）。亦可按官方要求，从 [SKEL 官网](https://skel.is.tue.mpg.de/download.php)
登录并获得 `skel_models_v1.1.zip`，确认内部版本 `v1.1.1`。将整个模型目录放到服务器
`~/workspace/datasets/HSMR/body_models/skel/`，其中有 `skel_male.pkl`、`skel_female.pkl`
和 `changelog_v1.1.1.txt`。不能仅靠源码子模块代替模型参数。

## 训练层次与当前硬件

Legion：RTX 3060 Laptop 6 GiB VRAM，WSL2，约 7.6 GiB RAM / 2 GiB swap。
官方 `configs/exp/hsmr/train.yaml` 是 8 GPU、每卡 batch=78、100 epochs，包含 SKELify-SPIN。
6 GiB 显存不足以按该配置全参数训练 ViT-H（约 6.3 亿骨干参数，Adam 状态占用很大）。

`legion_finetune.yaml` 使用发布 HSMR 权重、冻结 ViT-H、batch=1、混合精度、
后续单 COCO14 分片、100 步；不运行 SKELify 伪标签更新，关闭图片渲染日志。
这是工程验证/头部微调基线，不能宣称复现了论文完整训练和指标。
已使用自带图片和合成目标验证完整主损失前后向；真实数据训练仍需后续验证。

```bash
bash scripts/train_legion.sh
# 正式较长头部微调：
bash scripts/train_legion.sh exp_tag=head-v1 pl_trainer.max_steps=10000 pl_trainer.max_epochs=20
# 恢复完整训练状态（同时恢复 optimizer、步数）：
bash scripts/train_legion.sh exp_tag=head-v1 resume_path=/absolute/path/to/last.ckpt
# 官方训练入口（需要更大训练硬件、完整数据与 backbone）：
python exp/run_train.py exp=hsmr/train
```

`ckpt_path` 是初始模型权重；`resume_path` 是 Lightning 完整续训检查点，两者职责不同。
将每次微调的配置、代码提交、数据分片、初始权重、随机种子、指标记录在 Git 中；
大权重存服务器。为复现实验建立 `experiment/<name>` 分支，稳定版本可打 tag。

## 本地改动

1. 通过 `HSMR_INPUTS_ROOT` / `HSMR_OUTPUTS_ROOT` 分离代码与服务器数据路径。
2. 修正 WebDataset 的位置参数误传：原代码把 `epoch_size` 传给了 `resampled`。
3. 区分初始化权重和完整续训，CPU 加载初始化检查点，保留本机 SKEL 模型缓冲区。
4. 冻结骨干时持续保持 eval 模式，包括从发布模型初始化的路径。
5. macOS 不强制 EGL；Linux 保留 EGL 默认后端。
6. 修正配置入口重复添加命令行参数，恢复 Hydra `--cfg` 配置检查功能。
7. 增加双端环境安装、检查、数据校验、同步与微调入口。

尚未完成：真实数据微调、完整基准评测、论文指标复现。

## 验证记录

本地环境安装、`pip check`、训练模块导入已通过。
`python -m pytest -q tests`：4 项通过，覆盖 Hydra 配置、缩小规模的真实 ViT/预测头
前向反向、骨干冻结状态、外部数据路径和实际训练 CLI 配置导出。
这些检查无需 SKEL 参数，不代表完整 HSMR 推理或训练已通过。

## 仅使用权重跑通与测速

```bash
export HSMR_INPUTS_ROOT=$HOME/workspace/datasets/HSMR
export HSMR_OUTPUTS_ROOT=$HOME/workspace/checkpoints/HSMR
python scripts/run_pretrained.py --with-body --benchmark-steps 20
```

使用官方 ViT-H / 预测头原始权重，严格检查参数键，输入仓库自带 yoga1.jpg。
整图补成方形后裁剪，不依赖检测器；输出 46 维姿态、10 维体型、相机参数、
SKEL 关节、皮肤 OBJ、骨骼 OBJ 和关节投影预览。
测速用重复的自带图片与由预测值构造的合成目标，执行真实骨干前向、预测头反向、
SKEL 网格与原始 HSMR 主损失、AdamW；不保存微调模型，不形成科学评测指标。
没有数据 IO、SKELify 伪标签更新和 checkpoint 开销，因此实际训练会更慢。

## Legion 初次实测（2026-10-04）

- CUDA 和训练依赖检查通过，`pip check` 无冲突；服务器 4 项测试通过。
- 实际加载发布检查点（2,757,041,120 bytes），630,912,000 骨干参数和 39,547,037 预测头参数。
- 真实神经网络推理约 41–43 ms/图片（不包含检测、SKEL 几何与渲染）。
- 输出 44 个关节、6,890 个皮肤顶点和 247,252 个骨骼网格顶点，以及姿态、体型、相机和 OBJ 文件。
- batch=1，冻结 ViT-H，两次 20 步合成目标完整主损失测速：0.47–0.50 秒/步。
- CUDA 峰值分配约 3.24 GiB，保留约 3.33 GiB。
- 换算纯计算耗时：1,000 步约 8.4 分钟；10,000 步约 1.4 小时；50,000 步约 7 小时。
- 实际读取数据、保存检查点、验证和 SKELify 会增加时长；这不是全参数训练速度，也不是收敛时间保证。
- 官方压缩模型包加回归器合计下载约 2.83 GB；服务器保留归档和解压文件约 5.6 GiB。训练数据下载量为 0。

原始输出在服务器 `~/workspace/checkpoints/HSMR/pretrained-smoke.json`，
运行日志在 `~/workspace/logs/hsmr-pretrained-smoke.log`。公开仓库只记录去除预测数组的测速摘要。

代码规模：主仓库当前受控文件约 8.6 MB；SKEL 子模块文件约 38.6 MB；
本地全部检出（含 Git 历史、子模块与开发缓存）约 94 MiB。
