# 社交媒体行为模拟项目 (Social Behavior Simulation)

基于强化学习（GRPO - Group Relative Policy Optimization）的社交媒体互动预测与生成项目。

## 📋 项目概述

本项目实现了一个完整的社交媒体行为模拟系统，能够：
- 预测用户在社交媒体上的互动行为（评论、转发等）
- 生成符合用户兴趣和上下文的互动内容
- 使用强化学习优化模型性能

### 主要技术栈
- **框架**: VERL (Versatile Reinforcement Learning)
- **模型**: Qwen2.5-3B-Instruct
- **训练方法**: SFT (Supervised Fine-Tuning) + GRPO (Group Relative Policy Optimization)
- **并行策略**: FSDP (Fully Sharded Data Parallel)

## 🏗️ 项目结构

```
Social_simulation/
├── configs/                    # 配置文件
│   └── sft_config.yaml        # SFT训练配置
├── scripts/                    # 运行脚本
│   ├── run_sft.sh             # SFT训练启动脚本
│   ├── run_train_gen.sh       # GRPO训练启动脚本
│   ├── evaluate.sh            # 模型评估脚本
│   └── fuse.sh                # 模型融合脚本
├── src/
│   ├── verl/                  # VERL框架核心代码
│   │   ├── trainer/           # 训练器实现
│   │   ├── workers/           # 工作节点实现
│   │   └── utils/             # 工具函数
│   ├── data_process/          # 数据预处理
│   │   ├── make_sft_file.py   # SFT数据构造
│   │   ├── make_grpo_train.py # GRPO训练数据构造
│   │   └── make_grpo_val.py   # GRPO验证数据构造
│   ├── training/              # 训练脚本
│   │   └── train_grpo_gen_bfs.py  # GRPO训练主脚本
│   └── evaluation/            # 评估脚本
│       ├── model_evaluate.py  # 模型评估主脚本
│       └── f1_for_json.py    # F1分数计算
└── requirements.txt           # Python依赖包
```

## 🔧 环境配置

### 系统要求
- Python >= 3.8
- CUDA >= 11.8 (推荐)
- 多GPU环境（推荐4+ GPU）

### 安装步骤

1. **克隆仓库**
```bash
cd /path/to/social_simulation
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装额外依赖**（如需要）
```bash
# Flash Attention 2 (推荐，用于加速训练)
pip install flash-attn --no-build-isolation

# 如果遇到编译问题，可以跳过或使用预编译版本
```

5. **安装项目包**（以开发模式）
```bash
cd Social_simulation
pip install -e .
```

## 📊 数据准备

### 数据格式要求

**原始数据**: JSONL格式，每行包含一个社交媒体帖子及其互动信息
- 每条记录包含 `user`（作者信息）、`text_raw`（正文）、`created_at`（时间戳）
- 包含 `comments`（评论列表）和 `reposts`（转发列表）
- 需要用户画像文件（包含用户兴趣信息）和ID映射文件（如需要）

### 数据预处理流程

数据流程：**原始JSONL数据** → **rebuild.py** → **树形JSON** → **make_sft_file.py** → **SFT数据(Parquet)** → **make_grpo_train.py** → **GRPO数据(JSONL)**

#### 步骤1: 构建树形结构数据 (rebuild.py)

从原始JSONL数据构建树形结构的JSON文件，包含用户画像、互动次数等信息。

```bash
python src/data_process/rebuild.py \
    <输入文件1.jsonl> [输入文件2.jsonl ...] \
    <输出文件.json>
```

**说明**：
- 输入：一个或多个JSONL文件，每行是一个帖子记录
- 输出：单个JSON文件，包含所有帖子的树形结构
- 脚本会自动加载用户画像文件（需要修改脚本中的硬编码路径）
- 会进行一周时间窗口筛选，并统计互动次数

**修改脚本中的路径**（第300-303行）：
```python
profile_map = load_user_profile_map(
    '/path/to/user_profile.json',      # 用户画像文件路径
    '/path/to/id_mapping.json'         # ID映射文件路径（可选）
)
```

#### 步骤2: 生成SFT训练数据 (make_sft_file.py)

从树形JSON数据构造SFT训练数据（Parquet格式）。

```bash
python src/data_process/make_sft_file.py \
    --input <rebuild输出的JSON文件> \
    --output_dir <输出目录> \
    --noise_k_min_depth0 0.0 \
    --noise_k_max_depth0 3.0 \
    --noise_k_min_depth1 5.0 \
    --noise_k_max_depth1 11.0 \
    --shuffle_seed 42
```

**输出**：
- `train.parquet`: 训练集（约85%）
- `val.parquet`: 验证集（约15%）

**主要参数**：
- `--noise_k_min_depth0/max_depth0`: depth=0层的噪声倍数范围
- `--noise_k_min_depth1/max_depth1`: depth=1层的噪声倍数范围
- `--large_pool_threshold`: 候选池人数阈值（默认1000，超过则跳过）

#### 步骤3: 生成GRPO训练数据 (make_grpo_train.py)

从SFT数据构造GRPO训练数据（JSONL格式）。

```bash
python src/data_process/make_grpo_train.py \
    --val_sft_parquet <SFT训练数据路径(train.parquet)> \
    --val_output <输出JSONL路径>
```

**输出**：
- JSONL文件，每行包含 `messages`、`reward_model`、`root_potential` 等字段

#### 步骤4: 生成GRPO验证数据 (make_grpo_val.py)

从SFT验证数据构造GRPO验证数据。

```bash
python src/data_process/make_grpo_val.py \
    --val_sft_parquet <SFT验证数据路径(val.parquet)> \
    --val_output <输出JSONL路径>
```

## 🚀 使用指南

### 1. SFT (Supervised Fine-Tuning) 训练

**修改配置**：
- 编辑 `configs/sft_config.yaml`，更新数据路径和模型路径
- 修改 `scripts/run_sft.sh` 中的 `PROJECT_ROOT` 路径

**启动训练**：
```bash
bash scripts/run_sft.sh <GPU数量> [hydra_overrides...]
```

示例：
```bash
bash scripts/run_sft.sh 4 trainer.total_epochs=3
```

### 2. GRPO 训练

**修改脚本**：
- 编辑 `scripts/run_train_gen.sh`，更新以下路径：
  - `--data`: GRPO训练数据路径
  - `--model`: SFT模型检查点路径
  - `--tokenizer`: Tokenizer路径
  - `--output_dir`: 输出目录

**启动训练**：
```bash
bash scripts/run_train_gen.sh
```

**主要参数说明**：
- `--num_traj_per_root`: 每个根节点生成的轨迹数（默认4）
- `--epochs`: 训练轮数（默认3）
- `--lr`: 学习率（默认1e-5）
- `--max_input_tokens`: 最大输入token数（默认4096）
- `--gen_max_new_tokens`: 最大生成token数（默认2048）
- `--roots_per_update`: 每次更新的根节点数（默认4）
- `--use_lora`: 启用LoRA（可选）

### 3. 模型评估

**修改脚本**：
- 编辑 `scripts/evaluate.sh`，更新以下路径：
  - `SCRIPT_PATH`: 评估脚本路径
  - `DATA_PATH`: 验证数据路径
  - `MODEL_PATH`: 模型路径
  - `--checkpoint_pt`: GRPO检查点路径（可选）

**运行评估**：
```bash
bash scripts/evaluate.sh
```

**多卡评估日志合并**：

使用多GPU评估时，每个GPU会生成独立的日志文件：
- `detail.jsonl.rank0`, `detail.jsonl.rank1`, `detail.jsonl.rank2`, ... (详细评估结果)
- `io.jsonl.rank0`, `io.jsonl.rank1`, `io.jsonl.rank2`, ... (输入输出对比)

**需要手动合并这些文件**才能得到完整的评估结果：

```bash
# 合并 detail.jsonl 文件
cat detail.jsonl.rank* > detail.jsonl

# 合并 io.jsonl 文件（如果存在）
cat io.jsonl.rank* > io.jsonl

# 清理临时文件（可选）
rm detail.jsonl.rank* io.jsonl.rank*
```

**计算评估指标**：

合并后的 `detail.jsonl` 可用于计算F1等指标：

```bash
python src/evaluation/f1_for_json.py --input detail.jsonl
```

**注意事项**：
- 多卡评估时，数据会被分片到各个GPU，每个GPU处理不同的样本
- 确保所有GPU完成评估后再合并（脚本会使用 `dist.barrier()` 同步）
- 合并后的文件可以直接用于后续的指标计算

## ⚙️ 配置说明

### SFT配置 (`configs/sft_config.yaml`)

关键配置项：
- `data.train_files`: 训练数据路径
- `data.val_files`: 验证数据路径
- `model.partial_pretrain`: 基础模型路径
- `model.freeze.enable`: 是否冻结部分层
- `model.trainable_name_prefixes`: 可训练层前缀
- `trainer.total_epochs`: 训练轮数
- `trainer.save_freq`: 保存频率

### GRPO训练参数

在 `train_grpo_gen_bfs.py` 中可通过命令行参数调整：
- `--depth_limit`: 深度限制（默认0，只训练第一层）
- `--temperature`: 生成温度（默认1.0）
- `--gen_top_p`: Top-p采样参数（默认0.9）
- `--lora_r`: LoRA rank（默认8）

## 🐛 常见问题

### 1. 内存不足 (OOM)

**解决方案**：
- 减少 `--max_input_tokens`（如4096 → 2048）
- 减少 `--num_traj_per_root`（如4 → 2）
- 减少 `--roots_per_update`（如4 → 2）
- 减少 `--gen_max_new_tokens`（如2048 → 1024）
- 启用LoRA：添加 `--use_lora` 参数
- 脚本已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 优化内存

### 2. 路径错误

**解决方案**：
- 检查所有脚本中的硬编码路径
- 将 `/home/zss/Social_Behavior_Simulation` 替换为你的实际项目路径
- 确保数据路径和模型路径正确

### 3. 依赖缺失

**解决方案**：
- 检查 `requirements.txt` 是否包含所有必需包
- 某些包可能需要从源码安装（如flash-attn）
- VERL框架已包含在 `src/verl` 目录中

### 4. 分布式训练问题

**解决方案**：
- 确保所有GPU在同一节点
- 检查NCCL环境变量设置
- 使用 `torchrun` 而非 `python -m torch.distributed.launch`

## 📝 注意事项

1. **路径配置**: 项目中存在硬编码路径，需要根据实际环境修改
2. **数据格式**: 确保数据格式符合要求，特别是 `reward_model` 和 `root_potential` 字段
3. **GPU内存**: GRPO训练对显存要求较高，建议使用多GPU或启用LoRA
4. **检查点**: 定期保存检查点，训练过程可能较长
5. **日志**: 训练日志保存在 `checkpoints/` 目录下，可通过 `tail -f` 实时查看

## 🔄 训练流程

完整的训练流程如下：

```
原始JSONL数据（帖子+评论+转发）
  ↓
[rebuild.py] → 树形JSON数据（包含用户画像、互动次数）
  ↓
[make_sft_file.py] → SFT数据 (train.parquet, val.parquet)
  ↓
[SFT训练] → SFT模型检查点
  ↓
[make_grpo_train.py / make_grpo_val.py] → GRPO训练数据 (trainbrush.jsonl, valbrush.jsonl)
  ↓
[GRPO训练] → GRPO模型检查点
  ↓
[模型评估] → 评估结果
```

**关键步骤说明**：
1. **rebuild.py**: 将原始扁平数据构建为树形结构，添加用户画像和互动统计
2. **make_sft_file.py**: 从树形数据构造监督学习训练样本（添加噪声候选）
3. **SFT训练**: 使用VERL框架进行监督微调
4. **make_grpo_train.py**: 从SFT数据构造强化学习训练样本（包含reward_model）
5. **GRPO训练**: 使用Group Relative Policy Optimization进行强化学习训练

## 📈 性能优化建议

1. **使用Flash Attention**: 安装 `flash-attn` 可显著加速训练
2. **梯度累积**: 通过 `--roots_per_update` 控制有效批次大小
3. **混合精度**: 默认使用 bfloat16，可在配置中调整
4. **LoRA**: 对于大模型，启用LoRA可大幅减少显存占用
5. **数据加载**: 使用 `use_shm=true` 可加速数据加载（需要共享内存）

## 📄 许可证

请查看项目中的LICENSE文件（如存在）。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请通过Issue联系。

---

**最后更新**: 2025年12月

