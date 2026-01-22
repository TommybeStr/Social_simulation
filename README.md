# ReSim

A Reinforcement Learning-based system for predicting and generating social media interactions.

## 📋 Project Introduction

This project aims to predict and generate user interactions (comments, reposts) on social media platforms. It utilizes the **Qwen2.5-3B** model and is optimized using a pipeline of **SFT** (Supervised Fine-Tuning) and **GRPO** (Group Relative Policy Optimization).

## 📂 Project Structure

```text
Social_simulation/
├── demo_dataset/          # Demo data (Ready-to-use)
├── configs/               # Configuration files
├── scripts/               # Execution scripts (Modify MODEL_PATH here)
├── src/                   # Source code
│   ├── data_process/      # Data preprocessing
│   ├── training/          # GRPO training logic
│   ├── evaluation/        # Model evaluation
│   └── verl/              # VERL framework (Custom version)
└── checkpoints/           # Training outputs (Auto-created)
```

## 🚀 Quick Start

### Step 1: Environment Setup

**Requirements:**
* Python 3.8+
* PyTorch 2.6.0+
* CUDA 11.8+

```bash
# Clone or enter the directory
cd Social_simulation

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

### Step 2: Data Preparation

Choose **one** of the following options:

#### Option A: Use Demo Data (Recommended for Testing)
The project includes a ready-to-use dataset in `demo_dataset/`. **You can skip directly to Step 3.**

#### Option B: Use Custom Data
You need raw `.jsonl` files (each line containing a post and its comments) and a user profile file.

```bash
# 1. Build Tree Structure (Time window split & interaction stats)
python src/data_process/rebuild.py \
    --inputs your_data.jsonl \
    --profile user_profile.json \
    --train-out train.json \
    --val-out val.json \
    --test-out test.json

# 2. Generate SFT Data (JSON -> Parquet)
python src/data_process/make_sft_file.py --input train.json --output train.parquet
python src/data_process/make_sft_file.py --input val.json --output val.parquet

# 3. Generate GRPO Data
python src/data_process/make_grpo_train.py \
    --val_sft_parquet train.parquet \
    --val_output grpo_train.parquet
```

### Step 3: Model Training

#### 3.1 SFT (Supervised Fine-Tuning)

**For Demo / Quick Test (1 GPU):**
Run the following command to override config paths with demo data:

```bash
bash scripts/run_sft.sh 1 \
  data.train_files=demo_dataset/demo_train_sft.parquet \
  data.val_files=demo_dataset/demo_train_sft.parquet \
  model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct
```

**For Custom Data (Multi-GPU):**
1. Edit `configs/sft_config.yaml` to point to your data paths.
2. Run: `bash scripts/run_sft.sh 4`
3. Outputs are saved to `checkpoints/`.

#### 3.2 GRPO (Reinforcement Learning)

Open `scripts/run_train_gen.sh` and update the `MODEL_PATH`:

```bash
MODEL_PATH="${PROJECT_ROOT}/checkpoints/YOUR_SFT_CHECKPOINT_PATH"
```

Start GRPO training:

```bash
bash scripts/run_train_gen.sh
```

### Step 4: Evaluation

Open `scripts/evaluate.sh` and update the `MODEL_PATH`:

```bash
MODEL_PATH="${PROJECT_ROOT}/checkpoints/YOUR_FINAL_MODEL_PATH"
```

Run evaluation:

```bash
bash scripts/evaluate.sh
```

View metrics:

```bash
cd eval_results
# Merge distributed results
cat detail.jsonl.rank* > detail.jsonl
# Calculate statistics
python ../src/evaluation/statistics.py --input detail.jsonl
```
