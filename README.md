# Social Behavior Simulation Project (Social Behavior Simulation)

A social media interaction prediction and generation project based on Reinforcement Learning (GRPO - Group Relative Policy Optimization).

## 📋 Project Overview

This project implements a complete social media behavior simulation system capable of:
- Predicting user interaction behaviors on social media (comments, reposts, etc.)
- Generating interaction content that aligns with user interests and context
- Adopting a generative architecture to output structured prediction results in JSON format
- Supporting multi-level (depth 0/1) interaction prediction
- Optimizing generation quality using Reinforcement Learning (GRPO)

### Key Tech Stack
- **Framework**: VERL (Versatile Reinforcement Learning) - Custom Version
- **Base Model**: Qwen2.5-3B-Instruct
- **Training Methods**: 
  - SFT (Supervised Fine-Tuning) 
  - GRPO (Group Relative Policy Optimization)
- **Parallel Strategies**: 
  - FSDP (Fully Sharded Data Parallel) for SFT
  - DDP (Distributed Data Parallel) for GRPO
- **Optimization**: LoRA, Flash Attention 2, Gradient Checkpointing

## 🏗️ Project Structure

```text
Social_simulation/
├── configs/                   # Configuration files
│   └── sft_config.yaml        # SFT training configuration
├── scripts/                   # Execution scripts
│   ├── run_sft.sh             # SFT training launch script
│   ├── run_train_gen.sh       # GRPO training launch script
│   └── evaluate.sh            # Model evaluation script
├── src/
│   ├── verl/                  # VERL framework core code (Custom version)
│   │   ├── trainer/           # Trainer implementation
│   │   ├── workers/           # Distributed worker nodes
│   │   ├── models/            # Model implementation
│   │   ├── utils/             # Utility functions
│   │   └── tools/             # External tool integration
│   ├── data_process/          # Data preprocessing
│   │   ├── rebuild.py         # Tree structure construction (Raw JSONL -> JSON)
│   │   ├── make_sft_file.py   # SFT data construction (JSON -> Parquet)
│   │   ├── make_grpo_train.py # GRPO training data construction
│   │   └── make_grpo_val.py   # GRPO validation data construction
│   ├── training/              # Training scripts
│   │   └── train_grpo_gen_bfs.py  # GRPO main training script (Gather-Scatter architecture)
│   └── evaluation/            # Evaluation scripts
│       ├── model_evaluate.py  # Model evaluation main script (Supports LoRA)
│       └── statistics.py      # Metric calculation (ROUGE-L, BERTScore, etc.)
└── requirements.txt           # Python dependencies