# Social Behavior Simulation

> A Reinforcement Learning-based system for predicting and generating social media interactions.

## 📖 Project Introduction

This project aims to predict and generate user interactions (comments, reposts) on social media platforms. It utilizes the **Qwen2.5-3B** model as a foundation and is optimized using a robust pipeline of **SFT** (Supervised Fine-Tuning) and **GRPO** (Group Relative Policy Optimization) to simulate realistic social behaviors.

---

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