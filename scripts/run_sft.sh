#!/usr/bin/env bash
set -x

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <nproc_per_node> [hydra_overrides...]"
  exit 1
fi

# 读取 GPU 数
nproc_per_node=$1
shift 1

PROJECT_ROOT=/home/zss/Social_Behavior_Simulation
SAVE_DIR=${PROJECT_ROOT}/checkpoints/run${nproc_per_node}gpu
mkdir -p ${SAVE_DIR}  # ✅ 确保目录存在
LOG_FILE=${SAVE_DIR}/train_$(date "+%Y%m%d_%H%M%S").log

# 并行启动 SFT 训练
nohup torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path ${PROJECT_ROOT}/Config \
  --config-name sft_config \
  trainer.project_name=social-behavior-sft \
  trainer.experiment_name=run${nproc_per_node}gpu \
  "$@" > ${LOG_FILE} 2>&1 &

echo "Launched SFT on ${nproc_per_node} GPUs;"
echo "  - logs -> ${LOG_FILE}"
echo "  - checkpoints -> ${SAVE_DIR}"
