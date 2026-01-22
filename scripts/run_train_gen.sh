#!/bin/bash
# GRPO 训练启动脚本 (Gather-Scatter 架构版)
# 适配数据格式: Parquet (Root-Only + Look-ahead Noise)

# ========== 1. 路径配置 ==========
# Python 训练脚本路径 (请修改为你实际保存的文件名)
TRAIN_SCRIPT="/home/zss/Social_Behavior_Simulation/grpo_cls_bfs/train_gen_distributed.py"

# 数据路径 (使用由 build_grpo_from_raw_aligned.py 生成的 Parquet)
DATA_PATH="/home/zss/Social_Behavior_Simulation/abcreading/processed/grpo.parquet"

# 模型与输出路径
MODEL_PATH="/home/zss/Social_Behavior_Simulation/abcreading/sft_ckpt/global_step_6400"
TOKENIZER_PATH="/home/zss/Social_Behavior_Simulation/abcreading/sft_ckpt/tokenizer_with_spans"
OUTPUT_DIR="/home/zss/Social_Behavior_Simulation/abcreading/grpockpt/f1"

# 日志路径
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

# ========== 2. 显存优化配置 ==========
# 极其重要：Gather-Scatter 架构频繁申请释放显存，此配置能防止碎片化导致的伪 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== 3. 启动命令 ==========
echo "Starting Gather-Scatter GRPO Training..."
echo "Log file: $LOG_FILE"

nohup torchrun --nproc_per_node=4 \
    "$TRAIN_SCRIPT" \
    --data "$DATA_PATH" \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --do_sample \
    --roots_per_update 4 \
    --epochs 3 \
    --lr 1e-4 \
    --max_input_tokens 4096 \
    --max_logprob_tokens 2048 \
    --temperature 1.0 \
    --gen_max_new_tokens 2048 \
    --gen_top_p 0.9 \
    --use_lora \
    --lora_r 8 \
    --save_steps 40 \
    --resume_path /home/zss/Social_Behavior_Simulation/abcreading/grpockpt/f1/ckpt-step320-upd40 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training PID: $PID"

# ========== 4. 参数解释 (备忘) ==========
# --roots_per_update 8 : 
#    全局累计梯度步数。即 4张卡合力跑完 8 个不同的 Root 后，更新一次参数。
#    由于大家是并行的，这相当于 Global Batch Size = 8。
#    如果你觉得训练太慢，或者显存非常富裕，可以减小这个值（如 4），但 8 更稳。

# --num_traj_per_gpu 1 : 
#    Depth 0 阶段，每张卡生成几条轨迹。
#    4张卡 * 1条 = 针对该 Root 总共生成 4 条 Depth 0 轨迹。
#    设置为 1 是为了给后续 Depth 1 的动态展开留出最大的显存空间。