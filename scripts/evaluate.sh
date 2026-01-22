#!/bin/bash
# GRPO 评估启动脚本（LoRA adapter 版）
# 支持断开 SSH 连接后继续运行

# ========== 1. 配置路径与参数 ==========
SCRIPT_PATH="/home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/v3.0/model_evaluate.py"

DATA_PATH="/home/zss/Social_Behavior_Simulation/abcreading/processed/val.jsonl"

# Base 模型：SFT ckpt（一定要是训练 LoRA 时用的那个 base）
MODEL_PATH="/home/zss/Social_Behavior_Simulation/abcreading/sft_ckpt/global_step_6400"

# LoRA 训练保存目录（包含 adapter_config.json + adapter_model.safetensors）
ADAPTER_DIR="/home/zss/Social_Behavior_Simulation/abcreading/grpockpt/f1/ckpt-step960-upd240"

OUTPUT_DIR="/home/zss/Social_Behavior_Simulation/abcreading/evalresult/grpo_240"

NUM_GPUS=6

# ========== 2. 配置日志文件 ==========
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/eval.log"

# ========== 3. 环境变量配置 ==========
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# （可选）减少 NCCL 玄学卡死概率：网络/IB 环境不稳时很有用
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1

# ========== 4. 启动评估 ==========
echo "正在启动评估任务 (LoRA adapter) ..."
echo "Base model : $MODEL_PATH"
echo "Adapter dir: $ADAPTER_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Log file   : $LOG_FILE"

nohup torchrun --nproc_per_node=$NUM_GPUS \
    "$SCRIPT_PATH" \
    --data "$DATA_PATH" \
    --model "$MODEL_PATH" \
    --jsonl_detail "$OUTPUT_DIR/detail.jsonl" \
    --jsonl_io "$OUTPUT_DIR/io.jsonl" \
    --max_samples 10000 \
    --depth_limit 1 \
    --group_by_record_id \
    --root_chunk_size 40 \
    --adapter_dir "$ADAPTER_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
#--adapter_dir "$ADAPTER_DIR" \ --resume \
# ========== 5. 输出状态信息 ==========
echo "=========================================="
echo "✅ 评估任务已后台启动！"
echo "进程 ID (PID): $PID"
echo "日志文件: $LOG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "常用命令："
echo "  查看实时日志: tail -f $LOG_FILE"
echo "  查看进程状态: ps -p $PID"
echo "  停止任务: kill $PID"
echo ""
echo "✨ 现在你可以安全关闭 SSH 连接，程序会自动在后台运行。"
