#!/bin/bash
# GRPO 评估启动脚本 (参照你的训练脚本结构)
# 支持断开 SSH 连接后继续运行

# ========== 1. 配置路径与参数 (请修改这里) ==========
# 脚本路径
SCRIPT_PATH="/home/zss/Social_Behavior_Simulation/data_preprocess/scripts/tools/v3.0/model_evaluate.py"

# 输入数据路径 (你的 GRPO jsonl)
DATA_PATH="/home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/valbrush.jsonl"

# 模型路径
MODEL_PATH="/home/zss/Social_Behavior_Simulation/toothbrush/sft_ckpt/12.16/global_step_800"

# 输出目录
OUTPUT_DIR="/home/zss/Social_Behavior_Simulation/toothbrush/evalresult/eval_results_$(date +%Y%m%d_%H%M%S)"

# 显卡数量
NUM_GPUS=4

# ========== 2. 配置日志文件 ==========
# 自动创建日志目录
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/eval.log"

# ========== 3. 环境变量配置 ==========
# 关键：强制 Python 实时输出日志，否则 nohup.out 会是空的
export PYTHONUNBUFFERED=1

# 显存优化 (可选，保持和你训练脚本一致)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========== 4. 启动评估 ==========
# 使用 nohup 后台运行，输出重定向到日志文件
echo "正在启动评估任务..."

nohup torchrun --nproc_per_node=$NUM_GPUS \
    "$SCRIPT_PATH" \
    --data "$DATA_PATH" \
    --model "$MODEL_PATH" \
    --checkpoint_pt "/home/zss/Social_Behavior_Simulation/checkpoints/grpockpt/12.19/online_step40.pt" \
    --jsonl_detail "$OUTPUT_DIR/detail.jsonl" \
    --jsonl_io "$OUTPUT_DIR/io.jsonl" \
    --max_samples 10000 \
    --depth_limit 1 \
    > "$LOG_FILE" 2>&1 &

# 获取进程 ID
PID=$!

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