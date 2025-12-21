#!/bin/bash
# GRPO 文本生成训练启动脚本（不使用分类头，直接生成文本）
# 支持后台运行

# ========== 配置日志文件路径 ==========
# 方式1: 使用固定路径
LOG_FILE="/home/zss/Social_Behavior_Simulation/checkpoints/grpockpt/12.19/train_$(date +%Y%m%d_%H%M%S).log"

# 方式2: 使用带时间戳的日志文件名（取消下面的注释来使用）
# LOG_FILE="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_gen_ckpt/train_$(date +%Y%m%d_%H%M%S).log"

# 方式3: 分离标准输出和错误输出（取消下面的注释来使用）
# LOG_FILE="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_gen_ckpt/train.log"
# ERR_LOG_FILE="/home/zss/Social_Behavior_Simulation/checkpoints/grpo_gen_ckpt/train_error.log"

# 确保日志目录存在
mkdir -p "$(dirname "$LOG_FILE")"

# ========== CUDA 内存优化配置 ==========
# 设置 PyTorch CUDA 内存分配器以减少内存碎片
# expandable_segments: 允许内存段动态扩展，减少碎片（推荐用于解决 OOM 问题）
# max_split_size_mb: 限制最大内存块大小（可选，根据实际情况调整）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 可选：如果仍然遇到 OOM，可以尝试以下配置（取消注释）
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# 其他内存优化选项（根据需要取消注释）
# export CUDA_LAUNCH_BLOCKING=0  # 异步执行，可能减少内存峰值
# export PYTORCH_NO_CUDA_MEMORY_CACHING=0  # 保持默认的内存缓存

# 清理 GPU 缓存（可选，在启动前清理其他进程占用的显存）
# 注意：这会重置所有 GPU，请确保没有其他重要进程在运行
# nvidia-smi --gpu-reset || true

# ========== 启动训练 ==========
# 使用 nohup 后台运行，输出重定向到日志文件
# 注意：环境变量已通过 export 设置，会被子进程继承
nohup torchrun --nproc_per_node=4 \
    /home/zss/Social_Behavior_Simulation/grpo_cls_bfs/train_grpo_gen_bfs.py \
    --stage online \
    --data /home/zss/Social_Behavior_Simulation/data_preprocess/grpo_data/trainbrush.jsonl \
    --model /home/zss/Social_Behavior_Simulation/toothbrush/sft_ckpt/12.16/global_step_800 \
    --tokenizer /home/zss/Social_Behavior_Simulation/toothbrush/sft_ckpt/12.16/tokenizer_with_spans \
    --output_dir /home/zss/Social_Behavior_Simulation/checkpoints/grpockpt/12.19 \
    --num_traj_per_root 4 \
    --epochs 3 \
    --lr 1e-5 \
    --max_input_tokens 4096 \
    --depth_limit 0 \
    --temperature 1.0 \
    --gen_max_new_tokens 2048 \
    --gen_top_p 0.9 \
    --roots_per_update 4 \
    --save_steps 8 \
    > "$LOG_FILE" 2>&1 &

# 如果使用方式3（分离输出），使用下面的命令替代上面的：
# nohup torchrun --nproc_per_node=4 \
#     ... (相同参数) ... \
#     > "$LOG_FILE" 2> "$ERR_LOG_FILE" &

# 获取进程 ID
PID=$!
echo "=========================================="
echo "GRPO 文本生成训练已启动，进程 ID: $PID"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 模式: 文本生成（不使用分类头）"
echo "  - 数据: SFT 格式（真 gold + 噪声）"
echo "  - 深度限制: 0（只训练第一层）"
echo "  - LoRA: 启用（r=8）"
echo ""
echo "常用命令："
echo "  查看日志: tail -f $LOG_FILE"
echo "  查看最后100行: tail -n 100 $LOG_FILE"
echo "  实时查看日志: tail -f $LOG_FILE"
echo "  查看进程: ps aux | grep torchrun"
echo "  停止训练: kill $PID"
echo "  查看 GPU: watch -n 1 nvidia-smi"
echo ""
echo "内存优化提示："
echo "  - 已启用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 以减少内存碎片"
echo "  - 如果仍然遇到 OOM，可以尝试："
echo "    1. 减少 --max_input_tokens（当前: 4096）"
echo "    2. 减少 --num_traj_per_root（当前: 2）"
echo "    3. 减少 --roots_per_update（当前: 8）"
echo "    4. 减少 --gen_max_new_tokens（当前: 4096）"
echo "    5. 增加 --lora_r（当前: 8）以使用更多 LoRA 参数"
echo ""
echo "注意事项："
echo "  - 必须使用 SFT 格式的数据（make_sft_file.py 生成的 train.parquet）"
echo "  - depth_limit 固定为 0（只训练第一层）"
echo "  - 不使用分类头，直接生成文本并解析"
echo ""
echo "现在可以安全关闭 SSH 连接，训练会继续运行！"

