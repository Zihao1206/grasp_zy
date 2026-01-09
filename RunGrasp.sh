#!/bin/zsh

# 1. 设置工作目录为当前目录
WORK_DIR=/home/jet/zoneyung/grasp_static
echo "当前工作目录: $WORK_DIR"
cd "$WORK_DIR" || { echo "切换目录失败"; exit 1; }

# 2. 激活 Conda 环境
CONDA_ENV_NAME="zy_torch"
echo "正在激活 Conda 环境: $CONDA_ENV_NAME"

# 初始化 Conda（确保 Conda 可用）
source "$(conda info --base)/etc/profile.d/conda.sh"

# 激活环境
conda activate "$CONDA_ENV_NAME" || { echo "激活 Conda 环境失败"; exit 1; }

# 3. 执行 Python 脚本
PYTHON_SCRIPT="ctu_conn.py"
echo "正在运行 Python 脚本: $PYTHON_SCRIPT"

python "$PYTHON_SCRIPT" || { echo "运行 Python 脚本失败"; exit 1; }

echo "脚本运行完成"