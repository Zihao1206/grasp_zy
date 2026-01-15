#!/bin/bash
# ROS2 环境设置脚本（兼容 bash 和 zsh）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    设置 ROS2 环境${NC}"
echo -e "${GREEN}======================================${NC}"

# 检测当前 shell
if [ -n "$ZSH_VERSION" ]; then
    CURRENT_SHELL="zsh"
    SCRIPT_DIR="${0:a:h}"
elif [ -n "$BASH_VERSION" ]; then
    CURRENT_SHELL="bash"
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
else
    CURRENT_SHELL="unknown"
    SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
fi

echo -e "${GREEN}当前 Shell: $CURRENT_SHELL${NC}"

# 根据 shell 类型选择合适的 setup 文件
if [ -n "$ZSH_VERSION" ]; then
    SETUP_EXT="zsh"
elif [ -n "$BASH_VERSION" ]; then
    SETUP_EXT="bash"
else
    SETUP_EXT="bash"  # 默认
fi

echo -e "${GREEN}使用 setup.$SETUP_EXT 文件${NC}"

# 加载 ROS2
# Ubuntu 20.04 → Foxy (优先)
# Ubuntu 22.04 → Humble
if [ -f "/opt/ros/foxy/setup.$SETUP_EXT" ]; then
    source /opt/ros/foxy/setup.$SETUP_EXT
    echo -e "${GREEN}✓ 已加载 ROS2 Foxy (Ubuntu 20.04)${NC}"
elif [ -f "/opt/ros/humble/setup.$SETUP_EXT" ]; then
    source /opt/ros/humble/setup.$SETUP_EXT
    echo -e "${GREEN}✓ 已加载 ROS2 Humble (Ubuntu 22.04)${NC}"
elif [ -f "/opt/ros/galactic/setup.$SETUP_EXT" ]; then
    source /opt/ros/galactic/setup.$SETUP_EXT
    echo -e "${GREEN}✓ 已加载 ROS2 Galactic${NC}"
else
    echo -e "${YELLOW}⚠ 未找到 ROS2 安装${NC}"
    echo -e "${YELLOW}  Ubuntu 20.04 请安装: sudo apt install ros-foxy-desktop${NC}"
fi

# 加载工作空间
if [ -f "$SCRIPT_DIR/install/setup.$SETUP_EXT" ]; then
    source "$SCRIPT_DIR/install/setup.$SETUP_EXT
    echo -e "${GREEN}✓ 已加载工作空间环境 (setup.$SETUP_EXT)${NC}"
else
    echo -e "${YELLOW}⚠ 工作空间未编译，请先运行: ./build.sh${NC}"
fi

# 加载 Conda 环境（可选）
if command -v conda &> /dev/null; then
    if conda env list | grep -q "zy_torch"; then
        # zsh 和 bash 的 conda activate 语法相同
        conda activate zy_torch
        echo -e "${GREEN}✓ 已激活 Conda 环境: zy_torch${NC}"
    fi
fi

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}环境设置完成！${NC}"
echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo -e "  ros2 launch grasp_bringup grasp_system.launch.py    # 启动完整系统"
echo -e "  ros2 launch grasp_bringup vision_only.launch.py     # 仅启动视觉模块"
echo -e "  ros2 topic list                                      # 查看话题列表"
echo -e "  ros2 node list                                       # 查看节点列表"

