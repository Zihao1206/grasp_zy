#!/bin/bash
# ROS2 环境设置脚本

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    设置 ROS2 环境${NC}"
echo -e "${GREEN}======================================${NC}"

# 加载 ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo -e "${GREEN}✓ 已加载 ROS2 Humble${NC}"
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
    echo -e "${GREEN}✓ 已加载 ROS2 Foxy${NC}"
else
    echo -e "${YELLOW}⚠ 未找到 ROS2 安装${NC}"
fi

# 加载工作空间
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
    echo -e "${GREEN}✓ 已加载工作空间环境${NC}"
else
    echo -e "${YELLOW}⚠ 工作空间未编译，请先运行: ./build.sh${NC}"
fi

# 加载 Conda 环境（可选）
if command -v conda &> /dev/null; then
    if conda env list | grep -q "zy_torch"; then
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

