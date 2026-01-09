#!/bin/bash
# ROS2 工作空间编译脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    编译 ROS2 抓取系统工作空间${NC}"
echo -e "${GREEN}======================================${NC}"

# 检查是否在 Ros2 目录
if [ ! -f "grasp_interfaces/package.xml" ]; then
    echo -e "${RED}错误: 请在 Ros2 目录下运行此脚本${NC}"
    exit 1
fi

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}警告: 未检测到 ROS2 环境，尝试加载...${NC}"
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}已加载 ROS2 Humble${NC}"
    elif [ -f "/opt/ros/foxy/setup.bash" ]; then
        source /opt/ros/foxy/setup.bash
        echo -e "${GREEN}已加载 ROS2 Foxy${NC}"
    else
        echo -e "${RED}错误: 未找到 ROS2 安装${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}ROS2 发行版: $ROS_DISTRO${NC}"

# 清理旧的编译文件（可选）
if [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}清理旧的编译文件...${NC}"
    rm -rf build install log
fi

# 编译
echo -e "${GREEN}开始编译...${NC}"
colcon build --symlink-install

# 检查编译结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}    编译成功！${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "${YELLOW}运行以下命令加载环境：${NC}"
    echo -e "  source install/setup.bash"
    echo ""
    echo -e "${YELLOW}启动系统：${NC}"
    echo -e "  ros2 launch grasp_bringup grasp_system.launch.py"
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}    编译失败！${NC}"
    echo -e "${RED}======================================${NC}"
    exit 1
fi

