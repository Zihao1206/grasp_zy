#!/bin/bash
# 安装 systemd 服务脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    安装 ROS2 抓取系统服务${NC}"
echo -e "${GREEN}======================================${NC}"

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}错误: 请使用 sudo 运行此脚本${NC}"
    exit 1
fi

# 获取当前用户和路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REAL_USER=${SUDO_USER:-$USER}
SERVICE_FILE="$SCRIPT_DIR/ros2_grasp.service"

echo -e "${YELLOW}用户: $REAL_USER${NC}"
echo -e "${YELLOW}工作目录: $SCRIPT_DIR${NC}"

# 检查服务文件
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}错误: 未找到服务文件 ros2_grasp.service${NC}"
    exit 1
fi

# 更新服务文件中的路径和用户
echo -e "${YELLOW}更新服务文件...${NC}"
sed -i "s|User=jet|User=$REAL_USER|g" "$SERVICE_FILE"
sed -i "s|Group=jet|Group=$REAL_USER|g" "$SERVICE_FILE"
sed -i "s|WorkingDirectory=.*|WorkingDirectory=$SCRIPT_DIR|g" "$SERVICE_FILE"
sed -i "s|source /home/jet/zoneyung/grasp_static/Ros2/|source $SCRIPT_DIR/|g" "$SERVICE_FILE"

# 复制服务文件
echo -e "${YELLOW}安装服务文件...${NC}"
cp "$SERVICE_FILE" /etc/systemd/system/

# 重新加载 systemd
echo -e "${YELLOW}重新加载 systemd...${NC}"
systemctl daemon-reload

# 启用服务
echo -e "${YELLOW}启用服务...${NC}"
systemctl enable ros2_grasp.service

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    服务安装成功！${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo -e "  启动服务:    sudo systemctl start ros2_grasp.service"
echo -e "  停止服务:    sudo systemctl stop ros2_grasp.service"
echo -e "  重启服务:    sudo systemctl restart ros2_grasp.service"
echo -e "  查看状态:    sudo systemctl status ros2_grasp.service"
echo -e "  查看日志:    sudo journalctl -u ros2_grasp.service -f"
echo -e "  禁用自启:    sudo systemctl disable ros2_grasp.service"
echo ""
echo -e "${YELLOW}是否现在启动服务？ (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    systemctl start ros2_grasp.service
    sleep 2
    systemctl status ros2_grasp.service
fi

