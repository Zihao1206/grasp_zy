# ROS2 安装指南（Ubuntu 20.04）

## 检查是否已安装

```bash
# 检查 ROS2 版本
ls /opt/ros/

# 检查环境变量
echo $ROS_DISTRO
```

## 如果未安装 ROS2 Foxy

### 1. 设置源

```bash
# 设置 locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS2 apt 仓库
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2. 安装 ROS2 Foxy

```bash
# 更新软件包列表
sudo apt update
sudo apt upgrade

# 安装 ROS2 Foxy Desktop（推荐）
sudo apt install ros-foxy-desktop

# 或者安装 ROS2 Foxy Base（最小安装）
# sudo apt install ros-foxy-ros-base
```

### 3. 安装开发工具

```bash
# 安装 colcon（编译工具）
sudo apt install python3-colcon-common-extensions

# 安装其他开发工具
sudo apt install python3-rosdep python3-vcstool
```

### 4. 初始化 rosdep

```bash
# 初始化 rosdep（首次安装需要）
sudo rosdep init
rosdep update
```

### 5. 配置环境

```bash
# 添加到 ~/.zshrc（如果你用 zsh）
echo "source /opt/ros/foxy/setup.bash" >> ~/.zshrc
source ~/.zshrc

# 或添加到 ~/.bashrc（如果你用 bash）
# echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
# source ~/.bashrc
```

### 6. 验证安装

```bash
# 检查版本
echo $ROS_DISTRO
# 应该输出: foxy

# 测试命令
ros2 --help

# 运行演示
ros2 run demo_nodes_cpp talker
# 在另一个终端运行:
# ros2 run demo_nodes_cpp listener
```

## 安装项目依赖包

```bash
# 安装 cv_bridge（OpenCV 桥接）
sudo apt install ros-foxy-cv-bridge

# 安装 rqt 工具（可视化）
sudo apt install ros-foxy-rqt*

# 安装 image transport
sudo apt install ros-foxy-image-transport
```

## 常见问题

### Q1: GPG 错误

如果遇到 GPG 密钥错误：
```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys <KEY_ID>
```

### Q2: 网络问题（国内用户）

可以使用清华源：
```bash
# 修改 /etc/apt/sources.list.d/ros2.list
sudo nano /etc/apt/sources.list.d/ros2.list

# 将内容改为：
deb https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu focal main
```

### Q3: 依赖冲突

```bash
# 清理并重新安装
sudo apt autoremove
sudo apt clean
sudo apt update
sudo apt install --fix-broken
```

## 卸载 ROS2（如需重装）

```bash
# 卸载 ROS2 包
sudo apt remove ~nros-foxy-* && sudo apt autoremove

# 删除仓库
sudo rm /etc/apt/sources.list.d/ros2.list
sudo apt update
```

## 参考资料

- 官方文档: https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html
- 中文教程: https://www.guyuehome.com/

