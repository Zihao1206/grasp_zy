# Python 3.10 快速安装指南

## ⚠️ 重要提示

**torch==1.8.0+cu111 只支持 Python 3.6-3.9，不支持 Python 3.10！**

如果要用 Python 3.10，需要升级 PyTorch 版本。

---

## 一、创建新环境（Python 3.10）

```bash
# 创建名为 grasp_zy_py310 的环境，Python 3.10
conda create -n grasp_zy_py310 python=3.10 -y

# 激活环境
conda activate grasp_zy_py310
```

---

## 二、修改 requirements.txt

由于 PyTorch 1.8.0 不支持 Python 3.10，需要修改以下包的版本：

### 需要修改的包：

| 包名 | 原版本 | 新版本（Python 3.10） |
|------|--------|---------------------|
| torch | 1.8.0+cu111 | >=2.0.0 |
| torchvision | 0.9.0+cu111 | >=0.15.0 |
| numpy | 1.23.0 | >=1.24.0 |
| mmcv | 2.1.0 | >=2.0.0 |
| mmengine | 0.10.7 | >=0.10.0 |
| opencv-python | 4.12.0.88 | >=4.8.0 |
| open3d | 0.14.1 | >=0.17.0 |

### 修改后的 requirements.txt 内容：

```txt
# 核心深度学习（升级到支持 Python 3.10 的版本）
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# MMDetection相关
mmcv>=2.0.0
mmengine>=0.10.0
# Editable install with no version control (mmdet==3.3.0)
-e /home/zh/.local/lib/python3.10/site-packages

# 图像处理
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=9.5.0
matplotlib>=3.7.0

# 相机
pyrealsense2>=2.54.0

# 科学计算
scipy>=1.10.0
transforms3d>=0.3.1

# 工具库
PyYAML>=6.0
requests>=2.31.0
tqdm>=4.65.0
addict>=2.4.0
colorlog>=6.7.0

# 编译工具
Cython>=3.0.0
ninja>=1.11.0

# GraspNet相关
graspnetAPI @ file:///home/zh/zh/Vision-Language-Grasping/models/graspnetAPI
grasp_nms>=1.0.0
pointnet2>=0.0.0
open3d>=0.17.0
trimesh>=4.0.0
scikit-learn>=1.3.0

# 机械臂控制
pyserial>=3.5
crcmod>=1.7
```

---

## 三、安装步骤

### 方法 1：直接安装（推荐）

```bash
# 1. 创建并激活环境
conda create -n grasp_zy_py310 python=3.10 -y
conda activate grasp_zy_py310

# 2. 进入项目目录
cd /home/zh/zh/grasp_zy_zhiyuan

# 3. 安装依赖（使用上面的修改版本）
pip install -r requiments/requirements.txt
```

### 方法 2：分步安装（如果遇到问题）

```bash
# 1. 创建并激活环境
conda create -n grasp_zy_py310 python=3.10 -y
conda activate grasp_zy_py310

# 2. 先安装 PyTorch（自动安装 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 mmengine 和 mmcv
pip install mmengine mmcv -U

# 4. 安装其他依赖
pip install numpy opencv-python scikit-image Pillow matplotlib
pip install scipy transforms3d scikit-learn
pip install PyYAML requests tqdm addict colorlog
pip install Cython ninja
pip install pyrealsense2 pyserial crcmod
pip install open3d trimesh grasp_nms

# 5. 安装 pointnet2（需要编译）
cd graspnet-baseline/pointnet2
python setup.py install
cd ../..

# 6. 安装 graspnetAPI
cd graspnetAPI
pip install -e .
cd ..

# 7. 安装 mmdetection
cd models/mmdetection
pip install -v -e .
cd ../..
```

---

## 四、验证安装

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查其他包
python -c "import cv2, mmcv, mmdet, pyrealsense2, open3d; print('所有包安装成功！')"
```

---

## 五、可能的问题

### 问题 1：mmcv 版本不兼容

**解决方案**：
```bash
pip uninstall mmcv mmcv-full -y
pip install mmcv -U
```

### 问题 2：pointnet2 编译失败

**解决方案**：
```bash
# 确保安装了 CUDA 工具
conda install cudatoolkit

# 重新编译
cd graspnet-baseline/pointnet2
python setup.py install
```

### 问题 3：mmdet 路径错误

**解决方案**：
```bash
# 直接重新安装 mmdetection
cd models/mmdetection
pip install -v -e .
```

---

## 六、最简单的方式（如果不想改版本）

如果不想升级 PyTorch，**继续使用 Python 3.8**：

```bash
# 创建 Python 3.8 环境
conda create -n grasp_zy python=3.8 -y
conda activate grasp_zy

# 直接安装（使用现有的 requirements.txt）
pip install -r requiments/requirements.txt
```

---

**总结**：Python 3.10 需要升级 PyTorch 到 2.0+，这可能会影响代码兼容性。建议先测试，如果有问题可以回退到 Python 3.8。
