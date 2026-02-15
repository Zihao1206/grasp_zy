# Utils - 工具库

**注意：请用中文回答或者编写文档**

---

## 概述

通用工具函数库，包含数据处理、可视化、坐标变换等辅助功能。

---

## 文件结构

```
utils/
├── utils.py              # 主工具集 (769行) - BGR/RGB转换、绘图、NMS等
├── datasets.py           # 数据集处理 (616行) - Cornell/Grasp数据加载
├── parse_config.py       # 配置解析
├── timeit.py             # 计时工具
├── torch_utils.py        # PyTorch辅助函数
├── data/                 # 数据加载器
│   ├── cornell_data.py   # Cornell数据集
│   └── grasp_data.py     # Grasp数据集
├── dataset_processing/   # 数据预处理
│   ├── grasp.py          # 抓取标注处理 (443行)
│   ├── image.py          # 图像处理
│   └── evaluation.py     # 评估指标
└── visualisation/        # 可视化工具
```

---

## 核心函数

### utils.py

```python
# BGR <-> RGB 转换 (OpenCV vs Matplotlib)
bgr_to_rgb(img), rgb_to_bgr(img)

# 绘图工具
plot_results(images, titles)  # 多图并排显示
draw_grasp(img, grasp)        # 绘制抓取点

# 非极大值抑制
nms(boxes, scores, threshold)

# 图像预处理
letterbox(img, new_shape)     # 保持比例缩放
```

### datasets.py

```python
# 仿射变换矩阵组合 (重要!)
M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
# S=Shear, T=Translation, R=Rotation

# Cornell 数据集加载
Cornell(filepath)
  .get_depth()      # 获取深度图
  .get_rgb()        # 获取RGB图
  .get_grasp()      # 获取抓取标注
```

---

## 关键约定

### 1. 坐标系
- **OpenCV**: (row, column) = (y, x)
- **NumPy**: array[row, col]
- **图像**: height × width

### 2. 颜色格式
- **OpenCV**: BGR
- **Matplotlib**: RGB
- 转换函数：`bgr_to_rgb()` / `rgb_to_bgr()`

### 3. 矩阵运算顺序
```python
# datasets.py:563 - 顺序不可更改
M = S @ T @ R  # Shear @ Translation @ Rotation
```

---

## 反模式

### 禁止
- ❌ 更改矩阵乘法顺序 `S @ T @ R`
- ❌ 直接在BGR图像上使用Matplotlib显示

### 注意
- ⚠️ `cornell_data.py` 有两个版本（根目录和data/），可能有差异

---

## 依赖

```
numpy, opencv-python, matplotlib
torch, torchvision (torch_utils.py)
scipy (datasets.py)
```
