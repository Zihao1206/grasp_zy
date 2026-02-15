# AGENTS.md

This file provides guidance to agents when working with code in this repository.

注意，请用中文回答或者编写文档

## 项目特定命令

### 必须编译的自定义 CUDA 扩展
```bash
# pointnet2 扩展 (来自 votenet)
cd pointnet2
python setup.py install
cd ..

# KNN 扩展 (来自 pytorch_knn_cuda)
cd knn
python setup.py install
cd ..
```

### 训练命令
```bash
# RealSense 相机数据训练
CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root /data/Benchmark/graspnet

# Kinect 相机数据训练
CUDA_VISIBLE_DEVICES=0 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet
```

### 测试命令
```bash
# RealSense 模型测试
CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root /data/Benchmark/graspnet

# 快速推理(禁用碰撞检测)
CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root /data/Benchmark/graspnet --collision_thresh -1
```

### 演示命令
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path checkpoint-rs.tar
```

### 生成 tolerance 标签
```bash
cd dataset
python generate_tolerance_label.py --dataset_root /data/Benchmark/graspnet --num_workers 50
# 或运行脚本
sh command_generate_tolerance_label.sh
```

## 代码架构关键发现

### 导入路径处理
所有脚本使用特殊的 ROOT_DIR 模式来导入模块：
```python
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
```

### 模型架构
- **两阶段架构**：GraspNetStage1 (视角估计) + GraspNetStage2 (抓取生成)
- **相机特定模型**：checkpoint-rs.tar (RealSense) 和 checkpoint-kn.tar (Kinect) 不通用
- **碰撞检测**：ModelFreeCollisionDetector 使用 voxel_size=0.01

### 数据集特殊处理
- **跳过对象索引**：在 `load_grasp_labels()` 中明确跳过索引 18 (`if i == 18: continue`)
- **Tolerance 标签**：不包含在原始数据集中，必须单独生成或下载
- **数据分割**：train (0-99), test_seen (100-129), test_similar (130-159), test_novel (160-189)

### 关键参数
- **碰撞阈值**：`collision_thresh=0.01` (评估), `-1` (快速推理禁用)
- **点数**：默认 20000 个点
- **视角数**：默认 300 个视角
- **抓取表示**：12 个角度 × 4 个深度 = 48 种抓取配置

### 非标准实践
- **数据加载**：使用自定义的 `collate_fn` 处理嵌套列表结构
- **标签处理**：`process_grasp_labels()` 动态生成训练标签
- **可见性过滤**：`remove_invisible_grasp_points()` 移除被遮挡的抓取点
