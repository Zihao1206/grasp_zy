#!/usr/bin/env python3
"""
GraspNet 包装器模块
用于集成到现有的抓取系统中，替换 GQCNN
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import scipy.io as scio
import open3d as o3d

# 添加 GraspNet 路径
GRASPNET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(GRASPNET_ROOT)
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


class GraspNetWrapper:
    """
    GraspNet 包装器类
    提供与原有 GQCNN 类似的接口
    """
    
    def __init__(self, checkpoint_path=None, num_point=20000, num_view=300,
                 collision_thresh=0.01, voxel_size=0.01, device=None):
        """
        初始化 GraspNet 模型
        
        Args:
            checkpoint_path: 模型权重路径
            num_point: 点云采样数量
            num_view: 视角数量
            collision_thresh: 碰撞检测阈值
            voxel_size: 体素大小
            device: 计算设备 ('cuda' or 'cpu')
        """
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.net = self._init_model(checkpoint_path)
        
        # 相机内参（需要根据实际情况设置）
        self.intrinsic = None
        self.factor_depth = 1000.0
        
    def _init_model(self, checkpoint_path):
        """初始化 GraspNet 模型"""
        print("Initializing GraspNet...")
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(GRASPNET_ROOT, 'checkpoint-rs.tar')
        
        # 初始化网络
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], 
                       is_training=False)
        
        net.to(self.device)
        
        # 加载权重
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"-> loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        net.eval()
        return net
    
    def set_camera_params(self, intrinsic, factor_depth=1000.0):
        """
        设置相机参数
        
        Args:
            intrinsic: 3x3 内参矩阵
            factor_depth: 深度缩放因子
        """
        self.intrinsic = np.array(intrinsic)
        self.factor_depth = factor_depth
    
    def detect_grasps(self, depth_img, color_img, mask_img, num_grasps=10):
        """
        检测抓取姿态
        
        Args:
            depth_img: 深度图 (H, W)，单位：米
            color_img: 彩色图 (H, W, 3)，范围：[0, 1]
            mask_img: 物体mask (H, W)，255表示物体区域
            num_grasps: 返回的抓取数量
        
        Returns:
            grasp_list: 抓取列表，每个元素为 (row, col, angle, width, height)
            best_grasp: 最佳抓取 (row, col, angle, width, height)
        """
        if self.intrinsic is None:
            raise ValueError("Camera parameters not set. Call set_camera_params() first.")
        
        # 1. 准备数据
        depth = np.array(depth_img).astype(np.float32)  # 转换为毫米
        print(depth)
        color = np.array(color_img).astype(np.float32)
        
        # 确保深度图没有无效值
        depth = np.clip(depth, 0, 10000)
        
        # 2. 创建相机对象
        camera = CameraInfo(depth.shape[1], depth.shape[0], 
                           self.intrinsic[0,0], self.intrinsic[1,1], 
                           self.intrinsic[0,2], self.intrinsic[1,2], self.factor_depth)
        
        # 3. 生成点云
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        
        # 4. 应用 mask
        mask = (mask_img > 0) & (depth > 0)
        mask = mask.reshape(-1)
        cloud_masked = cloud.reshape(-1, 3)[mask]
        color_masked = color.reshape(-1, 3)[mask]
        
        # 5. 检查点云是否为空
        if len(cloud_masked) == 0:
            print("Warning: No valid points in mask")
            return [], None
        
        # 6. 采样点
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = cloud_masked[idxs].astype(np.float32)
        
        # 7. 准备网络输入
        end_points = {}
        cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_sampled_tensor = cloud_sampled_tensor.to(self.device)
        end_points['point_clouds'] = cloud_sampled_tensor
        
        # 8. 网络推理
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        
        # 9. 处理预测结果
        if len(grasp_preds) == 0 or grasp_preds[0] is None:
            print("Warning: No grasp predictions")
            return [], None
        
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        # 10. 碰撞检测
        if self.collision_thresh > 0:
            cloud_points = np.array(cloud_masked)
            mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=self.voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, 
                                              collision_thresh=self.collision_thresh)
            gg = gg[~collision_mask]
        
        # 11. 检查是否有剩余抓取
        if len(gg) == 0:
            print("Warning: All grasps filtered by collision detection")
            return [], None
        
        # 12. 排序并选择最佳抓取
        gg.sort_by_score()
        gg = gg[:50]  # 取前50个
        
        # 13. 转换为像素坐标
        grasp_list = self._convert_to_pixel_coords(gg, mask_img)
        
        # 14. 返回结果
        if len(grasp_list) == 0:
            print("Warning: No valid grasps inside mask")
            # 返回第一个抓取（可能在mask外）
            best_grasp = self._grasp_to_tuple(gg[0], self.intrinsic)
            return [], best_grasp
        
        best_grasp = grasp_list[0]
        return grasp_list, best_grasp
    
    def _convert_to_pixel_coords(self, gg, mask_img):
        """
        将抓取从3D坐标转换为像素坐标
        
        Args:
            gg: GraspGroup
            mask_img: mask图像
        
        Returns:
            grasp_list: 像素坐标下的抓取列表
        """
        grasp_list = []
        
        for grasp in gg:
            # 抓取中心在相机坐标系下的位置
            grasp_center_cam = grasp.translation
            
            # 投影到图像平面
            x_cam, y_cam, z_cam = grasp_center_cam
            
            if z_cam <= 0:
                continue
            
            # 使用内参投影
            column = int((x_cam * self.intrinsic[0,0] / z_cam) + self.intrinsic[0,2])
            row = int((y_cam * self.intrinsic[1,1] / z_cam) + self.intrinsic[1,2])
            
            # 检查是否在图像范围内
            if not (0 <= row < mask_img.shape[0] and 0 <= column < mask_img.shape[1]):
                continue
            
            # 检查是否在mask内
            if mask_img[row, column] == 0:
                continue
            
            # 获取抓取角度（绕Z轴旋转）
            rotation_matrix = grasp.rotation_matrix
            angle = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            
            # 获取抓取宽度（转换为像素）
            width_meters = grasp.width
            width_pixels = int(width_meters * self.intrinsic[0,0] / z_cam)
            
            # 确保宽度合理
            width_pixels = max(20, min(width_pixels, 200))
            
            # 添加到列表
            grasp_list.append((row, column, angle, width_pixels, width_pixels/2))
            
            if len(grasp_list) >= 10:  # 最多返回10个
                break
        
        return grasp_list
    
    def _grasp_to_tuple(self, grasp, intrinsic):
        """将抓取转换为tuple格式"""
        grasp_center_cam = grasp.translation
        x_cam, y_cam, z_cam = grasp_center_cam
        
        column = int((x_cam * intrinsic[0,0] / z_cam) + intrinsic[0,2])
        row = int((y_cam * intrinsic[1,1] / z_cam) + intrinsic[1,2])
        
        rotation_matrix = grasp.rotation_matrix
        angle = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        
        width_meters = grasp.width
        width_pixels = int(width_meters * intrinsic[0,0] / z_cam)
        width_pixels = max(20, min(width_pixels, 200))
        
        return (row, column, angle, width_pixels, width_pixels/2)
    
    def batch_detect(self, depth_imgs, color_imgs, mask_imgs):
        """
        批量检测抓取（用于测试）
        
        Args:
            depth_imgs: 深度图列表
            color_imgs: 彩色图列表
            mask_imgs: mask列表
        
        Returns:
            results: 抓取结果列表
        """
        results = []
        for depth, color, mask in zip(depth_imgs, color_imgs, mask_imgs):
            grasp_list, best_grasp = self.detect_grasps(depth, color, mask)
            results.append((grasp_list, best_grasp))
        return results


# 测试函数
def test_graspnet_wrapper():
    """测试 GraspNet 包装器"""
    print("Testing GraspNetWrapper...")
    
    # 创建包装器
    wrapper = GraspNetWrapper(
        checkpoint_path='checkpoint-rs.tar',
        num_point=10000,  # 减少点数以提高速度
        collision_thresh=0.01
    )
    
    # 设置相机参数（RealSense）
    intrinsic = np.array([
        [392.25048828, 0, 320.16729736],
        [0, 392.25048828, 242.32826233],
        [0, 0, 1]
    ])
    wrapper.set_camera_params(intrinsic, factor_depth=1000.0)
    
    # 加载测试数据
    try:
        depth_path = 'doc/try_data/zy_3d.tiff'
        color_path = 'doc/try_data/zy_3r.jpg'
        mask_path = 'doc/try_data/workspace_mask_0d.png'
        
        if os.path.exists(depth_path) and os.path.exists(color_path):
            depth_img = np.array(Image.open(depth_path))
            color_img = np.array(Image.open(color_path)) / 255.0
            mask_img = np.array(Image.open(mask_path))
            
            # 检测抓取
            grasp_list, best_grasp = wrapper.detect_grasps(depth_img, color_img, mask_img)
            
            print(f"Detected {len(grasp_list)} grasps")
            if best_grasp is not None:
                print(f"Best grasp: {best_grasp}")
            else:
                print("No grasp detected")
        else:
            print("Test data not found")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    test_graspnet_wrapper()