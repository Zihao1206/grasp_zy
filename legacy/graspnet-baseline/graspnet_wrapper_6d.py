#!/usr/bin/env python3
"""
GraspNet 6D 位姿包装器
直接将 GraspNet 输出转换为机械臂 6D 位姿 (x, y, z, rx, ry, rz)
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import cv2
from scipy.spatial.transform import Rotation as R

# 添加 GraspNet 路径
GRASPNET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(GRASPNET_ROOT)
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup


class GraspNet6DWrapper:
    """
    GraspNet 6D 位姿包装器
    直接输出机械臂可用的 6D 位姿
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
        
        # 相机到机械臂的变换矩阵（需要标定）
        self.Tcam2base = None
        
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
    
    def set_transform_matrix(self, Tcam2base):
        """
        设置相机到机械臂的变换矩阵
        
        Args:
            Tcam2base: 4x4 变换矩阵
        """
        self.Tcam2base = np.array(Tcam2base)
    
    def detect_grasps_6d(self, depth_img, color_img, mask_img, num_grasps=10):
        """
        检测抓取姿态并直接返回 6D 位姿
        
        Args:
            depth_img: 深度图 (H, W)，单位：毫米
            color_img: 彩色图 (H, W, 3)，范围：[0, 1]
            mask_img: 物体mask (H, W)，255表示物体区域
            num_grasps: 返回的抓取数量
        
        Returns:
            grasp_poses: 抓取位姿列表，每个元素为 (x, y, z, rx, ry, rz, width)
                        x,y,z: 位置（米）
                        rx,ry,rz: 欧拉角（弧度）
                        width: 抓取宽度（米）
        """
        if self.intrinsic is None:
            raise ValueError("Camera parameters not set. Call set_camera_params() first.")
        
        if self.Tcam2base is None:
            raise ValueError("Transform matrix not set. Call set_transform_matrix() first.")
        
        # 1. 准备数据（保持毫米单位）
        depth = np.array(depth_img).astype(np.float32)
        color = np.array(color_img).astype(np.float32)
        
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
            return []
        
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
            return []
        
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
            return []
        
        # 12. 排序并选择最佳抓取
        gg.sort_by_score()
        gg = gg[:50]  # 取前50个
        
        # 13. 转换为 6D 位姿
        grasp_poses = []
        for grasp in gg:
            # 抓取中心在相机坐标系下的位置
            position_cam = grasp.translation  # (x, y, z) in meters
            
            # 抓取方向（3x3 旋转矩阵）
            rotation_cam = grasp.rotation_matrix
            
            # 抓取宽度（米）
            width = grasp.width
            
            # 转换到机械臂基坐标系
            # 位置转换
            P_cam = np.append(position_cam, 1)
            P_base = self.Tcam2base @ P_cam
            position_base = P_base[:3]
            
            # 旋转转换
            R_base = self.Tcam2base[0:3, 0:3] @ rotation_cam
            
            # 将旋转矩阵转换为欧拉角（XYZ顺序，单位：弧度）
            r = R.from_matrix(R_base)
            euler_angles = r.as_euler('xyz')  # [rx, ry, rz]
            
            # 组合成 6D 位姿
            pose_6d = np.concatenate([position_base, euler_angles, [width]])
            
            grasp_poses.append(pose_6d)
            
            if len(grasp_poses) >= num_grasps:
                break
        
        return grasp_poses
    
    def detect_best_grasp(self, depth_img, color_img, mask_img):
        """
        检测最佳抓取位姿
        
        Args:
            depth_img: 深度图 (H, W)，单位：毫米
            color_img: 彩色图 (H, W, 3)，范围：[0, 1]
            mask_img: 物体mask (H, W)，255表示物体区域
        
        Returns:
            best_pose: 最佳抓取位姿 (x, y, z, rx, ry, rz, width)
                      如果检测失败返回 None
        """
        poses = self.detect_grasps_6d(depth_img, color_img, mask_img, num_grasps=1)
        
        if len(poses) > 0:
            return poses[0]
        else:
            print("Warning: No valid grasp detected")
            return None


# 测试函数
def test_graspnet_6d_wrapper():
    """测试 GraspNet6DWrapper"""
    print("Testing GraspNet6DWrapper...")
    
    # 创建包装器
    wrapper = GraspNet6DWrapper(
        checkpoint_path='checkpoint-rs.tar',
        num_point=10000,
        collision_thresh=0.01
    )
    
    # 设置相机参数（RealSense）
    intrinsic = np.array([
        [392.25048828, 0, 320.16729736],
        [0, 392.25048828, 242.32826233],
        [0, 0, 1]
    ])
    wrapper.set_camera_params(intrinsic, factor_depth=1000.0)
    
    # 设置变换矩阵（示例，需要实际标定）
    Tcam2base = np.array([
        [-0.01537554, -0.99988175, -0.00028888, 0.2070103],
        [ 0.9998815  ,-0.01537576 , 0.00076007,-0.03249003],
        [-0.00076442 ,-0.00027716,  0.99999967, 0.02642268],
        [0., 0., 0., 1.]
    ])
    wrapper.set_transform_matrix(Tcam2base)
    
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
            poses = wrapper.detect_grasps_6d(depth_img, color_img, mask_img, num_grasps=5)
            
            print(f"Detected {len(poses)} grasp poses:")
            for i, pose in enumerate(poses):
                print(f"  Grasp {i}: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}, "
                      f"rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}, width={pose[6]:.4f}")
            
            # 检测最佳抓取
            best_pose = wrapper.detect_best_grasp(depth_img, color_img, mask_img)
            if best_pose is not None:
                print(f"\nBest grasp: {best_pose}")
            else:
                print("No grasp detected")
        else:
            print("Test data not found")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_graspnet_6d_wrapper()