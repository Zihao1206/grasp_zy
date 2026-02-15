""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='/home/zh/zh/graspnet-baseline/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--dump_dir', type=str, default='logs/dump_rs', help='Directory to dump results')
parser.add_argument('--camera', type=str, default='realsense', help='Camera type')
parser.add_argument('--dataset_root', type=str, default='/data/Benchmark/graspnet', help='Root directory for dataset')
parser.add_argument('--log_dir', type=str, default='logs/log_rs', help='Directory to save logs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size [default: 2]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    # color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    color = np.array(Image.open(os.path.join(data_dir, 'zy_488r.jpg')).convert('RGB'), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'zy_488d.tiff')))
    
    # 使用物体mask（如果有的话），否则使用workspace_mask
    object_mask_path = os.path.join(data_dir, 'object_mask.png')
    if os.path.exists(object_mask_path):
        # 使用物体mask（通过空场景减法生成）
        print("Using object_mask.png")
        mask = np.array(Image.open(object_mask_path).convert('L'))
    else:
        # 回退到workspace_mask
        print("Using workspace_mask.png (object_mask.png not found)")
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')).convert('L'))
        mask = workspace_mask
    
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    meta = scio.loadmat(os.path.join(data_dir, 'meta_modified.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    # camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    print("Depth shape:", depth.shape)
    print("Color shape:", color.shape)
    print("Workspace mask shape:", workspace_mask.shape)
    print("Intrinsic matrix shape:", intrinsic.shape)
    print("Factor depth:", factor_depth)
    print("Depth value range: min={}, max={}, mean={}".format(depth.min(), depth.max(), depth.mean()))
    print("Workspace mask value range: min={}, max={}, mean={}".format(workspace_mask.min(), workspace_mask.max(), workspace_mask.mean()))
    
    # 检查深度值的分布
    depth_nonzero = depth[depth > 0]
    if len(depth_nonzero) > 0:
        print("Non-zero depth value range: min={}, max={}, mean={}".format(depth_nonzero.min(), depth_nonzero.max(), depth_nonzero.mean()))
    
    camera = CameraInfo(depth.shape[1], depth.shape[0], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    print("Cloud shape:", cloud.shape)
    print("Cloud value range: min={}, max={}, mean={}".format(cloud.min(), cloud.max(), cloud.mean()))

    # get valid points
    # 修复：抓取姿态在边框上的问题，原因是工作空间掩码包含了边框区域
    # 需要腐蚀工作空间掩码，去除边缘区域
    # 使用简单的numpy操作来腐蚀掩码，避免依赖cv2
    workspace_mask_eroded = workspace_mask.copy()
    # 将边缘像素设为0
    border_size = 5  # 边框大小
    # workspace_mask_eroded[:border_size+100, :] = 0  # 上边框
    # workspace_mask_eroded[-border_size-70:, :] = 0  # 下边框
    # workspace_mask_eroded[:, :border_size] = 0  # 左边框
    # workspace_mask_eroded[:, -border_size-30:] = 0  # 右边框
    
    mask = (workspace_mask_eroded > 0) & (depth > 0)
    mask = mask.reshape(-1)  # 确保 mask 是一维的
    print("Valid points before mask:", len(cloud.reshape(-1, 3)))
    print("Valid points after mask:", mask.sum())
    cloud_masked = cloud.reshape(-1, 3)[mask]
    color_masked = color.reshape(-1, 3)[mask]
    
    # 检查点云的范围
    if len(cloud_masked) > 0:
        print("Cloud masked range: x=[{}, {}], y=[{}, {}], z=[{}, {}]".format(
            cloud_masked[:, 0].min(), cloud_masked[:, 0].max(),
            cloud_masked[:, 1].min(), cloud_masked[:, 1].max(),
            cloud_masked[:, 2].min(), cloud_masked[:, 2].max()
        ))

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs].astype(np.float32)
    color_sampled = color_masked[idxs].astype(np.float32)

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = torch.from_numpy(color_sampled).to(device)

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    print("Grasp predictions shape:", grasp_preds[0].shape if grasp_preds[0] is not None else "None")
    gg_array = grasp_preds[0].detach().cpu().numpy()
    print("Grasp array shape:", gg_array.shape)
    print("Number of grasps:", len(gg_array))
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def filter_grasps_by_position(gg, cloud, position_threshold=0.15):
    """
    根据抓取位置过滤抓取框
    只保留靠近物体中心的抓取
    
    Args:
        gg: GraspGroup，抓取组
        cloud: 点云
        position_threshold: 位置阈值，只保留距离点云中心在这个范围内的抓取
    
    Returns:
        filtered_gg: 过滤后的抓取组
    """
    if len(gg) == 0:
        return gg
    
    # 计算点云的中心
    cloud_points = np.array(cloud.points)
    cloud_center = np.mean(cloud_points, axis=0)
    print(f"\nCloud center: {cloud_center}")
    
    # 获取所有抓取的位置
    grasp_positions = gg.translations  # 抓取的中心位置
    
    # 计算每个抓取位置到点云中心的距离
    distances = np.linalg.norm(grasp_positions - cloud_center, axis=1)
    
    # 只保留距离中心较近的抓取
    # 使用百分位数来动态确定阈值
    distance_threshold = np.percentile(distances, 70)  # 保留距离中心70%范围内的抓取
    print(f"Distance threshold (70th percentile): {distance_threshold:.4f}")
    
    # 过滤抓取
    mask = distances <= distance_threshold
    filtered_gg = gg[mask]
    
    print(f"Filtered {len(gg)} grasps -> {len(filtered_gg)} grasps")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    
    return filtered_gg

def vis_grasps(gg, cloud):
    # 按分数排序
    gg.sort_by_score()
    
    # 打印前10个抓取的分数
    print("\nTop 10 grasp scores:")
    for i in range(min(10, len(gg))):
        print(f"  Grasp {i}: score = {gg[i].score:.4f}")
    
    # 应用NMS（非极大值抑制）来去除重叠的抓取
    gg.nms()
    print(f"\nAfter NMS: {len(gg)} grasps remaining")
    
    # 根据位置过滤抓取，去除边缘抓取
    gg_filtered = filter_grasps_by_position(gg, cloud, position_threshold=0.15)
    
    # 根据分数设置不同的阈值来过滤抓取
    # 只显示分数高于阈值的抓取
    score_threshold = 0.3  # 降低阈值以显示更多抓取
    high_score_gg = gg_filtered[gg_filtered.scores > score_threshold]
    print(f"Grasps with score > {score_threshold}: {len(high_score_gg)}")
    
    # 如果没有高分抓取，显示前N个
    if len(high_score_gg) == 0:
        print("No grasps above threshold, showing top grasps")
        num_to_show = min(50, len(gg_filtered))
        high_score_gg = gg_filtered[:num_to_show]
    
    # 可视化
    grippers = high_score_gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    print("Point cloud shape:", end_points['point_clouds'].shape)
    print("Number of valid points:", len(np.array(cloud.points)))
    gg = get_grasps(net, end_points)
    print("Number of grasps before collision detection:", len(gg))
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
        print("Number of grasps after collision detection:", len(gg))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    # data_dir = 'doc/example_data'
    data_dir = 'doc/try_data'
    demo(data_dir)
