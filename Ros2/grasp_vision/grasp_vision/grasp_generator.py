#!/usr/bin/env python3
"""
抓取姿态生成服务节点
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import GenerateGrasp
from grasp_interfaces.msg import GraspPose
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Vector3
import sys
import os
import numpy as np
import cv2
import torch
import math

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from models.gqcnn_server.augment_cnn import AugmentCNN
import models.genotypes as gt
from utils.utils import scale_coords
from transforms3d.euler import mat2euler
import config as cfg


class GraspGenerator(Node):
    def __init__(self):
        super().__init__('grasp_generator')
        
        # 声明参数
        self.declare_parameter('gene_file', 'doc/single_new.txt')
        self.declare_parameter('cornell_data', 'dataset/cornell.data')
        self.declare_parameter('model_weights', 'models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        
        # 获取参数
        gene_file = self.get_parameter('gene_file').value
        cornell_data = self.get_parameter('cornell_data').value
        model_weights = self.get_parameter('model_weights').value
        device = self.get_parameter('device').value
        self.cam_width = self.get_parameter('camera_width').value
        self.cam_height = self.get_parameter('camera_height').value
        
        # 转换为绝对路径
        gene_file = os.path.join(project_root, gene_file)
        cornell_data = os.path.join(project_root, cornell_data)
        model_weights = os.path.join(project_root, model_weights)
        
        # 初始化设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f'使用设备: {self.device}')
        
        # 加载抓取模型
        self.get_logger().info('加载抓取模型...')
        with open(gene_file, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                gene = line
                getypr = gt.from_str(gene)
        
        self.model = AugmentCNN(cornell_data, 100, 4, 8, 5, False, getypr).to(self.device)
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        self.model.eval()
        self.get_logger().info('抓取模型加载完成')
        
        # 相机内参和变换矩阵
        self.setup_camera_params()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 创建服务
        self.srv = self.create_service(
            GenerateGrasp,
            'generate_grasp',
            self.generate_callback
        )
        
        self.get_logger().info('抓取姿态生成服务已启动')
    
    def setup_camera_params(self):
        """设置相机参数"""
        # RealSense D435内参
        self.camera_intr = np.array([
            [607.4404907226562, 0.0, 321.5382080078125],
            [0.0, 607.4632568359375, 244.00479125976562],
            [0.0, 0.0, 1.0]
        ])
        
        # 相机到基座的变换矩阵
        self.Tcam2base = np.array(cfg.Tcam2base)
        self.Rbase2cam = self.Tcam2base.T[0:3, 0:3]
    
    def get_coordinate(self, x, y, depth_img):
        """将像素坐标转换为相机坐标"""
        z = depth_img[int(y), int(x)]
        xpc, ypc, zpc = np.linalg.inv(self.camera_intr) @ np.array([x, y, 1]) * z
        return xpc, ypc, zpc
    
    def in_paint(self, depth_img):
        """深度图修复"""
        depth_img = np.array(depth_img).astype(np.float32)
        depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img == 0).astype(np.uint8)
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
        depth_img = depth_img[1:-1, 1:-1]
        return depth_img
    
    def to_tensor(self, depth_img, color_img):
        """转换为张量"""
        # 深度归一化
        depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)
        
        # 颜色归一化
        color_img = color_img[..., ::-1]  # BGR to RGB
        color_img = np.ascontiguousarray(color_img, dtype=np.float32)
        color_img /= 255
        color_img -= color_img.mean()
        
        # 合并
        img_in = np.concatenate((np.expand_dims(depth_img, 2), color_img), 2)
        img_in = np.transpose(img_in, (2, 0, 1))
        img_in = np.expand_dims(img_in, 0).astype(np.float32)
        img_in = torch.from_numpy(img_in).to(self.device)
        
        return img_in
    
    def letterbox(self, img, depth, height=100):
        """调整图像大小并填充"""
        color = (127.5, 127.5, 127.5)
        shape = img.shape[:2]
        ratio = float(height) / max(shape)
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
        dw = (height - new_shape[0]) / 2
        dh = (height - new_shape[1]) / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        depth = cv2.resize(depth, new_shape, interpolation=cv2.INTER_AREA)
        depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_REFLECT)
        
        return img, depth, ratio, dw, dh
    
    def generate_grasp_yolo(self, img_in, rgb_image, num, mask_img):
        """生成抓取姿态"""
        output = self.model.forward(img_in)
        output = output[output[:, :, 4] > 0]  # 移除低质量抓取
        
        if len(output) > 0:
            output = output[output[:, 4].sort(descending=True)[1]]
        
        output = output.detach().cpu().numpy()
        scale_coords(100, output, rgb_image.shape)
        
        grasp_select = []
        for k in range(num):
            if k >= len(output):
                break
            index = output[k, :]
            x, y = index[0], index[1]
            
            if mask_img[int(y), int(x)] == 255:
                row = int(index[1])
                column = int(index[0])
                angle = index[3]
                width = index[2]
                height = index[2] / 2
                quality = index[4]
                grasp_select.append((row, column, angle, width, height, quality))
                break
        
        if len(grasp_select) == 0 and len(output) > 0:
            self.get_logger().info("输出最高质量抓取!")
            index = output[0, :]
            row = int(index[1])
            column = int(index[0])
            angle = index[3]
            width = index[2]
            height = index[2] / 2
            quality = index[4]
            grasp_select.append((row, column, angle, width, height, quality))
        
        return grasp_select
    
    def grasp_img2real(self, color_img, depth_img, grasp, visualize=False):
        """将抓取姿态从图像坐标转换为机器人坐标"""
        row, column, angle, width, height, quality = grasp
        
        # 计算抓取矩形
        s = math.sin(angle)
        c = math.cos(angle)
        x1 = column - width / 2 * c
        x2 = column + width / 2 * c
        y1 = row + width / 2 * s
        y2 = row - width / 2 * s
        rect = np.array([
            [x1 - width / 4 * s, y1 - width / 4 * c],
            [x2 - width / 4 * s, y2 - width / 4 * c],
            [x2 + width / 4 * s, y2 + width / 4 * c],
            [x1 + width / 4 * s, y1 + width / 4 * c],
        ]).astype(int)
        
        # 计算夹爪宽度
        point1 = np.array(self.get_coordinate(rect[0][0], rect[0][1], depth_img))
        point2 = np.array(self.get_coordinate(rect[1][0], rect[1][1], depth_img))
        width_gripper = np.linalg.norm(point1[0:2] - point2[0:2], 2)
        
        # 获取深度并转换到基座坐标系
        z = depth_img[row, column] + 0.02
        xpc, ypc, zpc = np.linalg.inv(self.camera_intr) @ np.array([column + 70, row, 1]) * z
        
        Pobj2base = self.Tcam2base @ np.array([[xpc], [ypc], [zpc], [1]])
        Pobj2base = Pobj2base.squeeze()[:-1]
        
        # 计算旋转
        Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle - np.pi / 1000], dtype=np.float64))[0]).dot(
            self.Tcam2base[0:3, 0:3])
        
        # 边缘补偿
        t_tcp_flange = np.array([0, 0, 0.2])
        tcp_compensate = np.array([0, 0, 0.018])
        slope_angle = np.pi / 8
        column_left, column_right = 80, 480
        row_up, row_down = 120, 292
        
        slope_flag = False
        if column < column_left:
            if row < row_up:
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, slope_angle, 0], dtype=np.float64))[0])
                slope_flag = True
            elif row > row_down:
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, -slope_angle, 0], dtype=np.float64))[0])
                slope_flag = True
            else:
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, 0, 0], dtype=np.float64))[0])
                slope_flag = True
        elif column > column_right:
            if row < row_up:
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, slope_angle, 0], dtype=np.float64))[0])
                slope_flag = True
            elif row > row_down:
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, -slope_angle, 0], dtype=np.float64))[0])
                slope_flag = True
            else:
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, 0, 0], dtype=np.float64))[0])
                slope_flag = True
        elif row < row_up:
            R_rot = np.array(cv2.Rodrigues(np.array([0, slope_angle, 0], dtype=np.float64))[0])
            slope_flag = True
        elif row > row_down:
            R_rot = np.array(cv2.Rodrigues(np.array([0, -slope_angle, 0], dtype=np.float64))[0])
            slope_flag = True
        else:
            R_rot = np.eye(3)
            t_tcp_flange = np.array([0, 0, 0])
            tcp_compensate = np.array([0, 0, 0])
        
        if slope_flag:
            Robj2base = R_rot.dot(Robj2base)
            t_compesate = Robj2base @ t_tcp_flange
            t_z_translation = Robj2base @ tcp_compensate
            Pobj2base = Pobj2base - [t_compesate[0], t_compesate[1], 0] + t_z_translation
        
        # 可视化
        if visualize:
            color_img_vis = color_img.copy()
            color_img_vis = cv2.circle(color_img_vis, (column, row), 3, (0, 0, 255), -1)
            color_img_vis = cv2.arrowedLine(color_img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                                            thickness=2)
            color_img_vis = cv2.arrowedLine(color_img_vis, (int(x2), int(y2)), (int(x1), int(y1)), (0, 255, 0),
                                            thickness=2)
            cv2.imwrite('/tmp/grasp_visualization.png', color_img_vis)
        
        return Pobj2base, Robj2base, width_gripper, angle, slope_flag, quality
    
    def generate_callback(self, request, response):
        """生成抓取姿态回调"""
        try:
            # 转换图像
            color_img = self.bridge.imgmsg_to_cv2(request.color_image, desired_encoding='bgr8')
            depth_img = self.bridge.imgmsg_to_cv2(request.depth_image, desired_encoding='passthrough')
            
            # 图像预处理
            depth_img = self.in_paint(depth_img)
            depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0) / 1000
            depth_img = depth_img[:, 80:560]
            color_img = color_img[:, 80:560, :]
            
            color_img_raw = color_img.copy()
            depth_img_raw = depth_img.copy()
            
            # 反序列化掩码
            masks_shape = request.detection_result.masks_shape
            masks_data = np.array(request.detection_result.masks_data, dtype=np.uint8)
            masks = masks_data.reshape(masks_shape)
            labels = request.detection_result.labels
            class_names = request.detection_result.class_names
            
            # 生成目标物体的掩码
            black_image = np.zeros((color_img.shape[0], color_img.shape[1]), dtype=np.uint8)
            for j in range(len(labels)):
                if class_names[j] == request.target_label:
                    mask_circle = np.uint8(masks[j])
                    black_image[mask_circle > 0] = 255
            
            # 调整图像大小
            color_img_resized, depth_img_resized, ratio, padw, padh = self.letterbox(color_img, depth_img, 100)
            
            # 转换为张量
            img_in = self.to_tensor(depth_img_resized, color_img_resized)
            
            # 生成抓取姿态
            self.get_logger().info(f'生成抓取姿态: target={request.target_label}, top_k={request.top_k}')
            grasps = self.generate_grasp_yolo(img_in, color_img_raw, request.top_k, black_image)
            
            if len(grasps) == 0:
                response.success = False
                response.message = '未找到有效抓取姿态'
                return response
            
            # 转换为机器人坐标
            for grasp in grasps:
                Pobj2base, Robj2base, width_gripper, angle, slope_flag, quality = self.grasp_img2real(
                    color_img_raw, depth_img_raw, grasp, request.visualize
                )
                
                # 构建抓取姿态消息
                grasp_msg = GraspPose()
                grasp_msg.header.stamp = self.get_clock().now().to_msg()
                grasp_msg.header.frame_id = 'base_link'
                
                grasp_msg.row = grasp[0]
                grasp_msg.column = grasp[1]
                grasp_msg.angle = angle
                grasp_msg.width = grasp[3]
                grasp_msg.height = grasp[4]
                
                grasp_msg.position = Point(x=float(Pobj2base[0]), y=float(Pobj2base[1]), z=float(Pobj2base[2]))
                
                gesture = mat2euler(Robj2base, axes='sxyz')
                grasp_msg.orientation = Vector3(x=float(gesture[0]), y=float(gesture[1]), z=float(gesture[2]))
                
                grasp_msg.gripper_width = float(width_gripper)
                grasp_msg.quality = float(quality)
                grasp_msg.slope_flag = slope_flag
                
                response.grasp_poses.append(grasp_msg)
            
            response.success = True
            response.message = f'成功生成{len(grasps)}个抓取姿态'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'生成抓取姿态失败: {str(e)}'
            self.get_logger().error(response.message)
            import traceback
            traceback.print_exc()
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenerator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

