#!/usr/bin/env python3
"""
目标检测服务节点
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import DetectObjects
from grasp_interfaces.msg import DetectionResult
from cv_bridge import CvBridge
import sys
import os
import numpy as np
import torch

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

from mmdet.apis import init_detector, inference_detector
from models.nms import nms


class DetectionServer(Node):
    def __init__(self):
        super().__init__('detection_server')
        
        # 声明参数
        self.declare_parameter('config_file', 'models/mmdetection/configs/myconfig_zy.py')
        self.declare_parameter('checkpoint', 'models/weights/epoch_20_last.pth')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('nms_score_threshold', 0.8)
        self.declare_parameter('nms_iou_threshold', 0.9)
        
        # 获取参数
        config_file = self.get_parameter('config_file').value
        checkpoint = self.get_parameter('checkpoint').value
        device = self.get_parameter('device').value
        self.nms_score_thresh = self.get_parameter('nms_score_threshold').value
        self.nms_iou_thresh = self.get_parameter('nms_iou_threshold').value
        
        # 转换为绝对路径
        config_file = os.path.join(project_root, config_file)
        checkpoint = os.path.join(project_root, checkpoint)
        
        # 初始化检测模型
        self.get_logger().info('加载检测模型...')
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.det_model = init_detector(config_file, checkpoint, device=self.device)
        self.get_logger().info('检测模型加载完成')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 创建服务
        self.srv = self.create_service(
            DetectObjects,
            'detect_objects',
            self.detect_callback
        )
        
        self.get_logger().info('目标检测服务已启动')
    
    def detect_callback(self, request, response):
        """检测服务回调"""
        try:
            # 转换图像
            color_img = self.bridge.imgmsg_to_cv2(request.color_image, desired_encoding='bgr8')
            
            # 执行检测
            self.get_logger().info('执行目标检测...')
            pre = inference_detector(self.det_model, color_img)
            
            # 提取结果
            classes = self.det_model.dataset_meta['classes']
            bboxes = pre.pred_instances.bboxes.cpu().numpy()
            scores = pre.pred_instances.scores.cpu().numpy()[:, np.newaxis]
            masks = pre.pred_instances.masks.cpu().numpy()
            labels = pre.pred_instances.labels.cpu().numpy()[:, np.newaxis]
            
            # 组合预测结果
            predicts = np.concatenate((scores, bboxes, labels), axis=1)
            
            # NMS
            bboxes_nms, indics, labels_nms = nms(predicts, self.nms_score_thresh, self.nms_iou_thresh)
            masks_nms = masks[indics]
            
            # 构建响应
            response.result.header.stamp = self.get_clock().now().to_msg()
            response.result.class_names = [classes[int(label)] for label in labels_nms]
            response.result.bboxes = bboxes_nms.flatten().tolist()
            response.result.scores = bboxes_nms[:, 0].tolist()
            response.result.labels = labels_nms.astype(np.int32).tolist()
            
            # 序列化掩码
            response.result.masks_data = masks_nms.flatten().astype(np.uint8).tolist()
            response.result.masks_shape = list(masks_nms.shape)
            
            response.success = True
            response.message = f'检测到{len(labels_nms)}个物体'
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f'检测失败: {str(e)}'
            self.get_logger().error(response.message)
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectionServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

