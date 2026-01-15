#!/usr/bin/env python3
"""
测试目标检测服务
"""
import rclpy
from rclpy.node import Node
from grasp_interfaces.srv import DetectObjects
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys


class DetectionTester(Node):
    def __init__(self):
        super().__init__('detection_tester')
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        # 订阅相机图像
        self.image_sub = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # 创建检测服务客户端
        self.detect_client = self.create_client(DetectObjects, 'detect_objects')
        
        self.get_logger().info('检测测试节点已启动')
    
    def image_callback(self, msg):
        """图像回调"""
        self.latest_image = msg
    
    def test_detection(self):
        """测试检测服务"""
        print('\n' + '=' * 60)
        print('测试目标检测服务')
        print('=' * 60)
        
        # 等待图像
        print('\n等待相机图像...')
        timeout = 10
        for i in range(timeout * 10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is not None:
                break
        
        if self.latest_image is None:
            print('✗ 未收到图像（10秒超时）')
            print('  请确保相机节点正在运行: ros2 run grasp_vision camera_node')
            return False
        
        print('✓ 已收到图像')
        
        # 等待检测服务
        print('\n等待检测服务...')
        if not self.detect_client.wait_for_service(timeout_sec=10.0):
            print('✗ 检测服务不可用（10秒超时）')
            print('  请确保检测服务正在运行: ros2 run grasp_vision detection_server')
            return False
        
        print('✓ 检测服务已连接')
        
        # 调用检测服务
        print('\n调用检测服务...')
        request = DetectObjects.Request()
        request.color_image = self.latest_image
        
        future = self.detect_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        
        if not future.done():
            print('✗ 检测请求超时（30秒）')
            return False
        
        response = future.result()
        
        if not response.success:
            print(f'✗ 检测失败: {response.message}')
            return False
        
        # 显示结果
        print(f'\n✓ 检测成功: {response.message}')
        print('\n检测结果:')
        print('-' * 60)
        
        result = response.result
        print(f'检测到 {len(result.class_names)} 个物体:')
        for i, class_name in enumerate(result.class_names):
            score = result.scores[i]
            print(f'  [{i+1}] {class_name:15s} - 置信度: {score:.3f}')
        
        # 可视化结果
        try:
            # 转换图像
            color_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            
            # 反序列化掩码
            masks_shape = result.masks_shape
            masks_data = np.array(result.masks_data, dtype=np.uint8)
            
            if len(masks_data) > 0 and len(masks_shape) > 0:
                masks = masks_data.reshape(masks_shape)
                
                # 绘制结果
                vis_img = color_img.copy()
                
                # 为每个物体绘制掩码和标签
                for i, class_name in enumerate(result.class_names):
                    # 获取掩码
                    mask = masks[i]
                    
                    # 随机颜色
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    
                    # 绘制掩码
                    vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5
                    
                    # 找到掩码中心
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        # 绘制标签
                        label = f'{class_name} {result.scores[i]:.2f}'
                        cv2.putText(vis_img, label, (cx-50, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.circle(vis_img, (cx, cy), 5, color, -1)
                
                # 保存可视化结果
                save_path = '/tmp/ros2_detection_result.jpg'
                cv2.imwrite(save_path, vis_img)
                print(f'\n✓ 可视化结果已保存: {save_path}')
                
                # 显示图像
                cv2.imshow('Detection Result', vis_img)
                print('\n按任意键关闭窗口...')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            print(f'⚠ 可视化失败: {e}')
        
        print('\n' + '=' * 60)
        print('✓✓✓ 目标检测测试通过！')
        print('=' * 60)
        return True


def main(args=None):
    rclpy.init(args=args)
    
    print('=' * 60)
    print('目标检测服务测试')
    print('=' * 60)
    print('\n⚠️  请确保以下节点正在运行:')
    print('  1. ros2 run grasp_vision camera_node')
    print('  2. ros2 run grasp_vision detection_server')
    print('\n等待...\n')
    
    node = DetectionTester()
    
    success = node.test_detection()
    
    node.destroy_node()
    rclpy.shutdown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

