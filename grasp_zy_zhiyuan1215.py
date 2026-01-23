import torch
import config
import camera
import numpy as np
# from models.common import post_process_output
from skimage.feature import peak_local_max
# from models.VGG_GA3 import VGG
# from model.gqcnn_server.VGG_replace import VGG
import time
import cv2
import gripper_zhiyuan as gripper
# from gqcnn_server.resNet import ResNet, Bottleneck
# from model.gqcnn_server.network import GQCNN
import tkinter as tk
from tkinter import messagebox
from tkinter import font
from utils.utils import *
from models.gqcnn_server.augment_cnn import AugmentCNN
import models.genotypes as gt
import datetime
from Robotic_Arm.rm_robot_interface import *
from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t
from mmdet.apis import init_detector, inference_detector
from models.nms import nms
from mmdet.registry import VISUALIZERS
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from transforms3d.euler import mat2euler

class CollisionDetected(Exception):
    """Raised when the robot reports a collision event during motion."""

class Grasp:
    def __init__(self, hardware=False):
        # self.robot_speed = config.robot_speed 
        self.robot_speed = 20
        self.hardware = hardware
        self.init_pose = [-0.226100, -0.00309, 0.522900, -0.010, 0.028, 2.650]
        self.camera = camera.RS(640, 480)
        if self.hardware:
            # 新API: 创建机械臂对象并连接
            self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            self.robot_handle = self.robot.rm_create_robot_arm("192.168.127.101", 8080)
            self.robot.rm_set_collision_state(5)
            self.gripper = gripper.GripperZhiyuan(self.robot)
            self.gripper.gripper_initial()
            self.init_end_pos = np.array([0.003114, 0.071823, 0.34087])
            self.init_end_ori = np.array([
                [-0.98884, 0.12647, -0.078769],
                [0.027292, -0.36599, -0.93022],
                [-0.14647, -0.92199, 0.35845]
                ])
            self.init_pose = [86, -129, 127, -0.8, 71, -81]
            self.mid_pose = [0, -129, 127, -0.7, 71, -81]
            self.mid_pose1 = [0, -129, 80, -0.7, 100, -81]  # joint6 -28
            self.robot.rm_movej(self.init_pose, self.robot_speed, 0, 0, 1)
            
            self.lift2init_pose = [65, -129, 127, -0.7, 77, 1] ## 回转到初始状态
            self.place_mid_pose = [65, -129, 60, 0, 121, 1] ## 机械臂第三关节前倾
            self.place_mid_pose2 = [69, -129, 60, 0, 9, 1] ## 机械臂展开
            self.place_last_pose = [69, -104, 38, -2, 9, 1] ## 机械臂向下放置物体
            
        self.Tcam2base = np.array([
            [-0.01537554, -0.99988175, -0.00028888,0.2070103],
            [ 0.9998815  ,-0.01537576 , 0.00076007,-0.03249003],
            [-0.00076442 ,-0.00027716,  0.99999967,0.02642268],
            [0., 0., 0., 1., ]
            ])
        self.Rbase2cam = self.Tcam2base.T[0:3,0:3]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file_path = 'doc/single_new.txt'
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                gene = line
                getypr = gt.from_str(gene)
        model = AugmentCNN('dataset/cornell.data', 100, 4, 8, 5, False, getypr).cuda()

        model.load_state_dict(torch.load('models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00', map_location=self.device)) # scatch
        # 目标框检测模型加载
        config_file = 'models/mmdetection/configs/myconfig_zy.py'## 看一下
        check_point = 'models/weights/epoch_20_last.pth'
        inferencer = init_detector(config_file, check_point, device=self.device)
        self.det_model = inferencer

        self.net = model.eval()
        self.net.eval()
        self.rgb_include = 0

    def change_robot_speed(self, robot_speed):
        if robot_speed < 0:
            robot_speed = 0
        if robot_speed > 50:
            robot_speed = 50
        self.robot_speed = robot_speed

    def init_gripper(self):
        self.gripper.gripper_initial()
        
    def _movej_safe(self, pose, speed, *args, **kwargs):
        """安全移动函数，检测碰撞并抛出异常"""
        tag = self.robot.rm_movej(pose, speed, 0, 0, 1)
        if tag == 0:
            return tag, pose  # 成功返回
        # 检查碰撞码或关键词
        if "100D4" in str(tag) or "collision" in str(tag).lower() or "碰撞" in str(tag):
            raise CollisionDetected(f"碰撞检测到: {tag}")
        raise RuntimeError(f"MoveJ失败: {tag}")
        
    def _recover_from_collision(self):
        """碰撞恢复函数：停止、清除错误、回安全位"""
        print("执行碰撞恢复流程...")
        try:
            self.robot.rm_set_arm_stop()
            print("已发送停止命令")
        except Exception as e:
            print(f"停止命令失败: {e}")
            
        try:
            self.robot.rm_clear_system_err()
            print("已清除系统错误")
            self.robot.rm_set_collision_state(2)
        except Exception as e:
            print(f"清除错误失败: {e}")
            
        try:
            # 先回到中间安全位，再回初始位
            print(f"正在返回安全位置: {self.mid_pose1}")
            self.robot.rm_movej(self.mid_pose1, self.robot_speed, 0, 0, 1)
            self.gripper.gripper_position(1)
            time.sleep(1)
            self.gripper.gripper_position(0)
            time.sleep(1)
            self.gripper.gripper_position(0)
            print(f"正在返回初始位置: {self.init_pose}")
            self.robot.rm_movej(self.init_pose, self.robot_speed, 0, 0, 1)
            print("碰撞恢复完成")
        except Exception as e:
            print(f"返回安全位失败: {e}")

    @staticmethod
    def in_paint(depth_img):
        depth_img = np.array(depth_img).astype(np.float32)
        depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_img == 0).astype(np.uint8)

        depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
        # Back to original size and value range.
        depth_img = depth_img[1:-1, 1:-1]

        return depth_img

    def to_tensor(self, depth_img, color_img, rgb_include, depth_include):

        # depth normalize
        depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)

        # color normalize
        color_img = color_img[..., ::-1] # RGB
        color_img = np.ascontiguousarray(color_img, dtype=np.float32)
        color_img /= 255
        color_img -= color_img.mean()

        if rgb_include and depth_include:
            img_in = np.concatenate((np.expand_dims(depth_img, 2), color_img), 2)
        elif rgb_include and not depth_include:
            img_in = color_img
        elif depth_include and not rgb_include:
            img_in = np.expand_dims(depth_img, 2)
        
        
        img_in = np.transpose(img_in, (2, 0, 1))
        # depth only
        # img_in = np.expand_dims(depth_img, 2)
        # img_in = np.transpose(img_in, (2, 0, 1))
        # img_in = np.expand_dims(img_in, 0).astype(np.float32)
        # img_in = torch.from_numpy(img_in).to(self.device)

        img_in = np.expand_dims(img_in, 0).astype(np.float32)
        img_in = torch.from_numpy(img_in).to(self.device)
        # img_in = torch.from_numpy(img_in)

        return img_in

    def generate_grasp_yolo(self, img_in, rgb_image, num, indexrob, mask_img, pixel_wise_stride=1):
        t = time.time()
        output = self.net.forward(img_in)
        t1 = time.time()
        # torch.cuda.empty_cache()
        # print('forward time:{:.6f}'.format(t1 - t))
        # cv2.imshow("Q", q_out)
        # cv2.imshow("cos_A", np.cos(2*ang_out))
        # cv2.imshow("sin_A", np.sin(2*ang_out))
        # cv2.imshow("W", w_out)
        # from matplotlib import pyplot as plt
        # plt.figure(1)
        # plt.imshow(q_out, cmap='bone')
        # plt.axis('off')
        # plt.figure(2)
        # plt.imshow(ang_out, cmap='bone')
        # plt.axis('off')
        # plt.figure(3)
        # plt.imshow(w_out, cmap='bone')
        # plt.axis('off')
        # plt.show()

        output = output[output[:, :, 4] > 0]  # remove boxes < threshold

        if len(output) > 0:
            # Run NMS on predictions
            # detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            output = output[output[:, 4].sort(descending=True)[1]]
            # output = output[:1, :]

        output = output.detach().cpu().numpy()
        # rgb_image1 = rgb_image.copy()
        scale_coords(100, output, rgb_image.shape)
        # show_processed_image(rgb_image, output, num, indexrob)
        # show_processed_image(rgb_image1, output[:1, :], 0, 1)
        # print(index)
        grasp_select = []
        for k in range(num):
            index = output[k, :]
            x, y = index[0], index[1]
            # x, y = index[3] * 16 + 48, index[2] * 16 + 48
            point = torch.tensor([[x, y]])
            if mask_img[y.astype(np.int32), x.astype(np.int32)] == 255:
                row = int(index[1])
                column = int(index[0])
                # row = output[0][1]
                # column = output[0][0]
                angle = index[3]
                width = index[2]
                height = index[2]/2
                grasp_select.append((row, column, angle, width, height))
                break
        
        if (len(grasp_select)==0):
            print("输出最高质量抓取!")
            index = output[0, :]
            row = int(index[1])
            column = int(index[0])
            # row = output[0][1]
            # column = output[0][0]
            angle = index[3]
            width = index[2]
            height = index[2]/2
            grasp_select.append((row, column, angle, width, height))

        return grasp_select, (int(row * pixel_wise_stride + pixel_wise_stride//2),
                int(column * pixel_wise_stride + pixel_wise_stride//2),
                angle, width, height)

    def grasp_img2real_yolo(self, color_img, depth_img, grasp, slope_angle, topk, vis=False, color=(255, 0, 0),
                       note='', collision_check=False):
        row, column, angle, width, height = grasp

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

        point1 = np.array(self.camera.get_coordinate(rect[0][0], rect[0][1]))
        point2 = np.array(self.camera.get_coordinate(rect[1][0], rect[1][1]))

        width_gripper = np.linalg.norm(point1[0:2] - point2[0:2], 2)

        # coordinate = np.append(coordinate_cam[0:2], coordinate_cam[2])
        z = depth_img[row, column] + 0.02
        # xpc, ypc, zpc= np.linalg.inv(self.camera.intr) @ np.array([column+80, row, 1]) * z
        xpc, ypc, zpc= np.linalg.inv(self.camera.intr) @ np.array([column+70, row, 1]) * z
        # coordinate[2] -= 0.1
        Tcam2base = self.Tcam2base
        Pobj2base = Tcam2base @ np.array([[xpc], [ypc], [zpc], [1]])
        Pobj2base = Pobj2base.squeeze()[:-1]
        # Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle-np.pi/5.5], dtype=np.float64))[0]).dot(self.Tcam2base[0:3, 0:3])
        Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle-np.pi/1000], dtype=np.float64))[0]).dot(self.Tcam2base[0:3, 0:3])
        # Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle+config.angle1], dtype=np.float64))[0]).dot(self.Tcam2base[0:3, 0:3])
        t_tcp_flange = np.array([0, 0, 0.2])
        tcp_compensate = np.array([0, 0, 0.018])
        slope_flag = True
        # slope_angle = np.pi/5
        slope_angle = np.pi/8
        column_left, column_right = 80, 480
        # column_left, column_right = 80, 430
        row_up, row_down = 120, 292
        if column < column_left:
            if row < row_up:
                print("左上角处理")
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, slope_angle, 0], dtype=np.float64))[0])
            elif row > row_down:
                print("左下角处理")
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, -slope_angle, 0], dtype=np.float64))[0])
            else:
                print("左侧边缘处理") 
                R_rot = np.array(cv2.Rodrigues(np.array([slope_angle, 0, 0], dtype=np.float64))[0])
        elif column > column_right:
            if row < row_up:
                print("右上角处理")
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, slope_angle, 0], dtype=np.float64))[0])
            elif row > row_down:
                print("右下角处理")
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, -slope_angle, 0], dtype=np.float64))[0])
            else:
                print("右侧边缘处理")# 右侧边缘处理
                R_rot = np.array(cv2.Rodrigues(np.array([-slope_angle, 0, 0], dtype=np.float64))[0])
        elif row < row_up:
            print("上边缘处理")# 上边缘处理
            R_rot = np.array(cv2.Rodrigues(np.array([0, slope_angle, 0], dtype=np.float64))[0])
        elif row > row_down:
            print("下边缘处理")
            R_rot = np.array(cv2.Rodrigues(np.array([0, -slope_angle, 0], dtype=np.float64))[0])
        else:
            R_rot = np.eye(3)
            t_tcp_flange = np.array([0, 0, 0])
            tcp_compensate = np.array([0, 0, 0])
            slope_flag = False
        Robj2base = R_rot.dot(Robj2base)
        t_compesate = Robj2base @ t_tcp_flange  #末端绕基座旋转后,夹爪在末端坐标系下的偏移量导致夹住中心点偏离物体,需要补偿
        t_z_translation = Robj2base @ tcp_compensate
        Pobj2base = Pobj2base - [t_compesate[0], t_compesate[1], 0] + t_z_translation
        
        if vis:
            # grasp center
            color_img = cv2.circle(color_img, (column, row), 3, (0, 0, 255), -1)
            color_img = cv2.arrowedLine(color_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
            color_img = cv2.arrowedLine(color_img, (int(x2), int(y2)), (int(x1), int(y1)), color, thickness=2)
            if note != '':
                cv2.putText(color_img, note, (column, row), fontScale=1.2, fontFace=cv2.FONT_HERSHEY_SIMPLEX
                            , color=(255, 255, 255))
            cv2.imwrite('outputs/graspyolo_{}.png'.format(int(topk)), color_img)
        
        return Pobj2base, Robj2base,  width_gripper, angle, t_compesate[2], slope_flag
        # return 


    def letterbox1(self, img, depth, height=416,
                   color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded square
        shape = img.shape[:2]  # shape = [height, width]
        ratio = float(height) / max(shape)  # ratio  = old / new
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
        dw = (height - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        depth = cv2.resize(depth, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_REFLECT)  # padded square
        return img, depth, ratio, dw, dh


    def find_num_count_np(self, arr, target):
        count = np.sum(arr == target)
        return count
    
    def get_index(self, items, target):
        try:
            return items.index(target)
        except ValueError:
            return None

    ### 执行指定物体抓取
    def obj_grasp(self, label, vis=False):

        self.gripper.gripper_position(0)
        depth_img, color_img = self.camera.get_img()

        depth_img = self.in_paint(depth_img)
        depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0)/1000
        depth_img = depth_img[:, 80:560]
        color_img = color_img[:, 80:560, :]
        color_img_raw = color_img.copy()
        depth_img_raw = depth_img.copy()

        pre = inference_detector(self.det_model, color_img) # numpy array BGR

        classes = self.det_model.dataset_meta['classes']
        bboxes = pre.pred_instances.bboxes.cpu().numpy()
        scores = pre.pred_instances.scores.cpu().numpy()[:, np.newaxis]
        masks = pre.pred_instances.masks.cpu().numpy()
        labels = pre.pred_instances.labels.cpu().numpy()[:, np.newaxis]
        predicts = np.concatenate((scores, bboxes, labels), axis=1)
       
        bboxes, indics, labels = nms(predicts, 0.8, 0.9)
        masks = masks[indics]
        

        color_image = color_img.astype(np.uint8)
        
        black_image = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        background_img = cv2.imread('zy/background.png')
        for j in range(len(bboxes)):
            mask_circle = np.uint8(masks[j])

            if classes[labels[j]] == label:
                black_image[mask_circle > 0] = 255
                background_img[mask_circle > 0] = color_image[mask_circle > 0]

        color_img, depth_img, ratio, padw, padh = self.letterbox1(color_img, depth_img, 100)
        # color_img, depth_img, ratio, padw, padh = self.letterbox1(background_img, depth_img, 100)
        img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=1)
        #img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=0)
        k_max = 100
        # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, blank_depth_img, pixel_wise_stride=1)
        topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)

        coordinate, ori, width_gripper, angle, z_compensate, slope_flag = self.grasp_img2real_yolo(
                        color_img_raw, depth_img_raw, best_grasp, np.pi/7, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
                    )
        
        gesture = mat2euler(ori, axes='sxyz')
        pose = np.hstack((coordinate, gesture))
        # angle = self.grasp_imgshow(
        #         color_img_raw, best_grasp, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
        # )
        # if 0 <= angle <= np.pi/2:
        #     angle = np.pi/2 - angle
        # else:
        #     angle = -np.pi/2 - angle

        # mat = cv2.Rodrigues(np.array([0, 0, angle]).astype(np.float32))[0]
        # Tobj2cam = np.vstack((
        #                 np.hstack(
        #                     (np.dot(self.Rbase2cam, np.array(mat)),
        #                      np.expand_dims(coordinate, 0).T)
        #                 ), np.array([[0, 0, 0, 1]])
        #                 ))

        if self.hardware:
            self.robot.rm_movej(self.mid_pose, self.robot_speed, 0, 0, 1)
            self.robot.rm_movej(self.mid_pose1, self.robot_speed, 0, 0, 1)
            # Tobj2base = self.Tcam2base.dot(Tobj2cam)
            # position = Tobj2base[0:3, 3]
            # gesture = cv2.Rodrigues(Tobj2base[0:3, 0:3])[0].T.squeeze()
            # pose = np.hstack((position, gesture))
            # print(pose)
            if slope_flag:
                pose[2] = config.pose2_2
                # pose[1] += -0.02
                pose[0] += -0.02
                # pose = pose + [0, 0, -z_compensate-0.02, 0, 0, 0]
                # pose = pose + [0, 0, -z_compensate, 0, 0, 0]
                if pose[2] > 0.534:
                    pose[2] = config.pose2_2
                    # pose[2] = 0.553
                pose_up_to_grasp_position = pose + [0, 0, -0.08, 0, 0, 0]
                # [0, 0, -0.12, 0, 0, 0]
            else:
                # pose = pose + [0, 0, -0.197, 0, 0, 0]
                # pose = pose + [0, 0, -0.170, 0, 0, 0]
                # pose = pose + [0, 0, -0.170, 0, 0, 0]
                pose[2] = config.pose2
                if pose[2] > 0.534:
                    pose[2] = config.pose2
                pose_up_to_grasp_position = pose + [0, 0, -0.05, 0, 0, 0]
            # if pose[2] > 0.535:
            #     pose[2] = 0.532
            print(pose)
            # pose_up_to_grasp_position = pose + [0, 0, -0.12, 0, 0, 0]
            max_attempt = 2
            attempt = 0
            print(f"开始抓取尝试，最大尝试次数: {max_attempt}")
            
            while attempt < max_attempt:
                try:
                    print(f"\n--- 第 {attempt + 1} 次抓取尝试 ---")
                    
                    # 计算逆解
                    print(f"计算抓取位姿逆解: {pose}")
                    params = rm_inverse_kinematics_params_t(self.mid_pose1, pose, 1)
                    tag1, pose_joint = self.robot.rm_algo_inverse_kinematics(params)
                    if tag1 != 0:
                        print(f"✗ 机械臂逆解失败！请重新放置物体位置")
                        self.robot.rm_movej(self.init_pose, self.robot_speed, 0, 0, 1)
                        return False
                    
                    print(f"✓ 逆解成功: {pose_joint}")
                    
                    # 计算上方安全位置逆解
                    print(f"计算上方安全位置逆解: {pose_up_to_grasp_position}")
                    params_up = rm_inverse_kinematics_params_t(self.mid_pose1, pose_up_to_grasp_position, 1)
                    tag2, up_to_grasp_joint = self.robot.rm_algo_inverse_kinematics(params_up)
                    if tag2 != 0:
                        print(f"✗ 上方位置逆解失败！请重新放置物体位置")
                        self.robot.rm_movej(self.init_pose, self.robot_speed, 0, 0, 1)
                        return False
                    
                    print(f"✓ 上方位置逆解成功: {up_to_grasp_joint}")
                    
                    # 移动到上方安全位置
                    print("移动到上方安全位置...")
                    self._movej_safe(up_to_grasp_joint[0:6], self.robot_speed)
                    
                    # 打开夹爪
                    print("打开夹爪...")
                    self.gripper.gripper_position(1)
                    # self.robot.rm_set_collision_state(8)
                    time.sleep(1)
                    
                    # 移动到抓取位置
                    print("移动到抓取位置...")
                    self._movej_safe(pose_joint[0:6], self.robot_speed)
                    
                    # 闭合夹爪抓取
                    print("闭合夹爪抓取...")
                    self.gripper.gripper_position(0)
                    time.sleep(1)
                    
                    # 返回上方安全位置
                    print("返回上方安全位置...")
                    self._movej_safe(up_to_grasp_joint[0:6], self.robot_speed)
                    
                    # 返回中间位置
                    print("返回中间位置...")
                    self._movej_safe(self.mid_pose1, self.robot_speed)
                    
                    # 返回初始位置
                    print("返回初始位置...")
                    self._movej_safe(self.lift2init_pose, self.robot_speed)

                    # self.robot.rm_set_collision_state(4)

                    # 移动到放置位置
                    print("移动到放置位置...")
                    self._movej_safe(self.place_mid_pose, self.robot_speed)
                    self._movej_safe(self.place_mid_pose2, self.robot_speed)
                    self._movej_safe(self.place_last_pose, self.robot_speed)
                    
                    # 打开夹爪放置
                    print("打开夹爪放置...")
                    self.gripper.gripper_position(1)
                    time.sleep(1.0)
                    
                    # 闭合夹爪
                    print("闭合夹爪...")
                    self.gripper.gripper_position(0)
                    
                    # 返回初始位置
                    print("返回初始位置...")
                    self._movej_safe(self.place_mid_pose2, self.robot_speed)
                    self._movej_safe(self.place_mid_pose, self.robot_speed)
                    self._movej_safe(self.init_pose, self.robot_speed)
                    
                    print("✓ 抓取成功完成！")
                    return True
                    
                except CollisionDetected as exc:
                    print(f"✗ 第{attempt+1}次抓取碰撞: {exc}")
                    self._recover_from_collision()
                    attempt += 1
                    if attempt >= max_attempt:
                        print("✗ 连续碰撞，达到最大尝试次数，抓取失败")
                        return False
                    print(f"准备第{attempt+1}次重试...")
                    
                except Exception as exc:
                    print(f"✗ MoveJ执行异常: {exc}")
                    print("执行紧急恢复...")
                    self._recover_from_collision()
                    return False

    
if __name__ == '__main__':
    grasp = Grasp(hardware=True)


    # grasp.obj_grasp(label=config.type,vis=True)
    grasp.obj_grasp(label='carrot',vis=True)

 
# def on_submit():
#     label = entry.get()  # 获取用户输入的标签
#     if label:
#         grasp.action_yolo(label=label, vis=True)
#     else:
#         messagebox.showwarning("Input Error", "Please enter a label.")

# if __name__ == '__main__':
#     grasp = Grasp(hardware=True)

#     # 创建主窗口
#     root = tk.Tk()
#     root.title("机器人智能分拣界面")

# # 设置窗口的网格布局权重
#     root.grid_rowconfigure(0, weight=1)  # 第一行可拉伸
#     root.grid_columnconfigure(0, weight=1)  # 第一列可拉伸

#     custom_font = font.Font(family="微软雅黑", size=12, weight="bold")  # 字体类型、大小和粗细
#     custom_font1 = font.Font(family="Times New Roman", size=14, weight="bold")  # 字体类型、大小和粗细

#     # 创建输入框
#     label = tk.Label(root, text="请输入待抓取对象:", font=custom_font)
#     label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")  # 放置在网格中，并设置 sticky 参数

#     entry = tk.Entry(root, font=custom_font1)
#     entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")  # 放置在网格中，并设置 sticky 参数

#     # 创建提交按钮
#     submit_button = tk.Button(root, text="执行抓取", font=custom_font, command=on_submit)
#     submit_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")  # 放置在网格中，并设置 sticky 参数

#     # 运行主循环
#     root.mainloop()











