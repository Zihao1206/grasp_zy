import torch
import config
import camera
import numpy as np
from models.common import post_process_output
from skimage.feature import peak_local_max
# from models.VGG_GA3 import VGG
from gqcnn_server.VGG_replace import VGG
import time
import cv2
import gripper_zhiyuan as gripper
# from gqcnn_server.resNet import ResNet, Bottleneck
from gqcnn_server.network import GQCNN
import tkinter as tk
from tkinter import messagebox
from tkinter import font
from utils.utils import *
from gqcnn_server.augment_cnn import AugmentCNN
import genotypes as gt
import datetime
from robotic_arm_package.robotic_arm import *
from mmdet.apis import init_detector, inference_detector
from nms import nms
from mmdet.registry import VISUALIZERS
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2mat
from transforms3d.affines import compose
from transforms3d.euler import mat2euler

class Grasp:
    def __init__(self, hardware=False):
        self.robot_speed = config.robot_speed 
        # self.robot_speed = 20
        self.hardware = hardware
        self.init_pose = [-0.226100, -0.00309, 0.522900, -0.010, 0.028, 2.650]
        self.camera = camera.RS(640, 480)
        if self.hardware:
            self.robot = Arm(RM65, "192.168.127.101", 8080)
            self.robot.Set_Collision_Stage(4)
            # self.robot.Algo_Get_Joint_Max_Limit()
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
            self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)

            # self.lift2init_pose = [86, -129, 127, -0.7, 77, 1]
            self.lift2init_pose = [73, -129, 127, -0.7, 77, 1] ## 回转到初始状态
            # self.place_pose = [86, -129, 127, -0.7, 77, 62]
            self.place_mid_pose = [73, -129, 60, 0, 121, 1] ## 机械臂第三关节前倾
            # self.place_last_pose = [90, -69, 4.335, 0.9, 57.3, 62]
            self.place_mid_pose2 = [77, -129, 60, 0, 9, 1] ## 机械臂展开
            self.place_last_pose = [77, -104, 38, -2, 9, 1] ## 机械臂向下放置物体
            
        # self.Tcam2base = np.array([
        #     [-0.08473652, -0.9963165, 0.01315871, 0.224658],
        #     [0.99468956, -0.08535763, -0.0575044, -0.01471011],
        #     [0.05841577, 0.0082161, 0.99825853, 0.0296352],
        #     [0., 0., 0., 1., ]
        # ])
        # self.Rcam2base = np.array([
        #     [-0.08473652, -0.9963165, 0.01315871],
        #     [0.99468956, -0.08535763, -0.0575044],
        #     [0.05841577, 0.0082161, 0.99825853]])
        # self.Tcam2base = np.array([
        #     [0.02742095, -0.99940903, -0.0207286, 0.20841901],
        #     [0.9995487, 0.02766746, -0.01170045, -0.02848768],
        #     [0.01226705, -0.02039841, 0.99971667, 0.03739014],
        #     [0., 0., 0., 1., ]
        # ])
        # self.Rbase2cam = np.array([
        #     [0.02742095, 0.9995487, 0.01226705],
        #     [-0.99940903, 0.02766746, -0.02039841],
        #     [-0.0207286, -0.01170045, 0.99971667]])
        self.Tcam2base = np.array([
            [-0.01537554, -0.99988175, -0.00028888,0.2070103],
            [ 0.9998815  ,-0.01537576 , 0.00076007,-0.03249003],
            [-0.00076442 ,-0.00027716,  0.99999967,0.02642268],
            [0., 0., 0., 1., ]
            ])
        self.Rbase2cam = self.Tcam2base.T[0:3,0:3]


        # self.device = torch.device("cuda:0")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file_path = 'single_new.txt'
        # file_path = 'single_zy.txt'
        # file_path = 'single_rgb.txt'
        with open(file_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                gene = line
                getypr = gt.from_str(gene)
        model = AugmentCNN('cornell.data', 100, 4, 8, 5, False, getypr).cuda()
        # model = AugmentCNN('cornell.data', 100, 3, 8, 5, False, getypr).cuda()
        # model = AugmentCNN('C:/grasp_static/cornell.data', 100, 1, 8, 5, False, getypr)
        # model.load_state_dict(torch.load('weights/epoch_41_accuracy_1.00', map_location=self.device))  # single_zy.txt
        # model.load_state_dict(torch.load('weights/tune_epoch_19_loss_0.0362_accuracy_1.000', map_location=self.device)) #single-rgb
        # model.load_state_dict(torch.load('weights/tune_epoch_64_loss_0.0297_accuracy_1.000', map_location=self.device))  # single_zy.txt
        
        # model.load_state_dict(torch.load('weights/tune_epoch_11_loss_0.0297_accuracy_0.939', map_location=self.device))
        # model.load_state_dict(torch.load('weights/tune_epoch_20_loss_0.0478_accuracy_1.000', map_location=self.device))
        # model.load_state_dict(torch.load('test_250515_1053__zoneyung_rgb_last/tune_epoch_14_loss_0.0385_accuracy_0.980', map_location=self.device)) # fine_tune
        # model.load_state_dict(torch.load('test_250515_1057__zoneyung_rgb_last/tune_epoch_46_loss_0.0341_accuracy_0.970', map_location=self.device))
        # model.load_state_dict(torch.load('test_250515_1119__zoneyung_last/epoch_09_accuracy_1.00', map_location=self.device)) # scatch
        model.load_state_dict(torch.load('test_250927_1644__zoneyung_/epoch_84_accuracy_1.00', map_location=self.device)) # scatch
        # 目标框检测模型加载
        config_file = 'mmdetection/configs/myconfig_zy.py'
        check_point = 'weights/epoch_20_last.pth'
        inferencer = init_detector(config_file, check_point, device=self.device)
        self.det_model = inferencer

        # from torchsummary import summary
        # summary(model, (4, 480, 480))
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

    @staticmethod
    def vec2mat(rot_vec, trans_vec):
        """
        :param rot_vec: list_like [a, b, c]
        :param trans_vec: list_like [x, y, z]
        :return: transform mat 4*4
        """
        theta = np.linalg.norm(rot_vec, 2)
        if theta:
            rot_vec /= theta
        out_operator = np.array([
            [0, -rot_vec[2], rot_vec[1]],
            [rot_vec[2], 0, -rot_vec[0]],
            [-rot_vec[1], rot_vec[0], 0]
        ])
        rot_vec = np.expand_dims(rot_vec, 0)
        rot_mat = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * rot_vec.T.dot(rot_vec) + np.sin(
            theta) * out_operator

        trans_vec = np.expand_dims(trans_vec, 0).T

        trans_mat = np.vstack(
            (np.hstack((rot_mat, trans_vec)), [0, 0, 0, 1])
        )

        return trans_mat

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
        # depth_img = self.in_paint(depth_img)
        # scale = np.abs(depth_img).max()
        # depth_img /= scale
        depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)
        # depth_img = depth_img - depth_img.mean()

        # color normalize
        color_img = color_img[..., ::-1] # RGB
        color_img = np.ascontiguousarray(color_img, dtype=np.float32)
        # color_img = np.transpose(color_img, (2, 0, 1)).astype(np.float32)
        color_img /= 255
        color_img -= color_img.mean()

        # to tensor
        # img_in = np.vstack(
        #     (np.expand_dims(depth_img, 0), color_img)
        # )

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

    def to_tensor_grasp(self, depth_img, color_img):

        # depth normalize
        # depth_img = self.in_paint(depth_img)
        # scale = np.abs(depth_img).max()
        # depth_img /= scale
        depth_img = np.clip((depth_img - depth_img.mean()), -1, 1)
        # depth_img = depth_img - depth_img.mean()

        # color normalize
        # color_img = color_img[..., ::-1] # RGB
        # color_img = np.ascontiguousarray(color_img, dtype=np.float32)
        # # color_img = np.transpose(color_img, (2, 0, 1)).astype(np.float32)
        # color_img /= 255
        color_img -= color_img.mean()

        img_in = np.concatenate((np.expand_dims(depth_img, 2), color_img), 2)
        img_save = img_in.copy()
        img_in = np.transpose(img_in, (2, 0, 1))

        img_in = np.expand_dims(img_in, 0).astype(np.float32)
        img_in = torch.from_numpy(img_in).to(self.device)

        return img_in, img_save


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

        # for k in range(num):
        #     index = output[k, :]
        #     row = int(index[1])
        #     column = int(index[0])
        #     # row = output[0][1]
        #     # column = output[0][0]
        #     angle = index[3]
        #     width = index[2]
        #     height = index[2]/2
        #     grasp_select.append((row, column, angle, width, height))



        # row = int(output[0][1])
        # column = int(output[0][0])
        # # row = output[0][1]
        # # column = output[0][0]
        # angle = output[0][3]
        # width = output[0][2]
        # height = output[0][2]/2

        # return (row, column, angle,
        #         width, height)
        # return (int(row * pixel_wise_stride + pixel_wise_stride//2),
        #         int(column * pixel_wise_stride + pixel_wise_stride//2),
        #         angle,
        #         width, height)
        return grasp_select, (int(row * pixel_wise_stride + pixel_wise_stride//2),
                int(column * pixel_wise_stride + pixel_wise_stride//2),
                angle, width, height)

    def generate_grasp_yolo_top5(self, img_in, rgb_image, num, indexrob, topnum, pixel_wise_stride=1):
        t = time.time()
        output = self.net.forward(img_in)
        t1 = time.time()

        output = output[output[:, :, 4] > 0]  # remove boxes < threshold

        if len(output) > 0:
            # Run NMS on predictions
            # detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
            output = output[output[:, 4].sort(descending=True)[1]]
            output = output[:topnum, :]
            # output_copy = output.copy()
            label5 = output.clone()[:, :4]
        label5[:, :3] = label5[:, :3] / 100

        scale_coords(100, output, rgb_image.shape)
        grasps5 = []
        for i in range(topnum):
            row = int(output[i][1])
            column = int(output[i][0])
            row = int(row * pixel_wise_stride + pixel_wise_stride//2)
            column = int(column * pixel_wise_stride + pixel_wise_stride//2)
            angle = output[i][3]
            width = output[i][2]
            height = width/2
            grasps5.append((row, column, angle, width, height))
        return grasps5, label5

    
    def pose_to_matrix(self, degrees=False):
        # 创建Rotation对象（输入为绕XYZ轴的旋转，顺序需根据机械臂类型调整）
        _, joints, current_pose, _, _ = self.robot.Get_Current_Arm_State()
        rotation = R.from_euler('xyz', current_pose[3:], degrees=degrees)
        # 生成齐次矩阵
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = current_pose[:3]
        return matrix


    # 输入位姿：x,y,z为位置，rx,ry,rz为绕XYZ轴的旋转角度（单位：弧度）
    def get_current_end_pose(self):
        # 欧拉角转旋转矩阵（顺序为XYZ，对应RPY的ZYX需调整axes参数）
        _, joints, current_pose, _, _ = self.robot.Get_Current_Arm_State()
        rotation = euler2mat(current_pose[3], current_pose[4], current_pose[5], axes='sxyz')
        # 构造齐次变换矩阵
        translation = current_pose[:3]
        matrix = compose(translation, rotation, np.ones(3))
        return matrix

    def grasp_img2real_yolo(self, color_img, depth_img, grasp, slope_angle, topk, vis=False, color=(255, 0, 0),
                       note='', collision_check=False):
        row, column, angle, width, height = grasp

        # s = np.sin(angle)
        # c = np.cos(angle)
        #
        # x1 = column + width / 2 * c
        # x2 = column - width / 2 * c
        # y1 = row - width / 2 * s
        # y2 = row + width / 2 * s
        #
        # rect = np.array([
        #     [x1 - width / 4 * s, y1 - width / 4 * c],
        #     [x2 - width / 4 * s, y2 - width / 4 * c],
        #     [x2 + width / 4 * s, y2 + width / 4 * c],
        #     [x1 + width / 4 * s, y1 + width / 4 * c],
        # ]).astype(np.int)

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
        coordinate_cam = self.camera.get_coordinate(column+80, row)

        rotation_mat = cv2.getRotationMatrix2D((depth_img.shape[1] // 2,
                                                depth_img.shape[0] // 2),
                                               -float(angle) / np.pi * 180, 1)
        trans_mat = np.array([
            [1, 0, depth_img.shape[1] // 2 - column],
            [0, 1, depth_img.shape[0] // 2 - row]
        ]).astype(np.float32)

        full_mat = np.r_[rotation_mat, [[0, 0, 1]]].dot(np.r_[trans_mat, [[0, 0, 1]]])[:2, :]
        trans_rot_depth_img = cv2.warpAffine(
            depth_img, full_mat, (depth_img.shape[1], depth_img.shape[0])
        )

        croped_depth_img = trans_rot_depth_img[int(depth_img.shape[0] // 2 - width * 0.25):
                                               int(depth_img.shape[0] // 2 + width * 0.25),
                                               int(depth_img.shape[1] // 2 - width * 0.5):
                                               int(depth_img.shape[1] // 2 + width * 0.5)]

        grasp_depth = coordinate_cam[2]-0.18
        if grasp_depth > 0.630:  # z + 0.12286666
            grasp_depth_out = 0.625
            if collision_check:
                print('grasp depth prediction:', grasp_depth)
                print('collision with the plane!!!!')
        else:
            grasp_depth_out = grasp_depth

        # coordinate = np.append(coordinate_cam[0:2], coordinate_cam[2])
        z = depth_img[row, column] + 0.02
        xpc, ypc, zpc= np.linalg.inv(self.camera.intr) @ np.array([column+80, row, 1]) * z
        # coordinate[2] -= 0.1
        
        # Tend2base = u.pos_ori2mat(self.init_end_pos, self.init_end_ori)
        # getsure = cv2.Rodrigues(self.init_end_ori)[0].T.squeeze()
        Tcam2base = self.Tcam2base
        Pobj2base = Tcam2base @ np.array([[xpc], [ypc], [zpc], [1]])
        Pobj2base = Pobj2base.squeeze()[:-1]
        # R_rot = np.array(cv2.Rodrigues(np.array([0, np.pi/7, 0], dtype=np.float64))[0])
        Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle-np.pi/5.5], dtype=np.float64))[0]).dot(self.Tcam2base[0:3, 0:3])
        # Robj2base = np.array(cv2.Rodrigues(np.array([0, 0, -angle], dtype=np.float64))[0]).dot(self.Tcam2base[0:3, 0:3])
        # Robj2base_adjust = R_rot.dot(Robj2base)
        # t_compesate = Robj2base_adjust @ np.array([0, 0, 0.19])  #末端绕基座旋转后,夹爪在末端坐标系下的偏移量导致夹住中心点偏离物体,需要补偿
        # t_z_translation = Robj2base_adjust @ np.array([0, 0, 0.022])
        # Pobj2base = Pobj2base - [t_compesate[0], t_compesate[1], 0] + t_z_translation
        t_tcp_flange = np.array([0, 0, 0.2])
        # t_tcp_flange = np.array([0, 0, 0.2])
        tcp_compensate = np.array([0, 0, 0.018])
        # tcp_compensate = np.array([0, 0, 0.018])
        slope_flag = True
        # slope_angle = np.pi/5
        column_left, column_right = config.column_left, config.column_right
        # column_left, column_right = 80, 430
        # column_left, column_right = 40, 460
        # column_left, column_right = 80, 430
        row_up, row_down = 120, 292
        # row_up, row_down = 100, 300
        # row_up, row_down = 120, 292
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
            # color_img = cv2.line(color_img, tuple(rect[0]), tuple(rect[3]), color, thickness=2)
            # color_img = cv2.line(color_img, tuple(rect[2]), tuple(rect[1]), color, thickness=2)
            # cv2.drawContours(color_img, [rect], 0, color, 4)
            if note != '':
                cv2.putText(color_img, note, (column, row), fontScale=1.2, fontFace=cv2.FONT_HERSHEY_SIMPLEX
                            , color=(255, 255, 255))
            # from matplotlib import pyplot as plt
            # plt.imshow(depth_img)
            # plt.axis('off')
            # plt.show()
            # cv2.imshow('depth', depth_img)

            # cv2.imshow('grasp', color_img)
            # cv2.waitKey(0)  # 0 表示无限等待，直到用户按下任意键
            # cv2.destroyAllWindows()  # 关闭所有窗口
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


    ### 判断是否存在指定物体
    def detect_obj(self, label):
        depth_img, color_img = self.camera.get_img()

        depth_img = self.in_paint(depth_img)
        depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0)/1000
        depth_img = depth_img[:, 80:560]
        color_img = color_img[:, 80:560, :]
        color_img_raw = color_img.copy()
        depth_img_raw = depth_img.copy()

        # depth_img = cv2.imread('rgbd/241017_215857d.tiff', cv2.IMREAD_UNCHANGED)
        # depth_img = np.full((480, 480), 1, dtype=np.float32)
        # color_img = cv2.imread('zy/zy_rgb_319.jpg')
        # color_img_raw = color_img.copy()
        # depth_img_raw = depth_img.copy()

        pre = inference_detector(self.det_model, color_img) # numpy array BGR

        # img = color_img[..., ::-1]
        # visual = VISUALIZERS.build(self.det_model.cfg.visualizer)
        # visual.dataset_meta = self.det_model.dataset_meta
        # visual.add_datasample(
        #     'pre',
        #     img,
        #     data_sample=pre,
        #     draw_gt=True,
        #     show=True)
        # img_with_bbox = visual.get_image()
        # cv2.imwrite('outputs/1.jpg', img_with_bbox[:, :, ::-1])
        classes = self.det_model.dataset_meta['classes']
        bboxes = pre.pred_instances.bboxes.cpu().numpy()
        scores = pre.pred_instances.scores.cpu().numpy()[:, np.newaxis]
        masks = pre.pred_instances.masks.cpu().numpy()
        labels = pre.pred_instances.labels.cpu().numpy()[:, np.newaxis]
        predicts = np.concatenate((scores, bboxes, labels), axis=1)
        bboxes, indics, labels = nms(predicts, 0.8, 0.9)
        masks = masks[indics]

        obj_index = self.get_index(classes, label)
        num_obj = self.find_num_count_np(labels, obj_index)
        # exists = (labels == obj_index).any()

        return num_obj

    ### 执行指定物体抓取
    def obj_grasp(self, label, vis=False):

        # self.gripper.gripper_position(5*51200)
        self.gripper.gripper_position(1)
        depth_img, color_img = self.camera.get_img()

        depth_img = self.in_paint(depth_img)
        depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0)/1000
        depth_img = depth_img[:, 80:560]
        color_img = color_img[:, 80:560, :]
        color_img_raw = color_img.copy()
        depth_img_raw = depth_img.copy()

        # depth_img = cv2.imread('C:/grasp_static/rgbd/241017_215857d.tiff', cv2.IMREAD_UNCHANGED)
        # depth_img = np.full((480, 480), 1, dtype=np.float32)
        # color_img = cv2.imread('C:/grasp_static/zy/zy_rgb_2.png')
        # color_img_raw = color_img.copy()
        # depth_img_raw = depth_img.copy()

        pre = inference_detector(self.det_model, color_img) # numpy array BGR

        # img = color_img[..., ::-1]
        # visual = VISUALIZERS.build(self.det_model.cfg.visualizer)
        # visual.dataset_meta = self.det_model.dataset_meta
        # visual.add_datasample(
        #     'pre',
        #     img,
        #     data_sample=pre,
        #     draw_gt=True,
        #     show=True)
        # img_with_bbox = visual.get_image()
        # cv2.imwrite('outputs/1.jpg', img_with_bbox[:, :, ::-1])
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
            # contours, _ = cv2.findContours(mask_circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # (x,y), radius = cv2.minEnclosingCircle(contours[0])
            # cv2.circle(mask_circle, (int(x), int(y)), int(radius), (255, 255, 255), cv2.FILLED)
            if classes[labels[j]] == label:
                black_image[mask_circle > 0] = 255
                background_img[mask_circle > 0] = color_image[mask_circle > 0]
        # cv2.imwrite('C:/grasp_static/outputs/mask1.png', black_image)
        
        # plt.figure()
        # plt.imshow(background_img[..., ::-1])
        # plt.show()
        # cv2.imwrite('C:/grasp_static/outputs/depth_interest.png', black_image)

        color_img, depth_img, ratio, padw, padh = self.letterbox1(color_img, depth_img, 100)
        # color_img, depth_img, ratio, padw, padh = self.letterbox1(background_img, depth_img, 100)
        img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=1)
        #img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=0)
        k_max = 100
        # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, blank_depth_img, pixel_wise_stride=1)
        topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
        # best_grasp, feature_maps = self.generate_grasp(img_in)
        # pixel_wise_stride = depth_img.shape[0] / feature_maps[0].shape[0]
        coordinate, ori, width_gripper, angle, z_compensate, slope_flag = self.grasp_img2real_yolo(
                        color_img_raw, depth_img_raw, best_grasp, np.pi/7, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
                    )
        # coordinate, ori, width_gripper, angle, z_compensate, slope_flag = self.grasp_img2real_yolo(
        #               color_img_raw, depth_img_raw, best_grasp, np.pi/7, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
        #             )
        
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
            self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
            self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
            # Tobj2base = self.Tcam2base.dot(Tobj2cam)
            # position = Tobj2base[0:3, 3]
            # gesture = cv2.Rodrigues(Tobj2base[0:3, 0:3])[0].T.squeeze()
            # pose = np.hstack((position, gesture))
            # print(pose)
            if slope_flag:
                pose[2] = 0.540
                # pose = pose + [0, 0, -z_compensate-0.02, 0, 0, 0]
                # pose = pose + [0, 0, -z_compensate, 0, 0, 0]
                if pose[2] > 0.534:
                    pose[2] = 0.550
                    # pose[2] = 0.553
                pose_up_to_grasp_position = pose + [0, 0, -0.08, 0, 0, 0]
                # [0, 0, -0.12, 0, 0, 0]
            else:
                # pose = pose + [0, 0, -0.197, 0, 0, 0]
                # pose = pose + [0, 0, -0.170, 0, 0, 0]
                # pose = pose + [0, 0, -0.170, 0, 0, 0]
                pose[2] = 0.546
                if pose[2] > 0.534:
                    pose[2] = 0.546
                pose_up_to_grasp_position = pose + [0, 0, -0.05, 0, 0, 0]
            # if pose[2] > 0.535:
            #     pose[2] = 0.532
            print(pose)
            # pose_up_to_grasp_position = pose + [0, 0, -0.12, 0, 0, 0]
            tag, up_to_grasp_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose_up_to_grasp_position, 1)
            print(up_to_grasp_joint)
            if tag==0:
                # self.gripper.gripper_position(int((0.12-width_gripper)/0.12*3*51250))
                # self.gripper.gripper_position(int(51200))
                # pose_up_to_grasp_position = [0.12221, 0.026199, 0.49557, 0.0068711, 0.077718, -2.5144]
                self.robot.Movej_Cmd(up_to_grasp_joint[0:6], self.robot_speed)
                tag1, pose_joint = self.robot.Algo_Inverse_Kinematics(up_to_grasp_joint[0:6], pose, 1)
                print(pose_joint)
                # self.gripper.gripper_position(int(51200))
                self.gripper.gripper_position(0)
                time.sleep(2)
                self.robot.Movej_Cmd(pose_joint[0:6], self.robot_speed)
                # self.robot.Movel_Cmd(pose_up_to_grasp_position, 10)
                # self.robot.Movel_Cmd(pose, 10)
                # self.gripper.gripper_position(5*51200)
                self.gripper.gripper_position(1)
                time.sleep(1)
                self.robot.Movej_Cmd(up_to_grasp_joint[0:6], self.robot_speed)
                # self.robot.Movel_Cmd(pose_up_to_grasp_position, 20)
                self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed) 
                # self.robot.Movej_Cmd(self.mid_pose, self.robot_speed) 
                # self.robot.Movej_Cmd(self.init_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.lift2init_pose, self.robot_speed)
                # self.robot.Movej_Cmd(self.place_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.place_mid_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.place_mid_pose2, self.robot_speed)
                self.robot.Movej_Cmd(self.place_last_pose, self.robot_speed)
                self.gripper.gripper_position(0)
                time.sleep(1.0)
                # self.gripper.gripper_position(5*51200)
                self.gripper.gripper_position(1)
                self.robot.Movej_Cmd(self.place_mid_pose2, self.robot_speed)
                self.robot.Movej_Cmd(self.place_mid_pose, self.robot_speed)
                # self.gripper.gripper_position(5*51200)
                # self.robot.Movej_Cmd(self.place_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.init_pose, self.robot_speed)
            #     # self.gripper.gripper_position(100)
                # 分拣成功
                # self.camera.stop()
                return True
            else:
                print(f"机械臂逆解失败！请重新放置物体位置")
                # self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
                self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)
                # 分拣失败
                # self.camera.stop()
                return False

    
    def action_yolo(self, label, vis=False):

        depth_img, color_img = self.camera.get_img()

        depth_img = self.in_paint(depth_img)
        depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0)/1000
        depth_img = depth_img[:, 80:560]
        color_img = color_img[:, 80:560, :]
        color_img_raw = color_img.copy()
        depth_img_raw = depth_img.copy()

        # depth_img = cv2.imread('C:/grasp_static/rgbd/241017_215857d.tiff', cv2.IMREAD_UNCHANGED)
        # depth_img = np.full((480, 480), 1, dtype=np.float32)
        # color_img = cv2.imread('C:/grasp_static/zy/zy_rgb_2.png')
        # color_img_raw = color_img.copy()
        # depth_img_raw = depth_img.copy()
       
        # plt.figure(1)
        # plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        # plt.show()
        # print(color_img.shape)

        # color_img = 255 - color_img
        # cv2.imwrite('C:/grasp_static/outputs/object.png', color_img)
        # cv2.imwrite('depth_knife.png', depth_img/1000*255)
        # cv2.imshow('a', color_img)
        # # cv2.waitKey(0)
        # from matplotlib import pyplot as plt
        # plt.figure(1)
        # plt.imshow(depth_img/10000
        #            )
        # plt.axis('off')
        # # plt.figure(2)
        # # plt.imshow(color_img)
        # plt.axis('off')
        # plt.show()
        # cv2.imshow('a', self.in_paint(depth_img)/np.max(depth_img))
        t_start = time.time()

        pre = inference_detector(self.det_model, color_img) # numpy array BGR

        # img = color_img[..., ::-1]
        # visual = VISUALIZERS.build(self.det_model.cfg.visualizer)
        # visual.dataset_meta = self.det_model.dataset_meta
        # visual.add_datasample(
        #     'pre',
        #     img,
        #     data_sample=pre,
        #     draw_gt=True,
        #     show=True)
        # img_with_bbox = visual.get_image()
        # cv2.imwrite('outputs/1.jpg', img_with_bbox[:, :, ::-1])
        classes = self.det_model.dataset_meta['classes']
        bboxes = pre.pred_instances.bboxes.cpu().numpy()
        scores = pre.pred_instances.scores.cpu().numpy()[:, np.newaxis]
        masks = pre.pred_instances.masks.cpu().numpy()
        labels = pre.pred_instances.labels.cpu().numpy()[:, np.newaxis]
        predicts = np.concatenate((scores, bboxes, labels), axis=1)
       
        bboxes, indics, labels = nms(predicts, 0.8, 0.9)
        masks = masks[indics]

        print(len(bboxes))
        # print(indics)
        # print(pre.pred_instances.scores.cpu().numpy().tolist())
        # bboxes = bboxes[:4]

        # uint8
        color_image = color_img.astype(np.uint8)
        # i = 0
        # for bbox in bboxes:
        #     # mask = np.zeros(depth_image.shape).astype(np.uint8)
        #     x0, y0 = int(bbox[0]), int(bbox[1])
        #     x1, y1 = int(bbox[2]), int(bbox[3])
        #     # visualization
        #     start_point, end_point = (x0, y0), (x1, y1)
        #     color = (255, 255, 255)  # Red color in BGR
        #     thickness = 3  # Line thickness of 1 px
        #     # mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #     mask = masks[i].astype(np.uint8) * 255
        #     mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #     mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
        #     cv2.imwrite('C:/grasp_static/outputs/mask_bboxs_{}.png'.format(i), mask_bboxs)

        #     bbox_image = color_image[y0:y1, x0:x1]
        #     cv2.imwrite('C:/grasp_static/outputs/mask_color_{}.png'.format(i), bbox_image)
        #     i += 1
        
        # blank_depth_img = np.full((color_image.shape[0], color_image.shape[1]), 1, dtype=np.float32)
        # for j in range(len(bboxes)):
        #     mask_circle = np.uint8(masks[j])
        #     # contours, _ = cv2.findContours(mask_circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #     # (x,y), radius = cv2.minEnclosingCircle(contours[0])
        #     # cv2.circle(mask_circle, (int(x), int(y)), int(radius), (255, 255, 255), cv2.FILLED)
        #     blank_depth_img[mask_circle > 0] = depth_img[mask_circle > 0]
        
        black_image = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint8)
        background_img = cv2.imread('zy/background_0.png')
        for j in range(len(bboxes)):
            mask_circle = np.uint8(masks[j])
            # contours, _ = cv2.findContours(mask_circle, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # (x,y), radius = cv2.minEnclosingCircle(contours[0])
            # cv2.circle(mask_circle, (int(x), int(y)), int(radius), (255, 255, 255), cv2.FILLED)
            if classes[labels[j]] == label:
                black_image[mask_circle > 0] = 255
                background_img[mask_circle > 0] = color_image[mask_circle > 0]
        # cv2.imwrite('C:/grasp_static/outputs/mask1.png', black_image)
        
        # plt.figure()
        # plt.imshow(background_img[..., ::-1])
        # plt.show()
        # cv2.imwrite('C:/grasp_static/outputs/depth_interest.png', black_image)

        color_img, depth_img, ratio, padw, padh = self.letterbox1(color_img, depth_img, 100)
        # color_img, depth_img, ratio, padw, padh = self.letterbox1(background_img, depth_img, 100)
        img_in = self.to_tensor(depth_img, color_img, rgb_include=1, depth_include=0)
        k_max = 100
        # topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, blank_depth_img, pixel_wise_stride=1)
        topk_grasps, best_grasp = self.generate_grasp_yolo(img_in, color_img_raw, k_max, 0, black_image, pixel_wise_stride=1)
        # best_grasp, feature_maps = self.generate_grasp(img_in)
        # pixel_wise_stride = depth_img.shape[0] / feature_maps[0].shape[0]
        coordinate, width_gripper, angle = self.grasp_img2real_yolo(
                        color_img_raw, depth_img_raw, best_grasp, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
                    )
        # angle = self.grasp_imgshow(
        #         color_img_raw, best_grasp, 0, vis=vis, color=(0, 255, 0), note='', collision_check=True
        # )
        if 0 <= angle <= np.pi/2:
            angle = np.pi/2 - angle
        else:
            angle = -np.pi/2 - angle

        mat = cv2.Rodrigues(np.array([0, 0, angle]).astype(np.float32))[0]
        Tobj2cam = np.vstack((
                        np.hstack(
                            (np.dot(self.Rbase2cam, np.array(mat)),
                             np.expand_dims(coordinate, 0).T)
                        ), np.array([[0, 0, 0, 1]])
                        ))
        # k = 10
        # for grasp in topk_grasps:
        #     k += 1
        #     # note =  "top_{}grasp".format(int(k))
        #     # coordinate, width_gripper, angle = self.grasp_img2real_yolo(
        #     #     color_img_raw, depth_img_raw, grasp, k, vis=vis, color=(0, 255, 0), note= note, collision_check=True
        #     # )
        #     self.grasp_imgshow(
        #         color_img_raw, grasp, k, vis=vis, color=(0, 255, 0), collision_check=True
        #     )

        # dt = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        # cv2.imwrite('/home/server/grasp_jq/real_grasp_virtual_train2/'+dt+'.png', color_img_raw)


        if self.hardware:
            self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
            self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed, 0)
            Tobj2base = self.Tcam2base.dot(Tobj2cam)
            position = Tobj2base[0:3, 3]
            gesture = cv2.Rodrigues(Tobj2base[0:3, 0:3])[0].T.squeeze()
            pose = np.hstack((position, gesture))
            # print(pose)
            pose = pose + [0, 0, -0.197, 0, 0, 0]
            if pose[2] > 0.540:
                pose[2] = 0.538
            print(pose)
            pose_up_to_grasp_position = pose + [0, 0, -0.05, 0, 0, 0]
            tag, up_to_grasp_joint = self.robot.Algo_Inverse_Kinematics(self.mid_pose1, pose_up_to_grasp_position, 1)
            print(up_to_grasp_joint)
            if tag==0:
                # self.gripper.gripper_position(int((0.12-width_gripper)/0.12*3*51250))
                # self.gripper.gripper_position(int(51200))
                self.gripper.gripper_position(0)
                # pose_up_to_grasp_position = [0.12221, 0.026199, 0.49557, 0.0068711, 0.077718, -2.5144]
                self.robot.Movej_Cmd(up_to_grasp_joint[0:6], self.robot_speed)
                tag1, pose_joint = self.robot.Algo_Inverse_Kinematics(up_to_grasp_joint[0:6], pose, 1)
                print(pose_joint)
                self.robot.Movej_Cmd(pose_joint[0:6], self.robot_speed)
                # self.robot.Movel_Cmd(pose_up_to_grasp_position, 10)
                # self.robot.Movel_Cmd(pose, 10)
                # self.gripper.gripper_position(5*51200)
                self.gripper.gripper_position(1)
                time.sleep(0.7)
                self.robot.Movej_Cmd(up_to_grasp_joint[0:6], self.robot_speed)
                # self.robot.Movel_Cmd(pose_up_to_grasp_position, 20)
                self.robot.Movej_Cmd(self.mid_pose1, self.robot_speed) 
                # self.robot.Movej_Cmd(self.mid_pose, self.robot_speed) 
                # self.robot.Movej_Cmd(self.init_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.lift2init_pose, self.robot_speed)
                # self.robot.Movej_Cmd(self.place_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.place_mid_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.place_mid_pose2, self.robot_speed)
                self.robot.Movej_Cmd(self.place_last_pose, self.robot_speed)
                self.gripper.gripper_position(0)
                time.sleep(0.7)
                # self.gripper.gripper_position(5*51200)
                self.gripper.gripper_position(1)
                self.robot.Movej_Cmd(self.place_mid_pose, self.robot_speed)
                # self.gripper.gripper_position(5*51200)
                # self.robot.Movej_Cmd(self.place_pose, self.robot_speed)
                self.robot.Movej_Cmd(self.init_pose, self.robot_speed)
            #     # self.gripper.gripper_position(100)
            else:
                print(f"机械臂逆解失败！请重新放置物体位置")
                self.robot.Movej_Cmd(self.mid_pose, self.robot_speed, 0)
                self.robot.Movej_Cmd(self.init_pose, self.robot_speed, 0)



if __name__ == '__main__':
    grasp = Grasp(hardware=True)

    # while 1:
    # grasp.action_yolo(label='voltage', vis=True)  ## soap interrupter terminal limit voltage carrot
    grasp.obj_grasp(label='carrot',vis=True)
    #grasp.obj_grasp(label='voltage',vis=True)

    # exist_if = grasp.detect_obj(label='relay')
    # if exist_if:
    #     grasp.obj_grasp(label='orange')
    # else:
    #     print(f"不存在该类物体！")
    # if isinstance(num_obj, (int, float)):  # 判断是否是 int 或 float
    #     for i in range(int(num_obj)):
    #         grasp.obj_grasp(label='orange')
    # else:
    #     print(f"不存在该类物体: {num_obj}")
    # torch.cuda.empty_cache()
        # pass
        # gras/p.judge()
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

    # start_time = time.time()
    # grasp.action_yolo(label='banana', vis=True)
    # total_time = time.time() - start_time
    # print(f"推理时间：{total_time * 1000:.2f}毫秒")

    # num_runs = 100
    # totoal = 0
    # with torch.no_grad():
    #     for _ in range(num_runs):
    #         start_time = time.time()
    #         grasp.action_yolo(label='banana', vis=True)
    #         end_time = time.time()
    #         total_time = end_time - start_time
    #         print(f"推理时间：{total_time * 1000:.2f}毫秒")
    #         totoal += (end_time - start_time)
    
    # average = totoal / num_runs
    # print(f"Average inference time: {average * 1000:.2f} ms")

    # torch.cuda.empty_cache()
        # pass
        # gras/p.judge()
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












