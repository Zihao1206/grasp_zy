#!/usr/bin/env python
# coding=utf-8
#拍照 - 使用新 API 包 (RM_API2-1.1.1)
import logging
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys

sys.path.insert(0, '/home/zh/zh/grasp_zy_py310/RM_API2-1.1.1/Python')

from log_setting import CommonLog
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

CODE_fi = 65
HOST_fi = '192.168.127.101'

cam0_path = '/home/zh/zh/grasp_zy_py310/prepare/D435/data_collection_d435_win/images'

if not os.path.exists(cam0_path):
    os.makedirs(cam0_path)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)
count = 0

def callback(frame, arm):
    global count

    cv2.imshow("Capture_Video", frame)

    k = cv2.waitKey(30) & 0xFF
    if k == ord('s'):
        ret, state_dict = arm.rm_get_current_arm_state()
        logger_.info(f'获取状态：{"成功" if ret == 0 else "失败，错误码: {ret}"}')

        if ret == 0:
            pose_list = state_dict.get('pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            logger_.info(f'当前位姿为: {pose_list}')

            with open(os.path.join(cam0_path, 'pose.text'), 'a+') as f:
                pose_str = [str(i) for i in pose_list]
                new_line = f'{",".join(pose_str)}\n'
                f.write(new_line)

            cv2.imwrite(os.path.join(cam0_path, '{}.jpg'.format(count)), frame)
            count += 1
    else:
        pass


def displayD435(arm):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    global count

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image, arm)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logger_.info('初始化新 API 机械臂...')

    try:
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        arm_handle = arm.rm_create_robot_arm(HOST_fi, 8080)
        logger_.info(f'机械臂连接成功: handle={arm_handle}')

        arm.rm_set_collision_state(5)
        logger_.info('碰撞检测已设置 (stage=5)')
    except Exception as e:
        logger_.error(f'机械臂连接失败: {e}')
        sys.exit(1)

    displayD435(arm)
