# 
import numpy as np
import torch
# # 给定的速度分量
# vx = 0.2
# vy = -0.2

# # 计算合速度
# v = np.sqrt(vx**2 + vy**2)

# # 计算方向角度（以弧度为单位）
# theta = np.arctan2(vy, vx)

# # 将角度转换为度数
# theta_degrees = np.degrees(theta)

# # 取绝对值
# theta_abs_degrees = np.abs(theta_degrees)

# print(f"合速度: {v:.2f}")
# print(f"方向角度: {theta:.2f}")
# print(f"方向角度 (绝对值): {theta_abs_degrees:.2f}°")


import cv2
import numpy as np

# # 读取图像
# image = cv2.imread('C:/Users/SZJX/Desktop/230704_205444.png')

# # 转换为灰度图像
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 应用阈值分割
# _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# # 创建一个黑色背景
# black_background = np.zeros_like(image)

# # 将前景复制到黑色背景上
# result = cv2.bitwise_and(image, image, mask=thresh)

# # 显示结果
# cv2.imshow('Original Image', image)
# cv2.imshow('Black Background Image', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

depth = np.random.rand(480, 480)

depthImage = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
depth_max = torch.max(depthImage)
depth_min = torch.min(depthImage)

pose = torch.arange(5) / 5 * (depth_max - depth_min) + depth_min
pose = pose.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
pose = (pose - 0.5) / 0.6
depthImage = (depthImage - 0.5) / 0.6
depthImage = depthImage - pose  # [5, 1, 480, 480]
print(depthImage)