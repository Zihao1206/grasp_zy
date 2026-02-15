import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义长方体空间的边界
def is_within_bounds(point, bounds):
    return np.all(point >= bounds['min']) and np.all(point <= bounds['max'])

# 长方体空间的边界
bounds = {
    'min': np.array([-1, -0.26, 0]),
    'max': np.array([1, 0.26, 0.62])
}

# 初始位姿和目标位姿
initial_pose = np.array([0.23, -0.00022, 0.572])
target_pose = np.array([0.098, -0.0051, 0.568])

# 检查初始位姿和目标位姿是否在边界内
if not is_within_bounds(initial_pose, bounds) or not is_within_bounds(target_pose, bounds):
    raise ValueError("Initial or target pose is outside the bounds.")

# 路径规划（简单的直线插值）
def plan_path(initial_pose, target_pose):
    path = np.linspace(initial_pose, target_pose, 100)
    return path

# 调整路径以适应边界
def adjust_path_to_bounds(path, bounds):
    adjusted_path = []
    for point in path:
        adjusted_point = np.copy(point)
        for i in range(len(bounds['min'])):
            if adjusted_point[i] < bounds['min'][i]:
                adjusted_point[i] = bounds['min'][i]
            elif adjusted_point[i] > bounds['max'][i]:
                adjusted_point[i] = bounds['max'][i]
        adjusted_path.append(adjusted_point)
    return np.array(adjusted_path)

# 规划路径
path = plan_path(initial_pose, target_pose)

# 调整路径
adjusted_path = adjust_path_to_bounds(path, bounds)

# 可视化路径
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(adjusted_path[:, 0], adjusted_path[:, 1], adjusted_path[:, 2], label='Adjusted Path')
ax.scatter(initial_pose[0], initial_pose[1], initial_pose[2], color='r', label='Start')
ax.scatter(target_pose[0], target_pose[1], target_pose[2], color='g', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()