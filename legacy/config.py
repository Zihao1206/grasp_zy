place_last_pose = [77, -104, 38, -2, 9, 1] ## 机械臂向下放置物体

# 相机到基座的坐标变换矩阵
Tcam2base = [
            [0.02742095, -0.99940903, -0.0207286, 0.20841901],
            [0.9995487, 0.02766746, -0.01170045, -0.02848768],
            [0.01226705, -0.02039841, 0.99971667, 0.03739014],
            [0., 0., 0., 1., ]]
Rbase2cam = [
            [0.02742095, 0.9995487, 0.01226705],
            [-0.99940903, 0.02766746, -0.02039841],
            [-0.0207286, -0.01170045, 0.99971667]]

angle = 1/7

# 位置补偿计算
t_tcp_flange = ([0, 0, 0.2])
tcp_compensate = ([0, 0, 0.018])
slope_flag = True
# slope_angle = np.pi/5
column_left, column_right = 80, 480
# column_left, column_right = 80, 430
row_up, row_down = 160, 292

robot_speed =20

type = 'daikon'
## soap interrupter terminal limit voltage carrot banana daikon

# systemctl stop RunGraspd.service
# stop camera

# 机械臂默认速度
robot_speed =20
# Z轴（下降深度）
pose2_2=0.540
pose2=0.540
# /home/jet/zoneyung/grasp_static/grasp_zy_zhiyuan0828.py\