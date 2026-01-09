# 抓取姿态消息
std_msgs/Header header

# 抓取中心点在图像中的位置
int32 row
int32 column

# 抓取角度（弧度）
float64 angle

# 夹爪宽度（像素）
float64 width
float64 height

# 机器人基座坐标系下的位置
geometry_msgs/Point position

# 机器人基座坐标系下的姿态（欧拉角）
geometry_msgs/Vector3 orientation

# 夹爪应该张开的宽度（米）
float64 gripper_width

# 质量分数
float64 quality

# 是否需要边缘倾斜补偿
bool slope_flag

