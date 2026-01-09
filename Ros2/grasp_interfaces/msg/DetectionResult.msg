# 物体检测结果
std_msgs/Header header

# 检测到的物体类别
string[] class_names

# 每个物体的边界框 [x1, y1, x2, y2]
float32[] bboxes

# 每个物体的置信度
float32[] scores

# 每个物体的分割掩码（序列化的numpy数组）
uint8[] masks_data
int32[] masks_shape

# 每个物体的标签ID
int32[] labels

