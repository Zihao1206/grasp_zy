import tifffile
import numpy as np
import cv2

# ======== 1. 读取深度图 ========
depth_path = "doc/try_data/ed.tiff"          # 修改为你的深度图路径
depth = tifffile.imread(depth_path)      # uint16 深度图

print("Depth shape:", depth.shape)
print("Depth dtype:", depth.dtype)


# ======== 2. 过滤无效深度 ========
# RealSense 中常见：0 或特定值表示无效
valid_mask = depth > 0
valid_depth = depth[valid_mask]

if valid_depth.size == 0:
    raise RuntimeError("深度图全为 0，无有效像素！")


# ======== 3. 计算深度的中位数（代表桌面/容器平面） ========
median_depth = np.median(valid_depth)
print("Estimated median depth:", median_depth)


# ======== 4. 基于深度范围生成 workspace region ========
# 区域范围：中位数 ± 200 mm（可根据实际情况调整）
# 通常容器深度小于 200mm，这个范围最通用
lower = median_depth - 200
upper = median_depth + 200

workspace_mask = (depth > lower) & (depth < upper)
workspace_mask = workspace_mask.astype(np.uint8) * 255


# ======== 5. 可选：开闭操作，清理噪声（推荐） ========
kernel = np.ones((7,7), np.uint8)
workspace_mask = cv2.morphologyEx(workspace_mask, cv2.MORPH_CLOSE, kernel)
workspace_mask = cv2.morphologyEx(workspace_mask, cv2.MORPH_OPEN, kernel)


# ======== 6. 保存 mask ========
output_path = "doc/try_data/workspace_mask.png"
cv2.imwrite(output_path, workspace_mask)

print("Saved:", output_path)
