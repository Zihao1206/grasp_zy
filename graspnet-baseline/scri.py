import scipy.io as sio
import numpy as np

def modify_intrinsic_matrix(input_path, output_path):
    """
    修改MAT文件中的内参矩阵并保存
    
    Parameters:
    input_path: 输入MAT文件路径
    output_path: 输出MAT文件路径
    """
    # 读取MAT文件
    mat_data = sio.loadmat(input_path)
    
    # 定义新的内参矩阵参数
    # fx, fy = 392.25048828125, 392.25048828125
    # cx, cy = 320.16729736328125, 242.32826232910156
    fx, fy = 600.4671630859375, 600.12939453125
    cx, cy = 326.7477722167969, 249.3929901123047
    
    # 创建新的内参矩阵
    new_intrinsic_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]], 
                                   dtype=mat_data['intrinsic_matrix'].dtype)
    
    # 替换内参矩阵
    mat_data['intrinsic_matrix'] = new_intrinsic_matrix
    
    # 保存修改后的MAT文件
    sio.savemat(output_path, mat_data)
    
    print(f"内参矩阵修改完成!")
    print(f"原始文件: {input_path}")
    print(f"新文件: {output_path}")
    print(f"新的内参矩阵:\n{new_intrinsic_matrix}")

# 使用示例
modify_intrinsic_matrix("doc/example_data/meta.mat", "doc/try_data/meta_modified.mat")
