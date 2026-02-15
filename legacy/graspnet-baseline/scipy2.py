import scipy.io as sio
import numpy as np

# 读取MAT文件
def read_mat_file(file_path):
    try:
        # 加载MAT文件
        mat_data = sio.loadmat(file_path)
        
        # 打印所有变量名
        print("MAT文件中的变量:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # 过滤掉系统变量
                print(f"变量名: {key}, 类型: {type(mat_data[key])}, 形状: {mat_data[key].shape}")
        
        return mat_data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 示例使用
if __name__ == "__main__":
    # file_path = "doc/example_data/meta.mat"  # 替换为你的MAT文件路径
    file_path = "doc/try_data/meta_modified.mat"
    
    # 读取MAT文件
    data = read_mat_file(file_path)
    
    if data is not None:
        # 访问特定变量
        for key in data.keys():
            if not key.startswith('__'):
                variable = data[key]
                print(f"\n变量 '{key}' 的详细信息:")
                print(f"数据类型: {type(variable)}")
                print(f"形状: {variable.shape}")
                print(f"前几个元素: {variable.flat[:10] if variable.size > 10 else variable}")