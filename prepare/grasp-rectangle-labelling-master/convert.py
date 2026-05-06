import os
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder):
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入目录中的文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            # 读取PNG图片
            png_path = os.path.join(input_folder, filename)
            img = Image.open(png_path)
            
            # 转换颜色模式（解决PNG透明背景问题）
            rgb_img = img.convert('RGB')
            
            # 生成新的文件名和路径
            base_name = os.path.splitext(filename)[0]
            jpg_path = os.path.join(output_folder, f"{base_name}.jpg")
            
            # 保存为JPG格式（可调整质量参数）
            rgb_img.save(jpg_path, 'JPEG', quality=95)
            print(f"已转换：{filename} → {os.path.basename(jpg_path)}")

# 调用示例
input_dir = "C:/Users/SZJX/Desktop/grasp-rectangle-labelling-master/Images/zy_label"  # 替换为PNG图片所在目录
output_dir = "C:/Users/SZJX/Desktop/grasp-rectangle-labelling-master/Images/zy_jpg" # 替换为JPG输出目录
convert_png_to_jpg(input_dir, output_dir)