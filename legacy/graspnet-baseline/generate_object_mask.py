#!/usr/bin/env python3
"""
生成物体mask的工具脚本
使用空场景减法法：有物体的深度图 - 空场景深度图 = 物体mask
"""

import numpy as np
from PIL import Image
import argparse
import os

def generate_object_mask(empty_depth_path, object_depth_path, output_path, threshold=50):
    """
    通过空场景减法生成物体mask
    
    Args:
        empty_depth_path: 空场景深度图路径
        object_depth_path: 有物体的深度图路径
        output_path: 输出的mask路径
        threshold: 深度差异阈值，大于此值认为是物体
    """
    # 读取深度图
    empty_depth = np.array(Image.open(empty_depth_path))
    object_depth = np.array(Image.open(object_depth_path))
    
    print(f"Empty depth shape: {empty_depth.shape}, range: [{empty_depth.min()}, {empty_depth.max()}]")
    print(f"Object depth shape: {object_depth.shape}, range: [{object_depth.min()}, {object_depth.max()}]")
    
    # 确保尺寸相同
    if empty_depth.shape != object_depth.shape:
        raise ValueError(f"深度图尺寸不匹配: {empty_depth.shape} vs {object_depth.shape}")
    
    # 计算深度差异
    # 有物体的地方深度值会小于空场景（物体更近）
    depth_diff = empty_depth.astype(np.float32) - object_depth.astype(np.float32)
    
    # 创建mask：差异大于阈值的地方认为是物体
    # 同时排除无效深度（depth=0）
    object_mask = (depth_diff > threshold) & (object_depth > 0)
    
    # 将mask转换为0-255的图像
    mask_image = np.zeros_like(empty_depth, dtype=np.uint8)
    mask_image[object_mask] = 255
    
    # 保存mask
    Image.fromarray(mask_image).save(output_path)
    
    # 统计信息
    object_pixels = np.sum(object_mask)
    total_pixels = object_mask.size
    print(f"Object pixels: {object_pixels} / {total_pixels} ({object_pixels/total_pixels*100:.2f}%)")
    print(f"Mask saved to: {output_path}")
    
    return mask_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate object mask using empty scene subtraction')
    parser.add_argument('--empty', default='doc/try_data/ed.tiff', help='Path to empty scene depth image')
    parser.add_argument('--object', default='doc/try_data/d.tiff', help='Path to object scene depth image')
    parser.add_argument('--output', default='doc/try_data/maskoutput.png', help='Path to output mask image')
    parser.add_argument('--threshold', type=int, default=50, help='Depth difference threshold (default: 50)')
    
    args = parser.parse_args()
    
    generate_object_mask(args.empty, args.object, args.output, args.threshold)