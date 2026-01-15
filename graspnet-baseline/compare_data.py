#!/usr/bin/env python3
"""
比较 example_data 和 try_data 的差异
"""

import scipy.io as scio
import numpy as np
from PIL import Image
import os

def compare_mat_files():
    """比较 meta.mat 文件的差异"""
    print("=" * 60)
    print("比较 meta.mat 文件")
    print("=" * 60)
    
    # 读取 example_data 的 meta
    meta_example = scio.loadmat('doc/example_data/meta.mat')
    
    # 读取 try_data 的 meta
    meta_try = scio.loadmat('doc/try_data/meta_modified.mat')
    
    # 比较 intrinsic_matrix
    print("\n1. intrinsic_matrix 比较:")
    print("   example_data shape:", meta_example['intrinsic_matrix'].shape)
    print("   try_data shape:", meta_try['intrinsic_matrix'].shape)
    
    if meta_example['intrinsic_matrix'].shape == meta_try['intrinsic_matrix'].shape:
        diff = meta_example['intrinsic_matrix'] - meta_try['intrinsic_matrix']
        max_diff = np.max(np.abs(diff))
        print("   最大差异值:", max_diff)
        if max_diff > 1e-6:
            print("   example_data intrinsic_matrix:\n", meta_example['intrinsic_matrix'])
            print("   try_data intrinsic_matrix:\n", meta_try['intrinsic_matrix'])
            print("   差异:\n", diff)
    else:
        print("   形状不同！")
    
    # 比较 factor_depth
    print("\n2. factor_depth 比较:")
    print("   example_data:", meta_example['factor_depth'])
    print("   try_data:", meta_try['factor_depth'])
    
    diff_factor = meta_example['factor_depth'] - meta_try['factor_depth']
    if np.abs(diff_factor) > 1e-6:
        print("   差异:", diff_factor)
    else:
        print("   相同")

def compare_images():
    """比较图像文件的差异"""
    print("\n" + "=" * 60)
    print("比较图像文件")
    print("=" * 60)
    
    # 比较 workspace_mask.png
    print("\n1. workspace_mask.png 比较:")
    
    mask_example = np.array(Image.open('doc/example_data/workspace_mask.png').convert('L'))
    mask_try = np.array(Image.open('doc/try_data/workspace_mask.png').convert('L'))
    
    print("   example_data shape:", mask_example.shape)
    print("   try_data shape:", mask_try.shape)
    print("   example_data dtype:", mask_example.dtype)
    print("   try_data dtype:", mask_try.dtype)
    
    if mask_example.shape == mask_try.shape:
        # 统计值
        print("\n   example_data workspace_mask 统计:")
        print("     - 值为0的像素数:", np.sum(mask_example == 0))
        print("     - 值为255的像素数:", np.sum(mask_example == 255))
        print("     - 其他值的像素数:", np.sum((mask_example > 0) & (mask_example < 255)))
        
        print("\n   try_data workspace_mask 统计:")
        print("     - 值为0的像素数:", np.sum(mask_try == 0))
        print("     - 值为255的像素数:", np.sum(mask_try == 255))
        print("     - 其他值的像素数:", np.sum((mask_try > 0) & (mask_try < 255)))
        
        # 差异
        diff_mask = mask_example.astype(float) - mask_try.astype(float)
        print("\n   差异统计:")
        print("     - 平均差异:", np.mean(np.abs(diff_mask)))
        print("     - 最大差异:", np.max(np.abs(diff_mask)))
        print("     - 差异大于10的像素数:", np.sum(np.abs(diff_mask) > 10))
    
    # 比较深度图
    print("\n2. 深度图比较:")
    
    # example_data 使用 depth.png
    depth_example = np.array(Image.open('doc/example_data/depth.png'))
    print("   example_data depth.png shape:", depth_example.shape)
    print("   example_data depth.png dtype:", depth_example.dtype)
    print("   example_data depth.png range: [{}, {}]".format(depth_example.min(), depth_example.max()))
    
    # try_data 使用 d.tiff
    depth_try = np.array(Image.open('doc/try_data/d.tiff'))
    print("   try_data d.tiff shape:", depth_try.shape)
    print("   try_data d.tiff dtype:", depth_try.dtype)
    print("   try_data d.tiff range: [{}, {}]".format(depth_try.min(), depth_try.max()))
    
    if depth_example.shape == depth_try.shape:
        # 归一化后比较
        depth_example_norm = depth_example.astype(float) / depth_example.max()
        depth_try_norm = depth_try.astype(float) / depth_try.max()
        depth_diff = np.abs(depth_example_norm - depth_try_norm)
        print("   归一化后的平均差异:", np.mean(depth_diff))
        print("   归一化后的最大差异:", np.max(depth_diff))

def check_file_structure():
    """检查文件结构"""
    print("\n" + "=" * 60)
    print("文件结构检查")
    print("=" * 60)
    
    example_files = os.listdir('doc/example_data')
    try_files = os.listdir('doc/try_data')
    
    print("\nexample_data 文件:", example_files)
    print("try_data 文件:", try_files)
    
    print("\n文件差异:")
    only_in_example = set(example_files) - set(try_files)
    only_in_try = set(try_files) - set(example_files)
    
    if only_in_example:
        print("  只在 example_data 中的文件:", only_in_example)
    if only_in_try:
        print("  只在 try_data 中的文件:", only_in_try)

if __name__ == '__main__':
    compare_mat_files()
    compare_images()
    check_file_structure()