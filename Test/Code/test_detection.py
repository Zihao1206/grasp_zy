import torch
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from models.nms import nms

def test_detection():
    """测试目标检测模型"""
    
    # 1. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    config_file = 'models/mmdetection/configs/myconfig_zy.py'
    check_point = 'models/weights/epoch_20_last.pth'
    
    try:
        model = init_detector(config_file, check_point, device=device)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 2. 测试图像
    test_images = [
        'test_data/image1.jpg',
        'test_data/image2.jpg',
        'test_data/image3.jpg',
    ]
    
    for img_path in test_images:
        print(f"\n=== 测试图像: {img_path} ===")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"  无法读取图像")
            continue
        
        # 推理
        result = inference_detector(model, img)
        
        # 解析结果
        if hasattr(result, 'pred_instances'):
            bboxes = result.pred_instances.bboxes.cpu().numpy()
            scores = result.pred_instances.scores.cpu().numpy()
            labels = result.pred_instances.labels.cpu().numpy()
            
            # 获取类别名称
            classes = model.dataset_meta.get('classes', [])
            
            print(f"  检测到 {len(bboxes)} 个物体:")
            
            for i in range(min(5, len(bboxes))):  # 显示前5个
                if scores[i] > 0.5:
                    bbox = bboxes[i].astype(int)
                    class_id = int(labels[i])
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    
                    print(f"    物体 {i+1}: {class_name} (置信度: {scores[i]:.3f})")
                    print(f"      边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        
        # 可视化
        vis_img = img.copy()
        for i in range(len(bboxes)):
            if scores[i] > 0.5:
                bbox = bboxes[i].astype(int)
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 保存结果
        output_path = f"detection_{img_path.split('/')[-1]}"
        cv2.imwrite(output_path, vis_img)
        print(f"  结果保存到: {output_path}")

def test_nms():
    """测试NMS功能"""
    print("\n=== 测试NMS功能 ===")
    
    # 模拟预测数据
    predicts = np.array([
        [0.95, 100, 100, 200, 200, 0],  # 类别0，置信度0.95
        [0.90, 110, 110, 210, 210, 0],  # 类别0，置信度0.90（与第一个重叠）
        [0.80, 300, 300, 400, 400, 1],  # 类别1，置信度0.80
        [0.85, 120, 120, 220, 220, 0],  # 类别0，置信度0.85（与第一个重叠）
    ])
    
    try:
        bboxes, indics, labels = nms_module.nms(predicts, 0.8, 0.9)
        print(f"原始检测框: {len(predicts)}")
        print(f"NMS后检测框: {len(bboxes)}")
        print(f"保留索引: {indics}")
        print(f"保留标签: {labels}")
    except Exception as e:
        print(f"NMS测试失败: {e}")

if __name__ == "__main__":
    test_detection()
    test_nms()