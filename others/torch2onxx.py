import torch
import torch.onnx
import genotypes as gt
from gqcnn_server.augment_cnn import AugmentCNN
# 假设你有一个 PyTorch 模型
file_path = '/home/jet/zoneyung/grasp_static/single_zy.txt'
# file_path = 'C:/grasp_static/single_zy.txt'
# file_path = '/home/jet/zoneyung/grasp_static/single_rgb.txt'
with open(file_path, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        gene = line
        getypr = gt.from_str(gene)
model = AugmentCNN('/home/jet/zoneyung/grasp_static/cornell.data', 100, 3, 8, 5, False, getypr).cuda()
model.load_state_dict(torch.load('/home/jet/zoneyung/grasp_static/weights/tune_epoch_64_loss_0.0297_accuracy_1.000'))
model.eval()

# 创建一个示例输入
dummy_input = torch.randn(1, 3, 100, 100).cuda()

# 导出为 ONNX 格式
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=["input"], output_names=["output"], opset_version=11)