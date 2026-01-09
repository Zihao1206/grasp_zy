# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np

# # 创建 TensorRT 引擎
# def build_engine(onnx_file_path, engine_file_path):
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     parser = trt.OnnxParser(network, TRT_LOGGER)

#     with open(onnx_file_path, 'rb') as model:
#         parser.parse(model.read())

#     config = builder.create_builder_config()
#     config.set_flag(trt.BuilderFlag.FP16)
#     # config.set_flag(trt.BuilderFlag.INT8)
#     config.max_workspace_size = 1 << 30  # 1GB
#     builder.max_batch_size = 1

#     engine = builder.build_engine(network, config)

#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())

#     return engine

# # 构建引擎
# onnx_file_path = "model_finetune.onnx"
# engine_file_path = "model_finetune.trt"
# engine = build_engine(onnx_file_path, engine_file_path)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

# 定义 Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 创建 TensorRT 构建器和网络定义
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 解析 ONNX 模型
parser = trt.OnnxParser(network, TRT_LOGGER)
success = parser.parse_from_file("model_rgb.onnx")
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    raise RuntimeError("Failed to parse ONNX model")

# 配置构建器
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.INT8)

# 定义校准器类
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, calibration_data):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.calibration_data = calibration_data
        self.batch_size = 1
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.calibration_data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.calibration_data[self.current_index:self.current_index + self.batch_size]
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# 准备校准数据
calibration_data = [np.random.randn(1, 3, 100, 100).astype(np.float32) for _ in range(100)]  # 100 batches of random data

# 创建校准器
calibrator = Calibrator("calibration.cache", calibration_data)
config.int8_calibrator = calibrator

# 构建引擎
engine = builder.build_engine(network, config)

# 保存引擎
with open("model.engine", "wb") as f:
    f.write(engine.serialize())