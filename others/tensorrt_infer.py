# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import time

# output_shape = [1, 3750, 5]
# # 加载 TensorRT 引擎
# def load_engine(engine_file_path):
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         return runtime.deserialize_cuda_engine(f.read())

# # 创建执行上下文
# def create_context(engine):
#     return engine.create_execution_context()

# # 推理
# def infer(context, input_data):
#     # 分配输入和输出缓冲区
#     input_size = trt.volume(input_data.shape) * input_data.dtype.itemsize
#     output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
#     d_input = cuda.mem_alloc(input_size)
#     d_output = cuda.mem_alloc(output_size)

#     # 创建流
#     stream = cuda.Stream()

#     # 将输入数据复制到设备
#     cuda.memcpy_htod_async(d_input, input_data, stream)

#     # 执行推理
#     context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

#     # 将输出数据复制回主机
#     output_data = np.empty(output_shape, dtype=np.float32)
#     cuda.memcpy_dtoh_async(output_data, d_output, stream)

#     # 同步流
#     stream.synchronize()

#     return output_data

# # 加载引擎
# engine_file_path = "model_trt.engine"
# engine = load_engine(engine_file_path)
# context = create_context(engine)

# # 创建示例输入
# input_data = np.random.randn(1, 3, 100, 100).astype(np.float32)

# # 推理
# t = time.time()
# output_data = infer(context, input_data)
# t1 = time.time()
# forward_time = (t1 - t) * 1000  # 将时间差乘以 1000
# print('forward time:{:.6f} ms'.format(forward_time))
# print(output_data)

import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 定义一些常量
BATCH_SIZE = 1
INPUT_SHAPE = (3, 100, 100)  # 根据你的模型输入形状调整
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 100

# 加载 TensorRT 引擎
def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
def create_context(engine):
    return engine.create_execution_context()

# 分配 GPU 内存
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

# 定义 HostDeviceMem 类
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

# 执行推理
def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

# 主函数
def main():
    engine_path = 'model_rgb.engine'
    engine = load_engine(engine_path)
    context = create_context(engine)
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 生成随机输入数据
    input_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    input_data = np.ascontiguousarray(input_data)
    np.copyto(inputs[0].host, input_data.ravel())

    # 预热
    for _ in range(WARMUP_ITERATIONS):
        do_inference(context, bindings, inputs, outputs, stream)

    # 测量推理时间
    start_time = time.perf_counter()
    for _ in range(MEASUREMENT_ITERATIONS):
        do_inference(context, bindings, inputs, outputs, stream)
    end_time = time.perf_counter()

    # 计算平均推理时间
    avg_time = (end_time - start_time) / MEASUREMENT_ITERATIONS
    print(f"Average inference time: {avg_time * 1000:.2f} ms")

if __name__ == "__main__":
    main()

