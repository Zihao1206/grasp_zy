import tifffile
import numpy as np

depth = tifffile.imread("doc/try_data/d.tiff")
print(depth.dtype)  # 一般是 uint16
print(np.min(depth), np.max(depth))
