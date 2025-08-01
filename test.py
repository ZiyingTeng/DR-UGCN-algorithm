import numpy as np
import torch
import torch_scatter
import torch_sparse

print("=== 环境验证 ===")
print("NumPy版本:", np.__version__)  # 应该显示1.x
print("PyTorch版本:", torch.__version__)
print("设备:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 应该显示cpu
print("torch_scatter可用:", hasattr(torch_scatter, 'scatter_add'))
print("torch_sparse可用:", hasattr(torch_sparse, 'SparseTensor'))