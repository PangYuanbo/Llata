import torch

# 创建一个浮点张量
x = torch.randn(10, 10)

try:
    # 尝试将张量转换为 bf16，并执行一个简单操作
    x_bf16 = x.to(torch.bfloat16)
    y = x_bf16 + x_bf16
    print("CPU 支持 bf16 操作。")
except (RuntimeError, TypeError) as e:
    print("CPU 可能不支持 bf16 操作或缺少相关优化。")
    print("错误信息：", e)
