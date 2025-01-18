import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def count_dropouts(module):
    """递归统计给定模块及其子模块中 Dropout 层的个数。"""
    count = 0
    for submodule in module.modules():
        # 如果是 nn.Dropout（或自定义 CustomDropout），都可以在这里统计
        if isinstance(submodule, nn.Dropout):
            count += 1
    return count

def main():
    model_name = "meta-llama/Llama-3.2.3B"  # 请根据实际情况替换

    print(f"Loading model: {model_name}")
    # 如果需要身份验证
    # 注意在使用私有模型时需要设置 'use_auth_token=True' 或者在环境中设置 TOKEN
    # 例如: export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
    # 或者把 'use_auth_token="hf_XXXXX"' 写在 from_pretrained(...) 里
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    dropout_count = count_dropouts(model)
    print(f"Number of Dropout layers in '{model_name}': {dropout_count}")

if __name__ == "__main__":
    main()
