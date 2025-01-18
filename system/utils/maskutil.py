import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 定义 MaskManager
class MaskManager:
    def __init__(self):
        self.masks = []  # 存储所有 Dropout 层的掩码
        self.current_index = 0  # 当前掩码索引

    def add_mask(self, mask):
        self.masks.append(mask)

    def get_mask(self, index):
        if index < len(self.masks):
            return self.masks[index]
        else:
            raise IndexError(f"No mask found at index {index}")

    def reset(self):
        self.masks = []
        self.current_index = 0

# 2. 定义 CustomDropout
class CustomDropout(nn.Module):
    def __init__(self, p=0.5, mask_manager=None, mask_index=None):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.mask_manager = mask_manager
        self.mask_index = mask_index

    def forward(self, input, reuse_mask=False):
        if self.training:
            if reuse_mask:
                # 从 MaskManager 获取掩码
                mask = self.mask_manager.get_mask(self.mask_index)
            else:
                # 生成新的掩码
                mask = (torch.rand_like(input) > self.p).float() / (1.0 - self.p)
                # 将掩码存储到 MaskManager
                self.mask_manager.add_mask(mask)
            return input * mask
        else:
            return input

# 3. 替换函数
def replace_dropout(module, dropout_class, mask_manager, start_index=0):
    """
    递归地遍历模块，将所有 nn.Dropout 替换为自定义的 dropout_class。
    返回下一个可用的索引。
    """
    current_index = start_index
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            # 替换为 CustomDropout，并分配唯一索引
            setattr(module, name, dropout_class(p=child.p, mask_manager=mask_manager, mask_index=current_index))
            current_index += 1
        else:
            current_index = replace_dropout(child, dropout_class, mask_manager, current_index)
    return current_index

# 4. 加载模型和分词器
def load_model_and_tokenizer(model_name, mask_manager):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=True  # 确保已设置环境变量或通过其他方式进行身份验证
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 根据需要调整
        use_auth_token=True
    ).cuda()  # 将模型移动到 GPU

    # 5. 替换所有 Dropout 层
    replace_dropout(model, CustomDropout, mask_manager)

    return model, tokenizer

# 6. 定义前向传播函数，支持掩码复用
def forward_with_mask_reuse(model, input_ids, mask_manager, reuse_mask=False):
    """
    进行前向传播，控制是否复用掩码。
    """
    outputs = model(input_ids=input_ids)
    return outputs

# 7. 示例使用
def main():
    model_name = "meta-llama/Llama-3.1-8B"  # 确保您有访问权限
    mask_manager = MaskManager()

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_name, mask_manager)

    model.train()  # 设置为训练模式

    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

    input_ids = inputs['input_ids']

    # 第一次前向传播，生成并存储掩码
    print("=== 第一次前向传播（生成并存储掩码） ===")
    outputs1 = forward_with_mask_reuse(model, input_ids, mask_manager, reuse_mask=False)
    logits1 = outputs1.logits
    generated_ids1 = logits1.argmax(dim=-1)
    output_text1 = tokenizer.decode(generated_ids1[0], skip_special_tokens=True)
    print(f"输出1: {output_text1}")

    # 第二次前向传播，复用相同的掩码
    print("\n=== 第二次前向传播（复用相同的掩码） ===")
    outputs2 = forward_with_mask_reuse(model, input_ids, mask_manager, reuse_mask=True)
    logits2 = outputs2.logits
    generated_ids2 = logits2.argmax(dim=-1)
    output_text2 = tokenizer.decode(generated_ids2[0], skip_special_tokens=True)
    print(f"输出2: {output_text2}")

    # 第三次前向传播，生成新的掩码
    print("\n=== 第三次前向传播（生成新的掩码） ===")
    mask_manager.reset()  # 重置掩码管理器，清除所有已存储的掩码
    outputs3 = forward_with_mask_reuse(model, input_ids, mask_manager, reuse_mask=False)
    logits3 = outputs3.logits
    generated_ids3 = logits3.argmax(dim=-1)
    output_text3 = tokenizer.decode(generated_ids3[0], skip_special_tokens=True)
    print(f"输出3: {output_text3}")

if __name__ == "__main__":
    main()
