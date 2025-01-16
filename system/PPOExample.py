import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, create_reference_model


# -------------------------------------------------------------
# 1. 定义自定义奖励函数（示例：根据回复的单词数给予奖励）
# -------------------------------------------------------------
def custom_reward_fn(response_text: str) -> float:
    """
    根据具体需求定义奖励函数。此处仅以回复长度作为示例:
    - 如果回复很长，则奖励更高
    - 实际中，你可以接入更复杂的打分逻辑（如人工反馈、情感分析等）
    """
    # 简单的例子：根据单词数进行奖励（最多给 1.0 的奖励）
    word_count = len(response_text.split())
    reward = min(word_count / 20.0, 1.0)
    return reward


# -------------------------------------------------------------
# 2. 加载预训练的GPT-2模型与分词器
# -------------------------------------------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 因为 GPT-2 没有 pad_token，所以需要额外进行处理
tokenizer.pad_token = tokenizer.eos_token

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = create_reference_model(model)  # 创建参考模型，用于 PPO

# -------------------------------------------------------------
# 3. 配置 PPO
# -------------------------------------------------------------
ppo_config = PPOConfig(
    exp_name="PPOExample",  # 实验名称
    reward_model_path="EleutherAI/gpt-neo-2.7B",  # 奖励模型路径
    output_dir="./ppo_output",  # 输出目录
    batch_size=2,  # 每次进行 PPO 更新时同时处理的数量
    learning_rate=1e-5,  # 学习率可根据需要调整
)

# 创建 PPOTrainer
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
    **ppo_config.__dict__
)

# -------------------------------------------------------------
# 4. 定义训练使用的一些提示 (prompts)
# -------------------------------------------------------------
prompts = [
    "Question: What is your favorite color?\nAnswer:",
    "Question: How do you make a cup of tea?\nAnswer:",
    "Question: How to learn Python effectively?\nAnswer:",
]

# -------------------------------------------------------------
# 5. 训练循环
# -------------------------------------------------------------
EPOCHS = 3  # 示例中只训练 3 轮，你可以自行调整
for epoch in range(EPOCHS):
    print(f"=== Epoch {epoch + 1}/{EPOCHS} ===")

    for prompt in prompts:
        # 将 prompt 编码成输入张量
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.cuda() for k, v in batch.items()} if torch.cuda.is_available() else batch

        # 让模型生成一段文本
        generation_output = model.generate(
            **batch,
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1
        )

        # 解码生成结果，得到文本形式
        gen_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)

        # 计算自定义奖励
        reward = custom_reward_fn(gen_text)

        # 用 PPOTrainer 的 step 函数更新模型
        # step() 函数的输入:
        #   1) prompt_texts：原始的 prompt 列表
        #   2) generated_texts：本轮生成的响应列表
        #   3) rewards：对应生成回复的奖励列表
        ppo_trainer.step(
            [prompt],  # prompt_texts
            [gen_text],  # generated_texts
            [reward]  # rewards
        )

        print(f"Prompt: {prompt}")
        print(f"Generated: {gen_text}")
        print(f"Reward: {reward:.4f}")
        print("-" * 40)

print("训练完成！")
