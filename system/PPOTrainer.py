import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
# 1) 设置PPO超参数
ppo_config = PPOConfig(
    exp_name="PPOExample",  # 实验名称
    reward_model_path="EleutherAI/gpt-neo-2.7B",  # 奖励模型路径
    output_dir="./ppo_output",  # 输出目录
    batch_size=2,  # 每次进行 PPO 更新时同时处理的数量
    learning_rate=1e-5,  # 学习率可根据需要调整
    local_rollout_forward_batch_size=2,  # 本地 rollout 时的 batch size
)

# 2) 加载模型与分词器
#    注意：如果你想用 GPT-Neo，也可替换成EleutherAI/gpt-neo-2.7B等
model_name = "EleutherAI/gpt-neo-2.7B"
device = "cuda"

base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3) 构建 PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=base_model,         # 需要带有LMHead的模型
    ref_model=None,           # 若需要对比策略，可指定参考模型；这里简单用 None
    tokenizer=tokenizer
)
