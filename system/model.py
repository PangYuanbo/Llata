import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, create_reference_model

class Client:
    def __init__(self,
                 model_name_or_path: str = "gpt-neo-2.7B",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 use_flash_attention_2: bool = True):
        """
        初始化 Client 类，加载模型和 tokenizer。

        :param model_name_or_path: 模型名称或路径，默认使用 EleutherAI/gpt-neo-2.7B
        :param device: 使用的设备，如 "cuda" 或 "cpu"
        :param dtype: 使用的浮点精度，如 torch.float16
        :param use_flash_attention_2: 是否在 from_pretrained 中启用 Flash Attention 2
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.history = []
        self.rewards = 0.0
        self.steps = 0

        # 根据是否需要 Flash Attention 2，拼接相应的参数
        if use_flash_attention_2:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype
            )

        # 将模型移动到指定设备
        self.model.to(self.device)

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def generate(self,
                 prompt: str,
                 history: list = None,
                 max_new_tokens: int = 128,
                 temperature: float = 0.9) -> dict:
        """
        1) 使用 GPT-Neo (或其他兼容的 CausalLM) 生成文本
        2) 可根据需要做 JSON 解析并返回一个 dict

        :param prompt: 用户输入的请求
        :param history: 对话历史（列表），可选
        :param max_new_tokens: 本次生成的最大 token 数
        :param temperature: 采样时的温度
        :return: 一个字符串/字典，或其他结构化数据
        """
        self.steps += 1

        if history is None:
            history = []

        # 将 (history + prompt) 拼成完整的上下文文本
        context_text = ""
        for i, h in enumerate(history):
            context_text += f"Round {i + 1} - Model:\n{h}\n\n"
        context_text += f"User:\n{prompt}\n\nModel:\n"

        # 将上下文转为 token
        inputs = self.tokenizer(context_text, return_tensors="pt").to(self.device)

        # 生成
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

        # 解码得到文本
        raw_output = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        # 记录历史
        self.history.append({
            "input": context_text,
            "output": raw_output
        })

        # 简单做法：只截取 prompt 后模型新生成的部分
        new_text = raw_output[len(context_text):].strip()

        # 如果你想要将输出文本解析为 JSON，可在此处尝试:
        # 例如：
        # try:
        #     json_data = json.loads(new_text)
        #     return json_data
        # except json.JSONDecodeError:
        #     # 若解析失败，可考虑返回原始文本或抛出异常
        #     return {"error": "JSON parse error", "raw_text": new_text}

        return new_text

    def PPO_train (self, prompt: str, reward: float):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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
            model=self.model,
            ref_model=create_reference_model(self.model),
            tokenizer=tokenizer,
            **ppo_config.__dict__
        )
        # 计算自定义奖励
        reward = reward/self.steps
        EPOCHS = 3  # 示例中只训练 3 轮，你可以自行调整
        for epoch in range(EPOCHS):
            print(f"=== Epoch {epoch + 1}/{EPOCHS} ===")
            for _ in self.history:
                ppo_trainer.



# 例如，在主程序中可以这样使用：
if __name__ == "__main__":
    chat_neo = Client(
        model_name_or_path="EleutherAI/gpt-neo-2.7B",
        device="cuda",
        dtype=torch.float16,
        use_flash_attention_2=True
    )

    response = chat_neo.generate(prompt="你好，请介绍一下GPT-Neo模型的特点。")
    print("生成结果：", response)
