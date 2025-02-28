import json
import os
import random
import time

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# 在程序入口处加载 .env 文件
load_dotenv()

# 然后你就可以用 os.environ.get 来获取变量
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)
# 目标数据集文件路径
dataset_path = "training_dataset.jsonl"
import json
import re
from utils.datautil import get_gsm8k_question_solution_answer


def remove_repeated_question(response, question):
    # 找到问题第一次出现的位置
    question_position = response.find(question)
    if question_position != -1:
        # 截取问题之后的部分
        response = response[question_position + len(question):].strip()
    return response
def get_random_question():
    gen = get_gsm8k_question_solution_answer(split="train")
    questions = list(gen)  # 将生成器转换为列表
    random.shuffle(questions)  # 打乱问题的顺序
    return random.choice(questions)["question"]  # 返回随机题目

def send_request_cerebias():
    question = get_random_question()
    sys_prompt = (
        "You are a math solving assistant and need to use SymPy to help with calculations. "

        "Follow these step-by-step instructions using only symbols and expression (expr) operations:\n"
        "Step 1: Output only the symbol 'x'.\n"
        "Step 2: Output a mathematical expression involving 'x' (e.g., x**2 + 2*x + 1). "
        "Do not include variable assignment; only provide the expression itself.\n"
        "Step 3: Output only the name of a SymPy function you intend to use on the expression (e.g., expand, solve, Eq).\n"
        "Step 4: Output the arguments you would pass to that function, separated by commas (e.g., expr, x).\n\n"
        "step 5: Wheather you need to continue the next step(e.g., expr,x).\n\n"
        "Your final output should be a single JSON block like the example below:\n"
        """"```json
        {
            "symbol": "x",
            "expr": "x+8",
            "sympy_function": "solve",
            "function_args": "expr, x"
            "require_next_step": "True"
        }

        ```
        "Remember you don't need to do any additional calculation, just provide the json."
         "Do not output anything else."
        """
    )

    question_prompt = f"{sys_prompt}\nQuestion: {question}\n"
    messages = [
        {"role": "user", "content": question_prompt}
    ]
    response = client.chat.completions.create(
        model="llama3.1-70b",  # The scoring model you are using
        messages=messages,
        stream=False,  # Disable streaming output in JSON mode
        temperature=0.0001  # 设置温度为 0，确保生成确定性输出
    )

    content = response.choices[0].message.content

    cleaned_response = remove_repeated_question(content, question_prompt)
    # 使用正则表达式提取 JSON 块
    json_match = re.search(r"\{.*?\}", cleaned_response, re.DOTALL)
    if json_match:
        json_content = json_match.group(0)  # 提取匹配到的 JSON 块
        try:
            response_json = json.loads(json_content)
        except json.JSONDecodeError:
            response_json = {
                "error": "Failed to parse extracted JSON",
                "raw_response": cleaned_response,
                "extracted_json": json_content
            }
    else:
        response_json = {
            "error": "No JSON found in response",
            "raw_response": cleaned_response
        }
    print(question)
    # 打印最终结果
    print(json.dumps(response_json, indent=4, ensure_ascii=False))
    train_data_entry = {
        "input": question_prompt,
        "output": content
    }
    with open(dataset_path, "a") as f:
        f.write(json.dumps(train_data_entry, ensure_ascii=False) + "\n")
    return

def main():
    # 定义请求限制
    MAX_REQUESTS_PER_MINUTE = 25
    MAX_REQUESTS_PER_HOUR = 850
    MAX_REQUESTS_PER_DAY = 14400

    def send_request(request_id):
        """
        模拟请求的发送
        """
        print(f"发送请求: {request_id}")

    # 总请求数（假设要发送 10,000 个请求）
    total_requests = 10000

    # 计数器
    requests_sent = 0  # 已发送的请求数
    minute_counter = 0  # 当前分钟发送的请求数
    hour_counter = 0  # 当前小时发送的请求数

    start_time = time.time()  # 记录起始时间
    minute_start_time = start_time  # 当前分钟的起始时间
    hour_start_time = start_time  # 当前小时的起始时间

    while requests_sent < total_requests:
        current_time = time.time()

        # 检查是否需要重置分钟计数器
        if current_time - minute_start_time >= 60:
            minute_counter = 0
            minute_start_time = current_time

        # 检查是否需要重置小时计数器
        if current_time - hour_start_time >= 3600:
            hour_counter = 0
            hour_start_time = current_time

        # 如果没有超过限制，发送请求
        if minute_counter < MAX_REQUESTS_PER_MINUTE and hour_counter < MAX_REQUESTS_PER_HOUR and requests_sent < MAX_REQUESTS_PER_DAY:
            send_request_cerebias()
            requests_sent += 1
            minute_counter += 1
            hour_counter += 1
        else:
            # 如果超过限制，暂停一会儿
            time.sleep(1)  # 等待 1 秒再检查

    print("所有请求已发送完成！")


if __name__ == "__main__":
    main()