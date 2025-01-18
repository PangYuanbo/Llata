import json
import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# 在程序入口处加载 .env 文件
load_dotenv()

# 然后你就可以用 os.environ.get 来获取变量
client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

def llama_scoring_system(
    history_info,
    model_output,
    sympy_output,
    correct_solution,
    correct_answer,
    step=None,
    error_message=None
):
    content = {
        "history_info": history_info,
        "model_output": model_output,
        "sympy_output": sympy_output,
        "correct_solution": correct_solution,
        "correct_answer": correct_answer,
        "error_message": error_message
    }

    # Clearly specify in the system message to return only JSON
    system_message = (
        "You are a scoring and error hint model, please return in JSON format only, "
        "the JSON should contain two fields:\n"
        "1. \"error_hint\": a string providing hints about errors or shortcomings\n"
        f"2. \"score\": a double representing the score (0-{step})\n"
        "Please do not return anything other than JSON."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)}
    ]

    # Call the scoring model, disable streaming output
    response = client.chat.completions.create(
        model="llama3.1-8b",  # The scoring model you are using
        messages=messages,
        stream=False         # Disable streaming output in JSON mode
    )

    # Access the generated content
    model_raw_str = response.choices[0].message.content  # Updated to match correct structure

    # Attempt to parse the output as JSON
    try:
        score_and_feedback = json.loads(model_raw_str)
    except json.JSONDecodeError:
        error_response = {
            "failed_generation": model_raw_str
        }
        return error_response, 400

    if not isinstance(score_and_feedback, dict):
        error_response = {
            "failed_generation": model_raw_str
        }
        return error_response, 400

    if "error_hint" not in score_and_feedback or "score" not in score_and_feedback:
        error_response = {
            "failed_generation": model_raw_str
        }
        return error_response, 400

    return score_and_feedback, 200

def main():
    # 模拟测试数据
    history_info = "The user has been solving quadratic equations in the last few interactions."
    model_output = "x = 3"
    sympy_output = "x = 3"
    correct_solution = "x = 3"
    correct_answer = "x = 3"
    step = 10
    error_message = None

    # 调用评分系统
    result, status_code = llama_scoring_system(
        history_info=history_info,
        model_output=model_output,
        sympy_output=sympy_output,
        correct_solution=correct_solution,
        correct_answer=correct_answer,
        step=step,
        error_message=error_message
    )

    # 打印结果
    print(f"Status Code: {status_code}")
    print(f"Result: {json.dumps(result, indent=4, ensure_ascii=False)}")


if __name__ == "__main__":
    main()