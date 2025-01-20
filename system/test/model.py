from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from utils.datautil import get_gsm8k_question_solution_answer
from peft import PeftModel, PeftConfig
def remove_repeated_question(response, question):
    # 找到问题第一次出现的位置
    question_position = response.find(question)
    if question_position != -1:
        # 截取问题之后的部分
        response = response[question_position + len(question):].strip()
    return response

# 将 messages 拼接为单个字符串
def messages_to_string(messages):
    """
    将 messages 转换为单个字符串，保留对话上下文。
    """
    context = ""
    for message in messages:
        # 拼接角色和内容
        context += f"{message['role']}: {message['content']}\n"
    return context.strip()  # 移除最后一个多余的换行符

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=False)
    # 加载 LoRA 配置


    # 加载 LoRA 权重并附加到原始模型
    model = PeftModel.from_pretrained(model, "lora")

    # 加载分词器

    model = model.merge_and_unload()
    gen= get_gsm8k_question_solution_answer(split="train")

    question=next(gen)["question"]
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
    print(question)
    messages =messages_to_string( [
        {"role": "user", "content": question_prompt}
    ])
    inputs = tokenizer(messages, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=500,  # 确保足够长度以生成完整 JSON
        temperature=0.0001,  # 设置温度为 0，确保生成确定性输出
        top_k=1,  # 强制选择概率最高的 token
        num_return_sequences=1,  # 只生成一条回答
        eos_token_id=tokenizer.eos_token_id  # 设置结束标志
    )

    # 将模型输出解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_response = remove_repeated_question(response, question_prompt)
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

    # 打印最终结果
    print(json.dumps(response_json, indent=4, ensure_ascii=False))
    print("response:",cleaned_response)


if __name__ == "__main__":
    main()