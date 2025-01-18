from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=False)

    question = """What is the capital of France?
        Please provide the answer in JSON format like this:
        {
            "question": "What is the capital of France?",
            "answer": "Your answer here"
        }"""

    inputs = tokenizer(question, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.00001,  # 设置温度为 0，确保生成确定性输出
        top_k=1,  # 强制选择概率最高的 token
        eos_token_id=tokenizer.eos_token_id,  # 设置结束标志
        num_return_sequences=1  # 只生成一条回答
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        response_json = {"error": "Failed to parse JSON", "raw_response": response}

        # 打印结果
    print(json.dumps(response_json, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
