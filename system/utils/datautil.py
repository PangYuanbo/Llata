import re
from datasets import load_dataset


def parse_solution_and_answer(full_answer: str):
    """
    将 GSM8K 原始的 answer 文本拆分为：
      - solution: 除最后一行以外的内容 (step-by-step reasoning)
      - final_answer: 最后一行或最后一句 (最终答案)

    这是一个非常简单的拆分策略，可能需要你根据实际情况调整。
    """
    lines = full_answer.strip().split("\n")
    # 移除末尾空行
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) <= 1:
        # 若只有一行，就都算作 final_answer
        return "", lines[0] if lines else ""
    else:
        # 否则，前面作为 solution，最后一行作为 final_answer
        return "\n".join(lines[:-1]), lines[-1]


def get_gsm8k_question_solution_answer(split="train"):
    """
    迭代返回 GSM8K 的 {question, solution, final_answer}。
    :param split: "train" 或 "test"
    :return: 一个生成器，每次 yield 一条数据
    """
    gsm8k_dataset = load_dataset("gsm8k", "main", split=split)

    for sample in gsm8k_dataset:
        q = sample["question"]
        raw_ans = sample["answer"]
        solution, final_answer = parse_solution_and_answer(raw_ans)
        yield {
            "question": q,
            "solution": solution,
            "final_answer": final_answer
        }


if __name__ == "__main__":
    # 示例：从 train 集获取前 3 条看看
    gen = get_gsm8k_question_solution_answer(split="train")
    for i, item in zip(range(3), gen):
        print(f"--- Sample {i} ---")
        print("Question:", item["question"])
        print("Solution:\n", item["solution"])
        print("Final Answer:", item["final_answer"])
        print()
