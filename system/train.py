import sympy
from system.scoring import llama_scoring_system
import json
from model import generate_model_response_chat_neo


def solve_one_question(question, solution ,correct_answer, ppo_trainer):
    """
    针对单道题目的完整流程（简化示例）
    question: "请计算 x^2 + 2x + 1 在 x=2 时的值" (示例)
    correct_answer: 例如 9
    ppo_trainer: CleanRL PPO的训练器或相关包装类
    """

    # 这里的 history_info 可以记录每一次交互，比如:
    global score_feedback
    history_info = []

    # 1) 给模型的起始prompt，告诉它：
    #   "请输出 JSON，包含 step、sympy_function、sympy_equation、require_next_step"
    #   并提示它可以使用 sympy。示例（可根据需求自定义）：
    sys_prompt = (
        "You are a math solving assistant, and you need to use SymPy to help with calculations."
        "Please respond in JSON format with the following fields:\n"
        "1) step: What step are you currently on\n"
        "2) sympy_function: The SymPy function or method you will use\n"
        "3) sympy_equation: The specific expression or equation\n"
        "4) require_next_step: Whether you need to continue to the next step\n\n"
        "Example: {\"step\":\"Initialization\",\"sympy_function\":\"symbols\",\"sympy_equation\":\"x = sympy.Symbol('x')\",\"require_next_step\":true}\n\n"
    )

    question_prompt = f"{sys_prompt}\nQuestion: {question}\n"

    require_next_step = True
    round_idx = 0

    while require_next_step:
        round_idx += 1
        # 2) 调用模型生成答复（JSON）
        new_text = generate_model_response_chat_neo(
            prompt=question_prompt,
            history=history_info
        )
        # 尝试做 JSON 解析
        try:
            model_output = json.loads(new_text)
        except json.JSONDecodeError:
            # 如果解析失败，调用打分系统并传递原始输出
            error_message = "JSON format error"
            score_feedback = llama_scoring_system(
                history_info=history_info,
                model_output={"raw_text": new_text},
                sympy_output=None,
                correct_answer=correct_answer,
                error_message=error_message
            )
            print(f"[JSON 解析错误] round={round_idx}, score_feedback={score_feedback}")

            # 进行一次PPO训练更新，给予低分奖励
            ppo_trainer.update(policy_input=history_info, reward=-1)  # 低分奖励

            # 返回错误状态
            return False, None
        # 记录本轮输出
        history_info.append({
            "round": round_idx,
            "model_output": model_output
        })
        question_prompt += f"\nModel Response (Round {round_idx}):\n{new_text}\n"

        # 3) 尝试用 SymPy 执行
        sympy_function = model_output.get("sympy_function", "")
        sympy_equation = model_output.get("sympy_equation", "")
        require_next_step = model_output.get("require_next_step", False)

        # 4) SymPy 计算过程
        try:
            # 这里可以根据 model_output 的指令做实际计算
            # 示例：exec(sympy_equation) 或 自行 parse / eval
            # 注意安全性与 sandbox
            local_vars = {}
            exec(sympy_equation, {"sympy": sympy}, local_vars)  # 演示目的
            # 假设输出保存在 local_vars["result"] / local_vars["expr"] 之类
            sympy_result = local_vars.get("result", None)

            # 如果没有报错，则检查是否最终答案
            # 如果 require_next_step = True，则表示还有下一轮
            if not require_next_step:
                # 说明可能是最终答案
                # 可以将 sympy_result 与 correct_answer 对比
                pass

            sympy_output = f"result: {sympy_result}"

        except Exception as e:
            # 5) 如果SymPy执行报错，调用打分系统
            error_message = str(e)
            score_feedback = llama_scoring_system(
                history_info=history_info,
                model_output=model_output,
                sympy_output=None,
                correct_answer=correct_answer,
                error_message=error_message
            )
            print(f"[SymPy 执行报错] round={round_idx}, score_feedback={score_feedback}")

            # 6) 进行一次PPO训练更新
            reward = score_feedback.get("score", 0.0)
            error_hint = score_feedback.get("error_hint", "")
            ppo_trainer.update(policy_input=history_info, reward=0)

            # 7) 重新做这道题（return后再在上层控制循环或递归）
            return False, history_info, error_hint

        # 如果 SymPy 未报错，记录结果
        history_info[-1]["sympy_output"] = sympy_output

        # 如果这一轮回答说需要下一步，则继续循环
        # 如果不需要下一步，需要判断是否为最终答案，并打分
        if not require_next_step:
            # 最终答案 (sympy_result) vs correct_answer
            # 调用打分系统
            state=0
            while state!=200:
                score_feedback,state = llama_scoring_system(
                    history_info=history_info,
                    model_output=model_output,
                    sympy_output=sympy_output,
                    correct_solution=solution,
                    correct_answer=correct_answer
                )



            # 根据打分结果做一次 PPO 更新
            # 假设我们简单通过对比对错来给reward：
            reward = score_feedback.get("score",0.0)
            error_hint = score_feedback.get("error_hint", "")
            ppo_trainer.update(policy_input=history_info, reward=reward)

            # 如果答案错误，可以在外层判断后决定是否继续重复该题
            is_correct = reward==5
            return is_correct, sympy_result, error_hint

        else:
            # 如果需要下一步，则可以将当前中间结果再拼接到 prompt
            question_prompt += f"\n(Results of the previous round of calculations: {sympy_output})\n"


def training_loop(questions, ppo_trainer):
    """
    questions: [(question, correct_answer), ...]
    ppo_trainer: CleanRL PPO 训练器或包装
    """
    epoch = 0
    while True:
        epoch += 1
        print(f"=== EPOCH {epoch} ===")
        all_correct = True
        for idx, (q, ans) in enumerate(questions):
            print(f"--- Question {idx + 1}: {q} ---")
            is_correct, final_result = solve_one_question(q, ans, ppo_trainer)
            if not is_correct:
                # 如果错误，可以选择再重复做一次
                print("[RETRY] 该题答案错误，准备再做一次")
                solve_one_question(q, ans, ppo_trainer)
                all_correct = False
        if all_correct:
            print("本轮所有题目均正确，结束训练或进入下一个阶段。")
            break
