# main.py
# ============================================================
# 命令行调试入口：直接运行 Agent，打印每一步执行过程
# ============================================================
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import stream_agent
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("main")


def run_agent(query: str):
    """运行 Agent 并打印执行过程"""
    print("=" * 60)
    print(f"🤖 用户问题: {query}")
    print("=" * 60)

    final_answer = None
    step = 0

    for event in stream_agent(query):
        for node_name, value in event.items():
            step += 1
            print(f"\n{'─' * 40}")
            print(f"📍 步骤 {step}: {node_name}")
            print(f"{'─' * 40}")

            if node_name == "planner":
                action_data = value.get("action_data", {})
                action = action_data.get("action", "?")
                if action == "call_tool":
                    tool = action_data.get("tool_name", "?")
                    args = action_data.get("args", {})
                    print(f"  决策: 调用工具 🛠 {tool}")
                    print(f"  参数: {args}")
                else:
                    print(f"  决策: 直接回答 💬")
                    answer = action_data.get("answer", "")
                    if answer:
                        print(f"  回答预览: {answer[:80]}...")

            elif node_name == "tools":
                results = value.get("tool_results", [])
                for i, r in enumerate(results, 1):
                    preview = r[:120] + "..." if len(r) > 120 else r
                    print(f"  结果 {i}: {preview}")

            elif node_name == "reflection":
                need = value.get("need_continue", False)
                print(f"  反思结果: {'继续规划 🔄' if need else '信息足够 ✅'}")

            elif node_name == "final":
                final_answer = value.get("final_answer", "")
                print(f"  最终回答已生成")

    print("\n" + "=" * 60)
    print("📝 最终回答:")
    print("-" * 60)
    print(final_answer or "(未生成回答)")
    print("=" * 60)


if __name__ == "__main__":
    # 默认测试问题
    test_query = "计算 123 + 456 * 7"

    # 支持命令行参数传入问题
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])

    run_agent(test_query)
