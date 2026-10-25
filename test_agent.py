# test_agent.py
# ============================================================
# Agent 功能测试脚本：覆盖多个核心场景
# ============================================================
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import run_agent
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("test")

# 测试用例
CASES = [
    ("数学计算", "计算 123 + 456 * 7 等于多少"),
    ("直接回答", "你好，请简单介绍一下你自己"),
    ("时间查询", "现在几点了？今天星期几"),
    ("文件读取", "读取 data/test.txt 文件的内容"),
    ("字数统计", "统计一下 'Hello World 你好世界' 的字数"),
    ("JSON格式化", "格式化这个JSON：{\"name\":\"张三\",\"age\":25,\"skills\":[\"Python\",\"Java\"]}"),
    ("日期计算", "2024-01-15 加 100 天是几号"),
    ("Agent信息", "查看你的系统信息"),
]


def run_test(name: str, query: str) -> bool:
    """运行单个测试用例，返回是否成功"""
    print(f"\n{'=' * 60}")
    print(f"🧪 测试 [{name}]: {query}")
    print(f"{'=' * 60}")

    try:
        result = run_agent(query)
        final_answer = result.get("final_answer", "")
        trace = result.get("trace", [])

        if final_answer:
            print(f"  ✅ 成功 - 回答长度: {len(final_answer)} 字符")
            print(f"  📊 执行步数: {len(trace)}")
            print(f"  💬 回答预览: {final_answer[:80]}...")
            return True
        else:
            print(f"  ❌ 失败 - 未生成回答")
            return False

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        logger.error(f"测试 [{name}] 异常", exc_info=True)
        return False


def main():
    print("🚀 开始 Agent 功能测试")
    print(f"共 {len(CASES)} 个测试用例")

    results = []
    for name, query in CASES:
        success = run_test(name, query)
        results.append((name, success))

    # 汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} - {name}")

    print(f"\n总计: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  {total - passed} 个测试失败，请检查日志")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
