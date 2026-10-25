# quick_test.py - 快速验证基础功能
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("🧪 快速验证测试")
print("=" * 50)

# 1. 测试配置
print("\n1. 配置模块...")
from config import config
print(f"   ✅ 模型: {config.llm.model}")
print(f"   ✅ Ollama: {config.llm.is_ollama}")
print(f"   ✅ 最大步数: {config.agent.max_steps}")
print(f"   ✅ 安全目录: {config.tools.safe_dirs}")

# 2. 测试 utils
print("\n2. 工具函数...")
from utils.helpers import (
    clean_json_response,
    try_parse_json,
    truncate_text,
    format_tool_result,
    is_safe_path,
)
print("   ✅ utils.helpers 导入成功")

# 测试 JSON 清理
test_json = '```json\n{"action":"call_tool","tool_name":"calculator"}\n```'
parsed = try_parse_json(test_json)
assert parsed is not None, "JSON 解析失败"
assert parsed["action"] == "call_tool", "action 字段错误"
print("   ✅ JSON 清理解析正常")

# 测试文本截断
long_text = "a" * 5000
truncated = truncate_text(long_text, 1000)
assert len(truncated) <= 1000, "截断长度错误"
print("   ✅ 文本截断正常")

# 测试安全路径
assert is_safe_path("data/test.txt") == True
assert is_safe_path("../../etc/passwd") == False
print("   ✅ 路径安全检查正常")

# 3. 测试日志
print("\n3. 日志系统...")
from utils.logger import setup_logging, get_logger
setup_logging()
logger = get_logger("test")
logger.info("日志系统测试")
print("   ✅ 日志系统正常")

# 4. 测试工具注册系统
print("\n4. 工具注册系统...")
from agent.tools import list_tools, call_tool, get_tool_descriptions
tools = list_tools()
print(f"   ✅ 已注册工具: {len(tools)} 个")
for t in tools:
    print(f"      - {t['name']} [{t['category']}]")

# 5. 测试各个工具
print("\n5. 工具功能测试...")

# calculator
r = call_tool("calculator", {"expr": "123 + 456 * 7"})
assert "成功" in r and "3315" in r, f"计算器失败: {r}"
print(f"   ✅ calculator: {r[:50]}...")

# get_current_time
r = call_tool("get_current_time", {})
assert "成功" in r, f"时间查询失败: {r}"
print(f"   ✅ get_current_time: {r[:50]}...")

# word_count
r = call_tool("word_count", {"text": "Hello World 你好世界 测试文本"})
assert "成功" in r, f"字数统计失败: {r}"
print(f"   ✅ word_count: 正常")

# json_format
r = call_tool("json_format", {"json_str": '{"name":"张三","age":25}'})
assert "成功" in r, f"JSON格式化失败: {r}"
print(f"   ✅ json_format: 正常")

# date_calculate
r = call_tool("date_calculate", {"date_str": "2024-01-15", "days": 10})
assert "成功" in r, f"日期计算失败: {r}"
print(f"   ✅ date_calculate: 正常")

# list_files
r = call_tool("list_files", {"dir_path": "data"})
print(f"   ✅ list_files: 正常")

# run_python
r = call_tool("run_python", {"code": "print(1+1)"})
assert "2" in r, f"Python执行失败: {r}"
print(f"   ✅ run_python: 正常")

# get_agent_info
r = call_tool("get_agent_info", {})
assert "成功" in r, f"Agent信息失败: {r}"
print(f"   ✅ get_agent_info: 正常")

# 6. 测试 LLM 接口层（不实际调用）
print("\n6. LLM 接口层...")
from llm import get_llm, get_llm_provider
from llm.base import BaseLLM
llm = get_llm()
assert isinstance(llm, BaseLLM), "LLM 类型错误"
print(f"   ✅ LLM 提供商: {llm.provider_name}")
print(f"   ✅ LLM 模型: {llm.model_name}")

# 7. 测试 Agent 状态和图
print("\n7. Agent 核心...")
from agent.state import AgentState
from agent.graph import build_graph, graph
print("   ✅ Agent 状态定义正常")
print("   ✅ LangGraph 图构建正常")

print("\n" + "=" * 50)
print("🎉 所有基础测试通过！")
print("=" * 50)
print("\n💡 接下来可以:")
print("   1. 启动 Ollama 服务后运行 python app.py")
print("   2. 或运行 python main.py '你的问题' 命令行测试")
print("   3. 或运行 python test_agent.py 完整测试")
