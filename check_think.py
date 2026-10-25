# 验证 langchain ChatOpenAI 传递 think 和 max_tokens 参数
import time
from langchain_openai import ChatOpenAI

# 尝试 extra_body 传 think:false
try:
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="dummy",
        model="qwen3:4b",
        temperature=0.1,
        max_tokens=200,
        extra_body={"think": False},
    )
    t0 = time.time()
    r = llm.invoke("1+1等于几？只回答数字。")
    dt = time.time() - t0
    print(f"extra_body 方式: 成功! 耗时 {dt:.1f}s, 输出: {r.content[:100]!r}")
except TypeError as e:
    print(f"extra_body 不被支持: {e}")
except Exception as e:
    print(f"extra_body 调用失败: {type(e).__name__}: {e}")

# 对比：model_kwargs 方式
print("---- model_kwargs 方式 ----")
try:
    llm2 = ChatOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="dummy",
        model="qwen3:4b",
        temperature=0.1,
        max_tokens=200,
        model_kwargs={"think": False},
    )
    t0 = time.time()
    r2 = llm2.invoke("1+1等于几？只回答数字。")
    dt = time.time() - t0
    print(f"model_kwargs 方式: 成功! 耗时 {dt:.1f}s, 输出: {r2.content[:100]!r}")
except Exception as e:
    print(f"model_kwargs 调用失败: {type(e).__name__}: {e}")
