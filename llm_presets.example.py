# llm_presets.example.py
# ============================================================
# 模型/API 预设方案示例（不含真实 key，可安全上传到仓库）
#
# 使用方式：
#   1. 复制本文件为 llm_presets.py（本文件不进仓库，llm_presets.py 已被 .gitignore 忽略）
#   2. 在每个方案里填入自己的真实 API key
#   3. 界面下拉框即可一键切换，无需改 .env
#
# api_key 留空 "" 会沿用 .env 里的 LLM_API_KEY
# ============================================================

LLM_PRESETS = {
    "Ollama 本地 (qwen3:8b)": {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "dummy",           # 本地模型 key 无意义，占位即可
        "model": "qwen3:8b",
        "temperature": 0.1,
        "max_tokens": 512,
        "timeout": 120,
        "max_retries": 2,
        "ollama_think": False,
    },

    "DeepSeek (deepseek-chat)": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",                # 留空沿用 .env；或填 sk-你的key
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },

    "OpenAI (gpt-4o-mini)": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",                # 填 sk-你的key
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },

    "智谱 GLM-4": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "",                # 填 你的key
        "model": "glm-4",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },

    "Kimi (moonshot-v1-8k)": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": "",                # 填 sk-你的key
        "model": "moonshot-v1-8k",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },

    "通义千问 (qwen-plus)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",                # 填 sk-你的key
        "model": "qwen-plus",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },

    "自定义 (OpenAI 兼容)": {
        "base_url": "https://你的服务地址/v1",
        "api_key": "",
        "model": "your-model",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
    },
}
