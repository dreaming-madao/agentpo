import os

openai_api_key = ''

llm_model_dict = {
    "Qwen2.5-72b": "http://174.34.106.21:10001/v1",
    "Qwen2.5-14b": "http://174.34.106.20:10001/v1",
    "Qwen2.5-7b": "http://174.34.106.20:10004/v1",
    "Qwen3-4b": "http://174.34.106.20:10003/v1",
    "Llama-3.1-8B": "http://127.0.0.1:10006/v1",
    "Llama-3.2-3B": "http://127.0.0.1:10005/v1",
    "siliconflow-api": "https://api.siliconflow.cn/v1",
    "aliyuncs-api": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

configs = {
    "Qwen-max-uaes-1206": ["qwen-max", openai_api_key],
    "Qwen-plus-uaes-1206": ["qwen-plus", openai_api_key],
    "deepseek-r1-uaes-1206": ["deepseek-r1", openai_api_key],
    "deepseek-v3-uaes-1206": ["deepseek-v3", openai_api_key],
    "Qwen2.5-Math-7B-DashScope": ["qwen2.5-math-7b-instruct", os.environ.get("DASHSCOPE_API_KEY", "")],
    "Qwen2.5-7B-SiliconFlow": ["siliconflow-api", "Qwen/Qwen2.5-7B-Instruct", os.environ.get("SILICONFLOW_API_KEY", "")],
    "DeepSeek-V3-SiliconFlow": ["siliconflow-api", "deepseek-ai/DeepSeek-V3", os.environ.get("SILICONFLOW_API_KEY", "")],
    "qwq-32b-preview-uaes-1206": ["qwq-32b-preview", openai_api_key]
}

local_model_list = ["Qwen2.5-72b", "Qwen2.5-14b", "Qwen2.5-7b", "Qwen2.5-3b","Llama-3.2-3B", "Llama-3.1-8B", "Qwen3-4b"]
