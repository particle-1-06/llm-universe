import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

# 调试.env文件加载
env_path = find_dotenv()
print(f"Loading env from: {env_path}")  # 修正：使用f-string
load_dotenv(env_path)

# 安全获取环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")  # 修正：变量名
print(f"API key loaded: {bool(openai_api_key)}")  # 修正：f-string

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY未设置，请检查.env文件")

# 显式传递密钥（推荐）
llm = ChatOpenAI(
    temperature=0.6,
    openai_api_key=openai_api_key,  # 修正：参数名
    model="gpt-3.5-turbo"
)

# 测试调用
try:
    output = llm.invoke("请你自我介绍一下自己！")  # 修正：方法名
    print(output.content)  # 直接输出内容
except Exception as e:
    print(f"调用失败: {str(e)}")  # 修正：打印函数