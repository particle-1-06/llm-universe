import openai
from dotenv import load_dotenv
import os

load_dotenv()  # 加载 .env 中的密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, GPT!"}]
)
print(response.choices[0].message.content)