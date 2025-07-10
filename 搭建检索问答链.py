__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sys
sys.path.append("/workspaces/test_codespace/llm-universe/docs/C3") # 将父目录放入系统路径中

# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from openai import OpenAI

from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# 定义 Embeddings
embedding = OpenAI()

# 向量数据库持久化路径
persist_directory = '/workspaces/test_codespace/llm-universe/data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")
