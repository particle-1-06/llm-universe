__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 定义 Embeddings
embedding = OpenAIEmbeddings(
     api_key="sk-jYpyX1XfWIPKGTbaxunMyHK35MEa2HSpFloVw9sRUNU774Os",  # ✅ 你的 API Key
    base_url="https://xiaoai.plus/v1",  # 如果你用 OpenAI 官方直接服务就删掉这行
    model="text-embedding-3-small"  # 或 "text-embedding-3-large"
)

# 定义持久化路径
persist_directory = '/workspaces/test_codespace/llm-universe/data_base/vector_db/chroma'

from langchain_community.vectorstores import Chroma

vectordb = Chroma(
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

