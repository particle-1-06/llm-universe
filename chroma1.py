__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from Configuration import split_docs

# 配置 OpenAI Embedding
embedding = OpenAIEmbeddings(
     api_key="sk-jYpyX1XfWIPKGTbaxunMyHK35MEa2HSpFloVw9sRUNU774Os",  # ✅ 你的 API Key
    base_url="https://xiaoai.plus/v1",  # 如果你用 OpenAI 官方直接服务就删掉这行
    model="text-embedding-3-small"  # 或 "text-embedding-3-large"
)

# 设置向量数据库持久化路径
persist_directory = '/workspaces/test_codespace/llm-universe/data_base/vector_db'

# 创建 Chroma 向量库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)

# 输出当前向量数量
print(f"向量库中存储的数量：{vectordb._collection.count()}")


question="什么是大语言模型"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
