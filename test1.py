# 首先确保你已安装必要的库
# pip install langchain openai langchain-community chromadb

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
# from langchain_community.embeddings.openai import OpenAIEmbeddings

from langchain_community.document_loaders import TextLoader  # 或其他文档加载器
from langchain_community.vectorstores import Chroma

_ = load_dotenv(find_dotenv())

api_key=os.environ['OPENAI_API_KEY']
client = OpenAI(
    api_key=api_key,
    base_url="https://xiaoai.plus/v1"
    )
# 1. 首先你需要加载和分割文档
# 示例：使用文本文件
loader = TextLoader("/workspaces/test_codespace/llm-universe/untitled.md")  # 替换为你的文件路径
documents = loader.load()

# 分割文档
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 更智能的分割器

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # 目标块大小
    chunk_overlap=200,        # 块间重叠
    length_function=len,      # 计算长度的方法
    separators=["\n\n", "\n", "。", " ", ""]  # 优先按段落/句子分割
)
split_docs = text_splitter.split_documents(documents)

# 2. 定义Embeddings - 这里以OpenAI为例
# 需要设置你的OPENAI_API_KEY环境变量
from langchain_openai import OpenAIEmbeddings  # 替代原导入
embedding = OpenAIEmbeddings()

# 3. 定义持久化路径
persist_directory = "/workspaces/test_codespace/llm-universe/data_base/vector_db/chroma"  # 修改为你的本地路径

# 4. 创建向量数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")