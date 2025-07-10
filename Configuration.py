__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = '/workspaces/test_codespace/llm-universe/data_base/knowledge_db/prompt_engineering'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())

text = texts[1]
# print(f"每一个元素的类型：{type(text)}.", 
#     f"该文档的描述性数据：{text.metadata}", 
#     f"查看该文档的内容:\n{text.page_content[0:]}", 
#     sep="\n------\n")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

split_docs = text_splitter.split_documents(texts)


# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

# # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
# loader1 = PyMuPDFLoader("test_codespace/llm-universe/pumpkin_book.pdf")
# loader2 = UnstructuredMarkdownLoader("test_codespace/llm-universe/untitled.md")

# # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
# pdf_pages = loader1.load()
# md_pages = loader2.load()

# # print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
# # print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")

# pdf_page = pdf_pages[1]
# # print(f"每一个元素的类型：{type(pdf_page)}.", 
# #     f"该文档的描述性数据：{pdf_page.metadata}", 
# #     f"查看该文档的内容:\n{pdf_page.page_content}", 
# #     sep="\n------\n")

# # md_page = md_pages[0]
# # print(f"每一个元素的类型：{type(md_page)}.", 
# #     f"该文档的描述性数据：{md_page.metadata}", 
# #     f"查看该文档的内容:\n{md_page.page_content[0:][:200]}", 
# #     sep="\n------\n")


# import re
# pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
# pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
# print(pdf_page.page_content)
# pdf_page.page_content = pdf_page.page_content.replace('•', '')
# pdf_page.page_content = pdf_page.page_content.replace(' ', '')
# print(pdf_page.page_content)

# # md_page.page_content = md_page.page_content[0:][:200].replace('\n\n', '\n')
# # print(md_page.page_content)



# ''' 
# * RecursiveCharacterTextSplitter 递归字符文本分割
# RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
#     这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
# RecursiveCharacterTextSplitter需要关注的是4个参数：

# * separators - 分隔符字符串数组
# * chunk_size - 每个文档的字符数量限制
# * chunk_overlap - 两份文档重叠区域的长度
# * length_function - 长度计算函数
# '''
# #导入文本分割器
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# # 知识库中单段文本长度
# CHUNK_SIZE = 500

# # 知识库中相邻文本重合长度
# OVERLAP_SIZE = 50
# # 使用递归字符文本分割器
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=OVERLAP_SIZE
# )
# # a = text_splitter.split_text(pdf_page.page_content[0:1000])
# # print(a)

# split_docs = text_splitter.split_documents(pdf_pages)
# # print(f"切分后的文件数量：{len(split_docs)}")
# # print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

