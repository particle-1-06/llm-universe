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
folder_path = 'workspaces/test_codespace/llm-universe'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
#    elif file_type == 'md':
#        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())

text = texts[1]
print(f"每一个元素的类型：{type(text)}.", 
    f"该文档的描述性数据：{text.metadata}", 
    f"查看该文档的内容:\n{text.page_content[0:]}", 
    sep="\n------\n")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分文档
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500, chunk_overlap=50)

# split_docs = text_splitter.split_documents(texts)
# print(texts)