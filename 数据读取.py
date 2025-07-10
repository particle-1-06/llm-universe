from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader1 = PyMuPDFLoader("test_codespace/llm-universe/pumpkin_book.pdf")
loader2 = UnstructuredMarkdownLoader("test_codespace/llm-universe/untitled.md")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader1.load()
md_pages = loader2.load()

# print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
# print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")

pdf_page = pdf_pages[1]
# print(f"每一个元素的类型：{type(pdf_page)}.", 
#     f"该文档的描述性数据：{pdf_page.metadata}", 
#     f"查看该文档的内容:\n{pdf_page.page_content}", 
#     sep="\n------\n")

# md_page = md_pages[0]
# print(f"每一个元素的类型：{type(md_page)}.", 
#     f"该文档的描述性数据：{md_page.metadata}", 
#     f"查看该文档的内容:\n{md_page.page_content[0:][:200]}", 
#     sep="\n------\n")


import re
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
print(pdf_page.page_content)
pdf_page.page_content = pdf_page.page_content.replace('•', '')
pdf_page.page_content = pdf_page.page_content.replace(' ', '')
print(pdf_page.page_content)

# md_page.page_content = md_page.page_content[0:][:200].replace('\n\n', '\n')
# print(md_page.page_content)



''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
#导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 知识库中单段文本长度
CHUNK_SIZE = 500

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50
# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
a = text_splitter.split_text(pdf_page.page_content[0:1000])
print(a)

# split_docs = text_splitter.split_documents(pdf_pages)
# print(f"切分后的文件数量：{len(split_docs)}")
# print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

