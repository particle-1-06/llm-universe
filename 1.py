from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("test_codespace/llm-universe/untitled.md")
md_pages = loader.load()
print(f"载入后的变量类型为：{type(md_pages)}，",  f"该 Markdown 一共包含 {len(md_pages)} 页")
md_page = md_pages[0]
print(f"每一个元素的类型：{type(md_page)}.", 
    f"该文档的描述性数据：{md_page.metadata}", 
    f"查看该文档的内容:\n{md_page.page_content[0:][:200]}", 
    sep="\n------\n")



# （可选）处理 PDF 的代码保持注释或单独放在另一个文件
# from langchain_community.document_loaders import PyMuPDFLoader
# loader = PyMuPDFLoader("workspaces/test_codespace/lin-universe/pumpkin_book.pdf")
# pdf_pages = loader.load()
# print(f"嵌入后的变量类型为: {type(pdf_pages)}", f"该 PDF 一共包含 {len(pdf_pages)} 页")

# from langchain_community.document_loaders import PyMuPDFLoader

# # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
# loader = PyMuPDFLoader("workspaces/test_codespace/llm-universe/pumpkin_book.pdf")

# # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
# pdf_pages = loader.load()
# print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

# 示例：检查是否有递归调用
# def load(self):
#     # 错误示例：递归调用自身
#     return self.load()  # 会导致堆栈溢出

# from unstructured.partition.md import partition_md

# # 示例：解析 Markdown 文件
# elements = partition_md(filename="example.md")
# print(elements)