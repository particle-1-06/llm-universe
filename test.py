import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from typing import List
from langchain.schema import Document

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 设置代理（如果需要）
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def get_all_file_paths(folder_path: str) -> List[str]:
    """获取文件夹下所有文件路径"""
    folder_path = Path(folder_path).resolve()  # 转换为绝对路径
    if not folder_path.exists():
        raise ValueError(f"文件夹路径不存在: {folder_path}")
    
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def create_loaders(file_paths: List[str]) -> List:
    """根据文件类型创建对应的loader"""
    loaders = []
    for file_path in file_paths:
        try:
            file_type = file_path.lower().split('.')[-1]
            if file_type == 'pdf':
                loaders.append(PyMuPDFLoader(file_path))
            elif file_type == 'md':
                loaders.append(UnstructuredMarkdownLoader(file_path))
            # 可以继续添加其他文件类型的支持
        except Exception as e:
            print(f"无法为文件 {file_path} 创建loader: {str(e)}")
    return loaders

def load_documents(loaders: List) -> List[Document]:
    """使用loaders加载所有文档"""
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"加载文档失败: {str(e)}")
    return documents

def main():
    # 获取所有文件路径
    folder_path = 'workspaces/test_codespace'
    file_paths = get_all_file_paths(folder_path)
    print("找到的文件路径示例:", file_paths[:3])

    # 创建loaders
    loaders = create_loaders(file_paths)
    
    # 加载文档
    documents = load_documents(loaders)
    
    if not documents:
        print("没有加载到任何文档")
        return
    
    # 打印第一个文档的信息
    doc = documents[0]
    print(
        f"每一个元素的类型：{type(doc)}",
        f"该文档的描述性数据：{doc.metadata}",
        f"查看该文档的内容:\n{doc.page_content[:200]}...",  # 只显示前200字符
        sep="\n------\n"
    )

if __name__ == "__main__":
    main()