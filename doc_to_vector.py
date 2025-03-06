#! /usr/bin/python3
"""
将本地文档进行向量化，形成矢量数据库文件，用于 LLM 进行 RAG
"""

from langchain_community.document_loaders import TextLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

import logging.config
import os


def vectoring():
    # knowledge_dir = "/home/rd/software/ollama_models/test/"
    # knowledge_dir = "/home/rd/workspace/ind-mt/mt-cmn/src/main/java/cm/iot/cmn/"
    knowledge_file = "./1.csv"
    model_file = "../bge-large-zh-v1.5"
    vector_file = "./faiss_index"

    # 加载配置
    logging.config.fileConfig('logging.conf')

    # 创建 logger
    logger = logging.getLogger(__name__)

    # 加载知识库文件
    logger.info("load local doc {}".format(knowledge_file))
    # load word, PDF file
    # loader = UnstructuredLoader(file)
    # load txt file
    loader = TextLoader(knowledge_file, encoding='utf8')
    # load a directory
    # loader = DirectoryLoader(path=knowledge_dir, recursive=True, load_hidden=False,
    #                          loader_cls=TextLoader, glob="**/*.java")
    documents = loader.load()
    logger.info("loaded {} files, files name list as following".format(len(documents)))
    for doc in documents:
        print("\t\t{}".format(doc.metadata.get("source")))

    # 将文档分割成块
    logger.info("split doc")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 加载Embedding模型，进行自然语言处理
    logger.info("load embedding model")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_file,
        cache_folder='./bge-cache'
    )

    # 创建向量数据库
    logger.info("build vector db")
    db = FAISS.from_documents(texts, embeddings)
    # 保存向量存储库至本地，save_local() 方法将生成的索引文件保存到本地，以便之后可以重新加载
    logger.info("start save vector db to local file")

    db.save_local(vector_file)
    logger.info("vector db saved to local file {}".format(vector_file))


if __name__ == "__main__":
    # os.putenv("CUDA_VISIBLE_DEVICES", "1")
    # a = os.environ.get("CUDA_VISIBLE_DEVICES")
    # print(a)

    # os.environ["CUDA_VISIBLE_DEVICES"] = 0
    a = "hello"
    a += " world"
    a
    vectoring()
