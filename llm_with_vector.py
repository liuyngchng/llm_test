#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

import logging.config


def req_with_vector_db():
    """
    加载本地矢量数据库文件, 调用 LLM API, 进行 RAG, 输出结果
    """
    embedding_model = "../bge-large-zh-v1.5"
    api_url = "http://127.0.0.1:11434"
    faiss_index = "./faiss_index"
    question = "巴拉巴拉小魔仙是什么？"
    llm_name = "llama3.1:8b"
    #llm_name = "llama3.2:3b-text-q5_K_M"
    #llm_name = "deepseek-r1:7b"

    # 加载配置
    logging.config.fileConfig('logging.conf')

    # 创建 logger
    logger = logging.getLogger(__name__)

    # for test purpose only, read index from local file
    logger.info("embedding_model: {}".format(embedding_model))
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder='./bge-cache')
    logger.info("try to load index from local file")
    loaded_index = FAISS.load_local(faiss_index, embeddings,
                                    allow_dangerous_deserialization=True)
    logger.info("load index from local file finish")

    # 创建远程 Ollama API代理
    logger.info("get remote llm agent")
    llm = OllamaLLM(model=llm_name, base_url=api_url)
    # llm = OllamaLLM(model="codellama:7B", base_url=api_url)

    # 创建检索问答链
    logger.info("build retrieval")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=loaded_index.as_retriever())

    # 提问
    logger.info("invoke retrieval {}".format(question))
    result = qa.invoke(question)
    logger.info(result["result"])


if __name__ == "__main__":
    """
    ask the LLM for some private question not public to outside,
    let LLM retrieve the information from local vector database, 
    and the output the answer.
    """
    req_with_vector_db()
