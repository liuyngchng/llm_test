#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

import logging.config
from flask import Flask, request, jsonify, render_template

# 加载配置
logging.config.fileConfig('logging.conf')

# 创建 logger
logger = logging.getLogger(__name__)

app = Flask(__name__)


def req_with_vector_db(question: str) -> str:
    """
    加载本地矢量数据库文件, 调用 LLM API, 进行 RAG, 输出结果
    """
    embedding_model = "../bge-large-zh-v1.5"
    api_url = "http://127.0.0.1:11434"
    faiss_index = "./faiss_index"

    llm_name = "llama3.1:8b"
    # llm_name = "llama3.2:3b-text-q5_K_M"
    # llm_name = "deepseek-r1:7b"
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
    answer = result["result"]
    logger.info("answer is {}".format(answer))
    return answer


@app.route('/')
def index():
    """
     A index for static
    curl -s --noproxy '*' http://127.0.0.1:19000 | jq
    :return:
    """
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def get_data():
    """
    JSON submit, get data from application JSON
    curl -s --noproxy '*' -X POST  'http://127.0.0.1:19000/data' -H "Content-Type: application/json"  -d '{"msg":"who are you?"}'
    :return:
    """
    data = request.get_json()
    print(data)
    return jsonify({"message": "Data received successfully!", "data": data}), 200


@app.route('/submit', methods=['POST'])
def submit():
    """
    form submit, get data from form
    :return:
    """
    msg = request.form.get('msg')
    logger.info("rcv_msg: {}".format(msg))
    answer = req_with_vector_db(msg)
    logger.info("answer is：{}".format(answer))
    return answer

def test_req():
    """
    ask the LLM for some private question not public to outside,
    let LLM retrieve the information from local vector database, 
    and the output the answer.
    """
    my_question = "巴拉巴拉小魔仙是什么？"
    logger.info("invoke question: {}".format(my_question))
    answer = req_with_vector_db(my_question)
    logger.info("answer is {}".format(answer))


if __name__ == '__main__':
    # test_req()

    """
    just for test, not for a production environment.
    """
    app.run(host='0.0.0.0', port=19000)
