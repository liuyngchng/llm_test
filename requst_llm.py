#! /usr/bin/python3
# -*- coding: utf-8 -*-

import logging.config
from langchain_ollama import OllamaLLM


def req(question):
    """
    请求大模型 API，获取返回的信息
    :param question:
    :return:
    """
    # question = "hi"
    model_name = "deepseekR17B"
    llm_url = "http://127.0.0.1:11434"
    # 加载配置
    logging.config.fileConfig('logging.conf')

    # 创建 logger
    logger = logging.getLogger(__name__)
    llm = OllamaLLM(model=model_name, base_url=llm_url)
    logger.info("invoke question: {}".format(question))
    answer = llm.invoke(question)
    logger.info("answer is {}".format(answer))


if __name__ == "__main__":
    req("hi")
