#! /usr/bin/python3
import logging.config
from langchain_ollama import OllamaLLM

import getpass
import os


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")


def req(question):
    """
    请求大模型 API，获取返回的信息
    :param question:
    :return:
    """
    # question = "hi"
    # model_name = "deepseekR17B"
    # llm_url = "http://127.0.0.1:11434"

    model_name = "deepseek-r1"
    llm_url = "https://aiproxy.petrotech.cnpc/v1"
    api_key = "sk-myuqX43dtcWnjoXD8a54F80f6a0143E583Fb07E15b188c91"
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
