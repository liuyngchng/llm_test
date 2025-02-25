#! /usr/bin/python3
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM

# 初始化模型
llm = OllamaLLM(model="deepseekR1:7B", base_url='http://11.10.36.1:11435')

    
# 定义图节点
def chatbot(d):
    return {"messages": [llm.invoke(d)]}
# 创建一个 StateGraph 对象
graph_builder = StateGraph(dict)
# 定义图的入口和边
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()

# 执行图
user_input = '介绍你自己'
for event in graph.stream(user_input):
    for value in event.values():
        print("Assistant:", value["messages"])
