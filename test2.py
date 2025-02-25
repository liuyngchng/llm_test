#! /usr/bin/python3
from langgraph.graph import StateGraph, START, END
from graphviz import Digraph

        
def chatbot():
    return {"messages": ["this is a test"]}

def agent_node(state): return state
def tool_node(state): return state
def condition(state): return "route_a"

# 通过 graphviz 输出图
def export_graphviz(graph):
    dot = Digraph()
    # 添加节点
    for node in graph.nodes:
        dot.node(node, f"{node.upper()}节点")
    for src, dst in graph.edges:
        dot.edge(src, dst, xlabel=graph.edges[(src, dst)].get("condition", ""))

    return dot


def test():
    sg = StateGraph(dict)
    # 定义图的入口和边
    sg.add_node("chatbot", chatbot)
    sg.add_edge(START, "chatbot")
    sg.add_edge("chatbot", END)

    # 编译图
    graph = sg.compile()
    # 生成graph.pdf
    export_graphviz(graph.get_graph()).render("graph")

if __name__ == "__main__":
    test()
