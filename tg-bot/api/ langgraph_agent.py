from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from memory.memory import memory_tool
from tools.rag import rag_tool
from tools.qwen_vllm import qwen_tool
from tools.search import search_tool

def build_agent():
    builder = StateGraph()

    # Add memory to capture context
    builder.add_node("Memory", RunnableLambda(memory_tool))
    builder.add_node("RAG", RunnableLambda(rag_tool))
    builder.add_node("Search", RunnableLambda(search_tool))
    builder.add_node("Qwen", RunnableLambda(qwen_tool))

    # Flow: Memory → RAG → Search → Qwen → END
    builder.set_entry_point("Memory")
    builder.add_edge("Memory", "RAG")
    builder.add_edge("RAG", "Search")
    builder.add_edge("Search", "Qwen")
    builder.add_edge("Qwen", END)

    return builder.compile()
