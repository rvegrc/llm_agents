import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),          # File output
        logging.StreamHandler()                  # Console output
    ],
    force=True  # This overrides any prior logging config
)

logging.getLogger().info("Logging is set up.")

logging.info("Importing necessary modules for the application and load environment variables.")

from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI, requests
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from langgraph.prebuilt import ToolNode
from langgraph.graph import  END, START, MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
# from IPython.display import Image, display

import os
from dotenv import load_dotenv
load_dotenv()

from pprint import pprint

# Import necessary tools
from tools.memory import save_recall_memories, search_recall_memories
from tools.rag import vectorstore_collection_init, vectorstore_add_documents
from tools.llm import llm_chat_tool, llm_call
from tools.web_search import web_search_tool


QDRANT_URL = os.getenv("QDRANT_URL")
LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

logging.info("Modules and Environment variables loaded.")
logging.info("Initializing Qdrant client.")

# Initialize Qdrant client
client_qd = QdrantClient(url=QDRANT_URL)

logging.info("Qdrant client initialized.")

class State(MessagesState):
    question: BaseMessage
    messages: Optional[List[BaseMessage]] = None
    

logging.info("LLM and embeddings initializing.")

# emb_model_name = '/models/multilingual-e5-large-instruct'
# embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

emb_model_name = 'nomic-embed-text'

embeddings = OllamaEmbeddings(
    base_url=LLM_API_SERVER_URL,
    model=emb_model_name
)

logging.info(f"Using embeddings model: {emb_model_name}")


llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_base=f'{LLM_API_SERVER_URL}/v1', # for compatibility with OpenAI
    api_key="EMPTY",  # required by LangChain, but not used by Ollama
    temperature=0.2,
    max_tokens=1000
)

# for testing
# from langchain.chat_models import init_chat_model

# llm = init_chat_model(
#     model="gpt-4.1-mini"
#     ,model_provider="openai"
#     ,temperature=0.2
#     # ,max_tokens=1000
#     ,top_p=0.5
#     )



logging.info(f"LLM  and embeddings initialized.")

logging.info("Binding tools to LLM.")

tools = [search_recall_memories, web_search_tool]
llm_with_tools = llm.bind_tools(tools)

logging.info("Tools bound to LLM.")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("""
            You are a helpful assistant with advanced long-term memory capabilities. 
            Powered by a stateless LLM, you must rely on external tools and memory systems 
            to store information between conversations. You can also perform Retrieval-Augmented 
            Generation (RAG) to access relevant knowledge in real-time.

         
            ## MEMORY USAGE GUIDELINES
            
            1. Actively use memory tools to build a comprehensive understanding of the user.
            2. Make informed suppositions and extrapolations based on stored memories.
            3. Regularly reflect on past interactions to identify patterns and preferences.
            4. Update your mental model of the user with each new piece of information.
            5. Cross-reference new information with existing memories for consistency.
            6. Store emotional context and personal values alongside factual information.
            7. Use memory to anticipate needs and tailor responses to the user’s style.
            8. Recognize and acknowledge changes in the user's situation or perspective.
            9. Leverage memories to provide personalized examples and analogies.
            10. Recall past challenges or successes to inform current problem-solving.

            ## RECALL MEMORIES
            
            Recall memories are contextually retrieved based on the current conversation:
            {recall_memories}
            
            ## RAG USAGE GUIDELINES
            
            Use RAG when you need up-to-date, domain-specific, or context-specific information

            ## INTERNET SEARCH

            Use internet search to gather information from the web when needed.


            ## INSTRUCTIONS
           
            Engage with the user naturally, as a trusted colleague or friend. 
            Do not explicitly mention your memory or retrieval capabilities. 
            Instead, seamlessly integrate them into your responses. 
            Be attentive to subtle cues and underlying emotions. 
            Adapt your communication style to match the user's preferences, language and current emotional state. 
            If you use tools, call them internally and respond only after the tool operation 
            completes successfully. Respond only with russian language.
        """),

        HumanMessagePromptTemplate.from_template("user question: {question}"),
    ]
)

logging.info("Prompt template created.")

logging.info("Initializing vector store for recall memories...")

recall_memories = vectorstore_collection_init(
    client_qd=client_qd,
    collection_name='recall_memories',
    embeddings=embeddings,
    distance="Cosine"
)
logging.info("Vector store for recall memories initialized.")


logging.info("Create function for load memories from vector store...")

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    # Search for long-term memories in Qdrant
    search_recall_memories_runnable = RunnableLambda(search_recall_memories)
    recall_contents = search_recall_memories_runnable.invoke(
        state["question"].content
    )

    # If memories were found, merge them into one string
    if recall_contents:
        recall_text = "\n".join(recall_contents)

        # Wrap the recall text in an AIMessage so it matches the conversation format
        memory_msg = AIMessage(content=f"[recall_memory]\n{recall_text}")

        # Append this memory message to the existing conversation history
        messages = state.get("messages", []) + [memory_msg]
    else:
        # If no memories found, keep the same conversation history
        messages = state.get("messages", [])

    # Return updated state with the loaded memories added to the messages
    return State(
        messages=messages,
        question=state["question"]
    )

logging.info("Function for loading memories created.")


logging.info("Creating the agent and routing...")

def clean_response(response: str) -> str:
    """Remove any internal thinking tags from the response."""
    import re
    # Remove <think>...</think> blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # # Remove any other XML tags
    # response = re.sub(r'<\/?[a-z_]+>', '', response)
    return response.strip()

# def format_recall_memory(messages: list) -> str:
#     """Format relevant messages for recall memory."""
#     relevant_messages = []
    
#     for msg in messages:
#         # Skip system messages and tool outputs
#         if isinstance(msg, (SystemMessage, ToolMessage)):
#             continue
            
#         # Format human and AI messages
#         if isinstance(msg, HumanMessage):
#             relevant_messages.append(f"User: {msg.content}")
#         elif isinstance(msg, AIMessage):
#             relevant_messages.append(f"Assistant: {msg.content}")
    
#     return (
#         "<recall_memory>\n" + 
#         "\n".join(relevant_messages) + 
#         "\n</recall_memory>"
#     ) if relevant_messages else ""


def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | create_react_agent(llm, tools)

    recall_str = (
        "<recall_memory>\n" + "\n".join(str(m) for m in state["messages"]) + "\n</recall_memory>"
    )
    

    prediction = bound.invoke({
        "question": state["question"].content,
        "recall_memories": recall_str
    })['messages'][-1]

    if hasattr(prediction, 'content'):
        prediction.content = clean_response(prediction.content)

    return State(
        messages=state["messages"] + [prediction],
        question=state["question"]
    )

logging.info("Agent created.")


logging.info("Creating function for saving user interaction...")

def save_user_interaction(state: State, config: RunnableConfig) -> None:
    """Save the user interaction to recall memories.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        None
    """
    
    user_input = state["question"].content
    assistant_output = state["messages"][-1].content
    memory = f'''{{
        user_question: {user_input},
        assistant_output: {assistant_output}
    }}'''

    save_recall_memories(memory, config)

    return state


logging.info("Function for saving user interaction created.")


logging.info("Building the graph...")

def build_agent():
    builder = StateGraph(State)

    builder.add_node(load_memories)
    builder.add_node(agent)
    builder.add_node(save_user_interaction)
    # builder.add_node("tools", ToolNode(tools)) # llm with tools does it

    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    # builder.add_conditional_edges("agent", route_tools, ["tools", END]) # llm with tools does it
    # builder.add_edge("tools", "agent")
    builder.add_edge("agent", "save_user_interaction")
    builder.add_edge("save_user_interaction", END)

    memory = InMemorySaver()
  
    return builder.compile(checkpointer=memory)

graph = build_agent()

logging.info("Graph built.")

logging.info("Creating pretty print function...")

def pretty_print_stream_chunk(chunk):
    """Pretty print the stream chunk."""
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            if updates["messages"]:  # check not empty
                pprint(updates["messages"][-1].content)                
            else:
                print("<No messages in updates>")
        else:
            print(updates)

        print("\n")
        
logging.info("Pretty print function created.")

logging.info("Creating chat_with_agent function...")

def chat_with_agent(user_input: str, user_id: str, thread_id: str) -> str:
    """Send a user input string to the agent and return the agent's final response."""
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    # for debugging
    for chunk in graph.stream({'question': HumanMessage(content=user_input)}, config=config):
        pretty_print_stream_chunk(chunk)

    
    # last_msg should be a BaseMessage
    last_msg = chunk['save_user_interaction']["messages"][-1]

    return last_msg.content

logging.info("chat_with_agent function created.")

# for testing
if __name__ == '__main__':
    chat_with_agent("Какая погода в Пекине сегодня?", "user_123", "thread_456")
