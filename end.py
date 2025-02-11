from typing import Annotated

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults  # for more search result
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from typing_extensions import TypedDict

from langgraph.graph import MessagesState,StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")

tavily_api_key = os.getenv("Tavily_API_KEY")
# print("Tavily API Key:", os.getenv("TAVILY_API_KEY"))



llm = ChatGroq(   # poetry add langchain-groq
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
    
)
# response = llm.invoke("what is langgraph")
# print(response)

# def multiply(a:int, b:int):
#     """
#     Multiply a and b
#     Args:
#       a: first int
#       b: second int
#     """
#     return a*b


# def add(a:int, b:int):
#     """
#     add a and b
#     Args:
#       a: first int
#       b: second int
#     """
#     return a+b

# def divide(a:int, b:int):
#     """
#     divide a and b
#     Args:
#       a: first int
#       b: second int
#     """
#     return a/b

search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )


# response = search.invoke("who is a current president of USA? ")
# print(response)

tools = [search]

llm_with_tools = llm.bind_tools(tools)

# response = llm_with_tools.invoke("who is a current president of USA? ")
# print(response)

def reasoner(state:MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(MessagesState)

# add nodes
graph_builder.add_node("reasoner", reasoner)
graph_builder.add_node("tools", ToolNode(tools))

# add ages
graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", tools_condition)
graph_builder.add_edge("tools","reasoner" )
graph = graph_builder.compile()


def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    # config = {"configurable": {"thread_id": "1", "user_id": "1"}} # try this
    
    for event in graph.stream(state, config,):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
            
            
# generating response
response = stream_graph_update("who is the curent presedent of Iran")
print(response)