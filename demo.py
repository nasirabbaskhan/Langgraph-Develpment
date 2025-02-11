from typing import Annotated

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")

tavily_api_key = os.getenv("Tavily_API_KEY")
# print("Tavily API Key:", os.getenv("TAVILY_API_KEY"))
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )


# result = search_tool.invoke("What is the current temperature in Karachi?")
# print(result)
tools = [search_tool]
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

llm_with_tools = llm.bind(tools=tools)
# llm_with_tools = llm.bind_tools(tools)

# print("tolls are ", llm_with_tools)
query = "What is LangChain? Use the Tavily search tool if needed."
result = llm_with_tools.invoke(query)
print(result)


# def chatbot(state: State):
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages":[response]}
#     # return {"messages":[{"role":"assistant","content":response}]}


# graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools=[search_tool])
# graph_builder.add_node("tools", tool_node)

# graph_builder.add_conditional_edges("chatbot",tools_condition)
# # Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.set_entry_point("chatbot")
# graph = graph_builder.compile()


# def stream_graph_update(user_input:str):
#     # state = {"messages":[{"role":"user", "content":user_input}]}
#     state = {"messages":[user_input]}
#     config = {"configurable": {"thread_id": "1"}}
    
#     for event in graph.stream(state, config):  # stream_mode="values" can be removed
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)
        
    
# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "q", "exit"]:
#             print("Good By")
#             break
#         stream_graph_update(user_input)
#     except Exception as e:
#         print(f"An Error occurred: {e}")
#         break