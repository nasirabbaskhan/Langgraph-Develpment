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


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)
llm_with_tools = llm.bind_tools(tools)

query = "What is LangChain?"
result = llm_with_tools.invoke(query)
print(result)
# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}


# graph_builder.add_node("chatbot", chatbot)

# tool_node = ToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)

# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# # Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.set_entry_point("chatbot")
# graph = graph_builder.compile()