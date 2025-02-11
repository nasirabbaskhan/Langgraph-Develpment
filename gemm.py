from typing import Annotated
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
# from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("Tavily_API_KEY")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


# @tool
# def human_assistance(
#     name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
# ) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt(
#         {
#             "question": "Is this correct?",
#             "name": name,
#             "birthday": birthday,
#         },
#     )
#     if human_response.get("correct", "").lower().startswith("y"):
#         verified_name = name
#         verified_birthday = birthday
#         response = "Correct"
#     else:
#         verified_name = human_response.get("name", name)
#         verified_birthday = human_response.get("birthday", birthday)
#         response = f"Made a correction: {human_response}"

#     state_update = {
#         "name": verified_name,
#         "birthday": verified_birthday,
#         "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
#     }
#     return Command(update=state_update)



tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )
# result = tool.invoke("what is the current temperature in Karachi")  # to varyfi it is runnig or not
# print(result)
tools = [tool]
llm = ChatGroq(  
    # model="mixtral-8x7b-32768",
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
    
)
# result = llm.invoke("what is langchain")  # to varyfi it is runnig or not
# print(result)
# llm_with_tools = llm.bind(tools=tools)
llm_with_tools = llm.bind_tools(tools)
# result = llm_with_tools.invoke("who is presedent of Pakistan")
# print(result)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}
    # return {"messages":[{"role":"assistant","content":response}]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    # config = {"configurable": {"thread_id": "1", "user_id": "1"}} # try this
    
    for event in graph.stream(state, config,):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
        
    
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Good By")
            break
        stream_graph_update(user_input)
    except Exception as e:
        print(f"An Error occurred: {e}")
        break