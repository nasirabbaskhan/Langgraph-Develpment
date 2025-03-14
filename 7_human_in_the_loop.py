from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver # for story memory
from langgraph.types import Command, interrupt

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

tavily_api_key = os.getenv("Tavily_API_KEY")


#********************************* Defining llm ********************************

llm = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

#********************************* defining tools ********************************

# # tool for web searching
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )

# add tool
def add(a:int, b:int):
    """
    add a and b
    Args:
      a: first int
      b: second int
    """
    return a+b

# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

tools = [search_tool,add]

# # bind the toos with llm
llm_with_tools = llm.bind_tools(tools)

#****************************** defining state and nodes ********************************e

class State(TypedDict):
    messages: Annotated[list, add_messages]

    
# # defining chatbot function
# def chatbot(state:State):
#     # return {"messages": [llm_with_tools.invoke(state["messages"])]}
#     message = llm_with_tools.invoke(state["messages"])
#     # Because we will be interrupting during tool execution,
#     # we disable parallel tool calling to avoid repeating any
#     # tool invocations when we resume.
#     assert len(message.tool_calls) <= 1
#     return {"messages": [message]}

# defining human_feedback node
def human_feedback(query: str):
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# define the chatbot Node
# sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def chatbot(state: State):
   return {"messages": [llm_with_tools.invoke(state["messages"])]}


#******************************  adding nodes and edges ********************************e

# defining  StateGraph
graph_builder = StateGraph(State)

# # Add Nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START,"human_feedback")
graph_builder.add_edge("human_feedback", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools","human_feedback" )
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_conditional_edges("chatbot", tools_condition)
# graph_builder.add_edge("tools","chatbot" )


# # Finally, compile the graph with the provided checkpointer and interrupt_before.
memory = MemorySaver()
graph = graph_builder.compile(interrupt_before=["human_feedback"],checkpointer=memory)

#*************************** stream Update and run chatbot in loop  ********************************e

# # stream Update
def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    
    # Run the graph until the first interruption 
    for event in graph.stream(state, config):
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].pretty_print())
        for key, value in event.items():  # Ensure correct tuple unpacking
            if isinstance(value, dict) and "messages" in value:
                print("Assistant:", value["messages"][-1].pretty_print())
    
    # Get user input for update the state
    user_input = input("Tell me how you want to update your input:")  
    
    # # We now update the state as if we are the human_feedback node
    graph.update_state(config, {"messages": user_input}, as_node="human_feedback") 
    

    # # Continue the graph execution
    for event in graph.stream(None, config):
        # event["messages"][-1].pretty_print()   
        for key, value in event.items():
            if isinstance(value, dict) and "messages" in value:
                print("Assistant:", value["messages"][-1].pretty_print())
            
# generating response
# response = stream_graph_update("who is the curent presedent of Iran")
# print(response)    

# run chatbot in loop    
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