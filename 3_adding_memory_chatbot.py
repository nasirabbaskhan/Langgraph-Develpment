from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver # for story memory

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

tavily_api_key = os.getenv("Tavily_API_KEY")


# defining state
class State(TypedDict):
    messages:Annotated[list, add_messages]
   
    
# defining  StateGraph
graph_builder = StateGraph(State)


# defining llm  -->> ChatGoogleGenerativeAI have issue to call the tools
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0.2
# )


llm = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

# tool for web searching
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )

tools = [search_tool]

# bind the toos with llm
llm_with_tools = llm.bind_tools(tools)

# defining chatbot function
def chatbot(state:State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add Nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools","chatbot" )


# Finally, compile the graph with the provided checkpointer.
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)



# stream Update
def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(state, config):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
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