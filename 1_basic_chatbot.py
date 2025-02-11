from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# step 1: Defining State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
    
# step 2: inililizing StateGraph
graph_builder = StateGraph(State)

# step 3 initilize the LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# Defining the chatbot function
def chatbot(state:State):
    response = llm.invoke(state["messages"])
    return {"messages": [{"role": "assistant", "content": response}]}
    # return {"messages": [llm.invoke(state["messages"])]}


# Add Nodes and edges
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# compile the graph
graph = graph_builder.compile()

# stream Update
def stream_graph_updates(user_input:str):
    state = {"messages":[{"role":"user", "content":user_input}]}
    for event in graph.stream(state):
        for value in event.values():
            print("Assistant:", value["messages"][-1]["content"])

# run chatbot in loop
while True:
    try:
       user_input = input("User: ") 
       if user_input.lower() in ["quit","q","exit" ]:
           print("Goodby")
           break
       stream_graph_updates(user_input)
        
    except Exception as e:
        print(f"An Error occurred: {e}")
        break
        
    
        
        
    
    


