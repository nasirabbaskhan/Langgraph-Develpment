from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

from langgraph.types import Command, interrupt

#****************************** defining state and nodes ********************************e

class State(TypedDict):
    input: str

# node
def human_approval(state) -> Command[Literal["__end__", "call_agent"]]:
    print("---human_feedback---")

    is_approved = interrupt("Is this correct?")

    print("\n user respnse is: ", is_approved)

    if is_approved == "yes":
        return Command(goto="call_agent")
    else:
        return Command(goto="__end__")

# node
def call_agent(state):
    print("---call_agent 3---")
    pass

#******************************  adding nodes and edges **********************************

builder = StateGraph(State)
builder.add_node("human_approval", human_approval)
builder.add_node("call_agent", call_agent)

builder.add_edge(START, "human_approval")

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory)

#*************************** stream Update and run chatbot in loop  ********************************e
    
def stream_graph_update(user_input:str):
    state = {"input": user_input}
    # state = {"messages":[user_input]}
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption 
    for event in graph.stream(state, thread):
        print(event)
         
  # Get user input for update the state
    user_input = input("Do you want to continue? yes or no:  ")  
    
    
   # HOW TO RESUME
    # Continue the graph execution
    for event in graph.stream(Command(resume=user_input),thread):
        print(event)
            
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
