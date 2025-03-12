
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langgraph.graph import START, END, StateGraph


class State(TypedDict):
    input: str

def step_1(state: State) -> State:
    print("---Step 1---")
    return state

def step_2(state: State) -> State:
    # Let's optionally raise a NodeInterrupt if the length of the input is longer than 5 characters
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

    print("---Step 2---")
    return state

def step_3(state: State) -> State:
    print("---Step 3---")
    return state

graph_builder= StateGraph(State)
graph_builder.add_node("step_1", step_1)
graph_builder.add_node("step_2", step_2)
graph_builder.add_node("step_3", step_3)
graph_builder.add_edge(START, "step_1")
graph_builder.add_edge("step_1", "step_2")
graph_builder.add_edge("step_2", "step_3")
graph_builder.add_edge("step_3", END)

# Set up memory
memory: MemorySaver = MemorySaver()

# Compile the graph with memory
graph = graph_builder.compile(checkpointer=memory)



 # stream Update
def stream_graph_update(user_input:str):
    state = {"input": user_input}
    # state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption 
    for event in graph.stream(state, config):
        print(event)
         
  # Get user input for update the state
    user_input = input("Tell me how you want to update your input, your input should be less than 5 characters:")  
    
    # # We now update the state as if we are the human_feedback node
    graph.update_state(config,{"input": user_input},)
    
    # # Continue (resuming) the graph execution
    for event in graph.stream(None, config,):
        print(event)
   

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
