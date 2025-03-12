# langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
# langgraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.state import CompiledStateGraph
# env
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")


#********************************* Defining llm ********************************

llm = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

#********************************* defining tools ********************************

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)

#************************************ defining Nodes *******************************************************

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

#******************************  adding nodes and edges **********************************

# Graph
builder: StateGraph = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory: MemorySaver = MemorySaver()
graph: CompiledStateGraph = builder.compile(checkpointer=memory)


#*************************** stream Update and run chatbot in loop  ********************************e

# stream Update
def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(state, config):
        for value in event.values():
            # event['messages'][-1].pretty_print()
            print("Assistant:", value["messages"][-1].pretty_print())
            
    # We can also browse the state history of our agent.
    all_states = [s for s in graph.get_state_history(config)]

    # We can re-run (Replying) our agent from any of the prior steps.
    to_replay = all_states[-2] #  second last state
    
    # time travel
    # we want it should replay from second last state --> all_states[-2] 
    # because may be we want to debug at that point
    for event in graph.stream(None, to_replay.config):
        # event['messages'][-1].pretty_print()
        for value in event.values():
                # print("Assistant:", value["messages"][-1].content)
                print("Assistant:", value["messages"][-1].pretty_print())


# run chatbot in loop
while True:
    try:
       user_input = input("User: ") 
       if user_input.lower() in ["quit","q","exit" ]:
           print("Goodby")
           break
       
       stream_graph_update(user_input)
        
    except Exception as e:
        print(f"An Error occurred: {e}")
        break

