from IPython.display import Image, display
from langchain_core.tools import Tool
import uuid
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import merge_message_runs
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from pydantic import BaseModel, Field
from trustcall import create_extractor

from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# #********************************* Defining llm ********************************

# # initilize the LLM
llm = ChatGroq(   
    # model= "Gemma2-9b-It",
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

# #********************************* Defining schema and trustcall ********************************

# Memory schema
class Memory(BaseModel):
    content: str

    # class Config:
    #     title = "Memory"
    #     description = "A tool for storing and retrieving user memories. This tool is used to retain long-term information about the user."
  
  
        
# Create the Trustcall extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[Memory],
    tool_choice="Memory",
    # This allows the extractor to insert new memories
    enable_inserts=True,
)

# #****************************** defining instruction for nodes *********************************

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user. 

You have a long term memory which keeps track of information you learn about the user over time.

Current Memory (may include updated memories from this conversation): 

{memory}"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously:"""

# #****************************** defining nodes *********************************

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memories from the store and use them to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # Format the memories for the system prompt
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # Respond using memory as well as the chat history
    # response = llm.invoke([SystemMessage(content=system_msg)]+state["messages"])
    
    response = llm.invoke(input=[SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("memories", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Memory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    # updated_messages=llm.invoke(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"])
    updated_messages = llm.invoke(input=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"])


    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": updated_messages, 
                                        "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )



# #******************************  adding nodes and edges **********************************
# # Define the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# #*************************** stream Update and run chatbot in loop *****************************************

# # stream Update
def stream_graph_update(user_input:str):
    config = {"configurable": {"thread_id": "1", "user_id": "1"}} 
    # if we change thread_id then it also rember the previous memory due to long term memory 
    input_messages = [HumanMessage(content=user_input)]
    
    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    


# # run chatbot in loop    
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




