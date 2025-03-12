# langchain
import uuid
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import merge_message_runs,HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
# langgraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver # short term memory
from langgraph.store.memory import InMemoryStore  # long term memory
from langgraph.store.base import BaseStore  
from trustcall import create_extractor
# typing
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
# env
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

# #********************************* Defining llm ********************************

# # initilize the LLM
model = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)

# #********************************* Defining llm ********************************

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']
    
    
# #********************************* Defining schema, spy class and trustcall ********************************

#
# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(description="Personal connection of the user, such as family members, friends, or coworkers",default_factory=list)
    interests: list[str] = Field(description="Interests that the user has", default_factory=list)

# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).")
    deadline: Optional[datetime] = Field(description="When the task needs to be completed by (if applicable)",default=None)
    solutions: list[str] = Field(description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)", min_items=1,default_factory= list)
    status: Literal["not started", "in progress", "done", "archived"] = Field(description="Current status of the task",default="not started")

# Inspect which tool calls has been called by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Initialize the spy
spy = Spy()

# Create the Trustcall extractor for updating the user profile 
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# Add the spy as a listener
trustcall_extractor_see_all_tool_calls = profile_extractor.with_listeners(on_end=spy)


# #***************************** defining function for extract tool information *********************************

def extract_tool_info(tool_calls, schema_name="Memory"):
    
    """Extract information from tool calls for both patches and new memories.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)

# #****************************** defining instruction for nodes *********************************

# Chatbot instruction for choosing what to update and what tools to call 
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


# # #***************************************** defining nodes ************************************************
# # node 1
def task_mAIstro(state: MessagesState, config:RunnableConfig, store: BaseStore):
    
    """Load memories from the store and use them to personalize the chatbot's response."""
    user_id = config["configurable"]["user_id"]
    
    # Retrieve profile memory from the store of spasific user id
    namespace = ("profile",user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None
        
    # Retrieve task memory from the store of spasific user id
    namespace = ("task", user_id)
    memories = store.search(namespace)
    if memories:
        todo = "\n".join(f"{mem.value}" for mem in memories)
    else:
            todo = None
            
    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories =  store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = None
        
    sys_message = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile, todo=todo,instructions=instructions)
    
    # Respond using memory as well as the chat history
    response = model.bind_tools([UpdateMemory],parallel_tool_calls=False).invoke(SystemMessage(content=sys_message)+state["messages"])
    
    return {"messages": [response]}

# # node 2
def update_profile(state: MessagesState, config:RunnableConfig, store:BaseStore):
    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)
    
    tool_name = "Profile"
    if existing_items:
        for existing_item in existing_items:
            existing_memories = (existing_item.key, tool_name, existing_item.value) 
    else:
        None
        
    # Format the existing memories for the Trustcall extractor
    # tool_name = "Profile"
    # existing_memories = ([(existing_item.key, tool_name, existing_item.value)
    #                       for existing_item in existing_items]
    #                       if existing_items
    #                       else None
    #                     )


    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time = datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages = [SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]+state["messages"][:-1]))
     
    # Invoke the extractor
    result = profile_extractor.invoke({"messages":updated_messages,
                                       "existing":existing_memories })
    
    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0]['id']}]}
    
    
    # # node 3
def update_todos(state:MessagesState, config:RunnableConfig, store:BaseStore):
    
    """Reflect on the chat history and update the memory collection."""
        
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Define the namespace for the memories
    namespace = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace) 
    
     # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )
    
    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))
    
    # Initialize the spy
    spy = Spy()
    
    # Create the Trustcall extractor for updating the ToDo list 
    todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice=tool_name,
    enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({"messages": updated_messages, 
                                    "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id":tool_calls[0]['id']}]}

    
# # node 4
def update_instructions(state:MessagesState, config:RunnableConfig, store:BaseStore ):
    
    """Reflect on the chat history and update the memory collection."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    namespace = ("instructions", user_id)
    key = "user_instructions"
    existing_memory = store.get(namespace, key )
    if existing_memory:
        current_instructions = existing_memory.value
    else:
        current_instructions = None
        
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=current_instructions)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])
    
    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id":tool_calls[0]['id']}]}

# Conditional edge
def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_profile","update_todos", "update_instructions"]:
    
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_calls = message.tool_calls[0]
        if tool_calls["args"]["update_type"] == "user":
            return "update_profile"
        if tool_calls["args"]["update_type"] == "todo":
            return "update_todos"
        if tool_calls["args"]["update_type"] == "instructions":
            return "instructions"
        else:
            raise ValueError
        
        
# # #******************************  adding nodes and edges **********************************
# # # Define the graph
builder = StateGraph(MessagesState)

builder.add_node("task_mAIstro",task_mAIstro)
builder.add_node("update_profile", update_profile)
builder.add_node("update_todos", update_todos)
builder.add_node("update_instructions", update_instructions)

builder.add_edge(START, "task_mAIstro")
builder.add_conditional_edges("task_mAIstro", route_message)
builder.add_edge("update_profile","task_mAIstro")
builder.add_edge("update_todos","task_mAIstro")
builder.add_edge("update_instructions","task_mAIstro")


# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# We compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)


# # #*************************** stream Update and run chatbot in loop  ********************************e

# # # stream Update
def stream_graph_update(user_input:str):
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
    # if we change thread_id then it also rember the previous memory due to long term memory 
    input_messages = [HumanMessage(content=user_input)]
    
    # Run the graph
    for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    


# # # run chatbot in loop    
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



