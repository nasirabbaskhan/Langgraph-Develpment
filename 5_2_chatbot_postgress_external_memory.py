# langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_groq import ChatGroq
# typing
from typing_extensions import TypedDict
from typing import Annotated
# langgraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# postgress database
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
# env loading
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Tavily_API_KEY = os.getenv("Tavily_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

#********************************* setup postgress database ********************************

# Connection pool for efficient database access
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
# Create a persistent connection pool
pool = ConnectionPool(conninfo=DATABASE_URL, max_size=20, kwargs=connection_kwargs)
# Initialize PostgresSaver checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensure database tables are set up

#********************************* Defining llm ********************************

llm = ChatGroq(   
    # model= "Gemma2-9b-It"
    model="Llama3-8b-8192",
    temperature=0,
    stop_sequences=""
)


#********************************* defining tool for web searching ********************************

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

#****************************** defining state and nodes ********************************e

class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary:str

# define the chatbot Node
def chatbot(state:State):
     # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    return {"messages": [llm_with_tools.invoke(messages)]} 


# We'll define a node to produce a summary
def summarize_conversation(state:State):
    summary = state.get("summary", "")  # it will get summary or empty
    
    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"
         # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
    


#******************************  adding nodes and edges ********************************e

# defining  StateGraph
graph_builder = StateGraph(State)   

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("summarize_conversation", summarize_conversation)

graph_builder.add_edge(START, "summarize_conversation")
graph_builder.add_edge("summarize_conversation", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile(checkpointer=checkpointer)


#*************************** stream Update and run chatbot in loop  ********************************e

def stream_graph_update(user_input:str):
    # state = {"messages":[{"role":"user", "content":user_input}]}
    state = {"messages":[user_input]}
    config = {"configurable": {"thread_id": "1"}}
    
    for event in graph.stream(state, config):
        for value in event.values():
            # print("Assistant:", value["messages"][-1].content)
            # print("Assistant:", value["messages"][-1].pretty_print())
            messages = value.get("messages", [])
            if messages:
                print("Assistant:", messages[-1].pretty_print())
            else:
                print("Assistant: No response received.")
            
            
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



