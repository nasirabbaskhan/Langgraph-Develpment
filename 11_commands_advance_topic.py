# langchain
from langchain_core.messages import HumanMessage, AIMessage
# langgraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
# This is new
from langgraph.types import Command
# typping
import random
from typing import Annotated, TypedDict, Literal, Optional


######################### S1: Understand Command Basic Usage ########################

#******* defining state and nodes ********************

# Define graph state
class State(TypedDict):
    address: str # address where user wants to search homes for.
    nearby_homes: Optional[list[dict]] # list of nearby homes
    messages: Annotated[list, add_messages]
    
    
# Define the nodes (agents)
# node_a is decision maker
def node_a(state: State) -> Command[Literal["search_nearby_homes", "answer_question"]]:
    print("Called A")

    # THis is where llm decides which node shall I go next
    value = random.choice(["search_nearby_homes", "answer_question"])

    # This is a replacement for a conditional edge function
    if value == "search_nearby_homes":
        goto = "search_nearby_homes"
    else:
        goto = "answer_question"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is a replacement for an edge
        goto=goto,
    )


# Nodes search_nearby_homes and answer_question are unchanged (just like before)
def search_nearby_homes(state: State):
    print("Called search_nearby_homes!")
    return {"nearby_homes": [{"home_1": "Zia House"}, {"home_2": "Qasim House"}]}

# answer_question node
def answer_question(state: State):
    print("Called answer_question")
    return {"messages": [AIMessage(content="System is down - please try later")]}


# #****************************** adding nodes and edges ***********************************

graph_builder = StateGraph(State)

graph_builder.add_node(node_a)
graph_builder.add_node(search_nearby_homes)
graph_builder.add_node(answer_question)

graph_builder.add_edge(START, "node_a")
# NOTE: there are no edges between nodes A, B and C!

graph = graph_builder.compile()

# invoking
agent_response = graph.invoke({"address": "Karachi"})

print("\n[FINAL RESPONSE]\n", agent_response)

######################### S2: Update State using Command ########################
# After node_1 we can update our state as well. So if user is already 
# in our system we can add it's user data in the stage to personalize user experience.

#******* defining state and nodes ********************

# # Define graph state
# class State(TypedDict):
#     address: str # address where user wants to search homes for.
#     nearby_homes: Optional[list[dict]] # list of nearby homes
#     messages: Annotated[list, add_messages]
#     user_data: Optional[dict] # user data - for registered users we already have this
    
    
# # Define the nodes (agents)
# # node_a is decision maker
# def node_a(state: State) -> Command[Literal["search_nearby_homes", "answer_question"]]:
#     print("Called A")
    
#     # Get User Data From Memory or Data Store
#     fetched_user_data = {"name": "Ammen Alam"}

#     # THis is where llm decides which node shall I go next
#     value = random.choice(["search_nearby_homes", "answer_question"])

#     # This is a replacement for a conditional edge function
#     if value == "search_nearby_homes":
#         goto = "search_nearby_homes"
#     else:
#         goto = "answer_question"

#     # note how Command allows you to BOTH update the graph state AND route to the next node
#     return Command(
#         update={"user_data": fetched_user_data},
#         # this is a replacement for an edge
#         goto=goto,
#     )


# # Nodes search_nearby_homes and answer_question are unchanged (just like before)
# def search_nearby_homes(state: State):
#     print("Called search_nearby_homes!")
#     print("UserInfo", state["user_data"])
#     return {"nearby_homes": [{"home_1": "Zia House"}, {"home_2": "Qasim House"}]}

# # answer_question node
# def answer_question(state: State):
#     print("Called answer_question")
#     print("UserInfo", state["user_data"])
#     user_name = state["user_data"].get("name", "Guest")
#     return {"messages": [AIMessage(content=f"Hi, {user_name} Welcome to Homes AI Search Engine")]}


#****************************** adding nodes and edges ***********************************

# graph_builder = StateGraph(State)

# graph_builder.add_node(node_a)
# graph_builder.add_node(search_nearby_homes)
# graph_builder.add_node(answer_question)

# graph_builder.add_edge(START, "node_a")
# # NOTE: there are no edges between nodes A, B and C!

# graph = graph_builder.compile()

# # invoking
# agent_response = graph.invoke({"address": "Karachi"})

# print("\n[FINAL RESPONSE]\n", agent_response)


######################### S3: Run Nodes(Agents) in Parallel ########################
# Now our user requirement is to both find homes and have conversation with user.
# For this:

# - We shall call Both Nodes in Parrallel

#******* defining state and nodes ********************

# # Define graph state
# class State(TypedDict):
#     address: str # address where user wants to search homes for.
#     nearby_homes: Optional[list[dict]] # list of nearby homes
#     messages: Annotated[list, add_messages]
#     user_data: Optional[dict] # user data - for registered users we already have this
    
    
# # Define the nodes (agents)
# # node_a is decision maker
# def node_a(state: State) -> Command[Literal["search_nearby_homes", "answer_question"]]:
#     print("Called A")
    
#     # Get User Data From Memory or Data Store
#     fetched_user_data = {"name": "Ammen Alam"}

#     # note how Command allows you to BOTH update the graph state AND route to the next node
#     return Command(
#         update={"user_data": fetched_user_data},
#         # this is a replacement for an edge
#         goto=["search_nearby_homes", "answer_question"],
#     )


# # Nodes search_nearby_homes and answer_question are unchanged (just like before)
# def search_nearby_homes(state: State):
#     print("Called search_nearby_homes!")
#     print("UserInfo", state["user_data"])
#     return {"nearby_homes": [{"home_1": "Zia House"}, {"home_2": "Qasim House"}]}


# def answer_question(state: State):
#     print("Called answer_question")
#     print("UserInfo", state["user_data"])
#     user_name = state["user_data"].get("name", "Guest")
#     return {"messages": [AIMessage(content=f"Hi, {user_name} Welcome to Homes AI Search Engine")]}


#****************************** adding nodes and edges ***********************************

# graph_builder = StateGraph(State)

# graph_builder.add_node(node_a)
# graph_builder.add_node(search_nearby_homes)
# graph_builder.add_node(answer_question)

# graph_builder.add_edge(START, "node_a")
# # NOTE: there are no edges between nodes A, B and C!

# graph = graph_builder.compile()

# # invoking
# agent_response = graph.invoke({"address": "Karachi"})

# print("\n[FINAL RESPONSE]\n", agent_response)