# langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
# langgraph
from langgraph.graph import START,END
from langgraph.graph import StateGraph, add_messages
# typing
from typing import Annotated
from typing_extensions import TypedDict
# env
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
Tavily_API_KEY = os.getenv("Tavily_API_KEY")


#********************************* Defining llm ********************************

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

    
#***  defining state and nodes of search_web, search_wikipedia and generate_answer  *******************

# state
class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, add_messages]
    
# search_web node    
def search_web(state):

    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

# search_wikipedia node 
def search_wikipedia(state):

    # Search
    search_docs = WikipediaLoader(query=state['question'],
                                  load_max_docs=2).load()
     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

def generate_answer(state):

    """ Node to answer a question """

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question,
                                                       context=context)

    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])

    # Append it to state
    return {"answer": answer}


#******************************  adding nodes and edges **********************************

builder: StateGraph = StateGraph(State)

# Initialize each node with node_secret
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

# ivoking the graph
result = graph.invoke({"question": "what is langgaph"})
print(result['answer'].content)