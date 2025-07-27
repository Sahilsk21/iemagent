from flask import Flask, render_template, jsonify, request,session
from langchain_google_genai import GoogleGenerativeAI 
import markdown
from markupsafe import Markup
from source_code.helper import save_user_phone
from source_code.customretriver import HybridRetriever 
from source_code.promt import prompt_template
from functools import lru_cache
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from source_code.promt import * 
import os  
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma 
import json 
from langchain.docstore.document import Document 
from langchain.schema import Document 
from langgraph.graph import StateGraph,END,START,MessagesState

from langgraph.checkpoint.memory import MemorySaver 

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool 
from typing import List
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

google_api_key=  
mistral=

app = Flask(__name__) 
app.secret_key = os.urandom(24)

load_dotenv()
# api_key = os.getenv("google_api_key")
api_key=google_api_key
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# Initialize embeddings

#####bm25 copus###### 


# Initialize Pinecone client




#### chroma db code #### 
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#         api_key=,
#         model_name="BAAI/bge-large-en-v1.5")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=)


CHROMA_DIR = "chroma_store" 
### Load Chroma Vector Store
if os.path.exists(CHROMA_DIR):
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    with open("static/bm25_corpus2.json", "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    docs = [Document(page_content=chunk) for chunk in raw_chunks]
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    db.persist()

retriever = HybridRetriever(db=db, embeddings=embeddings,  k=5)


from langchain_google_genai import ChatGoogleGenerativeAI
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=)
#llm=ChatGroq(api_key=,model="deepseek-r1-distill-llama-70b")
#llm=ChatMistralAI(mistral_api_key=mistral,model="mistral-large-latest",temperature=0)
##### mentain the history ####### 
# print(custom_retriever.invoke('bca fee'))


# Set up the LLM (Google Gemini in this case)


# Set up the QA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=custom_retriever,  # Use the custom retriever
#     return_source_documents=True
# ) 

# Create Conversational Retrieval Chain



#### otp verification ####### 
# Twilio API Keys 
# TWILIO_ACCOUNT_SID = ""
# TWILIO_AUTH_TOKEN = ""
# TWILIO_PHONE_NUMBER = "" 

# @app.route("/send_otp", methods=["POST"])
# def send_otp():
#     """Send OTP to user's phone number."""
#     data = request.json
#     phone = data.get("phone")

#     if not phone:
#         return jsonify({"status": "error", "message": "Phone number required."})

#     # Store phone in session before OTP verification
#     session["phone"] = phone  

#     otp = generate_otp()
#     session["otp"] = otp
#     session["otp_verified"] = False

#     send_otp_sms(phone, otp)

#     return jsonify({"status": "OTP Sent"})


# @app.route("/verify_otp", methods=["POST"])
# def verify_otp():
#     """Verify user-entered OTP."""
#     data = request.json
#     user_otp = data.get("otp")

#     if "otp" in session and session["otp"] == user_otp:
#         session["otp_verified"] = True

#         # Ensure phone number exists before saving
#         phone = session.get("phone")
#         if phone:
#             save_user_phone(phone)  # Save phone number only if it's set

#         session.pop("otp", None)  # Remove OTP after verification
#         return jsonify({"status": "verified"})

#     return jsonify({"status": "invalid"})

# Add this near your other store/memory initialization




@app.route("/")
def index():
    return render_template('chat.html')

###### phone number verifiaction code ######## 
@app.route("/store_phone", methods=["POST"])
def store_phone():
    """Store phone number if it's valid."""
    data = request.json
    phone = data.get("phone")

    if not phone or len(phone) != 10 or not phone.isdigit():
        return jsonify({"status": "error", "message": "Invalid phone number"})

    session["phone_verified"] = True
    save_user_phone(phone)
    return jsonify({"status": "success"})

##### get the frindly answer from this two function ###### 


@tool
def rag_call(query: str) -> str:
    """Use this tool for university-related queries about courses, programs, admissions, faculty, or facilities.

    The tool uses a hybrid retrieval system combining vector search and BM25 for optimal results.
    """
    try:
        # 1. Retrieve documents using your hybrid retriever
        docs: List[Document] = retriever.get_relevant_documents(query)

        if not docs:
            return "No relevant information found in the university knowledge base."

        # 2. Format the response with confidence indicators
        response_parts = [
            f"Found {len(docs)} relevant results for: '{query}'\n"
            "----------------------------------------"
        ]

        for i, doc in enumerate(docs, 1):
            # Extract metadata with fallbacks
            source = doc.metadata.get('source', 'unknown source')
            page = doc.metadata.get('page', 'N/A')

            response_parts.append(
                f"\nResult {i} (Source: {source}, Page: {page}):\n"
                f"{doc.page_content}\n"
                "----------------------------------------"
            )

        # 3. Add retrieval summary
        response_parts.append(
            f"\nTip: For more specific information, try rephrasing your question "
            "with details like course codes or department names."
        )

        return "\n".join(response_parts)

    except Exception as e:
        return (
            "Failed to retrieve information. "
            f"Technical details: {str(e)}\n"
            "Please try again or ask a different question."
        )
@tool
def is_identity_question(query: str) -> bool:
    """Check if the question is asking about the chatbot's identity or capabilities."""
    identity_keywords = ["who are you", "what are you", "your name", "your purpose", "your function"]
    return any(keyword in query.lower() for keyword in identity_keywords)

@tool
def is_friendly_question(query: str) -> bool:
    """Check if the question is a friendly, non-academic conversation."""
    friendly_keywords = ["how are you", "what's up", "how's it going", "how do you feel"]
    return any(keyword in query.lower() for keyword in friendly_keywords)

@tool
def is_greeting(query: str) -> bool:
    """Check if the message is a greeting like hello, hi, etc."""
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(greeting in query.lower() for greeting in greetings)

tool=[rag_call,is_identity_question,is_friendly_question,is_greeting]

llm_tools=llm.bind_tools(tool)

tools_by_name = {tool.name: tool for tool in tool}


# Define our tool node
def tool_node(state: MessagesState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

# Define the node that calls the model 

def call_model(
    state:MessagesState ,
    config: RunnableConfig,
):
    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    # system_prompt = SystemMessage(
    #     "You are IEM Chatbot,a highly knowledgeable and **strictly factual** AI assistant for the Institute of Engineering and Management (IEM). Your role is to provide **precise, up-to-date, and sourced answers** based on the **retrieved documents** from the IEM knowledge base"
    # )
    response = llm_tools.invoke([prompt_template] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue" 
    
memory=MemorySaver() 

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Now we can compile and visualize our graph
graph = workflow.compile(checkpointer=memory) 

config = {"configurable": {"thread_id": "user12"}} 

def print_stream(stream):
    last_message = None
    for s in stream:
        messages = s["messages"]
        if messages:  # Check if the messages list is not empty
            last_message = messages[-1]

    if last_message:
        if isinstance(last_message, ToolMessage) and last_message.content:
            try:
                # Attempt to parse the content as JSON if it's a JSON string
                # This is often the case for ToolMessage content
                content_data = json.loads(last_message.content)
                # You might need to adjust this depending on the structure of your tool output
                # For example, if your tool returns a simple string, you can return that directly.
                return content_data
            except json.JSONDecodeError:
                # If it's not JSON, return the raw content
                return last_message.content
        elif isinstance(last_message, SystemMessage) or isinstance(last_message, AIMessage) or isinstance(last_message, HumanMessage):
             # If it's a different type of message, return the content
             return last_message.content
        else:
            # If it's another type of message or has no content
            return str(last_message) # Or handle as needed
    return None # Return None if no message was processed
@lru_cache(maxsize=100)         
def chatbot_response(query): 
    
    inputs = {"messages": [HumanMessage(content=query)]}
    result=print_stream(graph.stream(inputs,config=config, stream_mode="values"))
    return Markup(markdown.markdown(result))

 


@app.route("/get", methods=["POST"])
def chat(): 
    
    msg = request.form["msg"]
    input = msg
    print(input)
    

    # Handle greetings first
    response = chatbot_response(input)

    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True) 
   
