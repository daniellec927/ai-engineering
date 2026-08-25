import os
from openai import OpenAI
import gradio as gr
import chromadb
import re
import uuid
from pprint import pprint
from huggingface_hub import HfFileSystem
import requests
import json
import random

# ------------------------------------------------------
# Setup
# ------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API Key is missing.")
client = OpenAI()
HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------------------------------
# Document
# ------------------------------------------------------
hffs = HfFileSystem(token=os.getenv("HF_TOKEN"))
document_overview = hffs.read_text("datasets/daniellec927/digital-twin-document/overview.txt")
document_education = hffs.read_text("datasets/daniellec927/digital-twin-document/education.txt")
document_professional_experience = hffs.read_text("datasets/daniellec927/digital-twin-document/professional_experience.txt")

# ------------------------------------------------------
# Chunking Function
# ------------------------------------------------------
def chunk_text(text, max_chars=500, overlap=50):
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        hard_end = min(start + max_chars, n)
        if hard_end == n:
            end = n
        else:
            window = text[start:hard_end]
            halfway = len(window) // 2
            end = None
            idx = window.rfind("\n\n")
            if idx != -1 and idx >= halfway:
                end = start + idx + 2
            if end is None:
                idx = window.rfind("\n")
                if idx != -1 and idx >= halfway:
                    end = start + idx + 1
            if end is None:
                matches = list(re.finditer(r"[.!?](?=\s|$)", window))
                if matches and matches[-1].end() >= halfway:
                    end = start + matches[-1].end()
            if end is None:
                idx = window.rfind(" ")
                if idx != -1 and idx >= halfway:
                    end = start + idx + 1
            if end is None:
                end = hard_end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks

# ------------------------------------------------------
# RAG: Chunk, Embed & Store in ChromaDB
# ------------------------------------------------------
documents = [
    {"text": document_overview, "source": "Overview"},
    {"text": document_education, "source": "Education"},
    {"text": document_professional_experience, "source": "Professional Experience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    # Prepare the lists
    chunks_ = chunk_text(doc["text"], 300, 30)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source":doc["source"], "chunk_index": i} for i in range(len(chunks_))]

    # Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

# Print for logs
print(f"Created {len(chunks)} chunks: \n")

print(len(chunks))
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1} (ID: {ids[i]}, Source: {metadatas[i]["source"]}, Index: {metadatas[i]["chunk_index"]}, Length: {len(chunk)}):")
    print(chunk)
    print()

# Generate Embeddings for all chunks
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)
embeddings = [item.embedding for item in response.data]

# Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path = "./chroma_db_twin")
# Alternative: initialize ChromaDB client (in-memory storage)
# chroma_client = chromadb.Client()

# Get or Create + Empty the collection before adding new data (for testing purposes)
collection = chroma_client.get_or_create_collection(name = "digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

# Adding data to chromadb
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())

# ------------------------------------------------------
# Tools
# ------------------------------------------------------
tools = []

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "http://api.pushover.net/1/messages.json"

# Create send notification function
def send_notification(message: str):
    if pushover_user is None or pushover_token is None: # Handling of missing credentials
        return "Notification failed. Pushover not configured."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(url=pushover_url, data=payload)
    return f"Notification sent: {message}"

# Describe Pushover as an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real Danielle. Use this when: \
        1) Someone wants to get in touch, hire, or collaborate\
         - ALWAYS ask for their name and contact details first, then send notification to Danielle with the name and contact details\
            2) You don't know the answer to a question about Danielle - send AUTOMATICALLY without asking, include the question so she can add this info later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The notification message to send to the user's device"}
        },
        "required": ["message"]
    }
}

# Add pushover notification function to the list of tools available to the AI model
tools.append({"type": "function", "function": send_notification_function})

# Simulate rolling a single six-sided die
def dice_roll():
    result = random.randint(1, 6)
    return result

# Describe function for the LLM
roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, decision-making, or random number generation.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Add function to list of tools of LLM
tools.append({"type": "function", "function": roll_dice_function})

# ------------------------------------------------------
# Tool Handler
# ------------------------------------------------------
def handle_tool_call(tool_calls):
    tool_call_results = []
    content = ""

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # print(f"Calling function: {function_name}") # for future debugging

        # Route to the appropriate function based on the function name
        if function_name == "send_notification":
            content = send_notification(args["message"])
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
        # elif function_name == "insert_function_name_3":
        #     content = insert_function_name_3(args["message"])
        else:
            content = f"Unknown function: {function_name}"

        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id,
        }
        tool_call_results.append(tool_call_result)

    return tool_call_results

# ------------------------------------------------------
# System Message
# ------------------------------------------------------
system_message = """
You are a digital twin of Danielle Choi. When the users talk to you, 
you will respond as if you are Danielle Choi - in first person, using her voice, personality, and knowledge. 

Important: do not make things up. If you don't know an answer, say you don't know.
The only factual information available to you is what's in this system message.
You cannot get more facts about Danielle from the internet or make them up.

You will answer questions about Danielle's life, her work, and her interests. 
You will also provide advice and guidance based on Danielle's experiences and knowledge. 
You will be helpful, friendly, and engaging in your responses.

IMPORTANT: Whenever you don't know something about Danielle,
ALWAYS use the send_notification tool to alert the real Danielle - do this automatically without asking the user.
"""

# ------------------------------------------------------
# Main Response Function
# ------------------------------------------------------
def respond_ai(message, history):
    # RAG 
    #Embed the query using the same model we used for the chunks to ensure compatibility
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message]
    )
    query_embedding = response.data[0].embedding

    #Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    # Stitch retrieved chunks together to create the context for the response
    context = "\n---\n".join(results["documents"][0])

    # Print logs for debugging
    print("\n======================\n")
    print(f"User message: \n{message}\n")
    print("***Retrieved Chunks: ")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}:\n{a}\n")

    # Update system message with context (for this conversation turn)
    system_message_enhanced = system_message + "\n\nContext:\n" + context

    # Build messages for this turn
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]

    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message = response.choices[0].message

    # check if model wants to call the tool
    while message.tool_calls:
        pprint(message.tool_calls)
        tool_result = handle_tool_call(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
            # tool_choice="auto"
        )
        message = response.choices[0].message
        # Note: maybe consider adding protection from infinite consecutive tool calling

    return message.content

# ------------------------------------------------------
# Launch Gradio
# ------------------------------------------------------
gr.ChatInterface(
    fn=respond_ai,
    title="Danielle's Digital Twin",
    chatbot=gr.Chatbot(avatar_images=(None, "danielle.jpg")),
    description="Chat with an AI version of Danielle Choi. Ask about her experience, projects, or just say hi!",
    examples=["What's your background?","AI Experience","What is your education background?"]
    ).launch()
