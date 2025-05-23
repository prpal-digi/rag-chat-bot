import streamlit as st
import requests
import tempfile
import os
import pickle

import faiss
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ----------- Config ------------------
load_dotenv()
API_BASE_URL = "http://localhost:8000/api"  # Your backend API base URL

VECTORSTORE_PATH = "faiss_index.pkl"
EMBEDDING_DIM = 1536  # Google embeddings default dimension

embedding_model =  HuggingFaceEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ----------- Load or create vectorstore ----------

def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        with open(VECTORSTORE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # FAISS index with dimension matching Google embeddings
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        return FAISS(embedding_model.embed_query, index, {}, {})

vectorstore = load_vectorstore()

def save_vectorstore():
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

# ----------- Helper functions -------------

def add_subscription_docs_to_vectorstore():
    try:
        res = requests.get(f"{API_BASE_URL}/subscriptions")
        res.raise_for_status()
        subscriptions = res.json()
    except Exception as e:
        st.error(f"Error fetching subscriptions from API: {e}")
        return

    docs = []
    for sub in subscriptions:
        text = f"User ID: {sub['user_id']}, Plan ID: {sub['plan_id']}, Status: {sub.get('status', 'active')}"
        docs.append({"page_content": text, "metadata": {"type": "subscription", "user_id": sub['user_id'], "plan_id": sub['plan_id']}})

    if docs:
        vectorstore.add_documents(docs)
        save_vectorstore()

def add_pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    if docs:
        vectorstore.add_documents(docs)
        save_vectorstore()

# ----------- CRUD API calls --------------

def create_subscription_api(user_id, plan_id):
    try:
        res = requests.post(f"{API_BASE_URL}/subscriptions", json={"user_id": user_id, "plan_id": plan_id})
        if res.status_code == 201:
            return True, "Subscription created."
        else:
            return False, res.text
    except Exception as e:
        return False, str(e)

def get_subscription_api(user_id, plan_id):
    try:
        res = requests.get(f"{API_BASE_URL}/subscriptions/{user_id}/{plan_id}")
        if res.status_code == 200:
            return res.json()
        else:
            return None
    except Exception as e:
        return None

def update_subscription_api(user_id, plan_id, new_plan_id):
    try:
        res = requests.put(f"{API_BASE_URL}/subscriptions/{user_id}/{plan_id}", json={"plan_id": new_plan_id})
        if res.status_code == 200:
            return True, "Subscription updated."
        else:
            return False, res.text
    except Exception as e:
        return False, str(e)

def cancel_subscription_api(user_id, plan_id):
    try:
        res = requests.put(f"{API_BASE_URL}/subscriptions/{user_id}/{plan_id}/cancel")
        if res.status_code == 200:
            return True, "Subscription cancelled."
        else:
            return False, res.text
    except Exception as e:
        return False, str(e)

def get_all_subscriptions_api():
    try:
        res = requests.get(f"{API_BASE_URL}/subscriptions")
        if res.status_code == 200:
            return res.json()
        else:
            return []
    except Exception as e:
        return []

# ----------- Conversational logic -------------

def extract_ids(text):
    import re
    user_id = None
    plan_id = None
    user_match = re.search(r"user\s*id\s*[:=]?\s*(\w+)", text, re.I)
    plan_match = re.search(r"plan\s*id\s*[:=]?\s*(\w+)", text, re.I)
    if user_match:
        user_id = user_match.group(1)
    if plan_match:
        plan_id = plan_match.group(1)
    return user_id, plan_id

def handle_user_input(user_input, state):
    if state.get("awaiting") == "user_id":
        state["user_id"] = user_input.strip()
        state["awaiting"] = None
        return "Please enter plan ID:", state

    if state.get("awaiting") == "plan_id":
        state["plan_id"] = user_input.strip()
        state["awaiting"] = None
        return "Processing your request...", state

    text = user_input.lower()
    user_id, plan_id = extract_ids(user_input)

    if "create" in text and "subscription" in text:
        if not user_id:
            state["awaiting"] = "user_id"
            return "Please enter your user ID to create subscription:", state
        if not plan_id:
            state["awaiting"] = "plan_id"
            state["user_id"] = user_id
            return "Please enter plan ID to create subscription:", state
        success, msg = create_subscription_api(user_id, plan_id)
        if success:
            add_subscription_docs_to_vectorstore()
        return msg, state

    if "get" in text or "show" in text:
        if not user_id:
            state["awaiting"] = "user_id"
            return "Please enter your user ID to get subscription:", state
        if not plan_id:
            state["awaiting"] = "plan_id"
            state["user_id"] = user_id
            return "Please enter plan ID to get subscription:", state
        sub = get_subscription_api(user_id, plan_id)
        if sub:
            return f"Subscription details: User ID: {sub['user_id']}, Plan ID: {sub['plan_id']}, Status: {sub.get('status', 'active')}", state
        else:
            return "Subscription not found.", state

    if "update" in text:
        if not user_id:
            state["awaiting"] = "user_id"
            return "Please enter your user ID to update subscription:", state
        if not plan_id:
            state["awaiting"] = "plan_id"
            state["user_id"] = user_id
            return "Please enter current plan ID to update subscription:", state
        if "new_plan_id" not in state:
            state["awaiting"] = "new_plan_id"
            return "Please enter the new plan ID to update to:", state
        new_plan_id = user_input.strip()
        success, msg = update_subscription_api(user_id, plan_id, new_plan_id)
        state.pop("new_plan_id", None)
        if success:
            add_subscription_docs_to_vectorstore()
        return msg, state

    if "cancel" in text:
        if not user_id:
            state["awaiting"] = "user_id"
            return "Please enter your user ID to cancel subscription:", state
        if not plan_id:
            state["awaiting"] = "plan_id"
            state["user_id"] = user_id
            return "Please enter plan ID to cancel subscription:", state
        success, msg = cancel_subscription_api(user_id, plan_id)
        if success:
            add_subscription_docs_to_vectorstore()
        return msg, state

    if "all plans" in text or "list plans" in text:
        subs = get_all_subscriptions_api()
        if not subs:
            return "No subscriptions found.", state
        res = "\n".join([f"User: {s['user_id']} - Plan: {s['plan_id']} - Status: {s.get('status', 'active')}" for s in subs])
        return res, state

    # Fallback to vectorstore QA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=False)
    answer = qa_chain.run(user_input)
    return answer, state

# ----------- Streamlit UI ----------------

st.title("Conversational Subscription & PDF Chatbot (API-based)")

if "state" not in st.session_state:
    st.session_state.state = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_pdf = st.file_uploader("Upload PDF to ingest", type=["pdf"])
if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name
    add_pdf_to_vectorstore(pdf_path)
    st.success(f"PDF '{uploaded_pdf.name}' ingested.")

if st.button("Refresh Subscription Data in Vectorstore"):
    add_subscription_docs_to_vectorstore()
    st.success("Subscription data loaded into vectorstore.")

user_question = st.text_input("Ask me anything about subscriptions or PDFs:")

if user_question:
    response, st.session_state.state = handle_user_input(user_question, st.session_state.state)
    st.session_state.chat_history.append((user_question, response))

for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**User:** {q}")
    st.markdown(f"**Bot:** {a}")
