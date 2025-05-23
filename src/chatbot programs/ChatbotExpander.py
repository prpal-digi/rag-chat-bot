import os
import nltk
import requests
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Setup NLTK
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
for package in ['punkt', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

# Patch unstructured tokenizer
import unstructured.nlp.tokenize as tokenize
tokenize.download_nltk_packages = lambda: print("Skipping NLTK download")

# Load environment variables
load_dotenv()

# Fetch data from API
def fetch_api_data():
    try:
        res = requests.get("http://localhost:8080/getAll")
        res.raise_for_status()
        data = res.json()
        return [Document(page_content="\n".join([f"{k}: {v}" for k, v in d.items()])) for d in data]
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return []

# Vectorstore setup
def setup_vectorstore():
    docs = fetch_api_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# Chat chain setup
def create_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# API handler
def call_crud_api(method, endpoint, payload=None):
    url = f"http://localhost:8080/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    try:
        res = requests.request(method, url, json=payload if method in ["POST", "PUT"] else None, params=payload if method == "DELETE" else None, headers=headers)
        if res.ok:
            return f"Success: {res.json() if res.text else 'No content'}"
        else:
            return f"Failed: {res.status_code} - {res.text}"
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.set_page_config(page_title="🧠 Subscription ChatBot", page_icon="📂")
st.title("📊 Talk to Your Subscription System")

# Init session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about subscriptions, plans, or users...")

# Forms for CRUD
with st.expander("➕ Create Plan"):
    with st.form("create_form"):
        name = st.text_input("Plan Name")
        payment = st.text_input("Payment Method")
        submitted = st.form_submit_button("Create Plan")
        if submitted and name and payment:
            payload = {
                "userId": "user1",
                "planName": name,
                "startDate": "2025-04-30",
                "endDate": None,
                "active": True,
                "price": 0.0,
                "paymentMethod": payment,
                "status": "active",
                "lastModifiedDate": "2025-04-30"
            }
            msg = call_crud_api("POST", "create", payload)
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

with st.expander("✏️ Update Plan"):
    with st.form("update_form"):
        pid = st.text_input("Plan ID")
        new_name = st.text_input("New Plan Name")
        new_method = st.text_input("New Payment Method")
        submitted = st.form_submit_button("Update Plan")
        if submitted and pid:
            payload = {
                "id": int(pid),
                "planName": new_name or None,
                "paymentMethod": new_method or None,
                "lastModifiedDate": "2025-04-30"
            }
            clean_payload = {k: v for k, v in payload.items() if v is not None}
            msg = call_crud_api("PUT", f"change-plan?id={pid}", clean_payload)
            st.success(msg)

with st.expander("🗑️ Delete Plan"):
    pid_delete = st.text_input("Enter Plan ID to Delete")
    if st.button("Delete Plan") and pid_delete:
        msg = call_crud_api("DELETE", f"delete?id={pid_delete}")
        st.success(msg)

# Handle chat
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response = st.session_state.conversation_chain({"question": user_input})
    result = response["answer"]
    st.chat_message("assistant").markdown(result)
    st.session_state.chat_history.append({"role": "assistant", "content": result})