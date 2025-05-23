import os
import nltk
import requests
import streamlit as st
import regex as re
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- NLTK Setup ---
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
for package in ['punkt', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

# --- Patch unstructured tokenizer ---
import unstructured.nlp.tokenize as tokenize
tokenize.download_nltk_packages = lambda: print("Skipping NLTK download")

# --- Load Environment Variables ---
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="🧠 Subscription ChatBot", page_icon="📂")
st.title("📊 Talk to Your Subscription System")

# Initialize session state
for key, default in {
    "vectorstore": None,
    "conversation_chain": None,
    "chat_history": [],
    "pending_action": None,
    "pending_payload": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Fetch API data
def fetch_api_data():
    try:
        res = requests.get("http://localhost:8080/getAll")
        res.raise_for_status()
        data = res.json()
        documents = []
        for i, d in enumerate(data):
            content = "\n".join([f"{k}: {v}" for k, v in d.items() if v is not None])
            if content.strip():
                documents.append(Document(page_content=content, metadata={}))
            else:
                print(f"⚠️ Skipping empty content at index {i}: {d}")

        return documents
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

# Create conversation chain
def create_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Call CRUD API
def call_crud_api(method, endpoint, payload=None,params=None):
    url = f"http://localhost:8080/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    try:
        print("url",url)
        res = requests.request(
            method, url,
            json=payload if method in ["POST", "PUT"] else None,
            headers=headers
        )
        if res.ok:
            content_type = res.headers.get("Content-Type", "")

            if "application/json" in content_type:
                return res.json()

            elif res.text.strip():  # only try to parse JSON if not empty
                return res.text
            else:
                return "What's Next?"
        else:
            return f"Failed: {res.status_code} - {res.text}"
    except Exception as e:
        return f"Error: {e}"

def is_valid_plan_name(name):
    name = name.strip()
    # Only letters and single spaces between words
    return bool(re.match(r"^[A-Za-z]+(?: [A-Za-z]+)*$", name)) and 3 <= len(name) <= 50

# Dummy intent parser (replace with NLP if needed)
def parse_user_intent(text):
    text = text.lower()

    # Intent keywords
    if any(word in text for word in ["create", "add", "make", "new"]):
        intent = "create"
    elif any(word in text for word in ["update", "edit", "modify"]):
        intent = "update"
    elif any(word in text for word in ["delete", "remove", "erase"]):
        intent = "delete"
    elif any(word in text for word in ["cancel", "terminate"]):
        intent = "cancel"
    elif any(word in text for word in ["get", "show", "view"]):
        intent = "get"
    else:
        intent = None

    # Plan name (e.g., "create plan basic", "add a new plan called gold")
    # name_match = re.search(r"(named|called)?\s*subscription\s*subscribe\s*plan\s*(\w+)", text)
    name_match = re.search(r"(named|called)?\s*(subscribe|subscription|plan)\s*(\w+)", text)
    plan_name = name_match.group(3) if name_match else None
    # Plan ID for update/delete/get
    id_match = re.search(r"(subscribe|subscription|plan)\s*([a-zA-Z0-9]+)", text)
    plan_id = id_match.group(2) if id_match else None

    # Payment method detection
    payment_method = None
    for method in ["credit", "upi", "cash"]:
        if method in text:
            payment_method = method
            break

    return {
        "intent": intent,
        "id": plan_id,
        "planName": plan_name,
        "paymentMethod": payment_method
    }


# Set up vectorstore and chain on first load
if not st.session_state.vectorstore:
    st.session_state.vectorstore = setup_vectorstore()

if not st.session_state.conversation_chain:
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_input = st.chat_input("Ask like 'create plan', 'update plan 2', 'delete plan 3'...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Step-by-step collection
    if st.session_state.pending_action:
        action = st.session_state.pending_action
        payload = st.session_state.pending_payload

        if "planName" not in payload:
            if not is_valid_plan_name(user_input):
                msg = "❌ Invalid plan name. Please use 3-50 alphanumeric characters (letters, numbers, spaces, or dashes)."
            else:
                payload["planName"] = user_input
                st.session_state.pending_payload = payload
                msg = "💳 Please enter the payment method (e.g., credit, upi, cash)."

            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()  # ❗ Important to stop further parsing

        elif "paymentMethod" not in payload:
            payload["paymentMethod"] = user_input
            payload.update({"userId": "user1"})
            msg = call_crud_api("POST", "create", payload)

            st.session_state.pending_action = None
            st.session_state.pending_payload = {}

            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

    # Parse intent
    parsed = parse_user_intent(user_input)

    if parsed["intent"] == "create":
        if not parsed["planName"]:
            st.session_state.pending_action = "create"
            st.session_state.pending_payload = {}
            print("i am here calling plan lin 207")
            msg = "📝 Please enter the plan name."
        elif not parsed["paymentMethod"]:
            st.session_state.pending_action = "create"
            st.session_state.pending_payload = {"planName": parsed["planName"]}
            msg = "💳 Please enter the payment method (e.g., credit, upi, cash)."
        else:
            payload = {
                "userId": "user1",
                "planName": parsed["planName"],
                "startDate": "2025-04-30",
                "endDate": None,
                "active": True,
                "price": 0.0,
                "paymentMethod": parsed["paymentMethod"],
                "status": "active",
                "lastModifiedDate": "2025-04-30"
            }
            print("create api call 225")
            msg = call_crud_api("POST", "create", payload)

    elif parsed["intent"] == "update" and parsed["id"]:
        payload = {
            "planName": parsed["planName"],
            "paymentMethod": parsed["paymentMethod"]
        }
        clean_payload = {k: v for k, v in payload.items() if v is not None}
        msg = call_crud_api("PUT", f"change-plan/{parsed['id']}", clean_payload)

    elif parsed["intent"] == "get" and parsed["id"]:
        msg = call_crud_api("GET", f"get/{parsed['id'].strip()}")

    elif parsed["intent"] == "delete" and parsed["id"]:
        msg = call_crud_api("DELETE", f"delete/{parsed['id'].strip()}")

    elif parsed["intent"] == "cancel" and parsed["id"]:
        msg = call_crud_api("PUT", f"cancel/{parsed['id']}")

        # Handle Q&A on subscription data
    else:
        vectorstore = setup_vectorstore()
        if vectorstore:
            chain = create_chain(vectorstore)
            response = chain({"question": user_input})
            msg = response["answer"]
        else:
            msg = "⚠️ Unable to load data to answer your question."

    st.chat_message("assistant").markdown(msg)
    st.session_state.chat_history.append({"role": "assistant", "content": msg})