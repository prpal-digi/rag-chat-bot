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
    if any(phrase in user_input.lower() for phrase in["abort", "nevermind", "stop", "exit"]):
        return {"intent": "abort"}

        # Cancel a subscription
    if "cancel" in text:
        if any(word in text for word in ["status", "get", "show", "list", "fetch", "cancelled"]):
            return {"intent": None}
        elif "subscription" in text or "plan" in text:
            return {"intent": "cancel"}
        else:
            # Fall back to assume cancel flow if ambiguous
            return {"intent": "cancel"}

    if any(phrase in user_input.lower() for phrase in ["create", "subscribe", "available", "what plans", "list plans"]):
        intent = "create"
    elif any(word in text for word in ["update", "edit", "modify", "upgrade", "change"]):
        intent = "update"
    elif any(word in text for word in ["delete", "remove", "erase"]):
        intent = "delete"
    elif any(word in text for word in ["cancel", "terminate","end"]):
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

    # Payment method detection
    payment_method = None
    for method in ["credit", "upi", "cash"]:
        if method in text:
            payment_method = method
            break

    return {
        "intent": intent,
        "planName": plan_name,
        "paymentMethod": payment_method
    }

class SubscriptionData:
    available_plans = [
        {"name": "Basic", "price": 99, "duration": "1 Month"},
        {"name": "Pro", "price": 249, "duration": "3 Months"},
        {"name": "Premium", "price": 899, "duration": "1 Year"}
    ]

def extract_plan_name(user_input, available_plans):
    user_input_lower = user_input.lower()
    for plan in available_plans:
        if plan["name"].lower() in user_input_lower:
            return plan["name"]  # return string like "Pro"
    return None

if "active_intent" not in st.session_state:
    st.session_state.active_intent = None

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
    parsed = parse_user_intent(user_input)
    print("intent",parsed["intent"])

    # Abort handling
    if parsed["intent"] == "abort":
        if st.session_state.get("active_intent") in [None, "", "abort"]:
            msg = "⚠️ There's no active operation to cancel."
        else:
            msg = f"❌ Operation **'{st.session_state.active_intent}'** cancelled."
            st.session_state.active_intent = None
            st.session_state.pending_action = None
            st.session_state.pending_payload = {}

        st.chat_message("assistant").markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.stop()

    # Set new intent
    st.session_state.active_intent = parse_user_intent(user_input)

    # Handle general subscription query
    if parsed["intent"] == "create" and not st.session_state.get("pending_payload"):
        msg = "📦 Available Subscription Plans:\n"
        for plan in SubscriptionData.available_plans:
            msg += f"- **{plan['name']}**: ₹{plan['price']} ({plan['duration']})\n"
        msg += "\n👉 Please type the **name** of the plan you want to subscribe to."

        st.chat_message("assistant").markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.session_state.active_intent = "create"
        st.session_state.pending_action = "create"
        st.session_state.pending_payload = {}
        st.stop()

    # Handle plan selection
    if parsed["intent"] == "create" or (st.session_state.pending_action == "create"):
        payload = st.session_state.get("pending_payload", {})

        if "planName" not in payload:
            plan_name = extract_plan_name(user_input, SubscriptionData.available_plans)
            if plan_name:
                st.session_state.pending_payload["planName"] = plan_name
                msg = f"💳 You selected **{plan_name}**. Please enter your new payment method (credit, upi, or cash)."
            else:
                msg = "❌ Invalid plan selected. Please choose from the available options."
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

        elif "paymentMethod" not in payload:
            payment_method = user_input.strip().lower()
            if payment_method not in ["credit", "upi", "cash"]:
                msg = "❌ Invalid payment method. Please enter one of: credit, upi, or cash."
            else:
                payload["paymentMethod"] = payment_method
                msg = call_crud_api("POST", "create", payload)
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.session_state.pending_action = None
            st.session_state.pending_payload = {}
            st.stop()

    # Conversational UPDATE flow
    elif parsed["intent"] == "update" or (st.session_state.pending_action == "update"):
        # Begin update flow if just triggered
        if st.session_state.pending_action != "update":
            st.session_state.pending_action = "update"
            st.session_state.pending_payload = {}
            msg = "🔧 Sure! Please enter the subscription ID you want to update."
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

        # Step 1: Get subscription ID
        if "id" not in st.session_state.pending_payload:
            # Try to extract subscription ID from user input
            match = re.search(r"\b[a-fA-F0-9]{24}\b", user_input.strip())

            if match:
                st.session_state.pending_payload["id"] = match.group(0)
                print("matchhh",match.group(0))
                msg = "📦 Here are the available plans:\n"
                for plan in SubscriptionData.available_plans:
                    msg += f"- **{plan['name']}**: ₹{plan['price']} ({plan['duration']})\n"
                msg += "\n👉 Please type the name of the plan you'd like to switch to."
            else:
                msg = "🔧 Sure! Please enter the subscription ID you want to update."
            st.chat_message("assistant").markdown(msg)
            print("message", msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

        # Step 2: Get new plan name
        elif "planName" not in st.session_state.pending_payload:
            selected_plan = next((p for p in SubscriptionData.available_plans if p["name"].lower() == user_input.lower()), None)
            if selected_plan:
                st.session_state.pending_payload["planName"] = selected_plan["name"]
                msg = f"💳 You selected **{selected_plan['name']}**. Please enter your new payment method (credit, upi, or cash)."
            else:
                msg = "❌ Invalid plan selected. Please choose from the available options."
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

        # Step 3: Get payment method and make API call
        elif "paymentMethod" not in st.session_state.pending_payload:
            payment_method = user_input.strip().lower()
            if payment_method not in ["credit", "upi", "cash"]:
                msg = "❌ Invalid payment method. Please enter one of: credit, upi, or cash."
            else:
                st.session_state.pending_payload["paymentMethod"] = payment_method
                sub_id = st.session_state.pending_payload.pop("id")
                payload = st.session_state.pending_payload
                msg = call_crud_api("PUT", f"change-plan/{sub_id}", payload)
                st.session_state.pending_action = None
                st.session_state.pending_payload = {}
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()


    elif parsed["intent"] == "get" or (st.session_state.pending_action == "get"):

        # Step 1: Get subscription ID if not already present
        if "id" not in st.session_state.pending_payload:
            # Try to extract subscription ID from user input
            match = re.search(r"\b[a-fA-F0-9]{24}\b", user_input)
            if match:
                st.session_state.pending_payload["id"] = match.group(0)
            else:
                msg = "🔧 Sure! Please enter the subscription ID you want to fetch."
                st.chat_message("assistant").markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.stop()
        # Step 2: Make GET request using the ID
        sub_id = st.session_state.pending_payload.pop("id")
        msg = call_crud_api("GET", f"get/{sub_id}")
        # Step 3: Respond and reset session state
        st.session_state.pending_action = None
        st.session_state.pending_payload = {}
        st.chat_message("assistant").markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.stop()

    elif parsed["intent"] == "delete" or (st.session_state.pending_action == "delete"):

        if st.session_state.pending_action != "delete":
            st.session_state.pending_action = "delete"
            st.session_state.pending_payload = {}
            msg = "🔧 Sure! Please enter the subscription ID you want to delete."
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()

        if "id" not in st.session_state.pending_payload:
            match = re.search(r"\b[a-fA-F0-9]{24}\b", user_input)
            if match:
                st.session_state.pending_payload["id"] = match.group(0)
            else:
                msg = "🔧 Sure! Please enter the subscription ID you want to delete."
                st.chat_message("assistant").markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.stop()
        sub_id = st.session_state.pending_payload.pop("id")
        msg = call_crud_api("DELETE", f"delete/{sub_id}")
        print("message from delete"+msg)
        st.session_state.pending_action = None
        st.session_state.pending_payload = {}
        st.chat_message("assistant").markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.stop()

    elif parsed["intent"] == "cancel" or (st.session_state.pending_action == "cancel"):

        if st.session_state.pending_action != "cancel":
            st.session_state.pending_action = "cancel"
            st.session_state.pending_payload = {}
            msg = "🔧 Sure! Please enter the subscription ID you want to cancel."
            st.chat_message("assistant").markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.stop()
        if "id" not in st.session_state.pending_payload:
            match = re.search(r"\b[a-fA-F0-9]{24}\b", user_input)
            if match:
                st.session_state.pending_payload["id"] = match.group(0)
            else:
                msg = "🔧 Sure! Please enter the subscription ID you want to cancel."
                st.chat_message("assistant").markdown(msg)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.stop()
        sub_id = st.session_state.pending_payload.pop("id")
        msg = call_crud_api("PUT", f"cancel/{sub_id}")
        st.session_state.pending_action = None
        st.session_state.pending_payload = {}
        st.chat_message("assistant").markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.stop()

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

