from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters.character import  CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize session state
for key, default in {

    "create_chainforpdf":None,
    "chat_history": [],
    "pending_action": None,
    "pending_payload": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# reading the document
def load_doc(file_path):
    print("loading", file_path)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("done loadind doc")
    return documents

# vector embedding db , for similarity search
def setup_vectorstoreforpdf(documents):
    print("setting up vectorstore")
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        task_type="retrieval_query"
    )
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    vectorstore.save_local("faiss_gemini_store_pdf")
    print("done setting up vectorstore")
    return vectorstore

def load_existing_vectorstore():
    if os.path.exists("faiss_gemini_store_pdf/index.faiss"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/text-embedding-004',
            task_type="retrieval_query"
        )
        return FAISS.load_local("faiss_gemini_store_pdf", embeddings, allow_dangerous_deserialization=True)
    return None

# llm has context of pervious conversation
def create_chainforpdf(vectorstore):
    print("creating chain")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",convert_system_message_to_human=True)
    print("done creating chain")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

st.set_page_config(
    page_title="Doc Chat Bot",
    page_icon="🗎",
    layout="centered",
)

# Try to load existing store
if st.session_state.create_chainforpdf is None:
    print("here inside createchainforpdf")
    existing_store = load_existing_vectorstore()
    if existing_store:
        print("already loaded")
        st.session_state.create_chainforpdf = create_chainforpdf(existing_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with Gemini")

uploaded_files = st.file_uploader(label="Upload PDF",type=["pdf"],accept_multiple_files=True)
if uploaded_files:
    all_documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        print("Processing:", file_path)
        docs = load_doc(file_path)
        all_documents.extend(docs)

        st.session_state.vectorstore = setup_vectorstoreforpdf(load_doc(file_path))
        st.session_state.conversation_chain = create_chainforpdf(st.session_state.vectorstore)


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Gemini")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        response = st.session_state.create_chainforpdf({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})