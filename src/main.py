import os
import nltk


# Set up NLTK data path to avoid redownloading and 403 errors
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download necessary packages if not already present
for package in ['punkt', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'taggers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

# Patch unstructured to avoid online download attempts
import unstructured.nlp.tokenize as tokenize
def noop_download():
    print("Skipping NLTK download; resources are already available.")
tokenize.download_nltk_packages = noop_download


from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import  CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from numpy.f2py.crackfortran import verbose

#load the env variable
load_dotenv()
print("Groq API Key:", os.getenv("GROQ_API_KEY"))
working_dir = os.path.dirname(os.path.abspath(__file__))

# reading the document
def load_doc(file_path):
    print("loading", file_path)
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    print("done loadind doc")
    return documents

# vector embedding db , for similarity search
def setup_vectorstore(documents):
    print("setting up vectorstore")
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks,embeddings)
    print("done setting up vectorstore")
    return vectorstore

# llm has context of pervious conversation
def create_chain(vectorstore):
    print("creating chain")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose = True
    )
    print("done creating chain")
    return chain

st.set_page_config(
    page_title="Doc Chat Bot",
    page_icon="🗎",
    layout="centered",
)

st.title("🦙Chat with LLama3.0")

# initialize the chat history in streamlit session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload PDF",type=["pdf"])

# here file uploaded, read and stored
if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    print("file path",file_path)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_doc(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])          # for printing the history

user_input = st.chat_input("Ask Llama")

# here is displaying the icons and triggering the lamma for analysis of doc and answer
if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question":user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role":"assistant","content":assistant_response})
