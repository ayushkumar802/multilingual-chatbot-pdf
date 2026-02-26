import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import tempfile
import string
from huggingface_hub import login
from semantic_router.encoders import HFEndpointEncoder
from semantic_chunkers import StatisticalChunker
import os

def clean_str(text: str) -> str:
    # Allowed punctuation
    allowed_punctuations = ".,?()[]{}*%"
    allowed = string.ascii_letters + string.digits + string.whitespace + allowed_punctuations
    table = str.maketrans("", "", "".join(set(map(chr, range(0x110000))) - set(allowed)))
    cleaned = text.translate(table)
    # Collapse multiple spaces
    return " ".join(cleaned.split())

st.set_page_config(
    page_title="ChatPDF",    # change name as needed
    page_icon="🤖",
    layout="wide",           # <- this is where you make it wide
    initial_sidebar_state="expanded"
)


# --- Streamlit UI ---
st.sidebar.title("Document Q&A Chatbot")
st.sidebar.write("Upload a PDF or TXT file, and ask questions about it!")

key_token = st.sidebar.text_input("Hugging Face Token!",placeholder='enter your token')
if key_token:
    try:
        st.session_state['login'] = key_token
        login(st.session_state['login'])
        st.sidebar.success("Logged in successfully!")
    except ValueError:
        st.sidebar.error("Invalid Input: Token is incorrect or expired.")

HF_TOKEN = st.session_state.get('login', None)

uploaded_file = st.sidebar.file_uploader("Upload your file", type=["txt", "pdf"],key="upload_file")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-m3",
                task="feature-extraction",
                huggingfacehub_api_token=HF_TOKEN
    )



# --- 1. Document Processing Logic (Cached) ---
@st.cache_data
def process_uploaded_file(uploaded_file):
    # This function only runs ONCE per unique file uploaded
    if uploaded_file.type == "text/plain":
        string_data = uploaded_file.getvalue().decode("utf-8")
        raw_docs = [Document(page_content=string_data, metadata={"source": uploaded_file.name})]
    else: # PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
        os.remove(tmp_path)

    # Cleaning and Chunking
    full_text = " ".join([d.page_content for d in raw_docs])
    cleaned_text = clean_str(full_text)
    
    # Statistical Chunking (Using your existing compatible_encoder)
    compatible_encoder = HFEndpointEncoder(name="BAAI/bge-m3",
                huggingface_url="https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction",
                huggingface_api_key=HF_TOKEN
            )
    chunker = StatisticalChunker(min_split_tokens=250, max_split_tokens=350, encoder=compatible_encoder)
    chunks = chunker(docs=[cleaned_text])
    final_docs = [Document(page_content="".join(c.splits)) for c in chunks[0]]
    return final_docs

# --- 2. Vector Store Initialization (Cached Resource) ---
@st.cache_resource
def get_retriever(_docs, _embedding_func):
    # This only runs when _docs changes
    vectorstore = Chroma.from_documents(
        documents=_docs,
        embedding=_embedding_func,
        persist_directory="my_chroma_db",
        collection_name="bge_m3_data"
    )
    return vectorstore.as_retriever()

# --- 3. Main App Flow ---
if uploaded_file:
    # This block now runs instantly on reruns because of @st.cache
    docs = process_uploaded_file(uploaded_file)
    
    
    st.sidebar.success("System Ready: Documents Indexed")


    embedding_func = load_embeddings()
    retriever = get_retriever(docs, embedding_func)

    def text_extractor(docs):
        return " ".join(doc.page_content for doc in docs)




    chat_objects = []

    prompt = ChatPromptTemplate([
        ("system","You're a AI assistent that answers user's question from given context, you can create your language to make user undertsand his doubt. If the question is out of question, just say question is out of Topic, and also you can resposed to user's general messages like Hi, how are you CONTEXT:- {context}"),
        *chat_objects,
        ("human","{question}") 
    ])


    llm = HuggingFaceEndpoint(
        repo_id='Qwen/Qwen3-Next-80B-A3B-Instruct',
        task='text-generation'
    )


    model = ChatHuggingFace(llm=llm)

    parcer = StrOutputParser()



st.title("Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_objects" not in st.session_state:
    st.session_state["chat_objects"] = []



# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (only once, no while True)
if prompt_ := st.chat_input("Type your question...", key="user_chat_main"):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt_})
    st.session_state["chat_objects"].append(HumanMessage(content=prompt_))
    with st.chat_message("user"):
        st.markdown(prompt_)

    # Your chain logic here
    parallel_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])) | RunnableLambda(text_extractor),
        "question": RunnablePassthrough(),
        "chats": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parcer
    result = final_chain.invoke({"chats": st.session_state["chat_objects"], "question": prompt_})

    st.session_state["chat_objects"].append(AIMessage(content=result))

    # AI response
    response = f"{result}"
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Show latest AI response immediately
    with st.chat_message("assistant"):

        st.markdown(response)
