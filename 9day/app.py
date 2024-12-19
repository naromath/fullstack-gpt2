"""
    1. 이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
    2. 파일 업로드 및 채팅 기록을 구현합니다.
    3. 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
    4. st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
    
    
    a. 코드를 공개 Github 리포지토리에 푸시합니다.
    b. 단. OpenAI API 키를 Github 리포지토리에 푸시하지 않도록 주의하세요.
    c. 여기에서 계정을 개설하세요: https://share.streamlit.io/
    d. 다음 단계를 따르세요: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app#deploy-your-app-1
    e. 앱의 구조가 아래와 같은지 확인하고 배포 양식의 Main file path 에 app.py를 작성하세요.
    
"""

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st
import time

class ChatCallbackHandler(BaseCallbackHandler):
    
    message=""
       
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
                       
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
            
                
    
st.set_page_config(
    page_icon="🖥️",
    page_title="RAG PIPE"
)

st.title("RAG PIPE")


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cashe_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 600,
        chunk_overlap = 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cashed_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cashe_dir)
    vectorstore = FAISS.from_documents(docs, cashed_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
        
def save_message(message, role):
    st.session_state["message"].append({"message": message, "role" :role})
    

def paint_history():
    for message in st.session_state["message"]:
        send_message(message["message"], message["role"], save=False)
        
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
        

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Context: {context}",),
        ("human","{question}"),
    ]
)



st.markdown(
    """
    안녕하세요! 당신의 파일을 분석해주는 문서GPT 입니다.
            
    당신이 분석하고 싶은 문서를 왼쪽에 첨부해 주세요."""
    )

with st.sidebar:
    file = st.file_uploader(
        "Upload a. txt .pdf or .docx file",
        type=["pdf","txt","docx"])



if file:
    retriever = embed_file(file)
    
    

    send_message("당신의 OPENAI API KEY를 왼쪽 창에 입력해 주세요", "ai", save=False)    
      
    
    key = st.sidebar.text_input("OPENAI API KEY")
    
    if not key:
        send_message("아직 API Key가 입력되지 않았습니다. 키를 입력해주세요.","ai", save=False)
        st.stop()
    else:
        send_message("준비 됬어요! 궁금한 점을 물어보세요", "ai", save=False)             
        
    llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler(),],
        openai_api_key=key
    )
    
    paint_history()
    
    message= st.chat_input("Ask anything about your file...")
    
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
        
else:
    st.session_state["message"]=[]
    
    
        
        
        
        
        
