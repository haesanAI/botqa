#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import langchain 
import pyarrow as pyarrow
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
 
st.title("🦜🔗 Langchain Quickstart App")

st.write(langchain.__version__)
st.write(pd.__version__)
st.write(pyarrow.__version__)

with st.sidebar:
    openai_api_key= st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)


# Prompt 
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
#conversation({"question": "hi"})
# data loader
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import NotionDBLoader
from langchain.document_loaders import ConfluenceLoader

import logging
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    
    #loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    NOTION_TOKEN = "secret_wpKyZTkbCLz1KwwyEBTDSH9E2OBr8Hc3H7RhmjTznec"
    DATABASE_ID = "55ac61ce713b46708f6dec874a8ef8e1"
    loader = NotionDBLoader(
        integration_token=NOTION_TOKEN,
        database_id=DATABASE_ID,
        request_timeout_sec=60,  # optional, defaults to 10
    )
    data = loader.load()
    print(data)
except Exception as e:  # 'as e' 부분을 추가하여 예외 객체를 e 변수에 저장
    print("Error loading data")
    st.write(f"Exception occurred1: {str(e)}")
    logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록


# splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    
except Exception as e:
    print("Error splitting data")
    logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록

    
# vector db
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

## retriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from openai_embeddings import Embedder


# llm = ChatOpenAI()
def haesan_response(input_text):
    try:
        st.write("1")
        embedder = Embedder("korean")
        all_splits = embedder(all_splits)
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key,disallowed_special={"metadata"})
        st.write("4")
        try:
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
        except Exception as e:
            print("Error in Chroma.from_documents")
            logging.error(f"Exception occurred Chroma1: {str(e)}")
            st.write(f"Exception occurred1: {str(e)}")
            return  # 여기서 함수를 종료합니다.
        
        
        retriever = vectorstore.as_retriever(search_type="mmr")
        matched_docs = retriever.get_relevant_documents(input_text)
        for i, d in enumerate(matched_docs):
            print(f"\n## Document {i}\n")
            print(d.page_content)
        retriever = vectorstore.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        haesan = qa(input_text)
        st.info(haesan["answer"])
        
    except Exception as e:
        print("Error creating retriever")
        st.write(f"Exception occurred2: {str(e)}")
        logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록

def generate_response(input_text):
    llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form("my_form"):
    text = st.text_area("Enter text:", "김한준의 업무는?")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif submitted:
        haesan_response(text) 