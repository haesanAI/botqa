from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.title("🦜🔗 Langchain Quickstart App")

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI()

# Prompt 
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human. and always using korean"
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
import traceback

import logging
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    #loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    DATABASE_ID = os.getenv("DATABASE_ID")
    loader = NotionDBLoader(
        integration_token=NOTION_TOKEN,
        database_id=DATABASE_ID,
        request_timeout_sec=30,  # optional, defaults to 10
    )

    data = loader.load()
    print(data)

except Exception as e:  # 'as e' 부분을 추가하여 예외 객체를 e 변수에 저장
    print(f"Error loading data {str(e)}")
    logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록
    traceback.print_exc()  # 에러 내용을 자세히 출력합니다.
    exit(1)  # 프로그램을 종료합니다.


# splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
except Exception as e:
    print("Error splitting data")
    logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록

######
def filter_complex_metadata(metadata):
  """Removes complex metadata from a dictionary.

  Args:
    metadata: The metadata dictionary.

  Returns:
    A new dictionary without any complex metadata.
  """

  filtered_metadata = {}
  for key, value in metadata.items():
    if not isinstance(value, (list, tuple, dict)):
      filtered_metadata[key] = value
    else:
      filtered_metadata[key] = value[0]

  return filtered_metadata
######

# all_splits이 어떤 형태인지에 따라 필터링 방법이 달라질 수 있습니다.
#metadata = dict(all_splits)

filtered_splits = []

for doc in all_splits:
  filtered_splits.append({
      'page_content': doc.page_content,
      'id': doc.metadata['id']
  })


   
# vector db
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class Document:
  def __init__(self, page_content):
    self.page_content = page_content
    self.metadata = {}

def doc_to_string(doc):
  if isinstance(doc, Document):
    return doc.page_content
  else:
    return str(doc)

documents = []

for doc in filtered_splits:
  documents.append(Document(page_content=doc['page_content']))


vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
#vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
# retriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

import pinecone

try:
    # initialize pinecone
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    PINECONE_ENV=os.getenv("PINECONE_ENV")
    pinecone.init(
        api_key=(PINECONE_API_KEY),  # find at app.pinecone.io
        environment=(PINECONE_ENV),  # next to api key in console
    )
    
    active_indexes = pinecone.list_indexes()
    whoami = pinecone.whoami()
    index_name = "damda"
    index = pinecone.Index("damda")
    describe = index.describe_index_stats()
    '''
    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  
    )
    '''
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    embeddings=OpenAIEmbeddings()
    print(index)
    print("3")
    try:
        # docsearch = Pinecone.from_documents(all_splits, embeddings,index_name=index_name)
        docsearch = Pinecone(index, embeddings, all_splits)
    except Exception as e:
        print(f"Error Pinecone from_documents: {str(e)}")
        logging.error(f"Pinecone from_documents: {str(e)}")  # 에러 메시지를 로그 파일에 기록
        exit
    print("4")
    # if you already have an index, you can load it like this
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    #query = "What did the president say about Ketanji Brown Jackson"
    #docs = docsearch.similarity_search(query)
    try:
        print("0")
        retriever = vectorstore.as_retriever() #백터디비 변경위치
        print("1")
        '''
        matched_docs = retriever.get_relevant_documents(query)
        for i, d in enumerate(matched_docs):
            print(f"\n## Document {i}\n")
            print(d.page_content)
        ### retriever = vectorstore.as_retriever()
        '''
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever,memory=memory)
        print("2")
        result = qa("김한준의 담당하는 업무는?")
        print(result["answer"])
        st.write(f"답변: {result["answer"]}")
        
        print("3") 
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error creating retriever: {str(e)}")

        logging.error(f"Exception occurred: {str(e)}")  # 에러 메시지를 로그 파일에 기록
except Exception as e:
    traceback.print_exc()
    print(f"Error creating vectorstore: {str(e)}")
    print(whoami)
    print(active_indexes)
    print(describe)
    logging.error(f"Exception occurred: {str(e)}")
    





