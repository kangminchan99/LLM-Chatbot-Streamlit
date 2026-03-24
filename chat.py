import os

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langsmith import Client
from pinecone import Pinecone

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("소득세 챗봇 🤖")
st.caption("소득세 관련 질문에 답변해 드립니다. 질문을 입력해주세요.")
load_dotenv()


# 메시지 내용 저장
if 'message_list' not in st.session_state:
  # 메시지 리스트 초기화
  st.session_state.message_list = []

# 이전 메시지 출력
for message in st.session_state.message_list:
  with st.chat_message(message['role']):
    st.write(message['content'])

def get_ai_message(user_message):
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")
  index_name = 'tax-index'
  database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

  llm = ChatOpenAI(model="gpt-5.4")

  LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
  client = Client(api_key=LANGSMITH_API_KEY)
  prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
  retriever = database.as_retriever(search_kwargs={"k": 3})


  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=database.as_retriever(),
      chain_type_kwargs={"prompt": prompt}
  )

  dictionary = ["사람을 나타내는 표현 -> 거주자"]

  prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 그대로 반환해주세요.
    사전: {dictionary}

    질문: {{question}}
  """
  )

  dictionary_chain = prompt | llm | StrOutputParser()
  tax_chain = {"query": dictionary_chain} | qa_chain
  ai_message = tax_chain.invoke({"question": user_message})
  return ai_message['result']

# 사용자의 질문
if user_question := st.chat_input(placeholder='소득세에 대해 궁금한 점을 입력하세요.'):
  # 채팅 입력 시 화면에 나타나게 (사용자)
  with st.chat_message('user'):
    st.write(user_question)
  st.session_state.message_list.append({"role": "user", "content": user_question})

  with st.spinner('답변을 생성하는 중...'):
    # 챗봇의 답변
    ai_message = get_ai_message(user_question)
    with st.chat_message('ai'):
      st.write(ai_message)
    st.session_state.message_list.append({"role": "ai", "content": ai_message})