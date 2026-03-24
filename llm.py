## ========== 라이브러리 임포트 ==========

# LangChain 체인 관련: 대화 히스토리를 인식하는 리트리버 생성, 검색 기반 체인 생성
from langchain_classic.chains import (create_history_aware_retriever,
                                      create_retrieval_chain)
# 검색된 문서들을 하나로 합쳐서(stuff) LLM에 전달하는 체인 생성
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
# 세션별 채팅 메시지 히스토리를 메모리에 저장하는 클래스
from langchain_community.chat_message_histories import ChatMessageHistory
# 채팅 메시지 히스토리의 기본(base) 인터페이스 클래스
from langchain_core.chat_history import BaseChatMessageHistory
# LLM 출력을 문자열로 파싱하는 파서
from langchain_core.output_parsers import StrOutputParser
# 프롬프트 템플릿 생성 및 메시지 플레이스홀더(채팅 히스토리 삽입용)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 체인에 대화 히스토리 관리를 자동으로 추가해주는 래퍼
from langchain_core.runnables.history import RunnableWithMessageHistory
# OpenAI의 ChatGPT 모델과 텍스트 임베딩 모델 사용
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Pinecone 벡터 데이터베이스와 연동하는 벡터 스토어
from langchain_pinecone import PineconeVectorStore

## ========== 세션 히스토리 저장소 ==========

# 세션 ID별로 대화 히스토리를 메모리에 보관하는 딕셔너리
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    세션 ID에 해당하는 채팅 히스토리를 반환합니다.
    - 해당 세션이 처음이면 새로운 ChatMessageHistory를 생성하여 저장합니다.
    - 이미 존재하면 기존 히스토리를 반환합니다.
    - 이를 통해 같은 세션 내에서 이전 대화 맥락을 유지할 수 있습니다.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## ========== Pinecone 리트리버 생성 ==========

def get_retriever():
  """
  Pinecone 벡터 데이터베이스에서 관련 문서를 검색하는 리트리버를 생성합니다.
  - OpenAI의 text-embedding-3-large 모델로 텍스트를 벡터로 변환합니다.
  - 'tax-index'라는 이름의 Pinecone 인덱스에 연결합니다.
  - 질문과 가장 유사한 상위 3개(k=3) 문서를 검색하도록 설정합니다.
  """
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")
  index_name = 'tax-index'
  database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
  retriever = database.as_retriever(search_kwargs={"k": 3})
  return retriever


## ========== LLM(대규모 언어 모델) 생성 ==========

def get_llm(model="gpt-5.4"):
  """
  OpenAI ChatGPT 모델 인스턴스를 생성하여 반환합니다.
  - 기본 모델은 gpt-5.4이며, 필요 시 다른 모델명을 전달할 수 있습니다.
  """
  llm = ChatOpenAI(model=model)
  return llm


## ========== 사전 기반 질문 변환 체인 ==========

def get_dictionary_chain():
  """
  사용자의 질문을 도메인 용어 사전에 맞게 변환하는 체인을 생성합니다.
  - 예: "사람" 같은 일반 표현을 세법 용어인 "거주자"로 변환합니다.
  - 변환이 필요 없으면 원래 질문을 그대로 반환합니다.
  - 체인 구성: 프롬프트 → LLM → 문자열 파서 (prompt | llm | StrOutputParser)
  """
  # 도메인 용어 사전 정의 (일반 표현 -> 세법 용어)
  dictionary = ["사람을 나타내는 표현 -> 거주자"]

  llm = get_llm()

  # 사전을 참고하여 질문을 변환하도록 지시하는 프롬프트 템플릿
  prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 그대로 반환해주세요.
    사전: {dictionary}

    질문: {{question}}
  """
  )

  # 프롬프트 → LLM 호출 → 출력을 문자열로 파싱하는 체인
  dictionary_chain = prompt | llm | StrOutputParser()
  return dictionary_chain

## ========== RAG(검색 증강 생성) 체인 ==========

def get_rag_chain():
  """
  RAG(Retrieval-Augmented Generation) 체인을 생성합니다.
  이 체인은 다음 과정을 거쳐 답변을 생성합니다:
    1. 대화 히스토리를 고려하여 사용자 질문을 독립적인 질문으로 재구성
    2. 재구성된 질문으로 Pinecone에서 관련 문서 검색
    3. 검색된 문서(context)와 질문을 결합하여 LLM이 답변 생성
    4. 대화 히스토리를 세션별로 자동 관리
  """
  llm = get_llm()
  retriever = get_retriever()

  ## ----- 1단계: 질문 맥락화 (History-Aware Retriever) -----
  # 대화 히스토리를 참고하여, 이전 맥락에 의존하는 질문을
  # 독립적으로 이해 가능한 질문으로 재구성하는 시스템 프롬프트
  contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
  )

  # 질문 맥락화 프롬프트: 시스템 지시 + 대화 히스토리 + 사용자 입력
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", contextualize_q_system_prompt),
      MessagesPlaceholder("chat_history"),  # 이전 대화 내역이 자동 삽입됨
      ("human", "{input}"),                 # 사용자의 현재 질문
      ]
  )
  
  # 대화 히스토리를 인식하는 리트리버 생성
  # - 먼저 질문을 독립적으로 재구성한 뒤, 해당 질문으로 문서를 검색
  history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
  )
  
  ## ----- 2단계: 질의응답 체인 (Question-Answer Chain) -----
  # 검색된 문서(context)를 참고하여 답변을 생성하는 시스템 프롬프트
  # {context}에 검색된 문서 내용이 자동으로 삽입됩니다
  system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
  )

  # 질의응답 프롬프트: 시스템 지시(+검색 문서) + 대화 히스토리 + 사용자 입력
  qa_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      MessagesPlaceholder("chat_history"),  # 이전 대화 내역
      ("human", "{input}"),                 # 사용자의 현재 질문
    ]
  )

  # 검색된 문서들을 하나로 합쳐서 LLM에 전달하는 체인 생성 (stuff 방식)
  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

  # 리트리버 + 질의응답 체인을 결합한 RAG 체인 생성
  # - 질문 → 문서 검색 → 검색 결과와 질문을 합쳐서 답변 생성
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  ## ----- 3단계: 대화 히스토리 관리 추가 -----
  # 대화 히스토리 자동 관리 기능을 추가한 최종 체인
  # - input_messages_key: 사용자 입력 키 ('input')
  # - output_messages_key: AI 응답 키 ('answer')
  # - history_messages_key: 대화 히스토리 키 ('chat_history')
  # - .pick('answer'): 전체 출력 중 'answer' 값만 추출
  conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history,
  input_messages_key='input', output_messages_key='answer', history_messages_key='chat_history').pick('answer')

  return conversational_rag_chain


## ========== AI 응답 생성 (최종 진입점) ==========

def get_ai_response(user_message):
  """
  사용자의 질문을 받아 AI 응답을 스트리밍 방식으로 반환합니다.
  전체 처리 흐름:
    1. 사전 체인(dictionary_chain)이 사용자 질문의 용어를 세법 용어로 변환
    2. 변환된 질문이 RAG 체인(rag_chain)에 'input'으로 전달
    3. RAG 체인이 관련 문서를 검색하고, 대화 히스토리를 참고하여 답변 생성
    4. 결과를 스트리밍(stream) 방식으로 반환하여 실시간 출력 가능
  """
  dictionary_chain = get_dictionary_chain()
  rag_chain = get_rag_chain()

  # 사전 체인의 출력을 RAG 체인의 'input'으로 연결하는 전체 파이프라인
  # {"input": dictionary_chain} → 사전 체인 결과가 'input' 키에 매핑됨
  tax_chain = {"input": dictionary_chain} | rag_chain
  # 스트리밍 방식으로 AI 응답 생성 (session_id로 대화 히스토리 구분)
  ai_response = tax_chain.stream({"question": user_message}, config={"configurable": {"session_id": "abc123"}})
  return ai_response