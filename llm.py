# --- 임포트 ---
from langchain_classic.chains import (create_history_aware_retriever,
                                      create_retrieval_chain)
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    FewShotChatMessagePromptTemplate,
                                    MessagesPlaceholder)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import answer_examples

# 세션별 대화 히스토리 저장소
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # 세션 ID에 해당하는 대화 히스토리 반환 (없으면 새로 생성)
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
  # Pinecone에서 유사 문서 상위 3개를 검색하는 리트리버 생성
  embedding = OpenAIEmbeddings(model="text-embedding-3-large")
  index_name = 'tax-index'
  database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
  retriever = database.as_retriever(search_kwargs={"k": 3})
  return retriever


def get_history_retriever():
  # 대화 히스토리를 고려해 질문을 재구성한 뒤 문서를 검색하는 리트리버 생성
  llm = get_llm()
  retriever = get_retriever()

  # 이전 대화 맥락을 반영해 독립적인 질문으로 재구성하도록 지시
  contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
  )

  contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", contextualize_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
      ]
  )
  
  history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
  )

  return history_aware_retriever


def get_llm(model="gpt-5.4"):
  # OpenAI ChatGPT 모델 인스턴스 생성 (기본: gpt-5.4)
  llm = ChatOpenAI(model=model)
  return llm


def get_dictionary_chain():
  # 사전 기반으로 일반 용어를 세법 용어로 변환하는 체인 (예: 사람 → 거주자)
  dictionary = ["사람을 나타내는 표현 -> 거주자"]
  llm = get_llm()

  prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 그대로 반환해주세요.
    사전: {dictionary}

    질문: {{question}}
  """
  )

  dictionary_chain = prompt | llm | StrOutputParser()
  return dictionary_chain

def get_rag_chain():
  # RAG 체인: 질문 재구성 → 문서 검색 → 답변 생성 + 대화 히스토리 관리
  llm = get_llm()

  # Few-shot: 답변 예시를 제공해 응답 형식 유도
  example_prompts = ChatPromptTemplate.from_messages(
    [
      ("human", "{input}"),
      ("ai", "{answer}")
    ]
  )
  few_show_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompts,
    examples=answer_examples,
  )
  
  # 검색된 문서를 참고해 답변을 생성하는 시스템 프롬프트
  system_prompt = (
    "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해 주세요."
    "아래에 제공된 문서를 참고하여 질문에 답변해주시고"
    "답변을 알 수 없다면 모른다고 답변해주세요"
    "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
    "2-3 문장 정도의 간결한 답변을 제공해주세요."
    "\n\n"
    "{context}"
  )

  # 프롬프트 구성: 시스템 지시 + few-shot 예시 + 대화 히스토리 + 사용자 질문
  qa_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt),
      few_show_prompt,                      # few-shot 예시 (답변 형식 유도)
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
    ]
  )

  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  # 문서 합쳐서 LLM에 전달
  history_aware_retriever = get_history_retriever()
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # 검색 + 답변 결합

  # 대화 히스토리 자동 관리 추가, 응답에서 'answer'만 추출
  conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history,
  input_messages_key='input', output_messages_key='answer', history_messages_key='chat_history').pick('answer')

  return conversational_rag_chain


def get_ai_response(user_message):
  # 사전 변환 → RAG 체인 → 스트리밍 응답 반환
  dictionary_chain = get_dictionary_chain()
  rag_chain = get_rag_chain()

  # 사전 체인 결과를 'input'으로 RAG 체인에 전달
  tax_chain = {"input": dictionary_chain} | rag_chain
  ai_response = tax_chain.stream({"question": user_message}, config={"configurable": {"session_id": "abc123"}})
  return ai_response