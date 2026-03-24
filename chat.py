import streamlit as st
from dotenv import load_dotenv

from llm import get_ai_message

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