import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser, SystemMessage, HumanMessage
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
import os
import json

########################################
# 페이지 설정
########################################
st.set_page_config(
    page_icon="⁉️",
    page_title="Quiz GPT"
)

st.title("Quiz GPT - Test Yourself")


########################################
# JSON 파서 정의
########################################
class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.strip().replace("```", "").replace("json", "")
        return json.loads(text)

json_parser = JsonOutputParser()

########################################
# 파일 로더 및 위키 검색 함수
########################################
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as content_file:
        content_file.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


########################################
# 사이드바 설정 : API Key, 주제, 난이도 등
########################################
with st.sidebar:
    st.markdown("### Configuration")
    # 사용자 API키 입력받기
    user_api_key = st.text_input("Enter your OpenAI API Key:", type="password", help="Enter your personal OpenAI API key to run this app.")
    st.markdown("---")

    # 출처 선택 (파일 또는 위키)
    choice = st.selectbox(
        "Select the source of quiz content:",
        (
            "File",
            "Wikipedia Article",
        ),
        index=0
    )

    topic = None
    docs = None

    # 파일 로딩
    if choice == "File":
        file = st.file_uploader("Upload a .docx, .txt, or .pdf file", type=["docx", "txt", "pdf"])
        if file:
            docs = split_file(file)
            context_source = file.name
    else:
        # 위키 검색
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
            context_source = topic

    st.markdown("---")
    # 난이도 선택
    difficulty = st.selectbox("Select Difficulty Level:", ["Easy", "Hard"], index=0)
    st.markdown("---")
    # 깃허브 링크
    st.markdown("[View on GitHub](https://github.com/username/repo)")  
    st.markdown("---")


########################################
# 문서가 없으면 안내 메세지
########################################
if not user_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
elif not docs:
    st.markdown(
        """
        ### Welcome to Quiz GPT!

        1. 왼쪽 사이드바에서 주제를 선택하세요.
        2. File: 원하는 파일을 업로드하면 해당 내용으로 퀴즈를 생성.
        3. Wikipedia: 주제를 입력하고 엔터를 누르면 해당 주제로 퀴즈를 생성.
        4. 난이도(Easy/Hard)도 선택할 수 있습니다.
        """
    )
else:
    ########################################
    # 문서가 로드된 상태: 퀴즈 생성
    ########################################

    context = format_docs(docs)

    # 함수 호출 스펙 정의
    quiz_function = {
        "name": "generate_quiz",
        "description": "Generate 10 quiz questions from given context and difficulty",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "A list of 10 question objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The quiz question"},
                            "answers": {
                                "type": "array",
                                "description": "List of possible answers, one of which is correct",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {"type": "string"},
                                        "correct": {"type": "boolean"}
                                    },
                                    "required": ["answer", "correct"]
                                }
                            }
                        },
                        "required": ["question", "answers"]
                    }
                }
            },
            "required": ["questions"]
        }
    }

    # 시스템 메시지
    system_msg = f"""
    You are a helpful assistant role-playing as a teacher.
    Based ONLY on the following context, generate 10 questions testing the user's knowledge.
    Each question should have 4 answer choices, exactly one of them correct.
    The difficulty should be {difficulty}.
    The user wants the questions to be challenging if difficulty is 'Hard', and simpler if 'Easy'.
    Return the results by calling the 'generate_quiz' function with a JSON object containing a 'questions' field.
    The 'questions' field should be an array of 10 question objects, each with 'question' and 'answers'.
    Each 'answers' is an array of 4 objects: {{'answer': 'some text', 'correct': false/true}}.
    Make sure exactly one is true. 출력은 한글로 해주세요.
    """

    # 메시지를 Langchain용 포맷으로 변경
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Context:\n{context}")
    ]

    # LLM 인스턴스 생성 (사용자 API 키 사용)
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=user_api_key,
    )

    # 함수 호출
    response = llm(
        messages=messages,
        functions=[quiz_function],
        function_call={"name": "generate_quiz"}
    )

    # 함수 호출 결과 파싱
    if response.additional_kwargs and "function_call" in response.additional_kwargs:
        function_args = response.additional_kwargs["function_call"].get("arguments", "{}")
        questions_data = json.loads(function_args)
    else:
        st.error("No quiz questions generated.")
        st.stop()

    if "questions" not in questions_data or len(questions_data["questions"]) != 10:
        st.error("Failed to generate a valid quiz.")
        st.stop()

    # 퀴즈 진행 상태 관리
    if "attempts" not in st.session_state:
        st.session_state["attempts"] = 0

    if "user_answers" not in st.session_state:
        st.session_state["user_answers"] = {}

    quiz_form = st.form("quiz_form")
    for i, q in enumerate(questions_data["questions"]):
        question_text = q["question"]
        quiz_form.write(f"**Question {i+1}:** {question_text}")
        options = [a["answer"] for a in q["answers"]]
        user_choice = quiz_form.radio(f"Select an option (Q{i+1}):", options, key=f"q_{i}",index=None)
        st.session_state["user_answers"][i] = user_choice

    submit_button = quiz_form.form_submit_button("Submit")

    if submit_button:
        # 채점
        score = 0
        for i, q in enumerate(questions_data["questions"]):
            chosen = st.session_state["user_answers"][i]
            for ans in q["answers"]:
                if ans["answer"] == chosen and ans["correct"]:
                    score += 10
                    break

        st.session_state["attempts"] += 1
        st.write(f"Your score: {score} / 100")

        # 점수 100점이면 풍선 표시
        if score == 100:
            st.balloons()
            st.success("Perfect score! Congratulations!")
        else:
            # 100점 미만일 때 퀴즈 다시 풀기 버튼
            if st.button("Retake the quiz"):
                # 현재는 동일한 문제를 그대로 다시 풀 수 있게 구성
                # 다른 새로운 문제를 원하면 여기서 LLM 재호출 코드를 추가할 수도 있음
                for i in range(10):
                    st.session_state["user_answers"][i] = None
                st.experimental_rerun()
o