"""
Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
AI Gateway
Cloudflare Vectorize
Workers AI
사이트맵을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
"llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?"
"Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?"
"벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?"
유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
st.sidebar를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.

1. 데이터 로드 및 전처리
 - url data 불러오기
 - 전처리 하기
 
2. 데이터 전처리 및 임베딩
 - 문장 쪼개기(Chunking)
 - 
 
3. 벡터 데이터베이스(인덱스) 생성
 - faiss 사용하기
 - 저장하기
  
4. 검색 프로세스
 - 사용자 질의
 - 질의 임베딩
 - 벡터 인덱스에서 유사문서 topk 찾기
 - 검색된 문서(맥락)을 모델에게 전달
 - 모델이 답변 생성.
 
5 결과 활용
 - prompt 생성
 - 출력력
 
추가. llm 선정하기
 - api key 받기기
"""


from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate


import streamlit as st


llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
    )

# ----------------------------------------------------------------------
# 1. sitemap에서 페이지 링크 읽어오기 & 최적화(전처리, chunking)
# ----------------------------------------------------------------------

url = "https://developers.cloudflare.com/sitemap-0.xml"


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:  
        footer.decompose()
    return (
        str(soup.get_text()).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    )


@st.cache_data(show_spinner="loading website...")
def load_website(url):
    loader = SitemapLoader(
        url,
        filter_urls=[    
            "https://developers.cloudflare.com/ai-gatewa",
            "https://developers.cloudflare.com/vectorize",
            "https://developers.cloudflare.com/workers-ai",
            ],
        parsing_function=parse_page,
        )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    return retriever

# ----------------------------------------------------------------------
# 2. 프롬프트 및 검색 프로세스
# ----------------------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                   
    Then, give a score to the answer between 0 and 5.
    
    Examples:
    
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
    
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
    
    Your turn!
    
    Question: {question}    
    """
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answer_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )
    
    



# ----------------------------------------------------------------------
# 6. 메인 Streamlit 앱
# ----------------------------------------------------------------------

st.title("SiteGPT for Cloudflare")
st.write("This is a SiteGPT version for Cloudflare documentation.")

api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your personal OpenAI API key to run this app.")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()
    
retriever = load_website(url)
query = st.text_input("Enter your question:")
    
if query:    
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
        
    )
    
    result = chain.invoke(query)
    st.markdown(result.content)
