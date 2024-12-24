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

1. 데이터 벡터 저장
 - url data 불러오기
 - 임베이딩 저장하기
 - 인덱싱하기 
 
2. 데이터 전처리 및 임베딩
 - 문장 쪼개기(Chunking)
 - 임베딩하기
 
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
 
추가가. llm 선정하기
 - api key 받기기
"""

import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# ----------------------------------------------------------------------
# 1. sitemap에서 페이지 링크 읽어오기
# ----------------------------------------------------------------------
SITEMAP_URL = "https://www.langchain.com/sitemap.xml"

def get_links_from_sitemap(sitemap_url):
    """Sitemap XML을 파싱하여 <loc> 태그에 있는 URL 목록을 반환"""
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.text, "xml")  # "xml" 파서 사용
    loc_tags = soup.find_all("loc")
    links = [tag.text.strip() for tag in loc_tags]
    return links

# ----------------------------------------------------------------------
# 2. 각 페이지에서 텍스트 추출
# ----------------------------------------------------------------------
def fetch_page_text(url):
    """해당 URL의 HTML을 GET 요청 후, <body>에서 텍스트를 추출 (간단 예시)"""
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            # 본문 텍스트(간단히)만 가져오는 예시
            body = soup.get_text(separator=" ")
            # 필요 시, <script>, <style> 제거 등 추가 처리 가능
            return body
        else:
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# ----------------------------------------------------------------------
# 3. 문서 분할(Chunking)
# ----------------------------------------------------------------------
def split_documents(text, chunk_size=1000, chunk_overlap=200):
    """
    너무 긴 문서는 chunk_size 단위로 분할해 Embedding에 넣기 쉽게 만든다.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# ----------------------------------------------------------------------
# 4. 벡터 스토어 생성 및 저장
# ----------------------------------------------------------------------
def create_vector_store(documents, model, db_path=".cache-dir"):
    """
    문서들을 임베딩하고 벡터 스토어를 생성하여 저장.
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    
    embeddings = model.encode(documents, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, os.path.join(db_path, "vector_store.index"))

# ----------------------------------------------------------------------
# 5. 검색 프로세스
# ----------------------------------------------------------------------
def search(query, model, db_path=".cache-dir", top_k=3):
    """
    사용자 질의를 임베딩하고 벡터 스토어에서 유사 문서 top_k를 찾는다.
    """
    index = faiss.read_index(os.path.join(db_path, "vector_store.index"))
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = index.search(np.array(query_embedding), top_k)
    return I[0]

# ----------------------------------------------------------------------
# 6. 메인 Streamlit 앱
# ----------------------------------------------------------------------
def main():
    st.title("LangChain 사이트맵 RAG 데모")

    # OpenAI API Key 입력 받기 (예시)
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("OpenAI API Key를 입력하세요.")
        st.stop()

    # 6.1. 문서 구축 / 벡터DB 생성 단계 (최초 1회 또는 '재생성' 버튼)
    if st.button("1) Sitemap 크롤링 후, 벡터DB 생성하기"):
        with st.spinner("Sitemap 읽는 중..."):
            links = get_links_from_sitemap(SITEMAP_URL)
            st.write(f"총 {len(links)}개 링크 중 일부 가져옴...")

        with st.spinner("페이지 텍스트 수집 & 문서화..."):
            docs = []
            for link in links[:5]:  # 여기서 max_pages=5는 데모용
                text = fetch_page_text(link)
                if text:
                    chunks = split_documents(text)
                    docs.extend(chunks)

        with st.spinner("벡터 저장소 생성 중..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            create_vector_store(docs, model, db_path=".cache-dir")
        st.success("벡터 DB 생성 완료!")

    # 6.2. 질의응답 섹션
    user_query = st.text_input("질문을 입력하세요:")
    if user_query:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        indices = search(user_query, model, db_path=".cache-dir", top_k=3)

        st.markdown("### 검색된 문서")
        for i, idx in enumerate(indices):
            st.write(f"**{i+1}.** {docs[idx]}")

if __name__ == "__main__":
    main()
