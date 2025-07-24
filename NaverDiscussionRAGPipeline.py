import os
import json
import uuid
import time
import http.client
from http import HTTPStatus
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from webdriver_manager.chrome import ChromeDriverManager

class NaverDiscussionRAGPipeline:
    def __init__(self, json_path: str, db_path: str, collection_name: str):
        load_dotenv(override=True)
        self.json_path = json_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunked_docs = []
        self.documents = []

        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.llm = ChatClovaX(model="HCX-003", max_tokens=2048)
        self.retriever = None
        self.vectorstore = None

        self._init_clova_executor()

    def _init_clova_executor(self):
        self._clova_host = "clovastudio.stream.ntruss.com"
        self._clova_api_key = os.environ["NCP_CLOVASTUDIO_API_KEY"]

    def _send_segmentation_request(self, text):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._clova_api_key}',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
        }
        conn = http.client.HTTPSConnection(self._clova_host)
        request_data = {
            "postProcessMaxSize": 5000,
            "alpha": 0.0,
            "segCnt": -1,
            "postProcessMinSize": 2000,
            "text": text,
            "postProcess": True
        }
        conn.request("POST", "/testapp/v1/api-tools/segmentation", json.dumps(request_data), headers)
        response = conn.getresponse()
        status = response.status
        result = json.loads(response.read().decode("utf-8"))
        conn.close()
        if status == HTTPStatus.OK and "result" in result:
            return [' '.join(seg) for seg in result["result"]["topicSeg"]]
        else:
            raise ValueError(f"Segmentation 실패: {result}")

    def crawl_comments(self, stock_code="005930", max_scroll=10, output_path="./data/comments.json"):
        url = f"https://m.stock.naver.com/domestic/stock/{stock_code}/discussion"
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.get(url)
        time.sleep(1)

        comments = []
        prev_count = 0
        no_change_count = 0
        start_time = time.time()

        for _ in range(max_scroll):
            try:
                more_btn = driver.find_element(By.XPATH, '//*[@id="content"]/div[12]/div/button')
                if more_btn.is_displayed():
                    more_btn.click()
                    time.sleep(1)
            except (NoSuchElementException, ElementNotInteractableException):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

            elements = driver.find_elements(By.XPATH, '//li[contains(@class, "DiscussionPostWrapper_article")]')
            if len(elements) == prev_count:
                no_change_count += 1
                if no_change_count >= 3:
                    break
            else:
                no_change_count = 0
            prev_count = len(elements)

            if time.time() - start_time > 30:
                break

        for elem in elements:
            try:
                title = elem.find_element(By.XPATH, './div[2]/strong').text.strip()
            except:
                title = ""
            try:
                body = elem.find_element(By.XPATH, './div[2]/p').text.strip()
            except:
                body = ""
            if title or body:
                comments.append({"content": f"{title}\n{body}".strip()})

        driver.quit()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"댓글 {len(comments)}개 저장 완료: {output_path}")

    def _load_documents(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            comments = json.load(f)
        docs = []
        for i, item in enumerate(comments):
            docs.append(Document(
                page_content=item.get("content", ""),
                metadata={
                    "source": os.path.basename(self.json_path),
                    "id": str(i)
                }
            ))
        return docs

    def segment_documents(self):
        docs = self._load_documents()
        self.chunked_docs = []
        group_size = 10
        for i in tqdm(range(0, len(docs), group_size), desc="Segmentation 요청 처리"):
            group = docs[i:i+group_size]
            merged_discussion_text = "\n\n".join([d.page_content for d in group])
            merged_ids = [d.metadata.get("id") for d in group]
            try:
                result_data = self._send_segmentation_request(merged_discussion_text)
                for paragraph in result_data:
                    self.chunked_docs.append({
                        "page_content": paragraph,
                        "metadata": {"source_ids": merged_ids}
                    })
                time.sleep(1)
            except Exception as e:
                print(f"Segmentation 실패: {e}")
                time.sleep(3)
        
        # 파일 저장
        os.makedirs("./data", exist_ok=True)

        # JSON 저장
        with open("./data/comment.json", "w", encoding="utf-8") as f_json:
            json.dump(self.chunked_docs, f_json, ensure_ascii=False, indent=2)
            print("comment.json 저장 완료")

        # TXT 저장
        merged_text = "\n\n".join(item["page_content"] for item in self.chunked_docs)
        with open("./data/merged_discussion_text.txt", "w", encoding="utf-8") as f_txt:
            f_txt.write(merged_text.strip())
            print("merged_discussion_text.txt 저장 완료")
                
                

    def embed_and_store(self):
        if not self.chunked_docs:
            raise ValueError("세그멘테이션이 먼저 실행되어야 합니다.")

        self.documents = []
        for item in self.chunked_docs:
            self.documents.append(Document(
                page_content=item["page_content"],
                metadata={
                    "source": item["metadata"],
                    "id": str(uuid.uuid4())
                }
            ))

        client = chromadb.PersistentClient(path=self.db_path)
        client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        self.vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )

        texts, metadatas = [], []
        for doc in self.documents:
            original = doc.metadata
            source_ids = original.get("source", {}).get("source_ids", [])

            # 핵심 수정: 문자열 보장 + 길이 제한 + 널문자 제거
            text = str(doc.page_content).replace("\x00", "").strip()[:8000]
            
            if not text:
                continue  # 빈 텍스트는 건너뜀

            flattened_metadata = {
                "source_ids": ",".join(source_ids) if isinstance(source_ids, list) else str(source_ids),
                "id": original.get("id", "")
            }

            texts.append(text)
            metadatas.append(flattened_metadata)

        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"{len(texts)}개 문서가 ChromaDB에 저장되었습니다.")

    def query_opinion(self, question: str) -> str:
        if self.vectorstore is None:
            raise ValueError("임베딩이 먼저 수행되어야 합니다.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = PromptTemplate.from_template(
            """당신은 금융 전문가이자 투자 심리 분석가입니다.
아래에 제공된 문맥(context)은 특정 종목에 대한 최근 투자자들의 댓글과 여론입니다.
당신의 임무는 이 문맥에 포함된 여론을 객관적으로 요약하고, 투자 심리와 감정적 분위기를 평가하는 것입니다.

답변 시 다음 지침을 따르세요:
- 전반적인 여론을 '긍정적', '부정적', '혼재됨', '중립적' 중 하나로 분류하세요.
- 투자자들의 태도와 심리를 간단히 요약해 주세요.
- 비속어나 혐오 표현은 그대로 반복하지 않고, 해당 표현이 사용되었음을 언급만 해주세요.
- 문맥에 충분한 정보가 없으면 '주어진 정보에서 여론을 평가하기 어렵습니다'라고 답변해 주세요.

# Question:
{question}

# Context:
{context}

# Answer:"""
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        result = rag_chain_with_source.invoke(question)
        return result['answer']

def main():
    pipeline = NaverDiscussionRAGPipeline(
        json_path="./data/comments.json",
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_discussion_docs"
    )

    # 0단계: 댓글 크롤링부터 자동 실행
    pipeline.crawl_comments(stock_code="005930")  # 삼성전자

    # 1단계: CLOVA 세그멘테이션 수행
    pipeline.segment_documents()

    # 2단계: 임베딩 후 Chroma 저장
    pipeline.embed_and_store()

    # 3단계: 테스트 질의
    query = "삼성전자에 대한 최근 여론이 어때?"
    answer = pipeline.query_opinion(query)

    print("\n질문:", query)
    print("분석 결과:\n", answer)


# 메인 실행
if __name__ == "__main__":
    main()
