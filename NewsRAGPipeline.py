import os
import json
import uuid
import time
import http.client
import requests
import re
from http import HTTPStatus
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict, Optional
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

class NaverNewsRAGPipeline:
    def __init__(self, json_path: str, db_path: str, collection_name: str):
        load_dotenv(override=True)
        self.json_path = json_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunked_docs = []
        self.documents = []
        
        # 네이버 API 설정
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        self.naver_headers = {
            'X-Naver-Client-Id': self.naver_client_id,
            'X-Naver-Client-Secret': self.naver_client_secret,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # LangChain 모델 초기화
        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.llm = ChatClovaX(model="HCX-003", max_tokens=2048)
        self.retriever = None
        self.vectorstore = None

        self._init_clova_executor()

    def _init_clova_executor(self):
        """CLOVA Studio 세그멘테이션 API 초기화"""
        self._clova_host = "clovastudio.stream.ntruss.com"
        self._clova_api_key = os.environ["NCP_CLOVASTUDIO_API_KEY"]

    def _send_segmentation_request(self, text: str) -> List[str]:
        """CLOVA Studio 세그멘테이션 API 호출"""
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

    def search_naver_news(self, query: str = "삼성전자", display: int = 100) -> List[Dict]:
        """네이버 뉴스 검색 API를 통한 뉴스 수집"""
        try:
            url = "https://openapi.naver.com/v1/search/news.json"
            params = {
                'query': query,
                'display': display,
                'start': 1,
                'sort': 'date'  # 최신순
            }
            
            print(f"네이버 뉴스 검색 시작: {query}")
            response = requests.get(url, headers=self.naver_headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news_items = data.get('items', [])
                print(f"뉴스 검색 완료: {len(news_items)}개 기사 발견")
                return news_items
            else:
                print(f"네이버 API 오류: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"뉴스 검색 중 오류: {e}")
            return []

    def clean_html_tags(self, text: str) -> str:
        """HTML 태그 제거 및 텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        clean_text = re.sub(r'<[^>]+>', '', text)
        # HTML 엔티티 디코딩
        clean_text = clean_text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        clean_text = clean_text.replace('&quot;', '"').replace('&#39;', "'")
        # 연속된 공백 제거
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text

    def extract_article_content(self, url: str) -> Dict[str, str]:
        """기사 내용 크롤링"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 도메인별 기사 내용 추출
            domain = urlparse(url).netloc.lower()
            content_info = self._extract_by_domain(soup, domain, url)
            
            return content_info
            
        except Exception as e:
            print(f"기사 크롤링 오류 {url}: {e}")
            return {'content': '', 'publisher': '알 수 없음', 'error': str(e)}

    def _extract_by_domain(self, soup: BeautifulSoup, domain: str, url: str) -> Dict[str, str]:
        """도메인별 기사 내용 추출"""
        content = ""
        publisher = "알 수 없음"
        
        try:
            # 네이버 뉴스
            if 'news.naver.com' in domain:
                article_body = soup.find('div', {'id': 'dic_area'}) or soup.find('div', {'class': 'newsct_article'})
                if article_body:
                    content = article_body.get_text(strip=True)
                
                publisher_elem = soup.find('div', {'class': 'press_logo'}) or soup.find('img', {'class': 'press_logo'})
                if publisher_elem:
                    publisher = publisher_elem.get('alt', '알 수 없음')
            
            # 조선일보
            elif 'chosun.com' in domain:
                article_body = soup.find('div', {'class': 'article-body'}) or soup.find('div', {'id': 'news_body_id'})
                if article_body:
                    content = article_body.get_text(strip=True)
                publisher = "조선일보"
            
            # 중앙일보
            elif 'joongang.co.kr' in domain:
                article_body = soup.find('div', {'class': 'article_body'}) or soup.find('div', {'id': 'article_body'})
                if article_body:
                    content = article_body.get_text(strip=True)
                publisher = "중앙일보"
            
            # 한국경제
            elif 'hankyung.com' in domain:
                article_body = soup.find('div', {'class': 'article-body'}) or soup.find('div', {'id': 'articletxt'})
                if article_body:
                    content = article_body.get_text(strip=True)
                publisher = "한국경제"
            
            # 매일경제
            elif 'mk.co.kr' in domain:
                article_body = soup.find('div', {'class': 'news_cnt_detail_wrap'}) or soup.find('div', {'class': 'art_txt'})
                if article_body:
                    content = article_body.get_text(strip=True)
                publisher = "매일경제"
            
            # 기본 추출 로직
            else:
                selectors = [
                    'div.article-body', 'div.article_body', 'div.news-article-body',
                    'div.entry-content', 'div.post-content', 'div.content',
                    'article', 'div[id*="article"]', 'div[class*="article"]',
                    'div[id*="content"]', 'div[class*="content"]'
                ]
                
                for selector in selectors:
                    article_body = soup.select_one(selector)
                    if article_body:
                        content = article_body.get_text(strip=True)
                        break
                
                # 발행사 추출
                publisher_selectors = [
                    'meta[property="og:site_name"]',
                    'meta[name="author"]',
                    '.publisher', '.press', '.source'
                ]
                
                for selector in publisher_selectors:
                    pub_elem = soup.select_one(selector)
                    if pub_elem:
                        publisher = pub_elem.get('content') or pub_elem.get_text(strip=True)
                        break
            
            # 내용이 너무 짧으면 다른 방법 시도
            if len(content) < 100:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
            # 내용 정리
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'content': content[:10000],  # 내용 길이 제한
                'publisher': publisher,
                'url': url
            }
            
        except Exception as e:
            print(f"도메인별 추출 오류 {domain}: {e}")
            return {
                'content': '',
                'publisher': '알 수 없음',
                'url': url,
                'error': str(e)
            }

    def crawl_news(self, query: str = "삼성전자", max_articles: int = 10, output_path: str = "./data/news_articles.json"):
        """뉴스 크롤링 메인 함수"""
        print(f"뉴스 크롤링 시작: {query}")
        
        # 1. 네이버 뉴스 검색
        news_items = self.search_naver_news(query, display=max_articles)
        
        if not news_items:
            print("검색된 뉴스가 없습니다.")
            return []
        
        # 2. 각 뉴스 기사 상세 내용 크롤링
        crawled_articles = []
        
        for i, item in enumerate(tqdm(news_items, desc="기사 내용 크롤링")):
            try:
                # 기본 정보 추출
                title = self.clean_html_tags(item.get('title', ''))
                description = self.clean_html_tags(item.get('description', ''))
                original_link = item.get('originallink', '')
                naver_link = item.get('link', '')
                pub_date = item.get('pubDate', '')
                
                print(f"[{i+1}/{len(news_items)}] 크롤링 중: {title[:50]}...")
                
                # 기사 내용 크롤링
                target_url = original_link if original_link else naver_link
                content_info = {'content': '', 'publisher': '알 수 없음'}
                
                if target_url:
                    content_info = self.extract_article_content(target_url)
                    time.sleep(1)  # 크롤링 간격 조절
                
                # 최종 데이터 구성
                article = {
                    'id': str(uuid.uuid4())[:12],
                    'title': title,
                    'publisher': content_info.get('publisher', '알 수 없음'),
                    'content': content_info.get('content', ''),
                    'description': description,
                    'original_link': original_link,
                    'naver_link': naver_link,
                    'pub_date': pub_date,
                    'crawled_at': datetime.now().isoformat(),
                    'content_length': len(content_info.get('content', '')),
                    'keyword': query
                }
                
                # 내용이 있는 기사만 저장
                if len(article['content']) > 100:
                    crawled_articles.append(article)
                    print(f"✓ 완료: {len(article['content'])}자")
                else:
                    print(f"✗ 내용 부족: {len(article['content'])}자")
                
            except Exception as e:
                print(f"기사 처리 중 오류: {e}")
                continue
        
        # 3. 결과 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(crawled_articles, f, ensure_ascii=False, indent=2)
        
        print(f"뉴스 크롤링 완료: {len(crawled_articles)}개 기사 저장 → {output_path}")
        return crawled_articles

    def _load_documents(self):
        """JSON 파일에서 뉴스 기사 로드"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        docs = []
        for article in articles:
            # 제목과 내용을 결합한 전체 텍스트 생성
            full_content = f"제목: {article.get('title', '')}\n\n내용: {article.get('content', '')}"
            
            docs.append(Document(
                page_content=full_content,
                metadata={
                    "id": article.get('id', ''),
                    "title": article.get('title', ''),
                    "publisher": article.get('publisher', ''),
                    "pub_date": article.get('pub_date', ''),
                    "original_link": article.get('original_link', ''),
                    "keyword": article.get('keyword', ''),
                    "source": os.path.basename(self.json_path)
                }
            ))
        
        print(f"문서 로드 완료: {len(docs)}개 기사")
        return docs

    def segment_documents(self):
        """CLOVA Studio를 사용한 문서 세그멘테이션"""
        docs = self._load_documents()
        self.chunked_docs = []
        
        # 그룹별로 세그멘테이션 처리
        group_size = 5  # 뉴스는 내용이 길어서 그룹 크기 줄임
        
        for i in tqdm(range(0, len(docs), group_size), desc="세그멘테이션 처리"):
            group = docs[i:i+group_size]
            
            # 각 기사별로 개별 처리
            for doc in group:
                try:
                    # 기사 내용이 너무 길면 일부만 처리
                    content = doc.page_content[:15000]  # 15000자 제한
                    
                    result_segments = self._send_segmentation_request(content)
                    
                    for segment in result_segments:
                        if len(segment.strip()) > 50:  # 너무 짧은 세그먼트 제외
                            self.chunked_docs.append({
                                "page_content": segment,
                                "metadata": {
                                    "source_id": doc.metadata.get("id"),
                                    "title": doc.metadata.get("title"),
                                    "publisher": doc.metadata.get("publisher"),
                                    "pub_date": doc.metadata.get("pub_date"),
                                    "keyword": doc.metadata.get("keyword")
                                }
                            })
                    
                    time.sleep(1)  # API 호출 간격
                    
                except Exception as e:
                    print(f"세그멘테이션 실패: {e}")
                    # 실패한 경우 원본 내용을 그대로 청크로 사용
                    content = doc.page_content
                    # 길이에 따라 분할
                    chunk_size = 3000
                    for j in range(0, len(content), chunk_size):
                        chunk = content[j:j+chunk_size]
                        if len(chunk.strip()) > 50:
                            self.chunked_docs.append({
                                "page_content": chunk,
                                "metadata": {
                                    "source_id": doc.metadata.get("id"),
                                    "title": doc.metadata.get("title"),
                                    "publisher": doc.metadata.get("publisher"),
                                    "pub_date": doc.metadata.get("pub_date"),
                                    "keyword": doc.metadata.get("keyword")
                                }
                            })
                    time.sleep(3)
        
        # 결과 저장
        os.makedirs("./data", exist_ok=True)
        
        # JSON 저장
        with open("./data/news_segments.json", "w", encoding="utf-8") as f:
            json.dump(self.chunked_docs, f, ensure_ascii=False, indent=2)
            print(f"세그멘테이션 완료: {len(self.chunked_docs)}개 세그먼트 생성")
            print("news_segments.json 저장 완료")
        
        # 병합된 텍스트 저장
        merged_text = "\n\n".join(item["page_content"] for item in self.chunked_docs)
        with open("./data/merged_news_text.txt", "w", encoding="utf-8") as f:
            f.write(merged_text.strip())
            print("merged_news_text.txt 저장 완료")

    def embed_and_store(self):
        """임베딩 생성 및 ChromaDB 저장"""
        if not self.chunked_docs:
            raise ValueError("세그멘테이션이 먼저 실행되어야 합니다.")

        # Document 객체 생성
        self.documents = []
        for item in self.chunked_docs:
            self.documents.append(Document(
                page_content=item["page_content"],
                metadata={
                    "source_id": item["metadata"].get("source_id", ""),
                    "title": item["metadata"].get("title", ""),
                    "publisher": item["metadata"].get("publisher", ""),
                    "pub_date": item["metadata"].get("pub_date", ""),
                    "keyword": item["metadata"].get("keyword", ""),
                    "id": str(uuid.uuid4())
                }
            ))

        # ChromaDB 클라이언트 설정
        client = chromadb.PersistentClient(path=self.db_path)
        
        # 기존 컬렉션 삭제 후 새로 생성
        try:
            client.delete_collection(name=self.collection_name)
            print(f"기존 컬렉션 '{self.collection_name}' 삭제")
        except Exception:
            pass  # 컬렉션이 없으면 무시
        
        client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        
        # Chroma 벡터스토어 생성
        self.vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )

        # 텍스트와 메타데이터 준비
        texts, metadatas = [], []
        for doc in self.documents:
            # 텍스트 정리 (널문자 제거, 길이 제한)
            text = str(doc.page_content).replace("\x00", "").strip()[:8000]
            
            if not text or len(text) < 20:
                continue  # 너무 짧은 텍스트는 건너뜀

            # 메타데이터 평면화 (ChromaDB는 중첩 구조 지원 안함)
            flattened_metadata = {
                "source_id": str(doc.metadata.get("source_id", "")),
                "title": str(doc.metadata.get("title", ""))[:200],  # 길이 제한
                "publisher": str(doc.metadata.get("publisher", "")),
                "pub_date": str(doc.metadata.get("pub_date", "")),
                "keyword": str(doc.metadata.get("keyword", "")),
                "id": str(doc.metadata.get("id", ""))
            }

            texts.append(text)
            metadatas.append(flattened_metadata)

        # ChromaDB에 저장
        print(f"임베딩 및 저장 시작: {len(texts)}개 문서")
        
        # 배치 단위로 저장 (ChromaDB 안정성을 위해)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
            print(f"배치 {i//batch_size + 1} 저장 완료: {len(batch_texts)}개 문서")
            time.sleep(0.5)  # 안정성을 위한 대기

        print(f"임베딩 및 ChromaDB 저장 완료: 총 {len(texts)}개 문서")

    def query_news(self, question: str, k: int = 5) -> str:
        """뉴스 기반 질의응답"""
        if self.vectorstore is None:
            raise ValueError("임베딩이 먼저 수행되어야 합니다.")

        # 검색기 설정
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # 프롬프트 템플릿
        prompt = PromptTemplate.from_template(
            '''당신은 금융 투자 전문가이자 뉴스 분석 전문가입니다.
아래에 제공된 뉴스 기사들을 바탕으로 질문에 대해 정확하고 객관적으로 답변해주세요.

답변 시 다음 사항을 고려하세요:
1. 제공된 뉴스 기사의 내용만을 근거로 답변하세요
2. 기사의 발행사와 날짜 정보를 활용하세요
3. 여러 기사에서 언급된 공통 내용을 종합하세요
4. 추측이나 개인적 의견은 배제하세요

# 질문:
{question}

# 관련 뉴스 기사들:
{context}

# 답변:'''
        )

        def format_docs(docs: List[Document]) -> str:
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                title = metadata.get('title', '제목 없음')
                publisher = metadata.get('publisher', '발행사 미상')
                pub_date = metadata.get('pub_date', '날짜 미상')
                
                formatted_doc = f"""[기사 {i}]
제목: {title}
발행사: {publisher}
날짜: {pub_date}
내용: {doc.page_content[:1000]}...
"""
                formatted_docs.append(formatted_doc)
            
            return "\n\n".join(formatted_docs)

        # RAG 체인 구성
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        # 질의 실행
        print(f"질문 처리 중: {question}")
        result = rag_chain_with_source.invoke(question)
        
        return result['answer']

def main():
    """메인 실행 함수"""
    pipeline = NaverNewsRAGPipeline(
        json_path="./data/news_articles.json",
        db_path="./chroma_langchain_db",
        collection_name="naver_news_docs"
    )
    
    # 0단계: 뉴스 크롤링
    print("=== 1단계: 뉴스 크롤링 ===")
    pipeline.crawl_news(query="삼성전자", max_articles=30)
    
    # 1단계: CLOVA 세그멘테이션
    print("\n=== 2단계: 문서 세그멘테이션 ===")
    pipeline.segment_documents()
    
    # 2단계: 임베딩 및 ChromaDB 저장
    print("\n=== 3단계: 임베딩 및 DB 저장 ===")
    pipeline.embed_and_store()
    
    # 3단계: 질의응답 테스트
    print("\n=== 4단계: RAG 질의응답 테스트 ===")
    test_questions = [
        "삼성전자의 최근 실적은 어떤가요?",
        "삼성전자 주가에 영향을 주는 주요 요인은 무엇인가요?",
        "삼성전자의 신제품이나 기술 개발 소식이 있나요?"
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        print("-" * 50)
        answer = pipeline.query_news(question)
        print(f"답변: {answer}")
        print("=" * 80)

if __name__ == "__main__":
    main()