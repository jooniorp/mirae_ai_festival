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
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        text = re.sub(r'&[#\d]+;', '', text)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def extract_article_content(self, url: str) -> Dict[str, str]:
        """뉴스 기사 내용 추출"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            domain = urlparse(url).netloc
            
            return self._extract_by_domain(soup, domain, url)
            
        except Exception as e:
            print(f"기사 내용 추출 실패: {e}")
            return {'content': '', 'publisher': '알 수 없음'}

    def _extract_by_domain(self, soup: BeautifulSoup, domain: str, url: str) -> Dict[str, str]:
        """도메인별 기사 내용 추출"""
        content = ""
        publisher = "알 수 없음"
        
        # 네이버 뉴스
        if "news.naver.com" in domain:
            # 네이버 뉴스 본문
            article_body = soup.find('div', {'id': 'articleBody'}) or soup.find('div', {'id': 'articleBodyContents'})
            if article_body:
                content = article_body.get_text()
            
            # 언론사 정보
            press_info = soup.find('div', {'class': 'press_logo'}) or soup.find('a', {'class': 'press'})
            if press_info:
                publisher = press_info.get_text().strip()
        
        # 일반 언론사
        else:
            # 제목 추출
            title_selectors = ['h1', 'h2', '.title', '.headline', '.article-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    content += title_elem.get_text() + "\n\n"
                    break
            
            # 본문 추출
            content_selectors = [
                '.article-body', '.article-content', '.news-content', '.content',
                'article', '.post-content', '.entry-content', '.story-body'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content += content_elem.get_text()
                    break
            
            # 언론사 정보 추출
            publisher_selectors = ['.publisher', '.press', '.media', '.source']
            for selector in publisher_selectors:
                pub_elem = soup.select_one(selector)
                if pub_elem:
                    publisher = pub_elem.get_text().strip()
                    break
            
            return {
            'content': self.clean_html_tags(content),
            'publisher': publisher
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

    def analyze_news_impact(self, company_name: str) -> Dict:
        """뉴스 영향도 분석 및 주가 변동 가능성 판단"""
        try:
            # 뉴스 크롤링
            news_articles = self.crawl_news(query=company_name, max_articles=10)
            
            if not news_articles:
                return {
                    "trigger": False,
                    "reason": "뉴스를 찾을 수 없습니다.",
                    "analysis": "추가 분석이 필요하지 않습니다."
                }
            
            # 뉴스 내용을 하나의 텍스트로 결합
            news_text = ""
            for article in news_articles[:8]:  # 상위 8개 기사 분석 (더 많은 기사 분석)
                news_text += f"제목: {article.get('title', '')}\n"
                news_text += f"내용: {article.get('content', '')[:800]}...\n\n"
            
            # 주가 변동 가능성 분석 프롬프트
            impact_prompt = PromptTemplate(
                input_variables=["company_name", "news_text"],
                template="""
당신은 주식 투자 전문가입니다. 주어진 뉴스를 분석하여 해당 기업의 주가에 미칠 영향을 판단해주세요.

기업명: {company_name}

최근 뉴스:
{news_text}

⚠️ **중요한 필터링 기준**:
- **스포츠 관련 뉴스** (야구, 축구, 농구, 배구 등): 기업 주가와 무관하므로 "하"로 판단
- **연예/문화 관련 뉴스** (콘서트, 드라마, 영화 등): 기업 주가와 무관하므로 "하"로 판단
- **동명이인 관련 뉴스**: 같은 이름이지만 다른 분야의 인물/기관 관련 뉴스는 제외
- **기업의 스포츠팀/문화사업**: 해당 기업의 직접적인 경영 활동이 아닌 경우 "하"로 판단

다음 기준으로 분석해주세요:

1. **주가 변동 가능성** (상/중/하)
   - 상: 실적 발표, 대규모 계약, 경영진 변화, 규제 변화, 노조 교섭 결렬, 파업, 경영권 분쟁, 관세 정책, 수출입 제한 등
   - 중: 신제품 출시, 시장 동향, 경쟁사 소식, 인수합병 소식, 공급망 이슈, 원자재 가격 변동, 환율 변동 등
   - 하: 일반적인 업계 소식, 마케팅 활동, 사소한 업무 소식, 문화/사회 활동, 스포츠 관련, 연예 관련 등

2. **변동 방향** (상승/하락/중립)
   - 상승: 긍정적인 실적, 성장 전망, 호재, 성공적인 계약 체결 등
   - 하락: 부정적인 실적, 리스크 증가, 악재, 노조 분쟁, 파업 등
   - 중립: 영향이 미미하거나 양면적, 불확실한 상황

3. **변동 강도** (강/중/약)
   - 강: 5% 이상 주가 변동 예상 (노조 파업, 대규모 계약, 실적 발표 등)
   - 중: 2-5% 주가 변동 예상 (일반적인 경영 소식, 시장 동향 등)
   - 약: 2% 미만 주가 변동 예상 (사소한 업무 소식, 스포츠, 연예 등)

4. **추가 분석 필요성**
   - YES: 주가 변동 가능성이 높고 추가 분석이 필요 (노조 교섭, 실적 발표, 대규모 계약, 관세 정책, 공급망 이슈 등)
   - NO: 주가 변동 가능성이 낮아 추가 분석 불필요 (일반적인 업무 소식, 마케팅 활동, 문화/사회 활동, 스포츠, 연예 등)

분석 결과를 JSON 형식으로 출력해주세요:
{{
    "stock_impact": "상/중/하",
    "direction": "상승/하락/중립",
    "intensity": "강/중/약",
    "need_analysis": "YES/NO",
    "reason": "판단 근거",
    "key_events": ["주요 이벤트1", "주요 이벤트2"],
    "recommendation": "투자자에게 제안할 행동"
}}
"""
            )
            
            # LLM 분석 실행
            chain = impact_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "company_name": company_name,
                "news_text": news_text
            })
            
            # JSON 파싱
            try:
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group())
                else:
                    # JSON 파싱 실패 시 기본값
                    analysis_result = {
                        "stock_impact": "중",
                        "direction": "중립",
                        "intensity": "약",
                        "need_analysis": "NO",
                        "reason": "뉴스 분석 결과 추가 분석이 필요하지 않습니다.",
                        "key_events": [],
                        "recommendation": "현재 상태 유지"
                    }
            except:
                analysis_result = {
                    "stock_impact": "중",
                    "direction": "중립",
                    "intensity": "약",
                    "need_analysis": "NO",
                    "reason": "뉴스 분석 결과 추가 분석이 필요하지 않습니다.",
                    "key_events": [],
                    "recommendation": "현재 상태 유지"
                }
            
            # 트리거 판단
            trigger = analysis_result.get("need_analysis", "NO") == "YES"
            
            return {
                "trigger": trigger,
                "reason": analysis_result.get("reason", "분석 불가"),
                "analysis": analysis_result,
                "news_count": len(news_articles),
                "key_events": analysis_result.get("key_events", [])
            }
            
        except Exception as e:
            print(f"뉴스 영향도 분석 중 오류: {e}")
            return {
                "trigger": False,
                "reason": f"분석 중 오류 발생: {str(e)}",
                "analysis": {},
                "news_count": 0,
                "key_events": []
            }

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
        total_groups = (len(docs) + group_size - 1) // group_size
        
        print(f"세그멘테이션 배치 처리: 1-{min(group_size, len(docs))}/{len(docs)}")
        
        for i in range(0, len(docs), group_size):
            group = docs[i:i + group_size]
            batch_num = (i // group_size) + 1
            
            # 그룹의 모든 문서 내용을 결합
            combined_text = ""
            for doc in group:
                combined_text += doc.page_content + "\n\n"
            
            try:
                # CLOVA Studio 세그멘테이션 실행
                segments = self._send_segmentation_request(combined_text)
                
                # 세그먼트를 Document로 변환
                for j, segment in enumerate(segments):
                    if len(segment.strip()) > 100:  # 최소 길이 필터
                        chunked_doc = Document(
                            page_content=segment,
                            metadata={
                                "source": f"batch_{batch_num}_segment_{j+1}",
                                "original_docs": [doc.metadata.get("title", "") for doc in group]
                            }
                        )
                        self.chunked_docs.append(chunked_doc)
                
                print(f"배치 {batch_num}: {len(segments)}개 세그먼트 생성")
                    
            except Exception as e:
                print(f"배치 {batch_num} 세그멘테이션 실패: {e}")
                # 실패 시 원본 문서를 그대로 사용
                for doc in group:
                    self.chunked_docs.append(doc)
        
        print(f"총 {len(self.chunked_docs)}개의 segment 문서가 생성되었습니다.")
        return self.chunked_docs

    def embed_and_store(self):
        """문서 임베딩 및 벡터 저장소에 저장"""
        if not self.chunked_docs:
            self.segment_documents()
        
        # ChromaDB 벡터 저장소 초기화
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        )
        
        # 문서 추가
        self.vectorstore.add_documents(self.chunked_docs)
        
        # 검색기 설정
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        print(f"벡터 저장소에 {len(self.chunked_docs)}개 문서 저장 완료")

    def query_news(self, question: str, k: int = 5) -> str:
        """뉴스 질의응답"""
        try:
            # 벡터 저장소가 없으면 초기화
            if not self.vectorstore:
                self.embed_and_store()
            
            # 관련 문서 검색
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return "관련 뉴스를 찾을 수 없습니다."
            
            # RAG 프롬프트 생성
            rag_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
다음 뉴스 기사들을 참고하여 질문에 답변해주세요.

뉴스 기사:
{context}

질문: {question}

답변:
"""
        )

            # 문서 포맷팅
            def format_docs(docs: List[Document]) -> str:
                formatted_docs = []
                for i, doc in enumerate(docs, 1):
                    formatted_docs.append(f"{i}. {doc.page_content[:500]}...")
                return "\n\n".join(formatted_docs)
            
            # RAG 체인 실행
            rag_chain = (
                {"context": lambda x: format_docs(x), "question": RunnablePassthrough()}
                | rag_prompt
            | self.llm
            | StrOutputParser()
        )

            result = rag_chain.invoke(docs)
            return result
            
        except Exception as e:
            return f"뉴스 질의응답 중 오류: {str(e)}"

def main():
    """테스트용 메인 함수"""
    pipeline = NaverNewsRAGPipeline(
        json_path="./data/news_articles.json",
        db_path="./chroma_langchain_db",
        collection_name="naver_news_docs"
    )
    
    # 뉴스 크롤링 테스트
    articles = pipeline.crawl_news("삼성전자", max_articles=5)
    print(f"크롤링된 기사 수: {len(articles)}")
    
    # 뉴스 영향도 분석 테스트
    impact_result = pipeline.analyze_news_impact("삼성전자")
    print(f"주가 변동 가능성: {impact_result['trigger']}")
    print(f"이유: {impact_result['reason']}")

if __name__ == "__main__":
    main()
