import re
import fitz
import time
import uuid
import os
import http.client
import json
import chromadb
from typing import List
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel

class CLOVAStudioExecutor:
    def __init__(self, host, api_key):
        self._host = host
        self._api_key = api_key

    def _send_request(self, request, endpoint):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4())
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', endpoint, json.dumps(request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode('utf-8'))
        conn.close()
        return result
    
class ResearchRAGPipeline:
    def __init__(self, db_path, collection_name):
        load_dotenv(override=True)
        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        self.segmenter = CLOVAStudioExecutor(
            host="clovastudio.stream.ntruss.com",
            api_key=os.getenv("NCP_CLOVASTUDIO_API_KEY")
        )
        self.documents = []

    def _extract_metadata_from_text(self, text: str):
        metadata = {
            "company": "",
            "date": "",
            "opinion": "",
            "analyst": "",
            "price_target": ""
        }
        company_match = re.search(r"([가-힣]+(?:전자|화학|바이오|에너지|솔루션|건설|증권|은행|제약))", text)
        if company_match:
            metadata["company"] = company_match.group(1)

        date_match = re.search(r"(\d{4}[.\-년 ]\d{1,2}[.\-월 ]\d{1,2})", text)
        if date_match:
            metadata["date"] = date_match.group(1).replace("년", "-").replace("월", "-").replace(" ", "-").replace(".", "-").strip("-")

        opinion_match = re.search(r"(매수|중립|매도)", text)
        if opinion_match:
            metadata["opinion"] = opinion_match.group(1)

        analyst_match = re.search(r"(?:작성자|애널리스트)[\s:]*([가-힣]+)", text)
        if analyst_match:
            metadata["analyst"] = analyst_match.group(1)

        price_match = re.search(r"목표주가[^\d]*(\d{2,3}(?:,\d{3})*(?:\.\d{1,2})?)", text)
        if price_match:
            metadata["price_target"] = price_match.group(1).replace(",", "")

        return metadata

    def extract_from_pdf_folder(self, folder="./pdf_downloads"):
        path = Path(folder)
        if not path.exists() or not path.is_dir():
            raise ValueError("PDF 폴더 경로가 잘못되었습니다.")

        data_json = []
        merged_text = ""

        for file in path.glob("*.pdf"):
            print(f"PDF 처리 중: {file.name}")
            text = ""
            doc = fitz.open(str(file))
            for page in doc:
                text += re.sub(r'\s+', ' ', page.get_text()).strip() + "\n"
            doc.close()

            metadata = self._extract_metadata_from_text(text)
            document = Document(
                page_content=text.strip(),
                metadata=metadata
            )
            self.documents.append(document)

            # JSON용 구조 추가
            data_json.append({
                "file_name": file.name,
                "content": text.strip(),
                "metadata": metadata
            })

            # 텍스트 파일 누적
            merged_text += f"\n\n[파일명: {file.name}]\n{text.strip()}"

        print(f"총 {len(self.documents)}개 PDF 문서가 로드되었습니다.")

        # 저장 경로 보장
        os.makedirs("./data", exist_ok=True)

        # JSON 저장
        with open("./data/research.json", "w", encoding="utf-8") as f_json:
            json.dump(data_json, f_json, ensure_ascii=False, indent=2)
            print("research.json 저장 완료")

        # TXT 저장
        with open("./data/merged_research_text.txt", "w", encoding="utf-8") as f_txt:
            f_txt.write(merged_text.strip())
            print("merged_research_text.txt 저장 완료")

    def segment_documents(self):
        if not self.documents:
            raise ValueError("PDF에서 문서를 먼저 추출해야 합니다. (self.documents가 비어 있음)")

        segmented_docs = []
        for doc in tqdm(self.documents, desc="CLOVA 세그멘테이션 중"):
            try:
                segments = self.segmenter._send_request(
                    request={
                        "postProcessMaxSize": 5000,
                        "alpha": 0.0,
                        "segCnt": -1,
                        "postProcessMinSize": 2000,
                        "text": doc.page_content,
                        "postProcess": True
                    },
                    endpoint="/testapp/v1/api-tools/segmentation"
                )["result"]["topicSeg"]

                overlap = 1  # 1개의 세그먼트씩 겹치게
                for i in range(len(segments)):
                    seg_group = segments[i : i + 1 + overlap]
                    if not seg_group:
                        continue
                    combined = " ".join(" ".join(seg) if isinstance(seg, list) else seg for seg in seg_group)
                    segmented_docs.append(Document(
                        page_content=combined.strip(),
                        metadata=doc.metadata
                    ))
                    
            except Exception as e:
                print(f"segmentation 실패: {e}")

        self.documents = segmented_docs
        print(f"총 {len(self.documents)}개의 segment 문서가 생성되었습니다.")

    def embed_and_store(self):
        if not self.documents:
            raise ValueError("segmentation 문서가 없습니다. 먼저 segment_documents()를 실행하세요.")

        texts, metadatas = [], []
        for i, doc in enumerate(self.documents):
            text = doc.page_content

            if text is None:
                text = ""
            if not isinstance(text, str):
                text = str(text)
            text = text.replace("\x00", "").strip()

            if not text:
                continue

            meta = doc.metadata
            flattened = {
                "id": str(uuid.uuid4()),
                "company": meta.get("company", ""),
                "opinion": meta.get("opinion", ""),
                "date": meta.get("date", ""),
                "analyst": meta.get("analyst", ""),
                "price_target": meta.get("price_target", "")
            }
            flattened = {k: v for k, v in flattened.items() if v}

            texts.append(text)
            metadatas.append(flattened)

        # 저장
        success = 0
        for i, (text, meta) in enumerate(tqdm(zip(texts, metadatas), desc="임베딩 및 저장", total=len(texts))):
            try:
                embedding = self.embedding_model.embed_documents([text])[0]
                self.vectorstore.add_texts(
                    texts=[text],
                    metadatas=[meta],
                    ids=[f"doc_{i}"]
                )
                success += 1
            except Exception as e:
                if "429" in str(e):
                    print(f"429 Too Many Requests → 10초 대기 후 재시도 (doc_{i})")
                    time.sleep(10.0)
                    try:
                        embedding = self.embedding_model.embed_documents([text])[0]
                        self.vectorstore.add_texts(
                            texts=[text],
                            metadatas=[meta],
                            ids=[f"doc_{i}_retry"]
                        )
                        success += 1
                    except Exception as e2:
                        print(f"재시도 실패 (doc_{i}): {e2}")
                else:
                    print(f"저장 실패 (doc_{i}): {e}")
            time.sleep(2.0)
            if i % 3 == 0:
                import gc
                gc.collect()

        print(f"\n총 {success}개 문서가 ChromaDB에 저장되었습니다.")

    def query(self, question: str) -> str:
        if self.vectorstore is None:
            raise ValueError("vector store가 초기화되지 않았습니다. 먼저 embed_and_store()를 실행하세요.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})  # 🔧 문서 수 제한

        prompt = PromptTemplate.from_template(
            """당신은 금융 전문가이자 투자 심리 분석가입니다.
                아래에 제공된 문맥(context)은 특정 종목에 대한 전문가 리서치 보고서입니다.
                당신의 임무는 이 문맥을 기반으로 사용자의 질문에 대한 요약된 분석을 제공하는 것입니다.

                답변 시 다음 지침을 따르세요:
                - 정보의 출처가 명확하지 않으면 "정확한 정보가 확인되지 않습니다"라고 말하세요.
                - 핵심 요점을 간결하게 전달하세요.
                - 숫자나 회사명은 정확히 반복하고, 근거가 부족한 예측은 피하세요.

                # Question:
                {question}

                # Context:
                {context}

                # Answer:"""
                )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content[:1500] for doc in docs)  # 문서 길이 제한

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | ChatClovaX(model="HCX-003", max_tokens=2048)
            | StrOutputParser()
        )

        rag_pipeline = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain)

        result = rag_pipeline.invoke(question)
        return result["answer"]


def main():
    rag = ResearchRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_research_docs"
    )
    rag.extract_from_pdf_folder("./pdf_downloads")
    rag.segment_documents()
    rag.embed_and_store()

    question = "삼성전자의 목표주가는 얼마인가요?"
    answer = rag.query(question)
    print(f"\n질문: {question}")
    print("응답:", answer)


if __name__ == "__main__":
    main()
