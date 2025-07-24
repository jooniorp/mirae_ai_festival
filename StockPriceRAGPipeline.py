import requests
from urllib import parse
from ast import literal_eval
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_core.documents import Document

class StockPriceRAGPipeline:
    def __init__(self, db_path, collection_name):
        load_dotenv(override=True)
        self.embedding_model = ClovaXEmbeddings(model="bge-m3")
        self.db_path = db_path
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.db_path
        )
        self.documents = []
        self.today = datetime.today().strftime("%Y%m%d")

    # 1. 네이버 API로 주가 데이터 가져오기
    def get_sise(self, code, start_time, end_time, time_from='day'):
        get_param = {
            'symbol': code,
            'requestType': 1,
            'startTime': start_time,
            'endTime': end_time,
            'timeframe': time_from
        }
        url = f"https://api.finance.naver.com/siseJson.naver?{parse.urlencode(get_param)}"
        time.sleep(1)  # 서버 부하 방지
        response = requests.get(url)
        data = literal_eval(response.text.strip())
        if not data or len(data) < 2:
            raise ValueError("API 응답이 비어있거나 데이터 형식이 예상과 다릅니다.")
        columns, rows = data[0], data[1:]
        df = pd.DataFrame(rows, columns=columns)
        df['날짜'] = pd.to_datetime(df['날짜'], format="%Y%m%d")
        df.sort_values(by='날짜', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # 2. 2달치 데이터 fetch & 저장
    def fetch_and_save(self, code="005930"):
        # 오늘 날짜 기준 2달 전 계산
        end_date = datetime.today()
        start_date = end_date - timedelta(days=60)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        df = self.get_sise(code, start_str, end_str, "day")
        df.set_index("날짜", inplace=True)

        # ±1,2,3일 수익률 계산
        result_rows = []
        for base_date in df.index:
            row = {"기준일": base_date.strftime("%Y-%m-%d"), "종가": df.loc[base_date]["종가"]}
            base_price = df.loc[base_date]["종가"]
            for offset in [1, 2, 3]:
                for sign in ["+", "-"]:
                    target_date = base_date + pd.Timedelta(days=offset) if sign == "+" else base_date - pd.Timedelta(days=offset)
                    if target_date in df.index:
                        target_price = df.loc[target_date]["종가"]
                        chg = (target_price - base_price) / base_price * 100
                        row[f"chg_{sign}{offset}d"] = round(chg, 2)
                    else:
                        row[f"chg_{sign}{offset}d"] = None
            result_rows.append(row)
        df_result = pd.DataFrame(result_rows)

        # JSON/TXT 저장
        os.makedirs("./data", exist_ok=True)
        json_path = f"./data/stock_price_{self.today}.json"
        txt_path = f"./data/merged_stock_price_{self.today}.txt"
        df_result.to_json(json_path, orient="records", force_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            for row in result_rows:
                f.write(str(row) + "\n")

        # Document 객체로 변환
        self.documents = [
            Document(
                page_content=str(row),
                metadata={"date": row["기준일"], "stock_code": code}
            ) for row in result_rows
        ]
        print(f"주가 데이터 {len(self.documents)}건 로드 및 저장 완료: {json_path}, {txt_path}")

    # 3. (옵션) 세그멘테이션 (여기선 row 단위로 충분)
    def segment_documents(self):
        # 이미 row 단위로 segment됨
        pass

    # 4. 임베딩 및 ChromaDB 저장
    def embed_and_store(self):
        if not self.documents:
            raise ValueError("먼저 fetch_and_save()를 실행하세요.")
        texts = [doc.page_content for doc in self.documents]
        metadatas = [doc.metadata for doc in self.documents]
        ids = [f"stock_{i}_{self.today}" for i in range(len(self.documents))]
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"ChromaDB에 {len(texts)}건 저장 완료.")

    # 5. 질의 함수 (간단 요약)
    def query(self, question: str) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        # 간단 요약: 가장 유사한 5개 row를 연결
        answer = "\n".join([doc.page_content for doc in docs])
        return f"[주가 데이터 요약]\n{answer}"

if __name__ == "__main__":
    rag = StockPriceRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_stock_price_docs"
    )
    rag.fetch_and_save("005930")
    rag.embed_and_store()
    print(rag.query("최근 2달간 삼성전자 주가 변동 요약")) 