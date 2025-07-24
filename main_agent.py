from dotenv import load_dotenv
from langchain_community.chat_models import ChatClovaX
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
import re
import sys

load_dotenv(override=True)

# 1. LLM 설정
llm = ChatClovaX(model="HCX-003", max_tokens=2048)

# 2. Tool 함수 정의
from NaverDiscussionRAGPipeline import NaverDiscussionRAGPipeline 
from ResearchRAGPipeline import ResearchRAGPipeline

def run_discussion_analysis(question: str, stock_code="005930"):
    pipeline = NaverDiscussionRAGPipeline(
        json_path="./data/comments.json",
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_discussion_docs"
    )
    pipeline.crawl_comments(stock_code=stock_code)
    pipeline.segment_documents()
    pipeline.embed_and_store()
    return pipeline.query_opinion(question)

def run_research_analysis(question: str):
    pipeline = ResearchRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_research_docs"
    )
    pipeline.extract_from_pdf_folder("./pdf_downloads")
    pipeline.segment_documents()
    pipeline.embed_and_store()
    return pipeline.query(question)

# 3. Tool 등록 (이름-함수 매핑)
tool_map = {
    "DiscussionCrawlerTool": run_discussion_analysis,
    "ResearchRAGTool": run_research_analysis
}

tool_desc = """
- 종목 토론방 분석: DiscussionCrawlerTool
- 전문가 리서치 분석: ResearchRAGTool
"""

# 4. 프롬프트 템플릿 (Action/Observation/Final Answer 포맷 강조)
prompt_template = ChatPromptTemplate.from_template(
"""
당신은 금융 투자 분석 전문가입니다.

아래는 사용자의 질문입니다:
---------------------
{input}
---------------------

당신은 이를 위해 다음과 같은 도구들을 사용할 수 있습니다:
{tool_desc}

아래의 포맷을 반드시 따르세요:
- Action: 사용할 도구 이름
- Action Input: 도구에 전달할 입력값(질문, 종목코드 등)
- Observation: 도구 실행 결과(자동으로 제공됨)
- Thought: 다음 단계에 대한 생각(필요시)
- Final Answer: 최종 투자 판단 및 근거

반드시 Action/Observation 단계를 거친 후에만 Final Answer를 출력하세요.
한 번에 Action과 Final Answer를 동시에 출력하지 마세요.
""")

# 5. LLM 호출 함수
def call_llm(history: str) -> str:
    # LLM에 대화 이력(history)을 입력으로 전달
    response = llm.invoke(history)
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# 6. ReAct 루프 구현
def react_loop(user_question: str):
    # 1. 첫 프롬프트 생성
    history = prompt_template.format(input=user_question, tool_desc=tool_desc)
    print("[LLM 프롬프트]\n" + history + "\n")
    step = 1
    while True:
        print(f"\n[STEP {step}] LLM 호출 중...")
        llm_output = call_llm(history)
        print(f"\n[LLM 출력]\n{llm_output}\n")

        # 2. Action/Observation/Final Answer 파싱
        action_match = re.search(r"Action\s*:\s*(\w+)", llm_output)
        action_input_match = re.search(r"Action Input\s*:\s*(.+)", llm_output)
        final_answer_match = re.search(r"Final Answer\s*:\s*(.+)", llm_output, re.DOTALL)

        if final_answer_match:
            print("\n[최종 답변]")
            print(final_answer_match.group(1).strip())
            break
        elif action_match and action_input_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip()
            print(f"\n[도구 실행] {tool_name} (입력: {tool_input})")
            tool_func = tool_map.get(tool_name)
            if tool_func is None:
                observation = f"[ERROR] 지원하지 않는 도구: {tool_name}"
            else:
                try:
                    # 도구 입력값이 여러 개일 경우(예: 콤마로 구분) 처리
                    if tool_name == "DiscussionCrawlerTool" and ',' in tool_input:
                        q, code = tool_input.split(',', 1)
                        observation = tool_func(q.strip(), code.strip())
                    else:
                        observation = tool_func(tool_input)
                except Exception as e:
                    observation = f"[ERROR] 도구 실행 중 오류: {e}"
            # Observation을 history에 추가
            history += f"\nObservation: {observation}\n"
        else:
            print("[ERROR] LLM 출력에서 Action/Final Answer를 찾을 수 없습니다.\n출력:\n", llm_output)
            break
        step += 1

if __name__ == "__main__":
    user_question = "삼성전자 지금 사도 될까?"
    react_loop(user_question)
