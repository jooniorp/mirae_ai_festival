from dotenv import load_dotenv
from langchain_community.chat_models import ChatClovaX
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate
import re
import sys
from agent_memory import AgentMemory, run_memory_tool
import final_analysis

load_dotenv(override=True)

# 1. LLM 설정
llm = ChatClovaX(model="HCX-003", max_tokens=4096)  # CLOVA X 최대 토큰 제한

# 2. 메모리 인스턴스 생성
agent_memory = AgentMemory()

# 3. Tool 함수 정의
from NaverDiscussionRAGPipeline import NaverDiscussionRAGPipeline 
from ResearchRAGPipeline import ResearchRAGPipeline
from StockPriceRAGPipeline import StockPriceRAGPipeline

# 회사명과 종목코드 매핑
COMPANY_STOCK_MAP = {
    "삼성전자": "005930",
    "삼성": "005930",
    "sk하이닉스": "000660",
    "하이닉스": "000660",
    "sk": "000660",
    "lg에너지솔루션": "373220",
    "lg에너지": "373220",
    "lg": "373220",
    "네이버": "035420",
    "카카오": "035720",
    "현대차": "005380",
    "현대": "005380",
    "기아": "000270",
    "포스코홀딩스": "005490",
    "포스코": "005490",
    "lg화학": "051910",
    "lg화": "051910",
    "삼성바이오로직스": "207940",
    "삼성바이오": "207940",
    "카카오뱅크": "323410",
    "토스": "323410",
    "셀트리온": "068270",
    "아모레퍼시픽": "090430",
    "아모레": "090430",
    "신세계": "004170",
    "롯데케미칼": "051915",
    "롯데": "051915"
}

def extract_company_info(user_question: str):
    """사용자 질문에서 회사명과 종목코드를 추출"""
    question_lower = user_question.lower()
    
    # 회사명 찾기
    found_company = None
    for company, stock_code in COMPANY_STOCK_MAP.items():
        if company.lower() in question_lower:
            found_company = company
            break
    
    if not found_company:
        # 기본값으로 삼성전자 사용
        found_company = "삼성전자"
        stock_code = "005930"
        print(f"[경고] 질문에서 회사명을 찾을 수 없어 기본값 '{found_company}'를 사용합니다.")
    else:
        stock_code = COMPANY_STOCK_MAP[found_company]
        print(f"[회사 정보 추출] 회사: {found_company}, 종목코드: {stock_code}")
    
    return found_company, stock_code

def generate_tool_questions(company_name: str, user_question: str):
    """각 도구별로 적절한 질문 생성"""
    questions = {
        "NaverDiscussionRAGPipeline": f"{company_name}에 대한 최근 투자자 여론과 시장 관심도는 어때?",
        "ResearchRAGTool": f"최근 {company_name} 주가 분석",
        "StockPriceRAGTool": f"{company_name}의 현재 주가 상황과 최근 2달간의 가격 변화 분석"
    }
    return questions

def suggest_optimal_tools(user_question: str, agent_memory) -> str:
    """메모리에서 최적의 도구 순서 추천"""
    if len(agent_memory.memory_data["analyses"]) < 3:
        return ""
    
    # 유사한 질문들의 성공 패턴 분석
    question_keywords = set(user_question.split())
    successful_patterns = []
    
    for analysis in agent_memory.memory_data["analyses"][-20:]:  # 최근 20개 분석
        stored_keywords = set(analysis["question"].split())
        similarity = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
        
        if similarity > 0.2:  # 20% 이상 유사한 분석
            # 성공 여부 판단 (매수/매도/유지가 명확히 나온 경우)
            final_answer = analysis["final_answer"].lower()
            if any(keyword in final_answer for keyword in ["매수", "매도", "유지", "추천", "권장"]):
                successful_patterns.append({
                    "tools": analysis["tools_used"],
                    "similarity": similarity,
                    "answer": analysis["final_answer"]
                })
    
    if not successful_patterns:
        return ""
    
    # 가장 유사한 성공 패턴에서 도구 순서 추출
    best_pattern = max(successful_patterns, key=lambda x: x["similarity"])
    
    # 도구 순서를 문자열로 변환
    tool_order = " → ".join(best_pattern["tools"])
    
    return tool_order

def run_discussion_analysis(question: str, stock_code="005930"):
    pipeline = NaverDiscussionRAGPipeline(
        json_path="./data/discussion_comments.json",
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

def run_stock_price_analysis(question: str, stock_code="005930"):
    pipeline = StockPriceRAGPipeline(
        db_path="./chroma_langchain_db",
        collection_name="clovastudiodatas_stock_price_docs"
    )
    pipeline.fetch_and_save(stock_code)
    pipeline.embed_and_store()
    return pipeline.query(question)





# 4. Tool 등록 (이름-함수 매핑)
tool_map = {
    "NaverDiscussionRAGPipeline": run_discussion_analysis,
    "ResearchRAGTool": run_research_analysis,
    "StockPriceRAGTool": run_stock_price_analysis,
    "MemoryTool": run_memory_tool
}

tool_desc = """
- NaverDiscussionRAGPipeline: 종토방 여론 분석 (실시간 투자자 여론)
- ResearchRAGTool: 전문가 리서치 분석
- StockPriceRAGTool: 주가 데이터 분석 (최근 2달)
- MemoryTool: 메모리 관리 (save/recall/patterns)
"""

# 5. 프롬프트 템플릿 (최적화된 버전)
prompt_template = ChatPromptTemplate.from_template(
"""당신은 금융 투자 분석 전문가입니다.

사용자 질문: {input}

사용 가능한 도구: {tool_desc}

분석 전략: 종목 토론방 → 전문가 리서치 → 주가 데이터 순서로 분석하세요.
종목 토론방은 실시간 투자자 여론과 시장 관심도를 파악하는 데 집중하세요.

⚠️ 핵심 규칙:
1. 한 번에 하나의 Action만 출력 (절대 여러 개 동시 출력 금지!)
2. 절대로 Observation을 직접 생성하지 마세요 (시스템이 자동 제공)
3. Final Answer는 2개 이상 도구 실행 후에만
4. Action Input은 비워두세요 (자동 설정됨)
5. Action 실행 후 반드시 기다리세요!

답변 형식:
Thought: 다음 도구 선택 이유
Action: 도구이름
Action Input:

⚠️ 중요: 한 번에 하나의 Action만 출력하고 기다리세요!
⚠️ Final Answer는 모든 도구 실행 완료 후에만!
""")

# 6. LLM 호출 함수 (Rate Limit 방지)
def call_llm(history: str) -> str:
    import time
    import random
    
    # Rate Limit 방지를 위한 지연
    time.sleep(3 + random.uniform(0, 2))  # 3-5초 랜덤 대기
    
    try:
        response = llm.invoke(history)
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        if "429" in str(e):
            print("[API Rate Limit 감지] 10초 대기 후 재시도...")
            time.sleep(10)
            response = llm.invoke(history)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        else:
            raise e

# 7. 지능적 ReAct 루프 구현 (Token 제한에 영향받지 않는 완전한 분석)
def react_loop(user_question: str):
    # 회사 정보 추출
    company_name, stock_code = extract_company_info(user_question)
    tool_questions = generate_tool_questions(company_name, user_question)
    
    print(f"\n[분석 대상] 회사: {company_name} (종목코드: {stock_code})")
    print(f"[도구별 질문] {tool_questions}")
    
    # 🧠 메모리 기반 최적화: 유사한 이전 분석 확인
    print("\n[메모리 기반 최적화] 유사한 이전 분석 확인 중...")
    similar_analyses = agent_memory.recall_similar_analysis(user_question, top_k=2)
    if "유사한 분석 결과가 없습니다" not in similar_analyses:
        print("[메모리 힌트] 유사한 이전 분석 발견:")
        print(similar_analyses)
        
        # 이전 분석에서 성공한 도구 순서 추천
        optimal_tools = suggest_optimal_tools(user_question, agent_memory)
        if optimal_tools:
            print(f"[최적화 제안] 추천 도구 순서: {optimal_tools}")
            # 프롬프트에 최적화 힌트 추가
            optimization_hint = f"\n💡 메모리 기반 최적화 힌트: {optimal_tools} 순서로 분석하는 것을 권장합니다."
        else:
            optimization_hint = ""
    else:
        print("[메모리 힌트] 유사한 이전 분석이 없어 기본 전략을 사용합니다.")
        optimization_hint = ""
    
    history = prompt_template.format(input=user_question, tool_desc=tool_desc) + optimization_hint
    print("[LLM 프롬프트]\n" + history + "\n")
    step = 1
    action_observation_log = []
    used_tools = set()  # 사용된 도구 추적
    max_steps = 15  # 토큰 제한 고려하여 스텝 수 조정
    tool_quality_check = {}  # 도구별 품질 체크
    final_answer = None
    
    while step <= max_steps:
        print(f"\n[STEP {step}] LLM 호출 중...")
        
        # 토큰 사용량 최적화: 히스토리가 너무 길어지면 정리
        if len(history) > 8000:  # 8K 토큰 제한 고려
            print("[토큰 최적화] 히스토리 길이를 정리합니다.")
            lines = history.split('\n')
            # 프롬프트 부분은 유지하고, Action-Observation 부분만 정리
            prompt_end = 0
            for i, line in enumerate(lines):
                if line.startswith('Thought:') or line.startswith('Action:'):
                    prompt_end = i
                    break
            
            if prompt_end > 0:
                # 최근 6개의 Action-Observation만 유지
                action_obs_lines = lines[prompt_end:]
                if len(action_obs_lines) > 12:  # 6개 Action-Observation = 12줄
                    action_obs_lines = action_obs_lines[-12:]
                history = '\n'.join(lines[:prompt_end] + action_obs_lines)
        
        llm_output = call_llm(history)
        print(f"\n[LLM 출력]\n{llm_output}\n")

        # Thought, Action, Final Answer 파싱
        thought_match = re.search(r"Thought\s*:\s*(.+)", llm_output, re.DOTALL)
        
        # 여러 Action이 있는지 확인
        action_matches = re.findall(r"Action\s*:\s*(\w+)", llm_output)
        if len(action_matches) > 1:
            print(f"[경고] LLM이 {len(action_matches)}개의 Action을 한 번에 출력했습니다!")
            print("첫 번째 Action만 사용합니다.")
            action_match = re.search(r"Action\s*:\s*(\w+)", llm_output)
        else:
            action_match = re.search(r"Action\s*:\s*(\w+)", llm_output)
            
        action_input_match = re.search(r"Action Input\s*:?\s*(.*?)(?=\n|$)", llm_output, re.DOTALL)
        final_answer_match = re.search(r"Final Answer\s*:\s*(.+)", llm_output, re.DOTALL)

        # Thought 처리
        if thought_match:
            thought = thought_match.group(1).strip()
            print(f"\n[Thought] {thought}")
            history += f"\nThought: {thought}\n"
        
        # 가짜 Observation 감지 및 경고
        fake_observation_match = re.search(r"Observation\s*:\s*(.+)", llm_output, re.DOTALL)
        if fake_observation_match and not action_observation_log:
            print("[경고] LLM이 가짜 Observation을 생성했습니다!")
            print("실제 도구 실행 결과만 사용해야 합니다.")
            # 가짜 Observation 제거하고 다시 시도
            llm_output = re.sub(r"Observation\s*:\s*.*?(?=\n|$)", "", llm_output, flags=re.DOTALL)
            print(f"\n[수정된 LLM 출력]\n{llm_output}\n")
        
        # 가짜 Final Answer 감지 및 경고 (충분한 분석 전에 나온 경우)
        if final_answer_match and len(action_observation_log) < 2:
            print("[경고] LLM이 충분한 분석 전에 Final Answer를 생성했습니다!")
            print("최소 2개 이상의 도구 분석이 완료된 후에만 Final Answer를 출력해야 합니다.")
            # 가짜 Final Answer 제거
            llm_output = re.sub(r"Final Answer\s*:\s*.*?(?=\n|$)", "", llm_output, flags=re.DOTALL)
            print(f"\n[수정된 LLM 출력]\n{llm_output}\n")

        # 만약 Final Answer가 있고 충분한 분석이 완료되었으면 루프 종료
        if final_answer_match and len(action_observation_log) >= 2:  # 최소 2개 이상의 도구 실행 후에만 Final Answer 허용
            final_answer = final_answer_match.group(1).strip()
            print("\n[최종 답변]")
            print(final_answer)
            print("\n[분석에 사용된 툴 및 Observation 요약]")
            for idx, (tool, obs) in enumerate(action_observation_log, 1):
                print(f"{idx}. {tool}: {obs[:200]}{'...' if len(obs)>200 else ''}")
            break

        # Action이 있으면 도구 실행
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_input_match.group(1).strip() if action_input_match else ""
            
            print(f"\n[도구 실행] {tool_name} (입력: {tool_input})")
            tool_func = tool_map.get(tool_name)
            
            if tool_func is None:
                observation = f"[ERROR] 지원하지 않는 도구: {tool_name}"
            else:
                try:
                    # API Rate Limit 방지를 위한 지연
                    import time
                    time.sleep(2)  # 2초 대기
                    
                    if tool_name == "MemoryTool":
                        observation = tool_func(tool_input, user_question, agent_memory)
                    elif tool_name == "NaverDiscussionRAGPipeline":
                        # 종토방 분석: 회사별 질문과 종목코드 사용
                        question = tool_questions.get(tool_name, tool_input)
                        observation = tool_func(question, stock_code)
                    elif tool_name == "StockPriceRAGTool":
                        # 주가 분석: 회사별 질문과 종목코드 사용
                        question = tool_questions.get(tool_name, tool_input)
                        observation = tool_func(question, stock_code)
                    elif tool_name == "ResearchRAGTool":
                        # 리서치 분석: 회사별 질문 사용
                        question = tool_questions.get(tool_name, tool_input)
                        observation = tool_func(question)
                    else:
                        observation = tool_func(tool_input)
                    
                    # 도구 실행 결과 품질 체크 (MemoryTool 제외)
                    if tool_name != "MemoryTool":
                        quality_score = final_analysis._evaluate_tool_quality(tool_name, observation)
                        tool_quality_check[tool_name] = quality_score
                        print(f"[품질 점수] {tool_name}: {quality_score}/10")
                        
                        # 품질이 낮으면 재실행 요청
                        if quality_score < 5 and tool_name not in used_tools:
                            observation += f"\n[품질 경고] {tool_name}의 결과가 불충분합니다. 더 자세한 분석이 필요합니다."
                        
                except Exception as e:
                    observation = f"[ERROR] 도구 실행 중 오류: {e}"
                    if tool_name != "MemoryTool":
                        tool_quality_check[tool_name] = 0
            
            # Observation을 history에 추가
            history += f"\nObservation: {observation}\n"
            if tool_name != "MemoryTool":
                action_observation_log.append((tool_name, observation))
                used_tools.add(tool_name)
            print(f"\n[Observation 추가됨] {observation[:100]}...")
            print(f"[사용된 도구: {used_tools}]")
            
            # 분석 완성도 체크
            completion_status = final_analysis._check_analysis_completeness(action_observation_log, tool_quality_check)
            print(f"[분석 완성도] {completion_status}")
            
        else:
            print("[ERROR] LLM 출력에서 Action을 찾을 수 없습니다.")
            break
        step += 1
    
    # 최대 스텝에 도달했지만 Final Answer가 없는 경우
    if step > max_steps:
        print(f"\n[경고] 최대 스텝({max_steps})에 도달했습니다.")
        print(f"[사용된 도구: {used_tools}]")
        
        # 최종 종합 분석 실행
        if len(action_observation_log) >= 1:
            print("\n[최종 종합 분석 실행]")
            final_analysis_result = final_analysis.run_final_analysis(user_question, [obs for _, obs in action_observation_log], llm)
            final_answer = final_analysis_result.content if hasattr(final_analysis_result, 'content') else final_analysis_result
            print("\n[최종 투자 판단]")
            print(final_answer)
        
        print("\n[분석에 사용된 툴 및 Observation 요약]")
        for idx, (tool, obs) in enumerate(action_observation_log, 1):
            print(f"{idx}. {tool}: {obs[:200]}{'...' if len(obs)>200 else ''}")
    
    # 분석 결과를 메모리에 저장
    if final_answer and len(action_observation_log) >= 1:
        memory_result = agent_memory.add_analysis(
            user_question, 
            used_tools, 
            [obs for _, obs in action_observation_log], 
            final_answer
        )
        print(f"\n[메모리 저장] {memory_result}")
    
    # 토큰 사용량 최적화를 위한 히스토리 정리
    if len(history) > 10000:  # 히스토리가 너무 길어지면 정리
        print("[토큰 최적화] 히스토리 길이를 정리합니다.")
        # 최근 5개의 Action-Observation만 유지
        lines = history.split('\n')
        action_obs_sections = []
        for i, line in enumerate(lines):
            if line.startswith('Action:') or line.startswith('Observation:'):
                action_obs_sections.append(i)
        
        if len(action_obs_sections) > 10:
            keep_start = action_obs_sections[-10]
            history = '\n'.join(lines[:keep_start] + lines[keep_start:])



if __name__ == "__main__":
    # 다양한 회사로 테스트
    test_questions = [
        "삼성전자 지금 사도 될까?",
        "네이버 주식 어때?",
        "카카오 투자해도 될까?",
        "하이닉스 지금 매수 타이밍인가?"
    ]
    
    # 삼성전자로 테스트 (전체 시스템 검증)
    user_question = test_questions[0]  # "삼성전자 지금 사도 될까?"
    print(f"\n{'='*50}")
    print(f"테스트 질문: {user_question}")
    print(f"{'='*50}")
    react_loop(user_question)
