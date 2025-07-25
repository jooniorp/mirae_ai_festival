import json
import os
from datetime import datetime

class AgentMemory:
    def __init__(self, memory_file="./data/memory.json"):
        self.memory_file = memory_file
        self.memory_data = self.load_memory()
    
    def load_memory(self):
        """메모리 파일 로드"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"analyses": [], "patterns": [], "last_updated": ""}
        return {"analyses": [], "patterns": [], "last_updated": ""}
    
    def save_memory(self):
        """메모리 파일 저장"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        self.memory_data["last_updated"] = datetime.now().isoformat()
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_data, f, ensure_ascii=False, indent=2)
    
    def add_analysis(self, question, tools_used, observations, final_answer):
        """분석 결과 저장"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "tools_used": list(tools_used),
            "observations": observations,
            "final_answer": final_answer
        }
        self.memory_data["analyses"].append(analysis)
        self.save_memory()
        return f"분석 결과가 메모리에 저장되었습니다. (총 {len(self.memory_data['analyses'])}개)"
    
    def recall_similar_analysis(self, question, top_k=3):
        """유사한 분석 결과 조회"""
        if not self.memory_data["analyses"]:
            return "저장된 분석 결과가 없습니다."
        
        # 간단한 키워드 매칭 (실제로는 더 정교한 유사도 계산 가능)
        question_keywords = set(question.split())
        similar_analyses = []
        
        for analysis in self.memory_data["analyses"][-10:]:  # 최근 10개만 검색
            stored_keywords = set(analysis["question"].split())
            similarity = len(question_keywords & stored_keywords) / len(question_keywords | stored_keywords)
            if similarity > 0.3:  # 30% 이상 유사
                similar_analyses.append((similarity, analysis))
        
        similar_analyses.sort(key=lambda x: x[0], reverse=True)
        
        if not similar_analyses:
            return "유사한 분석 결과가 없습니다."
        
        result = "유사한 이전 분석 결과:\n"
        for i, (similarity, analysis) in enumerate(similar_analyses[:top_k]):
            result += f"\n{i+1}. 유사도: {similarity:.1%}\n"
            result += f"   질문: {analysis['question']}\n"
            result += f"   답변: {analysis['final_answer'][:100]}...\n"
            result += f"   날짜: {analysis['timestamp'][:10]}\n"
        
        return result
    
    def get_analysis_patterns(self):
        """분석 패턴 추출"""
        if len(self.memory_data["analyses"]) < 3:
            return "분석 데이터가 부족하여 패턴을 추출할 수 없습니다."
        
        # 간단한 패턴 분석
        tool_usage = {}
        common_questions = {}
        
        for analysis in self.memory_data["analyses"]:
            for tool in analysis["tools_used"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            question_type = "주가" if "사도 될까" in analysis["question"] else "분석"
            common_questions[question_type] = common_questions.get(question_type, 0) + 1
        
        patterns = f"분석 패턴 요약:\n"
        patterns += f"• 총 분석 횟수: {len(self.memory_data['analyses'])}\n"
        patterns += f"• 가장 많이 사용된 도구: {max(tool_usage.items(), key=lambda x: x[1])[0]}\n"
        patterns += f"• 질문 유형 분포: {dict(common_questions)}\n"
        
        return patterns

def run_memory_tool(action: str, data: str = "", memory_instance=None):
    """메모리 저장/조회 도구"""
    if memory_instance is None:
        memory_instance = AgentMemory()
    
    if action == "save":
        return "메모리 저장 준비 완료. 분석 완료 후 자동 저장됩니다."
    elif action == "recall":
        return memory_instance.recall_similar_analysis(data if data else "삼성전자")
    elif action == "patterns":
        return memory_instance.get_analysis_patterns()
    else:
        return f"지원하지 않는 메모리 액션: {action}" 