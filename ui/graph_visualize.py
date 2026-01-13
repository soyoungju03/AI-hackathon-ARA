# -*- coding: utf-8 -*-
"""
LangGraph 워크플로우 시각화 (개선 버전)
=======================================

이 스크립트는 실제 그래프 이미지를 생성합니다:
1. PNG/SVG 형식의 워크플로우 다이어그램 (LangGraph 내장 기능)
2. Mermaid 형식의 온라인 다이어그램
3. 상세한 HTML 보고서

필수 패키지:
    pip install langgraph langchain

선택 패키지 (더 나은 시각화):
    pip install pillow
    pip install graphviz

사용법:
    python visualize_workflow_improved.py
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_with_langgraph():
    """
    LangGraph의 내장 시각화 기능을 사용합니다.
    
    이것은 가장 간단하고 가장 정확한 방법입니다.
    당신의 워크플로우 구조를 PNG/SVG로 직접 변환합니다.
    """
    logger.info("=" * 80)
    logger.info("🎨 LangGraph 내장 시각화 기능 사용")
    logger.info("=" * 80)
    
    try:
        from app.graph.workflow import build_research_workflow
        
        logger.info("✓ 워크플로우 로드 성공")
        
        # 워크플로우 빌드
        workflow = build_research_workflow()
        logger.info("✓ 워크플로우 빌드 성공")
        
        # 그래프 가져오기
        graph = workflow.get_graph()
        logger.info("✓ 그래프 객체 획득 성공")
        
        # Mermaid 형식 생성 (텍스트 기반)
        try:
            mermaid_code = graph.draw_mermaid()
            
            # Mermaid 파일로 저장
            mermaid_output = "workflow_mermaid.md"
            
            mermaid_content = f"""# AI Research Assistant 워크플로우 다이어그램

## Mermaid 형식 다이어그램

```mermaid
{mermaid_code}
```

## 온라인에서 보기

위의 Mermaid 다이어그램을 다음 사이트에서 온라인으로 확인할 수 있습니다:

### 방법 1: Mermaid Live Editor
https://mermaid.live 에 접속하여 위의 코드를 복사해 붙여넣으세요.

### 방법 2: GitHub
이 파일을 GitHub에 커밋하면, README에 포함된 Mermaid 다이어그램이 자동으로 렌더링됩니다.

---

## 다이어그램 설명

이 워크플로우는 다음과 같이 작동합니다:

1. **시작**: 사용자 질문 수신
2. **분석**: LLM으로 질문 분석 및 키워드 추출
3. **첫 번째 Interrupt**: 키워드 확인 요청
4. **라우팅**: 사용자 응답에 따라 재분석 또는 다음 단계로 진행
5. **두 번째 Interrupt**: 논문 수 선택 요청
6. **검색**: arXiv에서 논문 검색
7. **필터링**: 연관성 평가 및 필터링
8. **요약**: LLM으로 논문 요약
9. **생성**: 최종 응답 생성

"""
            
            with open(mermaid_output, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
            
            logger.info(f"✓ Mermaid 다이어그램 파일 생성: {mermaid_output}")
            logger.info("  → 이 파일을 GitHub에 올리면 자동으로 렌더링됩니다")
            
        except Exception as e:
            logger.warning(f"⚠️  Mermaid 생성 실패: {str(e)}")
        
        # PNG 이미지 생성 시도
        try:
            logger.info("\n시도 중: PNG 이미지 생성...")
            
            png_data = graph.draw_mermaid_png()
            
            png_output = "workflow_diagram.png"
            
            with open(png_output, 'wb') as f:
                f.write(png_data)
            
            logger.info(f"✓ PNG 이미지 파일 생성: {png_output}")
            logger.info("  → 이미지 파일을 이미지 뷰어로 열어보세요")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  PNG 생성 실패: {str(e)}")
            logger.info("   원인: graphviz나 필요한 라이브러리가 설치되지 않았을 수 있습니다")
            logger.info("   대신 Mermaid Live Editor를 사용하세요: https://mermaid.live")
            return False
        
    except Exception as e:
        logger.error(f"✗ 시각화 실패: {str(e)}")
        return False


def create_simple_ascii_diagram():
    """
    간단한 ASCII 기반 다이어그램을 생성합니다.
    graphviz 없이도 터미널에서 바로 볼 수 있습니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 ASCII 기반 다이어그램 생성")
    logger.info("=" * 80)
    
    diagram = """
╔════════════════════════════════════════════════════════════════════════════╗
║               AI Research Assistant - 워크플로우 구조                      ║
║                     (LangGraph 기반)                                       ║
╚════════════════════════════════════════════════════════════════════════════╝


                            ┏━━━━━━━━━━━━━━┓
                            ┃    START     ┃
                            ┗━━━━┳━━━━━━━━┛
                                 │
                    ┌────────────▼────────────┐
                    │  receive_question      │
                    │  (사용자 질문 수신)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  analyze_question      │
                    │  (LLM으로 분석)        │
                    │  키워드 추출           │
                    └────────────┬────────────┘
                                 │
           ┏━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━┓
           ┃  request_keyword_confirmation          ┃
           ┃  🔴 첫 번째 Interrupt                   ┃
           ┃  (키워드가 맞는지 확인 요청)           ┃
           ┗━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┛
                │                │
                │                │
    ┌───────────▼──────┐  ┌──────▼───────────┐
    │ "다시" 선택      │  │ "확인" 선택      │
    │ (재분석)         │  │ (다음 단계)      │
    └───────────┬──────┘  └──────┬───────────┘
                │                │
                └────┬───────────┘
                     │
      ┌──────────────▼──────────────┐
      │process_keyword_confirmation │
      │_response                     │
      │(응답 처리)                   │
      └──────────────┬───────────────┘
                     │
      ┌──────────────▼──────────────┐
      │check_keyword_confirmation   │
      │_status                       │
      │(라우팅 노드 - 상태 검사)    │
      └──────────┬─────────────┬────┘
                 │             │
        ┌────────▼┐      ┌─────▼─────┐
        │재분석  │      │다음으로   │
        └────────┼┐     └─┬─────────┘
                 └┼───────┼─────────┐
                  │       │         │
                 ─┴───────▼─────────┼─
                          │        │
           ┏━━━━━━━━━━━━━━▼━━━━━━━━┓
           ┃  request_paper_count  ┃
           ┃  🔴 두 번째 Interrupt   ┃
           ┃  (논문 개수 선택: 1-10)┃
           ┗━━━━┳━━━━━━━━━━━━━━━━━━┛
                │
      ┌─────────▼──────────┐
      │사용자가 숫자 선택  │ (예: 5)
      └─────────┬──────────┘
                │
      ┌─────────▼──────────────────┐
      │process_paper_count_response│
      │(논문 수 응답 처리)         │
      └─────────┬──────────────────┘
                │
      ┌─────────▼──────────────┐
      │check_paper_count_status│
      │(라우팅 노드)           │
      └─────────┬──────────────┘
                │
      ┌─────────▼──────────┐
      │  search_papers     │
      │ (arXiv 논문 검색)  │
      └──────┬─────────┬───┘
             │         │
      ┌──────▼──┐  ┌───▼─────┐
      │검색 실패 │  │검색 성공 │
      └──────┬──┘  └───┬─────┘
             │         │
             │         │
      ┌──────▴────┬────▴──┐
      │            │       │
      ├────────────┤       │
      │             │      │
      ▼             │      ▼
    ┌──────────┐    │   ┌──────────────┐
    │generate_ │    │   │evaluate_     │
    │response  │◄───┘   │relevance     │
    │(응답    │        │(필터링)      │
    │생성)    │        └──────┬───────┘
    └──┬───────┘               │
       │            ┌──────────▼──────┐
       │            │summarize_papers│
       │            │(논문 요약)     │
       │            └──────────┬──────┘
       │                       │
       │         ┌─────────────▼────┐
       │         │ 최종 응답 생성   │
       │         └─────────────┬────┘
       │                       │
       └───────────┬───────────┘
                   │
             ┌─────▼─────┐
             │    END    │
             └───────────┘

════════════════════════════════════════════════════════════════════════════

📌 핵심 포인트:

🔴 두 개의 Interrupt 지점:
   1. request_keyword_confirmation: 키워드 확인
   2. request_paper_count: 논문 수 선택

🟠 두 개의 라우팅 노드:
   1. check_keyword_confirmation_status: 키워드 상태 검사
   2. check_paper_count_status: 논문 수 상태 검사

📊 상태 필드 (AgentState):
   • user_question: 사용자의 질문
   • extracted_keywords: 추출된 키워드 목록
   • paper_count: 검색할 논문 개수
   • waiting_for: "keyword_confirmation" | "paper_count_selection" | None
   • interrupt_stage: 0 (시작) | 1 (첫 번째 대기) | 2 (두 번째 대기)
   • waiting_for_user: True/False (사용자 입력 대기 여부)
   • final_response: 최종 응답
   • is_complete: 완료 여부

════════════════════════════════════════════════════════════════════════════
"""
    
    # 파일 저장
    output_path = "workflow_ascii_diagram.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(diagram)
    
    logger.info(f"✓ ASCII 다이어그램 파일 생성: {output_path}")
    
    # 터미널에도 출력
    print(diagram)
    
    return True


def create_detailed_html_report():
    """
    매우 상세한 HTML 보고서를 생성합니다.
    이것은 웹 브라우저에서 열 수 있는 완전한 문서입니다.
    """
    logger.info("\n" + "=" * 80)
    logger.info("📄 HTML 상세 보고서 생성")
    logger.info("=" * 80)
    
    html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Research Assistant - 워크플로우 분석</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        header p {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        .content {
            padding: 50px 40px;
        }
        
        .section {
            margin-bottom: 50px;
        }
        
        h2 {
            color: #667eea;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
            font-size: 2em;
        }
        
        .flow-chart {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 25px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            line-height: 1.8;
        }
        
        .node-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }
        
        .node {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .node:hover {
            transform: translateY(-8px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
            border-color: #764ba2;
        }
        
        .node h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .node p {
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }
        
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 12px;
            margin-right: 8px;
        }
        
        .badge-interrupt {
            background: #e74c3c;
            color: white;
        }
        
        .badge-routing {
            background: #f39c12;
            color: white;
        }
        
        .badge-input {
            background: #3498db;
            color: white;
        }
        
        .badge-process {
            background: #9b59b6;
            color: white;
        }
        
        .badge-output {
            background: #27ae60;
            color: white;
        }
        
        .state-table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .state-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            border-bottom: 3px solid #764ba2;
        }
        
        .state-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .state-table tr:hover {
            background: #f8f9fa;
        }
        
        .state-table code {
            background: #f1f1f1;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #d63031;
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .highlight-box h3 {
            color: #667eea;
            margin-bottom: 12px;
        }
        
        .highlight-box ul {
            margin-left: 25px;
            color: #555;
        }
        
        .highlight-box li {
            margin: 8px 0;
        }
        
        .interrupt-section {
            background: linear-gradient(135deg, #e74c3c15 0%, #c0392b15 100%);
            border-left: 5px solid #e74c3c;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .interrupt-section h3 {
            color: #e74c3c;
            margin-bottom: 15px;
        }
        
        .flow-step {
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .flow-step-number {
            background: #667eea;
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            flex-shrink: 0;
        }
        
        .flow-step-content {
            flex: 1;
        }
        
        .flow-step-content strong {
            color: #667eea;
        }
        
        footer {
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #999;
            border-top: 1px solid #ddd;
        }
        
        .tip {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI Research Assistant</h1>
            <p>LangGraph 워크플로우 완전 분석 보고서</p>
        </header>
        
        <div class="content">
            <!-- 개요 -->
            <div class="section">
                <h2>📋 워크플로우 개요</h2>
                <p>이 문서는 AI Research Assistant의 LangGraph 워크플로우를 완전하게 설명합니다. 시스템이 사용자의 질문으로부터 최종 응답을 생성하기까지의 전체 과정을 시각적으로 이해할 수 있습니다.</p>
                
                <div class="highlight-box">
                    <h3>🎯 핵심 특징</h3>
                    <ul>
                        <li><strong>12개의 노드</strong>로 구성된 복잡한 워크플로우</li>
                        <li><strong>2개의 Interrupt 포인트</strong>를 통한 Human-in-the-Loop 구조</li>
                        <li><strong>2개의 라우팅 노드</strong>를 통한 동적 경로 결정</li>
                        <li><strong>실시간 상태 관리</strong>로 현재 진행 상황 추적</li>
                    </ul>
                </div>
            </div>
            
            <!-- 노드 목록 -->
            <div class="section">
                <h2>🔵 모든 노드 목록</h2>
                <p>다음은 워크플로우를 구성하는 모든 노드들입니다:</p>
                
                <div class="node-grid">
                    <div class="node">
                        <h3>1️⃣ receive_question</h3>
                        <p>사용자의 질문을 수신하고 초기 분석을 시작합니다.</p>
                        <span class="badge badge-input">입력</span>
                    </div>
                    
                    <div class="node">
                        <h3>2️⃣ analyze_question</h3>
                        <p>LLM으로 질문을 분석하여 핵심 키워드, 의도, 도메인을 추출합니다.</p>
                        <span class="badge badge-process">처리</span>
                    </div>
                    
                    <div class="node">
                        <h3>3️⃣ request_keyword_confirmation</h3>
                        <p>추출된 키워드를 사용자에게 보여주고 확인을 받습니다.</p>
                        <span class="badge badge-interrupt">Interrupt 1️⃣</span>
                    </div>
                    
                    <div class="node">
                        <h3>4️⃣ process_keyword_confirmation_response</h3>
                        <p>사용자의 키워드 확인 응답("확인" 또는 "다시")을 처리합니다.</p>
                        <span class="badge badge-process">처리</span>
                    </div>
                    
                    <div class="node">
                        <h3>5️⃣ check_keyword_confirmation_status</h3>
                        <p>키워드 확인 상태를 검사하고 다음 경로를 결정합니다. 다시 분석하거나 다음 단계로 진행합니다.</p>
                        <span class="badge badge-routing">라우팅</span>
                    </div>
                    
                    <div class="node">
                        <h3>6️⃣ request_paper_count</h3>
                        <p>검색할 논문의 개수를 1-10 중에서 선택받습니다.</p>
                        <span class="badge badge-interrupt">Interrupt 2️⃣</span>
                    </div>
                    
                    <div class="node">
                        <h3>7️⃣ process_paper_count_response</h3>
                        <p>사용자가 선택한 논문 개수를 처리하고 상태에 저장합니다.</p>
                        <span class="badge badge-process">처리</span>
                    </div>
                    
                    <div class="node">
                        <h3>8️⃣ check_paper_count_status</h3>
                        <p>논문 개수가 올바르게 설정되었는지 검사하고 검색 단계로 진행합니다.</p>
                        <span class="badge badge-routing">라우팅</span>
                    </div>
                    
                    <div class="node">
                        <h3>9️⃣ search_papers</h3>
                        <p>arXiv API를 사용하여 추출된 키워드로 논문을 검색합니다.</p>
                        <span class="badge badge-process">액션</span>
                    </div>
                    
                    <div class="node">
                        <h3>🔟 evaluate_relevance</h3>
                        <p>검색된 논문들의 연관성을 평가하고 임계값 이상의 논문들을 선별합니다.</p>
                        <span class="badge badge-process">필터</span>
                    </div>
                    
                    <div class="node">
                        <h3>1️⃣1️⃣ summarize_papers</h3>
                        <p>선별된 논문들을 LLM으로 분석하여 구조화된 요약을 생성합니다.</p>
                        <span class="badge badge-process">처리</span>
                    </div>
                    
                    <div class="node">
                        <h3>1️⃣2️⃣ generate_response</h3>
                        <p>모든 정보를 종합하여 사용자에게 제공할 최종 응답을 생성합니다.</p>
                        <span class="badge badge-output">출력</span>
                    </div>
                </div>
            </div>
            
            <!-- 워크플로우 흐름 -->
            <div class="section">
                <h2>📊 워크플로우 실행 흐름</h2>
                <p>다음은 사용자 질문이 최종 응답으로 변환되는 전체 과정입니다:</p>
                
                <div class="flow-step">
                    <div class="flow-step-number">1</div>
                    <div class="flow-step-content"><strong>receive_question</strong> → 사용자 질문 수신</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">2</div>
                    <div class="flow-step-content"><strong>analyze_question</strong> → LLM이 질문 분석, 키워드 추출</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">3</div>
                    <div class="flow-step-content"><strong>request_keyword_confirmation</strong> → 🔴 첫 번째 Interrupt: "이 키워드가 맞나요?"</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">4</div>
                    <div class="flow-step-content">
                        <strong>사용자 응답 수신</strong>
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li>"확인" → 단계 6으로 진행</li>
                            <li>"다시" → 단계 2로 돌아가서 재분석</li>
                        </ul>
                    </div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">5</div>
                    <div class="flow-step-content"><strong>check_keyword_confirmation_status</strong> → 라우팅 결정</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">6</div>
                    <div class="flow-step-content"><strong>request_paper_count</strong> → 🔴 두 번째 Interrupt: "몇 개의 논문을 찾을까요? (1-10)"</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">7</div>
                    <div class="flow-step-content"><strong>사용자가 숫자 선택</strong> → 예: "5"</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">8</div>
                    <div class="flow-step-content"><strong>process_paper_count_response</strong> → 논문 수를 상태에 저장</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">9</div>
                    <div class="flow-step-content"><strong>check_paper_count_status</strong> → 상태 검사, 검색 단계로 진행</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">10</div>
                    <div class="flow-step-content"><strong>search_papers</strong> → arXiv에서 5개의 논문 검색</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">11</div>
                    <div class="flow-step-content"><strong>evaluate_relevance</strong> → 연관성으로 필터링</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">12</div>
                    <div class="flow-step-content"><strong>summarize_papers</strong> → 각 논문 요약 생성</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">13</div>
                    <div class="flow-step-content"><strong>generate_response</strong> → 최종 응답 생성</div>
                </div>
                
                <div class="flow-step">
                    <div class="flow-step-number">✅</div>
                    <div class="flow-step-content"><strong>END</strong> → 사용자에게 응답 반환</div>
                </div>
            </div>
            
            <!-- 두 가지 Interrupt -->
            <div class="section">
                <h2>🔔 두 가지 주요 Interrupt 포인트</h2>
                
                <div class="interrupt-section">
                    <h3>🔴 첫 번째 Interrupt: 키워드 확인</h3>
                    <p><strong>발생 노드:</strong> request_keyword_confirmation</p>
                    <p><strong>목적:</strong> 추출된 키워드가 사용자의 질문과 일치하는지 확인받습니다</p>
                    <p><strong>사용자가 선택할 수 있는 옵션:</strong></p>
                    <ul style="margin-left: 25px; margin-top: 10px;">
                        <li><strong>"확인"</strong> → request_paper_count로 진행</li>
                        <li><strong>"다시"</strong> → analyze_question으로 돌아가서 재분석</li>
                    </ul>
                    <div class="tip">
                        <strong>💡 팁:</strong> 이 Interrupt는 AI의 분석이 사용자의 의도와 일치하는지 확인하는 품질 검사 역할을 합니다.
                    </div>
                </div>
                
                <div class="interrupt-section" style="background: linear-gradient(135deg, #f39c1215 0%, #d68910-15 100%); border-left-color: #f39c12;">
                    <h3 style="color: #f39c12;">🟠 두 번째 Interrupt: 논문 수 선택</h3>
                    <p><strong>발생 노드:</strong> request_paper_count</p>
                    <p><strong>목적:</strong> 검색할 논문의 개수를 사용자로부터 선택받습니다</p>
                    <p><strong>사용자가 선택할 수 있는 옵션:</strong> 1부터 10 사이의 정수</p>
                    <p><strong>진행:</strong> 선택 후 즉시 search_papers로 진행하여 논문 검색이 시작됩니다</p>
                    <div class="tip">
                        <strong>💡 팁:</strong> 이 Interrupt는 API 비용과 응답 시간의 균형을 맞추기 위해 필요합니다.
                    </div>
                </div>
            </div>
            
            <!-- 상태 필드 -->
            <div class="section">
                <h2>📊 AgentState 상태 필드</h2>
                <p>워크플로우를 통해 전달되는 모든 상태 정보는 AgentState 타입으로 정의됩니다:</p>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">📥 입력 데이터</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>user_question</code></td>
                            <td>str</td>
                            <td>사용자가 입력한 원본 질문</td>
                        </tr>
                        <tr>
                            <td><code>session_id</code></td>
                            <td>str</td>
                            <td>각 사용자 대화를 추적하기 위한 고유 ID</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">🔍 분석 결과</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>extracted_keywords</code></td>
                            <td>List[str]</td>
                            <td>LLM이 추출한 핵심 검색 키워드</td>
                        </tr>
                        <tr>
                            <td><code>question_intent</code></td>
                            <td>str</td>
                            <td>사용자의 의도 (예: "최신 연구 동향")</td>
                        </tr>
                        <tr>
                            <td><code>question_domain</code></td>
                            <td>str</td>
                            <td>질문의 도메인 (예: "computer science")</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">⚙️ 검색 설정</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>paper_count</code></td>
                            <td>int (1-10)</td>
                            <td>검색할 논문의 개수 (사용자가 선택)</td>
                        </tr>
                        <tr>
                            <td><code>selected_sources</code></td>
                            <td>List[str]</td>
                            <td>검색 소스 (현재는 arXiv만)</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">📚 검색 결과</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>papers</code></td>
                            <td>List[Paper]</td>
                            <td>arXiv에서 검색한 모든 논문</td>
                        </tr>
                        <tr>
                            <td><code>relevant_papers</code></td>
                            <td>List[Paper]</td>
                            <td>연관성 필터링 후 선별된 논문</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">🔔 Interrupt & 대기 (가장 중요!)</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>waiting_for</code></td>
                            <td>Optional[str]</td>
                            <td>현재 대기 중인 것<br>• None (대기 아님)<br>• "keyword_confirmation" (키워드 확인 대기)<br>• "paper_count_selection" (논문 수 선택 대기)</td>
                        </tr>
                        <tr>
                            <td><code>interrupt_stage</code></td>
                            <td>int</td>
                            <td>현재 Interrupt 단계<br>• 0 (시작 또는 완료)<br>• 1 (첫 번째 Interrupt)<br>• 2 (두 번째 Interrupt)</td>
                        </tr>
                        <tr>
                            <td><code>user_response</code></td>
                            <td>Optional[str]</td>
                            <td>Interrupt에 대한 사용자의 응답</td>
                        </tr>
                        <tr>
                            <td><code>keyword_confirmation_response</code></td>
                            <td>Optional[str]</td>
                            <td>키워드 확인 응답<br>• "confirmed" (확인)<br>• "retry" (다시)</td>
                        </tr>
                        <tr>
                            <td><code>waiting_for_user</code></td>
                            <td>bool</td>
                            <td>사용자 입력 대기 중 여부</td>
                        </tr>
                    </tbody>
                </table>
                
                <h3 style="color: #667eea; margin-top: 25px; margin-bottom: 15px;">📤 출력</h3>
                <table class="state-table">
                    <thead>
                        <tr>
                            <th>필드명</th>
                            <th>타입</th>
                            <th>설명</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>final_response</code></td>
                            <td>str</td>
                            <td>사용자에게 제공할 최종 응답</td>
                        </tr>
                        <tr>
                            <td><code>error_message</code></td>
                            <td>Optional[str]</td>
                            <td>오류 발생 시 에러 메시지</td>
                        </tr>
                        <tr>
                            <td><code>is_complete</code></td>
                            <td>bool</td>
                            <td>워크플로우 완료 여부</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- 디버깅 가이드 -->
            <div class="section">
                <h2>🔧 디버깅 가이드</h2>
                
                <div class="highlight-box">
                    <h3>문제: 두 번째 Interrupt에서 반복된다</h3>
                    <p>종이 수를 선택했는데 검색이 시작되지 않고 다시 논문 수를 선택하도록 요청하는 경우:</p>
                    <ul>
                        <li><strong>확인할 필드:</strong> <code>waiting_for</code>, <code>waiting_for_user</code>, <code>paper_count</code></li>
                        <li><strong>원인 1:</strong> <code>waiting_for</code가 여전히 "paper_count_selection"인 경우</li>
                        <li><strong>원인 2:</strong> <code>waiting_for_user</code가 False로 설정되지 않은 경우</li>
                        <li><strong>원인 3:</strong> <code>paper_count</code가 유효한 범위 (1-10)를 벗어난 경우</li>
                        <li><strong>확인 방법:</strong> 터미널에서 <code>[CHECK_PAPER_COUNT_STATUS]</code> 로깅을 확인하세요</li>
                    </ul>
                </div>
                
                <div class="tip">
                    <strong>💡 로깅 확인 포인트:</strong>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><code>[CHECK_KEYWORD_CONFIRMATION_STATUS]</code> - 키워드 상태 검사</li>
                        <li><code>[CHECK_PAPER_COUNT_STATUS]</code> - 논문 수 상태 검사</li>
                        <li><code>[STAGE 0]</code> - 워크플로우 시작</li>
                        <li><code>[STAGE 1]</code> - 첫 번째 Interrupt 처리</li>
                        <li><code>[STAGE 2]</code> - 두 번째 Interrupt 처리</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <footer>
            <p>AI Research Assistant - LangGraph 워크플로우 완전 분석 보고서</p>
            <p>생성 일시: 2026-01-12</p>
        </footer>
    </div>
</body>
</html>
"""
    
    output_path = "workflow_detailed_report.html"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ HTML 상세 보고서 생성: {output_path}")
    logger.info("  → 웹 브라우저에서 열어보세요!")
    
    return True


def print_instructions():
    """최종 안내를 출력합니다."""
    logger.info("\n" + "=" * 80)
    logger.info("✅ 시각화 작업 완료!")
    logger.info("=" * 80)
    
    logger.info("\n📁 생성된 파일들:\n")
    
    files = [
        ("workflow_mermaid.md", "마크다운 형식 - GitHub에 올리면 자동 렌더링"),
        ("workflow_ascii_diagram.txt", "ASCII 텍스트 형식 - 터미널에서 바로 확인 가능"),
        ("workflow_detailed_report.html", "상세 HTML 보고서 - 🌐 웹 브라우저에서 열기!"),
    ]
    
    for filename, description in files:
        if Path(filename).exists():
            logger.info(f"  ✓ {filename:40} {description}")
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 추천 사항:")
    logger.info("=" * 80)
    logger.info("""
1. 📱 가장 상세한 정보 보기:
   → workflow_detailed_report.html을 웹 브라우저에서 열기

2. 📊 GitHub에 공유하기:
   → workflow_mermaid.md 를 README에 포함시키기

3. 🖥️  터미널에서 빠르게 확인:
   → cat workflow_ascii_diagram.txt

4. 🌐 온라인에서 Mermaid 다이어그램 보기:
   → https://mermaid.live 에서 코드 붙여넣기

""")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("🎨 LangGraph 워크플로우 시각화 도구 (개선 버전)")
    logger.info("=" * 80 + "\n")
    
    try:
        # 1. ASCII 다이어그램 생성
        create_simple_ascii_diagram()
        
        # 2. LangGraph 내장 시각화
        visualize_with_langgraph()
        
        # 3. 상세 HTML 보고서
        create_detailed_html_report()
        
        # 최종 안내
        print_instructions()
        
    except Exception as e:
        logger.error(f"✗ 시각화 작업 중 오류 발생: {str(e)}", exc_info=True)
        logger.info("\n이 오류를 무시해도 텍스트 기반 다이어그램은 생성되었습니다.")