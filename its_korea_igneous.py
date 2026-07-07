# -*- coding: utf-8 -*-
"""
================================================================================
 한반도 화성암의 시대적·공간적 변천사  ─  개념기반학습 지능형 튜터링 시스템(ITS)
--------------------------------------------------------------------------------
 - 교육과정 : 2022 개정 고등학교 '지구과학'
 - 단원     : Ⅱ. 지구의 역사와 한반도의 암석
 - 성취기준 : [12지구02-03] 변동대에서 마그마가 생성되고, 그 조성에 따라
              다양한 화성암이 생성됨을 설명할 수 있다.
 - 핵심 지침: "화성암의 종류보다는 화성암이 생성되는 고유의 환경이 가지는
              의미를 이해하는 데 중점을 두며, 한반도에 나타나는 대표적인 지형과
              연계해서 수업을 전개한다." (교육과정 '성취기준 적용 시 고려 사항')
--------------------------------------------------------------------------------
 구성 : 좌측(화성암 형성 시뮬레이터) + 우측(구성주의 AI 튜터 챗봇, Gemini)
 실행 : streamlit run its_korea_igneous.py
--------------------------------------------------------------------------------
 ⚠️ 이 파일을 수정하는 모든 AI 에이전트/개발자는 같은 폴더의 CLAUDE.md를
    먼저 읽어야 한다. CLAUDE.md는 이 코드의 교육과정 경계(§1)·지질학 사실
    (§2)·교수설계 불변식(§3)을 정의하는 가드레일이며, 새 세션이 시작되어도
    이 문서를 기준으로 작업해야 한다. CLAUDE.md §8의 "변경 전 체크리스트"를
    통과하지 못하는 수정은 하지 말 것.
================================================================================
"""

import os
import streamlit as st

# ------------------------------------------------------------------------------
# 0-1. .env 파일 로드(python-dotenv)  ─  누락 시 안전하게 무시(환경변수/사이드바로 대체)
# ------------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 의 GEMINI_API_KEY/GOOGLE_API_KEY 를 os.environ 에 적용
except Exception:
    pass

# ------------------------------------------------------------------------------
# 0-2. 외부 라이브러리(google-generativeai) 임포트  ─  누락 시 안전하게 처리
# ------------------------------------------------------------------------------
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    GENAI_IMPORT_ERROR = None
except Exception as _import_err:  # ImportError 등 모든 임포트 실패 포괄
    genai = None
    GENAI_AVAILABLE = False
    GENAI_IMPORT_ERROR = str(_import_err)

# ------------------------------------------------------------------------------
# 0-3. 결정 조직(텍스처) 시각화용 numpy/matplotlib/scipy 임포트  ─  누락 시 안전하게 처리
#      (그림 없이도 기존 텍스트 결과는 그대로 동작해야 한다)
#      scipy.spatial.Voronoi 는 결정 알갱이를 "빈틈없이 맞물린" 보로노이 테셀레이션으로
#      그리기 위해 사용한다(실제 화성암의 결정질 조직을 더 정확히 재현하기 위함).
# ------------------------------------------------------------------------------
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Streamlit 서버에는 GUI 백엔드가 없으므로 비대화형 백엔드 고정
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi
    VISUAL_AVAILABLE = True
    VISUAL_IMPORT_ERROR = None
except Exception as _visual_import_err:
    np = None
    plt = None
    Voronoi = None
    VISUAL_AVAILABLE = False
    VISUAL_IMPORT_ERROR = str(_visual_import_err)


# ==============================================================================
# 1. 전역 상수 및 환경 설정
# ==============================================================================

# Gemini 모델명 (요구사항: gemini-1.5-flash 또는 gemini-1.5-pro)
#  ※ Google이 구형 모델을 단계적으로 종료할 수 있습니다. 만약 아래 모델명으로
#    호출이 실패하면 'gemini-2.0-flash', 'gemini-2.5-flash' 등 현재 사용 가능한
#    모델명을 변경하려면 이 상수만 바꾸면 된다.
DEFAULT_MODEL = "gemini-2.5-flash"

# 화성암 분류 SiO2 임계값 (교과서 표 II-1 기준: 염기성 ≤52% < 중성 < 63% ≤ 산성)
SIO2_MAFIC_MAX = 52.0          # 이 값 이하 → 염기성암(현무암질)
SIO2_FELSIC_MIN = 63.0         # 이 값 이상 → 산성암(유문암질)

# 냉각(관입) 깊이 구간 (km)
DEPTH_SURFACE = 0              # 지표 분출 → 화산암(세립질)
DEPTH_SHALLOW_MAX = 3          # 0<깊이<3 → 천부 관입(반상질, 반심성암)
                              # 3 이상 → 심성암(조립질)

# 시뮬레이터 슬라이더 범위 (결정 텍스처 시각화의 연속 보간 기준값으로도 사용)
SIO2_SLIDER_MIN = 45.0
SIO2_SLIDER_MAX = 75.0
DEPTH_SLIDER_MAX = 12.0

# 결정 텍스처 시각화용 색상 (연속 보간 양 끝값 — §2-1 염기성/산성의 대표색)
MAFIC_RGB = (61, 58, 58)        # 어두운색(현무암질) 대표색 — 배경색에만 사용
FELSIC_RGB = (230, 219, 194)    # 밝은색(유문암질) 대표색 — 배경색에만 사용

# 광물별 특성 색 팔레트 (PPL 박편/육안 기준, (기반R, 기반G, 기반B, 개체별_변동폭) 정수 0~255)
# XPL 간섭색(무지개색 등)이 아닌 실제 육안/PPL 수준 색을 단순화해 교육용으로 표현한다.
# 어두운 광물(휘석·각섬석·흑운모)과 밝은 광물(사장석·석영·정장석)의 명암 대비가 핵심.
_MINERAL_PALETTE = {
    # (기반R, 기반G, 기반B, 변동폭) — 육안/PPL 기준
    # var≤14 : 채널 독립 흔들림을 억제해 명도 변화만 남기고 색조 무작위화를 방지
    "pyroxene":    ( 42,  40,  37,  8),   # 거의 흑색 (휘석, PPL/육안)
    "hornblende":  ( 38,  42,  32,  8),   # 거의 흑색~암녹색 (각섬석)
    "olivine":     ( 62,  70,  36, 12),   # 어두운 올리브 녹색 (감람석, 육안)
    "biotite":     ( 48,  34,  16,  8),   # 매우 어두운 갈색 (흑운모, 육안)
    "plagioclase": (158, 155, 152, 12),   # 회백색 (사장석, PPL/육안) — 어둡게 낮춤
    "quartz":      (162, 162, 170, 10),   # 유리광택 회색 (석영, 육안)
    "k_feldspar":  (188, 162, 142, 14),   # 살색~분홍 (정장석, 육안)
    "glass":       ( 28,  26,  24,  4),   # 흑색 유리질
}
# SiO2 구간별 광물 조성비 — 광물학 비율을 교육 목적으로 단순화
_MINERAL_MODES = {
    "mafic":        [("pyroxene", 0.40), ("olivine", 0.10), ("plagioclase", 0.50)],
    "intermediate": [("hornblende", 0.22), ("biotite", 0.08), ("plagioclase", 0.55), ("quartz", 0.15)],
    "felsic":       [("hornblende", 0.05), ("biotite", 0.10), ("plagioclase", 0.28), ("k_feldspar", 0.24), ("quartz", 0.33)],
}

# 교육과정 메타데이터 (UI 상단 표시용)
CURRICULUM_META = {
    "subject": "2022 개정 고등학교 「지구과학」",
    "unit": "Ⅱ. 지구의 역사와 한반도의 암석",
    "standard_code": "[12지구02-03]",
    "standard_text": (
        "변동대에서 마그마가 생성되고, 그 조성에 따라 "
        "다양한 화성암이 생성됨을 설명할 수 있다."
    ),
    "level_A": (
        "변동대에서 마그마가 생성되는 과정과 마그마의 조성에 따라 다양한 화성암이 "
        "생성됨을 이해하고, 화성암에 나타나는 다양한 지질 구조의 형성 과정을 설명하며, "
        "이를 한반도에 나타나는 대표적인 화성암 지형과 관련지을 수 있다."
    ),
    "big_idea": (
        "한반도에 분포하는 화성암은 그것이 생성될 당시의 변동대 환경(판 경계의 성격)과 "
        "마그마가 식은 환경을 기록하고 있으며, 이를 통해 한반도의 지질학적 변천 과정을 "
        "추론할 수 있다."
    ),
}


# ==============================================================================
# 2. AI 튜터 시스템 프롬프트
#    (기존 '마그마 성질·화산 모양' 루브릭을 '판 운동·생성 환경·한반도 분포'
#     루브릭으로 전면 교체 — 평가 로직/상태머신 구조는 그대로 유지)
# ==============================================================================
SYSTEM_PROMPT = """
[Role & Core Philosophy]
당신은 고등학교 '지구과학' 과목의 [Ⅱ. 지구의 역사와 한반도의 암석] 단원, 그중 성취기준 [12지구02-03] "변동대에서 마그마가 생성되고, 그 조성에 따라 다양한 화성암이 생성됨을 설명할 수 있다"를 담당하는 베테랑 교사이자, 구성주의(Constructivism) 기반의 AI 촉진자(Facilitator)입니다.
당신의 목적은 학생에게 지식을 주입하는 것이 아니라, 좌측 시뮬레이터(SiO2 함량·냉각 깊이)와 한반도의 실제 화성암 지형(제주도, 한탄강, 울릉도·독도, 설악산, 북한산, 월출산 등)을 단서로 삼아, 학생이 '자신만의 언어'로 핵심 인과관계를 스스로 깨닫고 아래의 일반화(Big Idea)에 도달하도록 정교한 비계(Scaffolding)를 제공하는 것입니다.

[Big Idea : 도달 목표 일반화]
"한반도에 분포하는 화성암은 그것이 생성될 당시의 변동대 환경(판 경계의 성격)과 마그마가 식은 환경을 기록하고 있으며, 이를 통해 우리는 한반도의 지질학적 변천 과정을 추론할 수 있다."
※ 2022 개정 교육과정은 '화성암의 종류 암기'가 아니라 '화성암이 생성되는 고유한 환경이 가지는 의미'를 한반도의 대표 지형과 연계해 이해하는 데 중점을 둡니다. 모든 발문은 이 방향을 향해야 합니다.

[Evaluation Pivot : 핵심 평가 축]
학생의 답변을 다음 두 가지 인과축을 기준으로 평가하십시오. 교과서적 전문 용어가 아니더라도 일상어(예: 끈적하다, 알갱이가 굵다, 펑펑 터졌다, 천천히 식었다)로 맥락이 통한다면 모두 정답으로 인정합니다.
1. 물질적(조성) 인과축 ─ '무엇으로 만들어졌나':
   변동대(판 경계)의 마그마 생성 환경  →  마그마의 조성(SiO2 함량)  →  화성암의 화학적 종류(현무암질 ↔ 유문암질 계열)
   · 발산 경계(해령·열곡대): 맨틀 물질 상승 → 압력 감소 → 부분 용융 → 현무암질 마그마
   · 수렴 경계(섭입대): 섭입하는 지각의 함수 광물에서 물 공급 → 용융점 하강 → 현무암질 마그마 / 상승한 마그마가 대륙 지각을 가열·용융 → 유문암질 마그마 / 마그마 혼합 또는 결정 분화(정출) → 안산암질 마그마
2. 시공간적(냉각·분포) 인과축 ─ '어디서·언제 식었나':
   마그마가 식은 위치(지표 ↔ 지하 심부)  →  냉각 속도  →  암석의 조직(세립질 ↔ 조립질, 화산암 ↔ 심성암)  →  한반도의 시대별 분포
   · 신생대 화산 활동 → 지표 분출·급랭 → 세립질 화산암(현무암) → 제주도·한탄강·울릉도·독도
   · 중생대 지각 변동 → 지하 심부 서랭 → 조립질 심성암(화강암) → 설악산·북한산·월출산 (이후 융기·삭박으로 지표에 노출)

[Learning State Machine : 학습 상태 정의] (성취수준 A~E와 연계)
학생의 이전 대화 내역과 현재 답변을 종합하여 아래 5가지 상태 중 하나로 판정하십시오.
- State 0 (인지적 공백): "모르겠어요", "힌트 주세요" 또는 핵심과 무관한 침묵 상태.
- State 1 (단편적 사실 나열): 인과관계 없이 단편적 사실("제주도는 현무암이다", "북한산은 화강암이다", "현무암은 검다")만 언급한 상태. (성취수준 E~D)
- State 2 (조성축 성공 / 시공간축 미달): 판 경계 환경 → 마그마 조성 → 암석 종류의 관계는 이해했으나, 냉각 위치 → 조직 → 한반도 시대별 분포의 원리는 아직 미흡한 상태.
- State 3 (시공간축 성공 / 조성축 미달): 냉각 속도 → 조직 → 화산암/심성암 → 한반도 분포의 관계는 이해했으나, 판 경계 환경 → 마그마 조성의 원리는 아직 미흡한 상태.
- State 4 (최종 일반화 달성, 성취수준 A): 두 가지 인과축을 모두 자신의 언어로 명확하게 연결하고, 이를 한반도의 대표 화성암 지형과 관련지어 설명할 수 있는 상태.

[Thinking Process Constraints : AI 내부 사고 프로세스]
최종 대답을 출력하기 전, 반드시 백그라운드에서 다음 3단계를 거쳐 생각하십시오. (이 사고 과정 자체는 절대 출력하지 마십시오.)
1. 학생의 답변에서 '의미적으로' 매칭되는 핵심 개념 추출 (어떤 인과축의 어느 고리를 건드렸는지 파악).
2. 현재 학생의 학습 상태(State 0 ~ 4) 판정.
3. 해당 State에 지정된 발문 지침을 확인하고 오직 '하나의 질문'만 선정.

[State별 상세 행동 지침 및 발문 루브릭]
1. State 0 (인지적 공백) 대응:
   - 지침: 한반도의 친숙한 지형을 직관적으로 대비시키는 비유적 질문을 던지십시오.
   - 발문 템플릿 예시: "괜찮아, 처음엔 어려울 수 있어! 그럼 쉬운 것부터 가 보자. 제주도 돌하르방을 만드는 돌은 까맣고 구멍이 송송 뚫려 있는데, 북한산 인수봉의 거대한 바위는 희고 알갱이가 굵직하잖아. 같은 '돌'인데 왜 이렇게 생김새가 다를까?"
2. State 1 (단편적 사실 나열) 대응:
   - 지침: 학생이 제시한 사실을 인정한 뒤, 그것이 일어난 '이유(Why)'로 한 걸음 들어가는 징검다리 질문을 제공하십시오.
   - 발문 템플릿 예시: "정확해! 제주도엔 현무암이, 북한산엔 화강암이 있지. 그렇다면 화강암의 알갱이(결정)가 굵직하다는 건, 그 마그마가 지표에서 빨리 식었다는 증거일까, 아니면 땅속 깊은 곳에서 아주 천천히 식었다는 증거일까?"
3. State 2 (조성축 성공 / 시공간축 미달) 대응:
   - 지침: 성공적으로 연결한 '환경 → 조성 → 암석' 인과관계를 구체적으로 칭찬하고, 시선을 '식는 과정(조직과 분포)'으로 돌려주십시오.
   - 발문 템플릿 예시: "와, 섭입대에서 물이 공급되면 현무암질 마그마가, 그 마그마가 대륙 지각을 녹이면 유문암질 마그마가 생긴다는 흐름을 정확히 짚었어! 그럼 이번엔 '식는 장소'로 시선을 옮겨 볼까? 똑같은 마그마라도 지하 깊은 곳에서 아주 천천히 식는다면, 돌 속 알갱이(결정)의 크기는 어떻게 될까?"
4. State 3 (시공간축 성공 / 조성축 미달) 대응:
   - 지침: 성공적으로 연결한 '냉각 → 조직 → 한반도 분포' 인과관계를 구체적으로 칭찬하고, 시선을 '마그마의 성분(생성 환경)'으로 돌려주십시오.
   - 발문 템플릿 예시: "맞아, 천천히 식을수록 결정이 굵어져 심성암이 되고, 한반도에선 중생대에 그렇게 만들어진 화강암이 융기해 드러났다는 걸 멋지게 연결했어! 그렇다면 애초에 제주도의 현무암질 마그마와 화강암을 만든 유문암질 마그마는, 각각 어떤 판 경계 환경에서 만들어졌기에 성분이 그렇게 달랐을까?"
5. State 4 (최종 도달, 성취수준 A) 대응:
   - 지침: 개념적 도달을 축하하고, '개념 전이(Transfer) 과제'를 최종 미션으로 부여하십시오.
   - 발문 템플릿 예시: "놀라워! 변동대의 환경이 마그마의 성분을 정하고, 식는 장소가 조직을 정하며, 그 결과가 한반도 곳곳의 암석으로 남았다는 핵심 원리를 모두 찾아냈구나. 그렇다면 마지막 미션이야. 신생대에 만들어진 한탄강의 현무암 평야와 중생대에 만들어진 설악산의 화강암 봉우리를, 각각 '언제·어떤 환경에서·어떻게 식어' 지금의 모습이 되었는지 한 친구에게 차례대로 설명해 줄 수 있겠니?"

[Strict Policy : 절대 엄수 규칙]
- 학생에게 정답(SiO2 임계값, 판 경계의 이름, 마그마 조성명, 화산암·심성암, 암석명, 지질 시대 등 핵심 용어 및 결론)을 절대로 먼저 제시하지 마십시오.
- 한 번의 출력에 의문문은 오직 하나만 포함해야 합니다.
- 학생의 대답에 "틀렸습니다", "오답입니다"와 같은 단정을 사용하지 마십시오.
- '[시뮬레이터 현재 상태]'로 시작하는 줄은 학생이 화면에서 보고 있는 참고용 시스템 정보이며, 학생의 발화가 아닙니다. 이 정보를 활용해 발문의 맥락을 잡되, 결코 정답을 흘리지 마십시오.
- 한반도의 대표 지형(제주도·한탄강·울릉도·독도의 현무암, 설악산·북한산·월출산의 화강암 등)을 직관적 단서로 적극 활용하십시오.
""".strip()


# ==============================================================================
# 3. 시뮬레이터 계산 로직  (각 함수는 축약 없이 완전한 형태로 작성)
# ==============================================================================

def classify_magma_composition(sio2):
    """
    SiO2 함량(%)에 따라 마그마의 조성을 분류한다. (교과서 표 II-1 기준)

    Parameters
    ----------
    sio2 : float
        마그마의 SiO2 함량(%).

    Returns
    -------
    dict
        magma_type : 마그마 조성명(현무암질/안산암질/유문암질 마그마)
        rock_class : 암석 분류명(염기성암(고철질)/중성암/산성암(규장질))
        rock_color : 대표 색(어두운색/중간색/밝은색)
        main_minerals : 대표 광물 설명
    """
    if sio2 <= SIO2_MAFIC_MAX:
        return {
            "magma_type": "현무암질 마그마",
            "rock_class": "염기성암(고철질암)",
            "rock_color": "어두운색",
            "main_minerals": "감람석·휘석·각섬석 등 어두운색 광물이 많음",
        }
    elif sio2 < SIO2_FELSIC_MIN:
        return {
            "magma_type": "안산암질 마그마",
            "rock_class": "중성암",
            "rock_color": "중간색",
            "main_minerals": "어두운색·밝은색 광물이 비교적 고르게 섞임",
        }
    else:
        return {
            "magma_type": "유문암질 마그마",
            "rock_class": "산성암(규장질암)",
            "rock_color": "밝은색",
            "main_minerals": "석영·정장석 등 밝은색 광물이 많음",
        }


def infer_generation_environment(magma_type):
    """
    마그마 조성으로부터 '주요 생성 환경(판 경계)'을 추론한다. (교과서 그림 II-21 기준)
    ※ 이 단원의 핵심인 '생성 환경의 의미'를 학생에게 노출하기 위한 출력.

    Parameters
    ----------
    magma_type : str
        classify_magma_composition() 가 반환한 마그마 조성명.

    Returns
    -------
    dict
        boundary : 대표 판 경계/생성 위치 요약
        mechanism : 마그마 생성 메커니즘 설명
    """
    if magma_type == "현무암질 마그마":
        return {
            "boundary": "발산 경계(해령·열곡대) 또는 수렴 경계(섭입대) 하부",
            "mechanism": (
                "발산 경계에서는 맨틀 물질이 상승하며 **압력이 낮아져** 부분 용융되고, "
                "섭입대에서는 섭입하는 지각의 함수 광물에서 **빠져나온 물이 맨틀에 공급**되어 "
                "용융점이 낮아지며 맨틀이 부분 용융됩니다."
            ),
        }
    elif magma_type == "유문암질 마그마":
        return {
            "boundary": "수렴 경계(섭입대) ─ 대륙 지각 하부",
            "mechanism": (
                "섭입대에서 상승한 현무암질 마그마가 **대륙 지각 하부를 가열**하여 "
                "대륙 지각이 부분 용융되면 유문암질 마그마가 생성됩니다."
            ),
        }
    else:  # 안산암질 마그마
        return {
            "boundary": "수렴 경계(섭입대) ─ 마그마 혼합·결정 분화",
            "mechanism": (
                "섭입대에서 **현무암질 마그마와 유문암질 마그마가 섞이거나**, "
                "현무암질 마그마가 식는 과정에서 먼저 만들어진 광물이 분리(정출)되어 "
                "조성이 변하면 안산암질 마그마가 생성됩니다."
            ),
        }


def classify_cooling(depth_km):
    """
    냉각(관입) 깊이로부터 냉각 속도·조직·산출 형태를 분류한다.

    Parameters
    ----------
    depth_km : int or float
        마그마가 식은 깊이(km). 0이면 지표 분출.

    Returns
    -------
    dict
        cooling_rate : 냉각 속도(빠름/보통/느림)
        texture : 암석의 조직(세립질/반상질/조립질)
        rock_mode : 산출 형태(화산암/반심성암/심성암)
        place_desc : 냉각 위치 설명
        is_volcanic : 화산암 계열 여부(bool)  ← 암석명 결정에 사용
    """
    if depth_km == DEPTH_SURFACE:
        return {
            "cooling_rate": "빠름",
            "texture": "세립질",
            "rock_mode": "화산암",
            "place_desc": "지표로 분출하거나 지표 가까운 곳에서 빠르게 냉각",
            "is_volcanic": True,
        }
    elif depth_km < DEPTH_SHALLOW_MAX:
        return {
            "cooling_rate": "보통",
            "texture": "반상질(세립질 바탕 + 큰 결정 혼재)",
            "rock_mode": "반심성암(천부 관입)",
            "place_desc": "지표에 비교적 가까운 천부에서 관입·냉각",
            "is_volcanic": True,  # 화산암 계열로 분류(세립질 바탕)
        }
    else:
        return {
            "cooling_rate": "느림",
            "texture": "조립질",
            "rock_mode": "심성암",
            "place_desc": "지하 깊은 곳에서 아주 서서히 냉각",
            "is_volcanic": False,
        }


def determine_rock_name(sio2, is_volcanic):
    """
    SiO2 함량과 산출 형태(화산암/심성암)로 최종 암석명을 결정한다. (교과서 표 II-1)

    Parameters
    ----------
    sio2 : float
        마그마의 SiO2 함량(%).
    is_volcanic : bool
        화산암 계열이면 True, 심성암이면 False.

    Returns
    -------
    str
        현무암 / 안산암 / 유문암 / 반려암 / 섬록암 / 화강암 중 하나.
    """
    # 조성 구분 (mafic / intermediate / felsic)
    if sio2 <= SIO2_MAFIC_MAX:
        composition = "mafic"
    elif sio2 < SIO2_FELSIC_MIN:
        composition = "intermediate"
    else:
        composition = "felsic"

    category = "volcanic" if is_volcanic else "plutonic"

    rock_table = {
        ("mafic", "volcanic"): "현무암",
        ("intermediate", "volcanic"): "안산암",
        ("felsic", "volcanic"): "유문암",
        ("mafic", "plutonic"): "반려암",
        ("intermediate", "plutonic"): "섬록암",
        ("felsic", "plutonic"): "화강암",
    }
    return rock_table[(composition, category)]


def get_korea_context(rock_name):
    """
    최종 암석명에 따른 '한반도 대표 산출지·지질 시대·형성 과정·관련 지질 구조'를 반환한다.
    (교과서 그림 II-22 및 교육과정 '한반도 대표 지형 연계' 지침 기반)

    ※ 교육과정·교과서가 '대표 사례'로 강조하는 것은 신생대의 현무암과 중생대의
       화강암입니다. 나머지 암석은 분류표상의 종류로서 학습 맥락을 보충하되,
       대표 산출지의 강조 정도를 달리하여 정확성을 유지합니다.

    Returns
    -------
    dict
        era, locality, process, structure, emphasized(bool)
    """
    data = {
        "현무암": {
            "era": "신생대",
            "locality": "제주특별자치도, 한탄강 일대, 울릉도·독도",
            "process": "신생대에 일어난 화산 활동으로 현무암질 마그마가 지표로 분출한 뒤 "
                       "빠르게 식어 생성되었습니다.",
            "structure": "주상 절리 ─ 용암이 빠르게 식으며 부피가 수축해 만들어진 "
                         "다각형 기둥 모양의 절리(화산암에서 잘 발달).",
            "emphasized": True,
            "landmarks": [
                "**제주 중문대포 주상절리대** — 수십만 년 전 현무암질 용암이 바다와 만나 급랭하며 "
                "수축·균열된 6각형 기둥이 해안 절벽을 이룸.",
                "**제주 성산일출봉** — 얕은 바다 속 화산 폭발(수성화산)로 쌓인 현무암질 응회구. "
                "유네스코 세계자연유산.",
                "**한탄강 재인폭포** — 약 15만 년 전 분출한 현무암 용암이 굳어 만든 협곡과 "
                "주상절리 폭포.",
                "**울릉도·독도** — 신생대 화산활동으로 형성된 섬. 울릉도는 조면안산암 계열도 "
                "포함하며, 독도의 기반암도 현무암·안산암질 화산암.",
            ],
        },
        "화강암": {
            "era": "중생대",
            "locality": "설악산(비선대), 북한산(인수봉), 월출산(구정봉) "
                        "─ 쥐라기 대보 화강암·백악기 불국사 화강암 계열",
            "process": "중생대의 지각 변동으로 생성된 유문암질 마그마가 지하 깊은 곳에서 "
                       "서서히 식어 생성되었고, 이후 융기하여 위를 덮고 있던 암석이 "
                       "제거(삭박)되면서 지표에 드러났습니다.",
            "structure": "판상 절리 ─ 지하 심부의 암석이 융기·삭박으로 압력이 감소하며 "
                         "만들어진 얇은 판 모양의 절리(심성암에서 잘 발달).",
            "emphasized": True,
            "landmarks": [
                "**북한산 인수봉(서울·경기)** — 중생대에 지하에서 굳은 화강암이 융기·삭박으로 "
                "드러난 절벽. 판상 절리가 발달해 있음.",
                "**설악산 비선대(강원)** — 화강암이 오랜 풍화·침식을 받아 만들어진 경관. "
                "대보 화강암 계열.",
                "**월출산 구정봉(전남)** — 화강암에 발달한 판상 절리와 풍화 구멍(타포니)이 "
                "독특한 경관을 만듦.",
                "**경주 일대** — 백악기 불국사 화강암이 기반암을 이루며, 석굴암·불국사 등 "
                "문화재의 돌감(석재)으로도 활용됨.",
            ],
        },
        "안산암": {
            "era": "중생대(주로 백악기)",
            "locality": "경상 분지(경남·경북) 일대 백악기 화산암류, 울릉도(조면안산암 계열)",
            "process": "섭입대 환경에서 마그마 혼합 또는 결정 분화로 생성된 안산암질 "
                       "마그마가 지표 부근에서 식어 생성됩니다.",
            "structure": "중성 화산암(분류표 기준) — 교과서 대표 사례는 현무암·화강암이며, "
                         "안산암은 분류표 학습용으로 다룹니다.",
            "emphasized": False,
            "landmarks": [
                "**울릉도(조면안산암)** — 울릉도의 암석은 엄밀히는 알칼리성 '조면안산암' 계열로, "
                "안산암질 화성암의 한 종류. 나리분지 알봉이 대표적.",
                "**경상 분지 일대** — 중생대 백악기에 활발한 화산활동이 있었던 지역으로 "
                "안산암·유문암 등 화산암류가 넓게 분포.",
                "한반도에서 '안산암'이라고 명명되는 전형적인 관광 명소는 현무암·화강암에 비해 "
                "드뭅니다.",
            ],
        },
        "유문암": {
            "era": "중생대",
            "locality": "한라산 모세왓(제주), 경상 분지 일대 산성 화산암류",
            "process": "대륙 지각의 용융 등으로 만들어진 유문암질 마그마가 지표 부근에서 "
                       "식어 생성됩니다. 같은 마그마가 지하에서 천천히 식으면 화강암이 됩니다.",
            "structure": "산성 화산암(분류표 기준) — 유문암질 마그마의 냉각 위치에 따라 "
                         "지표→유문암, 지하→화강암으로 나뉩니다.",
            "emphasized": False,
            "landmarks": [
                "**한라산 모세왓(제주, 천연기념물 제455호)** — 한라산 백록담 남서쪽에 약 2.3 km "
                "구간의 유문암질 각력암 지대. 일반적인 관광 명소보다는 지질학적 명소에 가까움.",
                "**경상 분지 일대** — 백악기 화산활동으로 형성된 유문암·응회암이 경남·경북 "
                "여러 지역에 분포.",
                "한반도에서 유문암 자체가 전면에 드러나는 유명 지형은 매우 드물며, "
                "같은 마그마로 만들어진 **화강암**(설악산·북한산 등)이 훨씬 더 넓게 노출됩니다.",
            ],
        },
        "반려암": {
            "era": "—",
            "locality": "강원 화천·경기 양평 일대 소규모 관입체 등",
            "process": "현무암질 마그마가 지하 깊은 곳에서 서서히 식어 생성됩니다. "
                       "같은 마그마가 지표에서 빠르게 식으면 현무암이 됩니다.",
            "structure": "염기성 심성암(분류표 기준) — 지표에서는 현무암, 지하에서는 "
                         "반려암이 됩니다.",
            "emphasized": False,
            "landmarks": [
                "한반도에서 반려암이 대규모로 노출되어 관광 명소가 된 사례는 거의 없습니다.",
                "**강원 화천·경기 양평** 등 일부 지역에 소규모 관입체로 분포하나, "
                "주로 지질 학술 조사 대상입니다.",
                "반려암은 현무암과 같은 마그마 조성에서 출발하므로, 냉각 위치(지표 vs 지하 심부)가 "
                "암석 종류를 결정한다는 점이 핵심 학습 내용입니다.",
            ],
        },
        "섬록암": {
            "era": "—",
            "locality": "경북 영덕 화강섬록암 해안(국가지질공원) 등",
            "process": "안산암질 마그마가 지하 깊은 곳에서 서서히 식어 생성됩니다. "
                       "같은 마그마가 지표에서 빠르게 식으면 안산암이 됩니다.",
            "structure": "중성 심성암(분류표 기준) — 지표에서는 안산암, 지하에서는 "
                         "섬록암이 됩니다.",
            "emphasized": False,
            "landmarks": [
                "**경북 영덕 해안(동해안 국가지질공원)** — '화강섬록암' 해안으로, "
                "섬록암보다 산성 성분이 약간 높은 중간 계열 심성암. 해식애·파식대지 등 "
                "침식 지형이 발달.",
                "한반도에서 전형적인 '섬록암' 지형 명소는 매우 드뭅니다.",
                "섬록암은 안산암과 같은 마그마 조성에서 출발하므로, 냉각 위치(지표 vs 지하 심부)가 "
                "암석 종류를 결정한다는 점이 핵심 학습 내용입니다.",
            ],
        },
    }
    return data[rock_name]


def compute_simulation(sio2, depth_km):
    """
    두 슬라이더 값(SiO2, 깊이)으로부터 모든 결과를 종합 계산하여 하나의 dict로 반환한다.
    (좌측 시뮬레이터 출력과 우측 챗봇 컨텍스트 모두 이 결과를 공유한다.)
    """
    comp = classify_magma_composition(sio2)
    env = infer_generation_environment(comp["magma_type"])
    cool = classify_cooling(depth_km)
    rock_name = determine_rock_name(sio2, cool["is_volcanic"])
    korea = get_korea_context(rock_name)

    results = {
        "sio2": sio2,
        "depth_km": depth_km,
        # 물질적(조성) 인과축
        "magma_type": comp["magma_type"],
        "rock_class": comp["rock_class"],
        "rock_color": comp["rock_color"],
        "main_minerals": comp["main_minerals"],
        "env_boundary": env["boundary"],
        "env_mechanism": env["mechanism"],
        # 시공간적(냉각·분포) 인과축
        "cooling_rate": cool["cooling_rate"],
        "texture": cool["texture"],
        "rock_mode": cool["rock_mode"],
        "place_desc": cool["place_desc"],
        # 결과
        "rock_name": rock_name,
        "era": korea["era"],
        "locality": korea["locality"],
        "process": korea["process"],
        "structure": korea["structure"],
        "emphasized": korea["emphasized"],
        "landmarks": korea["landmarks"],
    }
    return results


def _clip_polygon_to_unit_square(poly_points):
    """
    Sutherland-Hodgman 알고리즘으로 임의의 볼록 다각형을 단위 정사각형 [0,1]x[0,1]에 클리핑한다.
    보로노이 셀은 항상 볼록(convex)이므로 이 알고리즘으로 충분하다.

    Returns
    -------
    numpy.ndarray or None
        클리핑된 정점 배열(시계/반시계 순서 유지). 결과가 삼각형 미만이면 None.
    """
    def _clip_edge(points, inside_fn, intersect_fn):
        if not points:
            return []
        result = []
        n = len(points)
        for i in range(n):
            cur = points[i]
            prev = points[i - 1]
            cur_in = inside_fn(cur)
            prev_in = inside_fn(prev)
            if cur_in:
                if not prev_in:
                    result.append(intersect_fn(prev, cur))
                result.append(cur)
            elif prev_in:
                result.append(intersect_fn(prev, cur))
        return result

    def _intersect_x(a, b, x_val):
        ax, ay = a
        bx, by = b
        t = 0.0 if bx == ax else (x_val - ax) / (bx - ax)
        return (x_val, ay + t * (by - ay))

    def _intersect_y(a, b, y_val):
        ax, ay = a
        bx, by = b
        t = 0.0 if by == ay else (y_val - ay) / (by - ay)
        return (ax + t * (bx - ax), y_val)

    pts = [tuple(p) for p in poly_points]
    pts = _clip_edge(pts, lambda p: p[0] >= 0.0, lambda a, b: _intersect_x(a, b, 0.0))
    pts = _clip_edge(pts, lambda p: p[0] <= 1.0, lambda a, b: _intersect_x(a, b, 1.0))
    pts = _clip_edge(pts, lambda p: p[1] >= 0.0, lambda a, b: _intersect_y(a, b, 0.0))
    pts = _clip_edge(pts, lambda p: p[1] <= 1.0, lambda a, b: _intersect_y(a, b, 1.0))
    if len(pts) < 3:
        return None
    return np.array(pts)


def _voronoi_cells_with_neighbors(points):
    """
    scipy.spatial.Voronoi 는 경계가 없는(무한히 뻗는) 영역을 만들 수 있으므로,
    입력 점들을 단위 정사각형의 4변에 대해 거울 대칭시킨 "유령 점"을 추가해
    모든 실제 점의 보로노이 영역이 정사각형 안에서 닫히도록 강제한다(표준 기법).

    단순히 셀 모양만 구하는 것이 아니라, 각 변이 "어느 이웃 알갱이와 맞닿아
    있는지"도 함께 돌려준다(scipy의 ridge_points/ridge_vertices를 이용한 실제
    인접 관계). 이는 한 알갱이 안에서도 변마다 자형(곧은 결정면)/타형(불규칙한
    면)이 섞여 나타나는 실제 결정형을 표현하려면, 그 변을 "곧게 그릴지 불규칙하게
    그릴지"를 양쪽 알갱이의 결정형 값으로부터 함께 결정해야 하기 때문이다.

    Returns
    -------
    (list of (numpy.ndarray or None), list of (list or None))
        cell_polygons[i] : 점 i 의 (아직 직선인) 보로노이 셀 정점 배열.
        cell_neighbors[i] : cell_polygons[i] 의 각 변(k번째 정점→k+1번째 정점)에
            대응하는 이웃 점 인덱스 리스트. 캔버스 경계에 닿아 실제 이웃이 없으면 None.
        둘 다 영역을 만들 수 없는 경우(축퇴 등) 해당 위치는 None.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return [], []

    mirror_left = points.copy(); mirror_left[:, 0] = -points[:, 0]
    mirror_right = points.copy(); mirror_right[:, 0] = 2.0 - points[:, 0]
    mirror_bottom = points.copy(); mirror_bottom[:, 1] = -points[:, 1]
    mirror_top = points.copy(); mirror_top[:, 1] = 2.0 - points[:, 1]
    all_points = np.vstack([points, mirror_left, mirror_right, mirror_bottom, mirror_top])

    try:
        vor = Voronoi(all_points)
    except Exception:
        return [None] * n, [None] * n

    # 변(두 보로노이 정점으로 이루어진 ridge) -> 그 변을 사이에 둔 두 점 인덱스.
    ridge_lookup = {}
    for (pa, pb), (va, vb) in zip(vor.ridge_points, vor.ridge_vertices):
        if va == -1 or vb == -1:
            continue
        ridge_lookup[frozenset((va, vb))] = (int(pa), int(pb))

    cell_polygons = []
    cell_neighbors = []
    for i in range(n):
        try:
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            if not region or -1 in region:
                cell_polygons.append(None)
                cell_neighbors.append(None)
                continue
            verts = vor.vertices[region]
            centroid = verts.mean(axis=0)
            angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
            order = np.argsort(angles)
            ordered_region = [region[k] for k in order]
            ordered_verts = verts[order]

            m = len(ordered_region)
            neighbors = []
            for k in range(m):
                pair = ridge_lookup.get(frozenset((ordered_region[k], ordered_region[(k + 1) % m])))
                if pair is None:
                    neighbors.append(None)
                    continue
                other = pair[0] if pair[1] == i else (pair[1] if pair[0] == i else None)
                # 유령(거울) 점이면(인덱스 >= n) 실제 이웃 알갱이가 아니라 캔버스
                # 경계와 맞닿은 변이므로 이웃 없음(None)으로 처리한다.
                neighbors.append(other if (other is not None and other < n) else None)

            cell_polygons.append(ordered_verts)
            cell_neighbors.append(neighbors)
        except Exception:
            cell_polygons.append(None)
            cell_neighbors.append(None)
    return cell_polygons, cell_neighbors


def _resolve_edge_amplitude(cell_idx, neighbor_idx, habit_by_point, is_glass_by_point,
                             low_amp, high_amp, glass_amp, base_seed):
    """
    한 변의 굴곡 폭(부호 포함)과 스타일 분류("bold"/"medium"/"faint"/"glass")를
    함께 정한다. 스타일 분류는 호출부(_build_wavy_polygon)가 테두리 선 굵기·
    진하기를 다르게 그리는 데 쓴다 — 굴곡 폭 차이만으로는 사람 눈에 거의 안
    띄어서(실측 후 확인된 문제), 결정면처럼 보이는 변은 굵고 진하게, 불규칙한
    변은 거의 안 보이게 옅게 그려 자형/반자형/타형의 차이를 한눈에 알아볼 수
    있게 한다.

    처음엔 변마다 "이 변을 결정면처럼 그릴지"를 동전 던지듯 무작위로 정했더니,
    한 알갱이 둘레에 진한 선과 흐린 선이 의미 없이 뒤섞여 "금이 간 것처럼"
    보이는 문제가 있었다(실측 후 확인된 문제). 결정면은 "둘 중 더 잘 발달한
    알갱이 쪽"이 보이는 것이므로, 무작위 동전 던지기 대신 두 알갱이의 결정형
    값 중 더 큰 값(max)을 결정론적으로 쓴다 — 그 결과 자형 알갱이는 이웃이
    무엇이든 자기 둘레 전체가 한결같이 또렷하고, 타형(흐릿한 바탕) 알갱이는
    자형 이웃과 맞닿은 그 변만 도드라지고 나머지(타형 이웃과 맞닿은 변)는
    흐려진다 — 실제 박편 사진에서 "또렷한 결정 몇 개 + 그 사이를 채운 흐릿한
    바탕"으로 읽히는 것과 같은 효과를 낸다.

    key 를 (min(cell_idx, neighbor_idx), max(...)) 로 정규화하므로, 이 변을
    공유하는 두 알갱이가 각자의 관점에서 이 함수를 호출해도 완전히 같은
    난수열을 얻어 — 같은 굴곡 폭·부호가 나와 셀 사이에 틈이나 겹침이 생기지 않는다.
    """
    habit_a = float(habit_by_point[cell_idx])
    if neighbor_idx is None:
        key = (cell_idx, -1)
        habit_b = habit_a
    else:
        key = (min(cell_idx, neighbor_idx), max(cell_idx, neighbor_idx))
        habit_b = float(habit_by_point[neighbor_idx])

    edge_seed = hash((key, base_seed)) & 0xFFFFFFFF
    edge_rng = np.random.default_rng(seed=edge_seed)
    sign = float(edge_rng.choice([-1.0, 1.0]))
    magnitude_factor = float(edge_rng.uniform(0.7, 1.0))

    is_glass_edge = bool(is_glass_by_point[cell_idx]) or (
        neighbor_idx is not None and bool(is_glass_by_point[neighbor_idx])
    )
    if is_glass_edge:
        # 유리질(비정질)은 결정면 자체가 없으므로 거의 매끈하게(아주 약한 불규칙함만) 그린다.
        return glass_amp * sign * magnitude_factor, "glass"

    strength = max(habit_a, habit_b)
    base_amp = low_amp + (high_amp - low_amp) * (1.0 - strength)
    if strength >= 0.7:
        style = "bold"
    elif strength >= 0.42:
        style = "medium"
    else:
        style = "faint"

    return base_amp * sign * magnitude_factor, style


def _wavy_edge_points(p_a, p_b, signed_amplitude_fraction):
    """
    한 변(p_a→p_b)의 중간에 완만한 굴곡(저주파 사인 굽이)을 주는 점 2개를 계산한다.
    굴곡의 크기·방향(signed_amplitude_fraction)은 호출하는 쪽(_resolve_edge_amplitude)에서
    이미 두 이웃 알갱이의 결정형 값으로부터 대칭적으로 정해서 넘겨준다 — 이 함수는
    그 값을 "양 끝점 좌표만으로 정해지는 변 방향"에 일관되게 투영하는 역할만 한다.

    핵심 설계: 이웃한 두 보로노이 셀은 이 변을 정확히 공유한다(같은 좌표). 양 끝점
    p_a, p_b 를 "작은 점 → 큰 점" 순서로 정규화해 변의 방향(법선)을 고정하기 때문에,
    이웃 셀이 같은 변을 반대 방향으로 순회하며 이 함수를 호출해도 완전히 동일한
    좌표의 굴곡점을 얻는다 — 그 결과 셀 사이에 빈틈이나 겹침 없이 맞물린 채로
    변만 자연스럽게 굽는다(실제 암석의 봉합선 같은 결정 경계 재현). 변 양 끝
    (t=0, t=1)에서는 굴곡이 0이므로 꼭짓점에서 이웃 변과 매끄럽게 이어져 별 모양
    같은 인공적인 뾰족함이 생기지 않는다.
    """
    key_a = (round(float(p_a[0]), 5), round(float(p_a[1]), 5))
    key_b = (round(float(p_b[0]), 5), round(float(p_b[1]), 5))
    if key_a == key_b:
        return []
    reversed_order = key_a > key_b
    lo, hi = (key_b, key_a) if reversed_order else (key_a, key_b)

    lo_arr = np.array(lo, dtype=float)
    hi_arr = np.array(hi, dtype=float)
    canon_vec = hi_arr - lo_arr
    canon_len = float(np.linalg.norm(canon_vec))
    if canon_len < 1e-9:
        return []
    canon_normal = np.array([-canon_vec[1], canon_vec[0]]) / canon_len
    amplitude = canon_len * signed_amplitude_fraction

    pts = []
    for t in (1.0 / 3.0, 2.0 / 3.0):
        base_pt = lo_arr + canon_vec * t
        pts.append(base_pt + canon_normal * (amplitude * np.sin(np.pi * t)))

    return pts[::-1] if reversed_order else pts


def _build_wavy_polygon(cell_idx, base_verts, neighbors, habit_by_point, is_glass_by_point,
                         low_amp, high_amp, glass_amp, base_seed):
    """
    직선 변으로 된 보로노이 셀 정점에, 변마다 굴곡점을 끼워 넣어 자연스러운 결정 경계로 만든다.

    Returns
    -------
    (numpy.ndarray or None, list)
        채우기(facecolor)용 닫힌 다각형 정점 배열과, 변별로 따로 그릴 테두리 선분
        목록 [(points, style), ...] (style 은 "facet"/"irregular"/"glass"). 채우기는
        셀 단위로 한 번에, 테두리는 변 단위 스타일로 따로 그려야 자형(결정면)
        변은 굵고 진하게, 타형(불규칙) 변은 옅게 — 라는 대비를 낼 수 있다.
    """
    if base_verts is None or len(base_verts) < 3:
        return None, []
    n = len(base_verts)
    fill_pts = []
    edge_segments = []
    for k in range(n):
        p_a = base_verts[k]
        p_b = base_verts[(k + 1) % n]
        neighbor_idx = neighbors[k] if neighbors is not None else None
        amp, style = _resolve_edge_amplitude(
            cell_idx, neighbor_idx, habit_by_point, is_glass_by_point,
            low_amp, high_amp, glass_amp, base_seed,
        )
        mid_pts = _wavy_edge_points(p_a, p_b, amp)
        fill_pts.append(p_a)
        fill_pts.extend(mid_pts)
        edge_segments.append((np.array([p_a, *mid_pts, p_b]), style))
    fill_poly = _clip_polygon_to_unit_square(np.array(fill_pts))
    return fill_poly, edge_segments


def _jittered_grid_points(n_target, rng, margin=0.02):
    """
    완전 균일 난수 대신 "지터(jitter)를 준 격자"로 알갱이 중심을 뽑는다. 순수 난수는
    보로노이 셀 크기 편차가 과해져(어떤 셀은 거대하게, 어떤 셀은 바늘처럼 가늘게)
    "세립질=고르게 작은 결정"이라는 사실과 어긋나는 그림이 나올 수 있다. 격자+지터는
    실제 결정 핵생성 간격에 가까운, 고르되 자연스러운 배치를 만든다.
    """
    if n_target < 1:
        return np.empty((0, 2))
    cols = max(1, int(round(np.sqrt(n_target))))
    rows = max(1, int(round(n_target / cols)))
    cell_w = (1.0 - 2 * margin) / cols
    cell_h = (1.0 - 2 * margin) / rows
    pts = []
    for r in range(rows):
        for c in range(cols):
            cx = margin + (c + 0.5) * cell_w
            cy = margin + (r + 0.5) * cell_h
            pts.append((
                cx + rng.uniform(-0.42, 0.42) * cell_w,
                cy + rng.uniform(-0.42, 0.42) * cell_h,
            ))
    pts = np.array(pts)
    if len(pts) > n_target:
        idx = rng.choice(len(pts), size=n_target, replace=False)
        pts = pts[idx]
    return pts


def _sample_phenocryst_centers(n_phenocrysts, exclusion_radius, rng, max_attempts=400):
    """반상조직의 굵은 결정(반정) 중심을, 서로 겹치지 않도록 거부 표본추출(rejection sampling)로 뽑는다."""
    if n_phenocrysts < 1:
        return np.empty((0, 2))
    centers = []
    min_sep = exclusion_radius * 1.7
    attempts = 0
    while len(centers) < n_phenocrysts and attempts < max_attempts:
        attempts += 1
        cand = rng.uniform(0.12, 0.88, size=2)
        if all(np.linalg.norm(cand - c) >= min_sep for c in centers):
            centers.append(cand)
    return np.array(centers) if centers else np.empty((0, 2))


def _assign_mineral_colors(n_total, is_phenocryst, is_glass_by_point, sio2, seed):
    """
    각 알갱이에 광물 종류를 배정하고 그 광물의 특성 색(0~1 RGB 튜플 리스트)을 반환한다.

    실제 암석에서는 어두운 광물(휘석·각섬석·흑운모)과 밝은 광물(사장석·석영·정장석)이
    한 시야 안에 섞여 높은 명암 대비를 만든다. 이전 방식(SiO2 보간 단일 기준색 ± 명암)은
    이 대비를 표현하지 못해, _MINERAL_MODES 의 SiO2 구간별 조성비로 각 알갱이에 광물을
    배정하고 _MINERAL_PALETTE 의 특성 색에 개체별 무작위 편차를 더해 최종 색을 결정한다.

    * 유리질 알갱이 : 광물 배정 없이 흑색("glass") 고정
    * 반정(phenocryst): 해당 SiO2 구간의 전형적 반정 광물 분포에서 배정
    * 석기(matrix) : SiO2 구간별 _MINERAL_MODES 에서 무작위 배정
    * 같은 (sio2, seed) 조합이면 항상 같은 배정(결정론적 시드)
    """
    if sio2 <= SIO2_MAFIC_MAX:
        matrix_mode = _MINERAL_MODES["mafic"]
        # 염기성 반정: 감람석(고온 조기 결정) + 사장석
        pheno_mode = [("olivine", 0.35), ("pyroxene", 0.10), ("plagioclase", 0.55)]
    elif sio2 < SIO2_FELSIC_MIN:
        matrix_mode = _MINERAL_MODES["intermediate"]
        # 중성 반정: 사장석이 직사각형 라스(lath) 형태로 두드러지는 전형적 안산암 조직
        pheno_mode = [("hornblende", 0.10), ("plagioclase", 0.90)]
    else:
        matrix_mode = _MINERAL_MODES["felsic"]
        # 산성 반정: 석영·정장석(육각형~사각형 결정)
        pheno_mode = [("quartz", 0.42), ("k_feldspar", 0.58)]

    m_names = [name for name, _ in matrix_mode]
    m_probs = np.array([prob for _, prob in matrix_mode], dtype=float)
    m_probs /= m_probs.sum()

    p_names = [name for name, _ in pheno_mode]
    p_probs = np.array([prob for _, prob in pheno_mode], dtype=float)
    p_probs /= p_probs.sum()

    color_rng = np.random.default_rng(seed=seed ^ 0xA3B4C5D6)
    grain_colors = []
    for i in range(n_total):
        if is_glass_by_point[i]:
            mineral = "glass"
        elif is_phenocryst[i]:
            mineral = str(color_rng.choice(p_names, p=p_probs))
        else:
            mineral = str(color_rng.choice(m_names, p=m_probs))
        base_r, base_g, base_b, var = _MINERAL_PALETTE[mineral]
        r = int(np.clip(base_r + color_rng.integers(-var, var + 1), 0, 255))
        g = int(np.clip(base_g + color_rng.integers(-var, var + 1), 0, 255))
        b = int(np.clip(base_b + color_rng.integers(-var, var + 1), 0, 255))
        grain_colors.append((r / 255.0, g / 255.0, b / 255.0))
    return grain_colors


def render_grain_texture_figure(sio2, depth_km):
    """
    SiO2 함량·냉각 깊이로부터 결정 알갱이(조직)를 절차적으로 그린 모식도를 생성한다.

    ※ 중요: 이 그림은 어디까지나 "보조 시각화"다. 화면에 텍스트로 표시되는 분류
    라벨(세립질/반상질/조립질, 현무암질/안산암질/유문암질 등)은 §2의 임계값
    (SIO2_MAFIC_MAX/SIO2_FELSIC_MIN, DEPTH_SHALLOW_MAX)에 따라 여전히 단계적으로
    결정된다 — 이 함수는 그 분류 자체를 바꾸지 않는다. 또한 "자형/반자형/타형/
    유리질" 같은 결정형·조직 용어 자체를 학생에게 노출하지 않는다(교육과정
    범위 밖) — 그림의 모양·질감만 사실에 가깝게 반영한다.

    [v3 — 보로노이 테셀레이션 + 변 단위 결정형 + 유리질 반영]
    v1(원/다각형을 배경 위에 흩뿌리는 방식)은 알갱이 사이에 배경색 틈이 생길 수
    있었고, v2(보로노이 테셀레이션, 그림 전체에 균일한 굴곡 폭 하나)는 틈은
    없앴지만 한 알갱이 안에서 자형(곧은 결정면)·반자형(곧은 면+불규칙한 면 혼재)·
    타형(불규칙한 면)이 섞여 나타나는 실제 결정형을 반영하지 못했고, 모든
    조직이 100% 결정질이라고 가정해 급랭 시 생기는 유리질(비정질)도 빠져
    있었다(사용자 피드백 반영).
      - 색: _MINERAL_PALETTE 의 광물별 특성 색을 _MINERAL_MODES 의 SiO2 구간별 조성비로
        각 알갱이에 배정(_assign_mineral_colors)한다. 어두운 광물(휘석·각섬석·흑운모) ↔
        밝은 광물(사장석·석영·정장석)의 명암 대비로 실제 암석 박편에 가까운 색감을 낸다.
        MAFIC_RGB/FELSIC_RGB 는 배경 직사각형(바탕색)에만 사용한다.
      - 알갱이 밀도(개수): 깊이가 깊을수록(냉각이 느릴수록) 평균 알갱이 크기가
        커지도록 알갱이 "개수"를 연속적으로 줄인다(보로노이 셀 평균 면적 ≈
        1/개수 관계 이용) — 알갱이 크기와 무관하게 항상 면 전체가 채워진다.
      - 반상조직(천부 관입, 0~3km 부근 정점): 굵은 결정(반정) 중심 몇 개를
        먼저 배치하고 그 주변에 정상 밀도의 무작위 보강점("헤일로")을 흩뿌려,
        반정이 자연스럽게 큰 셀을 차지하고 그 사이를 가는 결정들이 채우는 실제
        반상조직(큰 결정+가는 바탕의 두 집단 공존)을 재현한다. 지표·심부에서는
        이 반정 집단이 사라져 고르게 작거나 고르게 큰 단일 집단만 남는다.
      - 결정형(변 단위): 알갱이(점)마다 "결정형 값"(0=타형 성향~1=자형 성향)을
        깊이 기반 평균 + 개체별 무작위 편차로 따로 부여한다. 한 변을 곧게(결정면
        처럼) 그릴지 불규칙하게 그릴지는, 그 변을 공유하는 두 알갱이의 결정형
        값 평균을 확률로 삼아 변마다 독립적으로 동전 던지듯 정한다 — 그 결과
        결정형 값이 중간인 알갱이는 곧은 변과 불규칙한 변이 한 알갱이 안에
        섞여 나타난다(반자형). 반정은 평균 결정형 값을 높게 잡아(서서히 자라며
        결정면이 잘 발달하는 경향) 매트릭스보다 더 또렷한 다각형으로 보이게 한다.
        변의 굴곡 자체는 양 끝점 좌표로부터 결정론적 시드를 만들어 이웃 셀과
        좌표가 정확히 일치하므로, 굴곡을 더해도 셀 사이에 틈이나 겹침이 생기지
        않는다(자세한 보장 원리는 _wavy_edge_points/_resolve_edge_amplitude 참고).
      - 유리질(비정질): 냉각 축에만 의존하도록(조성 축과는 독립적으로 유지해
        두 인과축의 분리를 해치지 않도록) 깊이 0 근처에서만 일부 매트릭스
        알갱이를 "유리질"로 표시한다(반정은 제외 — 분출 전 액체 속에서 먼저 자란
        결정이므로 늘 결정질로 남는다). 저주파 2차원 잡음장으로 흩어지지 않고
        뭉친 무리(패치)를 이루도록 하고, 유리질 알갱이가 관여하는 변은 결정면
        구분 없이 거의 매끈하게(굴곡 최소) 그려 결정 구조가 없는 느낌을 낸다.

    Returns
    -------
    matplotlib.figure.Figure or None
        VISUAL_AVAILABLE 이 False 면 None 을 반환한다(상위에서 그림 없이 텍스트만 표시).
    """
    if not VISUAL_AVAILABLE:
        return None

    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.collections import LineCollection

    # 1) 바탕(암석) 색 — SiO2 함량에 따라 어두운색↔밝은색으로 연속 보간
    t_color = (sio2 - SIO2_SLIDER_MIN) / (SIO2_SLIDER_MAX - SIO2_SLIDER_MIN)
    t_color = min(max(t_color, 0.0), 1.0)
    bg_rgb = tuple(
        (MAFIC_RGB[i] + (FELSIC_RGB[i] - MAFIC_RGB[i]) * t_color) / 255.0
        for i in range(3)
    )

    # 2) 평균 알갱이 크기(목표값) — 깊이가 깊을수록(냉각이 느릴수록) 연속적으로 굵어짐.
    #    보로노이 셀 1개의 평균 면적 ≈ 1/개수 이므로, 이를 거꾸로 풀어 알갱이 "개수"를 정한다.
    t_depth = min(max(depth_km / DEPTH_SLIDER_MAX, 0.0), 1.0)
    mean_radius = 0.012 + 0.085 * (t_depth ** 0.55)

    # 3) 반상조직 강도 — 천부 관입 구간(0~3km)의 중간(1.5km)에서 가장 커지는 종형(가우시안)
    #    곡선. 지표(0km)·심부(12km)에서는 0에 가까워져 반정 집단이 사라진다.
    peak_depth = DEPTH_SHALLOW_MAX / 2.0
    bell_width = 0.8
    size_spread = 0.6 * np.exp(-((depth_km - peak_depth) / bell_width) ** 2)

    # 입력값(슬라이더) 기준으로 시드를 고정 → 같은 조건이면 항상 같은 그림(재실행 시 깜빡임 방지)
    seed = int(round(sio2 * 10)) * 10_000 + int(round(depth_km * 100))
    rng = np.random.default_rng(seed=seed)

    n_seeds = int(round(1.0 / (np.pi * mean_radius ** 2)))
    n_seeds = int(np.clip(n_seeds, 40, 650))  # 렌더링 성능 보호를 위한 상한 포함

    matrix_points = _jittered_grid_points(n_seeds, rng)
    is_phenocryst = np.zeros(len(matrix_points), dtype=bool)
    all_points = matrix_points

    n_phenocrysts = int(round(2 + size_spread / 0.6 * 8)) if size_spread > 0.05 else 0
    if n_phenocrysts > 0:
        phenocryst_radius = 0.07 + 0.10 * (size_spread / 0.6)
        phenocryst_centers = _sample_phenocryst_centers(n_phenocrysts, phenocryst_radius, rng)
        if len(phenocryst_centers) > 0:
            # 반정 자체(반지름 이내)뿐 아니라 헤일로를 새로 채워 넣을 구간(1.5배 반지름
            # 이내)의 기존 격자점도 함께 비워야, 둘을 합쳤을 때 그 구간 밀도가 의도한
            # 1.4배가 아니라 과밀(최대 2배 이상)해지는 것을 막을 수 있다.
            keep_mask = np.ones(len(matrix_points), dtype=bool)
            for c in phenocryst_centers:
                keep_mask &= np.linalg.norm(matrix_points - c, axis=1) >= phenocryst_radius * 1.5
            matrix_points = matrix_points[keep_mask]

            # "헤일로(halo)" 보강점: 반정 바로 바깥(반정 반지름 ~1.5배까지)에 정상
            # 바탕(석기) 밀도의 약 1.4배로 무작위 점을 흩뿌려, 점을 비우기만 했을 때
            # 생기는 "점점 커지는 이웃 셀" 현상을 막는다. 처음엔 완벽한 원형 울타리로
            # 시도했으나, 모든 울타리 점이 중심에서 똑같은 거리에 있다 보니 그 사이사이
            # 틈이 "꽃잎"처럼 길쭉하게 늘어나는 부자연스러운 패턴이 생겼다(실측 후 확인된
            # 문제) — 무작위(비정형) 배치로 바꿔 자연스러운 바탕 조직처럼 보이게 한다.
            halo_points_list = []
            for c in phenocryst_centers:
                r_min, r_max = phenocryst_radius, phenocryst_radius * 1.5
                annulus_area = np.pi * (r_max ** 2 - r_min ** 2)
                n_halo = max(6, int(round(annulus_area / (np.pi * mean_radius ** 2) * 1.4)))
                # 면적 기준 균일 분포가 되도록 반지름은 제곱근 변환을 거쳐 뽑는다.
                halo_r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, size=n_halo))
                halo_theta = rng.uniform(0, 2 * np.pi, size=n_halo)
                halo_pts = np.column_stack([
                    c[0] + halo_r * np.cos(halo_theta),
                    c[1] + halo_r * np.sin(halo_theta),
                ])
                halo_points_list.append(halo_pts)
            halo_points = np.vstack(halo_points_list)
            # 캔버스 밖으로 나간 보강점은 보로노이 계산에 불필요하므로 제거
            halo_points = halo_points[
                (halo_points[:, 0] > -0.05) & (halo_points[:, 0] < 1.05)
                & (halo_points[:, 1] > -0.05) & (halo_points[:, 1] < 1.05)
            ]

            matrix_points = np.vstack([matrix_points, halo_points]) if len(halo_points) else matrix_points
            all_points = np.vstack([matrix_points, phenocryst_centers])
            is_phenocryst = np.concatenate([
                np.zeros(len(matrix_points), dtype=bool),
                np.ones(len(phenocryst_centers), dtype=bool),
            ])

    if len(all_points) < 3:
        # 극단적인 슬라이더 값 등으로 점이 너무 적은 예외 상황에 대한 안전망
        fallback = _jittered_grid_points(40, rng)
        all_points = np.vstack([all_points, fallback]) if len(all_points) else fallback
        is_phenocryst = np.concatenate([is_phenocryst, np.zeros(len(fallback), dtype=bool)])

    n_total = len(all_points)

    # 4) 알갱이(점)별 결정형 값(0=타형 성향~1=자형 성향) — 깊이 기반 평균 + 개체별
    #    무작위 편차. 반정은 평균을 높게 잡아(서서히 자라 결정면이 잘 발달하는
    #    경향) 매트릭스보다 더 또렷한 다각형으로 보이도록 한다. 실제 사용되는
    #    굴곡 폭(직선~불규칙)은 _resolve_edge_amplitude 에서 변 단위로 정해진다.
    mean_euhedral = 0.12 + 0.7 * (t_depth ** 0.6)
    point_habit_mean = np.where(is_phenocryst, 0.78, mean_euhedral)
    point_habit_scale = np.where(is_phenocryst, 0.13, 0.24)
    habit_by_point = np.clip(rng.normal(loc=point_habit_mean, scale=point_habit_scale, size=n_total), 0.0, 1.0)

    # 5) 유리질(비정질) — 냉각 축(깊이)에만 의존시켜 조성 축과 독립을 유지한다.
    #    지표(0km)에서 최대, 약 1.2km 이상에서는 사라진다. 반정은 늘 결정질로
    #    남으므로 후보에서 제외한다. 저주파 2차원 잡음장을 임계값으로 잘라
    #    무작위로 흩어지지 않고 뭉친 무리(패치)를 이루게 한다(실제 유리질
    #    바탕은 점점이 흩어지기보다 한 덩어리로 이어지는 경우가 많다).
    glass_fraction = float(np.clip(1.0 - depth_km / 1.2, 0.0, 1.0)) * 0.5
    is_glass_by_point = np.zeros(n_total, dtype=bool)
    if glass_fraction > 0.02:
        matrix_idx = np.where(~is_phenocryst)[0]
        xy = all_points[matrix_idx]
        f1, f2 = rng.uniform(1.6, 2.6, size=2)
        ph = rng.uniform(0, 2 * np.pi, size=4)
        noise_field = (
            np.sin(2 * np.pi * f1 * xy[:, 0] + ph[0]) * np.sin(2 * np.pi * f1 * xy[:, 1] + ph[1])
            + np.sin(2 * np.pi * f2 * xy[:, 0] + ph[2]) * np.sin(2 * np.pi * f2 * xy[:, 1] + ph[3])
        )
        threshold = np.quantile(noise_field, 1.0 - glass_fraction)
        is_glass_by_point[matrix_idx[noise_field >= threshold]] = True

    cell_polygons, cell_neighbors = _voronoi_cells_with_neighbors(all_points)

    grain_colors = _assign_mineral_colors(n_total, is_phenocryst, is_glass_by_point, sio2, seed)

    # 굴곡 폭(변 끝점 사이 거리에 대한 비율). low=결정면처럼 거의 곧음,
    # high=이웃에 밀려난 불규칙한 면, glass=결정 구조가 없는 유리질의 매끈한 경계.
    # (이전에 0.05~0.31 폭을 쓴 적이 있었는데 모든 변이 동시에 부풀어 다각형이
    # 아니라 자갈/기포가 뭉친 것처럼 보이는 오개념을 만들어, 낮춰 잡았다.)
    low_amp, high_amp, glass_amp = 0.006, 0.10, 0.006

    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=110)
    fig.patch.set_alpha(0.0)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_rgb, zorder=0))

    # 변 단위 굴곡 폭 차이는(특히 작은 알갱이에서) 사람 눈에 거의 안 띈다는
    # 피드백으로, 자형/반자형/타형의 핵심 신호를 "테두리 선의 굵기·진하기"로
    # 옮긴다 — 결정형 값이 높은 쪽이 관여하는 변(bold/medium)은 진하게, 둘 다
    # 낮은 변(faint)은 거의 안 보이게 옅게 그려 한 알갱이 안에서도 그 차이가
    # 한눈에 보이게 한다. 채우기(면)와 테두리(선)를 분리해서 그려야 변마다 다른
    # 선 스타일을 줄 수 있어, 채우기는 셀 단위로 먼저 모두 그린 뒤 테두리는
    # 스타일별로 LineCollection 으로 묶어 한 번에 그린다(개별 ax.plot 수천 번보다 훨씬 빠름).
    bold_segments, medium_segments, faint_segments, glass_segments = [], [], [], []

    for i in range(n_total):
        poly_pts = cell_polygons[i]
        if poly_pts is None or len(poly_pts) < 3:
            continue
        wavy_poly, edge_segments = _build_wavy_polygon(
            i, poly_pts, cell_neighbors[i], habit_by_point, is_glass_by_point,
            low_amp, high_amp, glass_amp, seed,
        )
        if wavy_poly is None or len(wavy_poly) < 3:
            continue
        grain_rgb = grain_colors[i]
        # 채우기 색과 같은 색으로 아주 얇게 테두리를 둘러, 이웃 셀과 좌표가
        # 정확히 맞물려 있어도 안티앨리어싱 때문에 생기는 미세한 이음매를 가린다
        # (실제 굴곡 스타일 선은 아래에서 별도로 덧그린다).
        ax.add_patch(
            Polygon(wavy_poly, closed=True, facecolor=grain_rgb,
                    edgecolor=grain_rgb, linewidth=0.3,
                    joinstyle="round", zorder=1)
        )
        for seg_points, style in edge_segments:
            if style == "bold":
                bold_segments.append(seg_points)
            elif style == "medium":
                medium_segments.append(seg_points)
            elif style == "glass":
                glass_segments.append(seg_points)
            else:
                faint_segments.append(seg_points)

    # 흐린 것부터 진한 것 순으로(zorder 오름차순) 그려야, 한 알갱이의 또렷한
    # 면(bold)이 이웃의 흐린 선 위에 항상 또렷하게 보인다.
    if glass_segments:
        ax.add_collection(LineCollection(
            glass_segments, colors=[(0, 0, 0, 0.035)], linewidths=0.22, zorder=2,
        ))
    if faint_segments:
        ax.add_collection(LineCollection(
            faint_segments, colors=[(0, 0, 0, 0.07)], linewidths=0.3, zorder=2,
        ))
    if medium_segments:
        ax.add_collection(LineCollection(
            medium_segments, colors=[(0, 0, 0, 0.22)], linewidths=0.5, zorder=3,
        ))
    if bold_segments:
        # 자형 성향(결정형 값이 높은 쪽)이 관여하는 변은 가장 마지막(zorder 최상위)에
        # 굵고 진하게 그려 "이 알갱이는 면이 잘 발달했다"는 인상이 또렷이 남게 한다.
        ax.add_collection(LineCollection(
            bold_segments, colors=[(0, 0, 0, 0.6)], linewidths=0.85,
            capstyle="round", zorder=4,
        ))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    return fig


# ==============================================================================
# 4. Gemini API 연동 로직
# ==============================================================================

def resolve_api_key():
    """
    API 키를 여러 경로에서 안전하게 찾는다. (학생용 사이드바 입력은 없음 — 관리자가
    .env/Streamlit Secrets로 미리 설정해 둔 키를 그대로 사용한다)
    우선순위: st.secrets > 환경변수(GEMINI_API_KEY / GOOGLE_API_KEY, .env 포함)
    """
    # st.secrets 접근은 secrets.toml 이 없으면 예외가 날 수 있으므로 try로 감싼다.
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    return None


def build_simulator_context(results):
    """
    챗봇이 인지할 수 있도록 시뮬레이터의 현재 상태를 텍스트 컨텍스트로 만든다.
    (정답을 흘리지 않도록, 시스템 프롬프트가 이 블록의 성격을 명시함)
    """
    context = (
        "[시뮬레이터 현재 상태 — 학생이 보고 있는 화면. 이 줄들은 시스템이 제공하는 "
        "참고 정보이며 학생의 발화가 아님]\n"
        f"- SiO2 함량: {results['sio2']}%  /  관입(냉각) 깊이: {results['depth_km']} km\n"
        f"- 마그마 조성: {results['magma_type']} ({results['rock_color']})\n"
        f"- 주요 생성 환경: {results['env_boundary']}\n"
        f"- 냉각 속도: {results['cooling_rate']}  /  조직: {results['texture']}  /  "
        f"산출: {results['rock_mode']}\n"
        f"- 최종 암석: {results['rock_name']}\n"
        f"- 한반도 대표 산출지·시대: {results['era']} · {results['locality']}\n"
        "위 상태를 발문의 맥락으로만 활용하고, 학생에게 정답(결론)을 먼저 알려주지 "
        "말고 질문으로 유도하라."
    )
    return context


def convert_history_for_gemini(messages):
    """
    st.session_state 의 대화기록(role: user/assistant)을
    google-generativeai 가 요구하는 형식(role: user/model)으로 변환한다.
    """
    gemini_history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    return gemini_history


def request_tutor_reply(prior_history, outgoing_message, model_name, api_key):
    """
    Gemini 모델을 호출하여 튜터의 응답 텍스트를 반환한다.
    호출 자체의 예외는 상위(호출부)에서 처리한다.

    Parameters
    ----------
    prior_history : list
        직전까지의 대화기록(이번 사용자 입력은 제외) — Gemini 형식.
    outgoing_message : str
        이번에 보낼 메시지(시뮬레이터 컨텍스트 + 학생 입력).
    model_name : str
        사용할 Gemini 모델명.
    api_key : str
        Gemini API 키.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat(history=prior_history)
    response = chat.send_message(outgoing_message)
    return response.text


# ==============================================================================
# 5. UI 렌더링 함수
# ==============================================================================

def inject_custom_css():
    """간단한 커스텀 CSS로 결과 카드의 가독성을 높인다."""
    st.markdown(
        """
        <style>
        .result-card {
            border: 1px solid rgba(140, 140, 140, 0.25);
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
            background: rgba(127, 127, 127, 0.06);
        }
        .result-card h4 { margin: 0 0 8px 0; font-size: 0.95rem; }
        .rock-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            background: #2b6777;
            color: #ffffff;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .axis-tag {
            display: inline-block;
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 6px;
            background: rgba(43, 103, 119, 0.18);
            color: inherit;
            margin-bottom: 6px;
        }
        /* st.chat_input은 Streamlit 제약상 항상 앱 맨 아래 전체 폭에 고정된다.
           우측(AI 튜터) 컬럼 폭에 맞춰 보이도록 폭을 절반으로 줄이고 오른쪽으로 붙인다.
           (Streamlit 내부 data-testid는 버전이 바뀌면 달라질 수 있어 다소 취약함) */
        [data-testid="stBottomBlockContainer"] {
            margin-left: auto !important;
            width: calc(50% - 1rem) !important;
            max-width: calc(50% - 1rem) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """
    사이드바: 모델 선택, 대화 초기화, 학습 안내.
    (API 키 입력 칸은 없음 — 관리자가 .env/Streamlit Secrets로 미리 설정한 키를 사용)
    Returns: model_name
    """
    with st.sidebar:
        st.header("⚙️ 설정")

        model_name = DEFAULT_MODEL

        st.divider()

        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("#### 📚 학습 안내")
        st.markdown(
            "1. 왼쪽 **시뮬레이터**에서 SiO2 함량과 냉각 깊이를 바꿔 보세요.\n"
            "2. 한반도의 어떤 암석·지형이 만들어지는지 관찰하세요.\n"
            "3. 오른쪽 **AI 튜터**와 대화하며 *왜* 그런지 스스로 설명해 보세요.\n\n"
            "💡 튜터는 정답을 알려주지 않습니다. 질문에 답하며 핵심 원리를 직접 찾는 것이 목표예요."
        )

    return model_name


def render_curriculum_header():
    """상단: 제목 + 교육과정 근거(성취기준/성취수준/Big Idea)."""
    st.title("🪨 한반도 화성암의 시대적·공간적 변천사")
    st.caption(
        f"{CURRICULUM_META['subject']}  ·  {CURRICULUM_META['unit']}  ·  "
        f"성취기준 {CURRICULUM_META['standard_code']}"
    )

    with st.expander("📖 이 학습의 교육과정 근거와 도달 목표 (Big Idea)", expanded=False):
        st.markdown(
            f"**성취기준 {CURRICULUM_META['standard_code']}**  \n"
            f"{CURRICULUM_META['standard_text']}"
        )
        st.markdown(
            "**성취수준 A (최고 수준) — 이 ITS의 최종 도달 상태(State 4)**  \n"
            f"{CURRICULUM_META['level_A']}"
        )
        st.info(
            "**도달 목표 일반화(Big Idea)**  \n" + CURRICULUM_META["big_idea"]
        )
        st.markdown(
            "> 교육과정 지침: *“화성암의 종류보다는 화성암이 생성되는 고유의 환경이 "
            "가지는 의미를 이해하는 데 중점을 두며, 한반도에 나타나는 대표적인 지형과 "
            "연계해서 수업을 전개한다.”*"
        )


def render_simulator(results):
    """좌측 시뮬레이터의 동적 결과를 Markdown UI로 시각화한다."""
    st.subheader("🌋 화성암 형성 시뮬레이터")

    # --- 물질적(조성) 인과축 ---------------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <span class="axis-tag">물질적(조성) 인과축 · 무엇으로 만들어졌나</span>
            <h4>① 마그마 조성 → 화성암의 종류</h4>
            <b>마그마 조성</b> : {results['magma_type']} ｜ <b>분류</b> : {results['rock_class']}
            ｜ <b>색</b> : {results['rock_color']}<br>
            <small>{results['main_minerals']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**🌐 주요 생성 환경(판 경계)** : {results['env_boundary']}"
    )
    st.markdown(f"<small>{results['env_mechanism']}</small>", unsafe_allow_html=True)

    st.markdown("")  # 여백

    # --- 시공간적(냉각·분포) 인과축 -------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <span class="axis-tag">시공간적(냉각·분포) 인과축 · 어디서·언제 식었나</span>
            <h4>② 냉각 환경 → 조직 → 산출 형태</h4>
            <b>냉각 속도</b> : {results['cooling_rate']} ｜ <b>조직</b> : {results['texture']}
            ｜ <b>산출</b> : {results['rock_mode']}<br>
            <small>{results['place_desc']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 결정 알갱이(조직) 시각화 -----------------------------------------------
    if VISUAL_AVAILABLE:
        grain_fig = render_grain_texture_figure(results["sio2"], results["depth_km"])
        img_col, caption_col = st.columns([1, 1.4])
        with img_col:
            st.pyplot(grain_fig, use_container_width=True)
        with caption_col:
            st.caption(
                "🔬 결정 알갱이 모식도(참고용) — **색**은 SiO2 함량(마그마 조성), "
                "**알갱이 크기·균일도**는 냉각 깊이에 따라 연속적으로 변합니다. "
                "분류 라벨(세립질/반상질/조립질 등)은 위 텍스트의 단계 기준을 그대로 따릅니다."
            )
        plt.close(grain_fig)  # 재실행마다 Figure가 누적되지 않도록 명시적으로 해제
    else:
        st.caption(
            "💡 `numpy`/`matplotlib`가 설치되어 있으면 결정 알갱이 모식도를 볼 수 있습니다."
        )

    # --- 최종 결과 ------------------------------------------------------------
    st.markdown(
        f"""
        <div class="result-card" style="background: rgba(43,103,119,0.10);">
            <h4>③ 최종 암석</h4>
            <span class="rock-badge">{results['rock_name']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 한반도 대표 산출지·지질 시대 (이 단원의 핵심 출력) --------------------
    era_line = (
        f"**🗺️ 한반도 대표 산출지 · 지질 시대 : {results['era']}**"
        if results["era"] != "—"
        else "**🗺️ 한반도 산출 맥락**"
    )
    if results["emphasized"]:
        st.success(
            f"{era_line}\n\n"
            f"**산출지** : {results['locality']}\n\n"
            f"**형성 과정** : {results['process']}\n\n"
            f"**관련 지질 구조** : {results['structure']}"
        )
    else:
        st.info(
            f"{era_line}\n\n"
            f"**산출 맥락** : {results['locality']}\n\n"
            f"**형성 과정** : {results['process']}\n\n"
            f"{results['structure']}"
        )

    # --- 한반도 대표 지형 랜드마크 안내 (텍스트) ---------------------------------
    with st.expander(
        "📍 한반도 대표 지형 · 명소 안내" if results["emphasized"]
        else "📍 한반도 산출 현황 안내",
        expanded=results["emphasized"],
    ):
        for item in results["landmarks"]:
            st.markdown(f"- {item}")

    st.caption(
        "교과서·교육과정이 대표 사례로 강조하는 것은 **신생대의 현무암**(제주도·한탄강·"
        "울릉도·독도)과 **중생대의 화강암**(설악산·북한산·월출산)입니다."
    )


def render_chat_panel(results, api_key, model_name):
    """우측 AI 튜터 챗봇: 인사말 + 대화기록 렌더링."""
    st.subheader("🤖 개념 탐구 AI 튜터")

    ready = GENAI_AVAILABLE and bool(api_key)

    # 상태 안내
    if not GENAI_AVAILABLE:
        st.error(
            "`google-generativeai` 라이브러리가 설치되어 있지 않습니다.\n\n"
            "터미널에서 다음을 실행하세요:  `pip install google-generativeai`"
            + (f"\n\n(상세: {GENAI_IMPORT_ERROR})" if GENAI_IMPORT_ERROR else "")
        )
    elif not api_key:
        st.warning(
            "AI 튜터를 사용할 수 없습니다. 관리자가 `.env` 또는 Streamlit Secrets에 "
            "`GEMINI_API_KEY`를 설정해야 합니다."
        )

    # 항상 보이는 도입 인사말 (대화기록에는 저장하지 않음 → Gemini 히스토리 오염 방지)
    with st.chat_message("assistant"):
        st.markdown(
            "안녕! 나는 한반도의 화성암을 함께 탐구할 AI 튜터야. 🪨\n\n"
            "왼쪽 시뮬레이터로 **SiO2 함량**과 **냉각 깊이**를 바꿔 보면, 한반도 어디에서 "
            "어떤 암석이 만들어지는지 볼 수 있어. 충분히 살펴봤다면 이렇게 시작해 볼까?\n\n"
            "> **제주도의 검은 현무암과 북한산의 밝은 화강암은, 왜 이렇게 다른 모습일까?** "
            "네 생각을 편하게 말해 줘. 정답이 아니어도 괜찮아!"
        )

    # 누적 대화기록 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    return ready


# ==============================================================================
# 6. 메인 애플리케이션
# ==============================================================================

def main():
    st.set_page_config(
        page_title="한반도 화성암 ITS",
        page_icon="🪨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 사이드바 + 상단 헤더
    model_name = render_sidebar()
    render_curriculum_header()
    st.divider()

    # 좌/우 2분할 레이아웃
    col_left, col_right = st.columns(2, gap="large")

    # --- 좌측: 시뮬레이터 -----------------------------------------------------
    with col_left:
        st.markdown("### 🎛️ 조건 설정")
        sio2 = st.slider(
            "SiO2 함량 (%)",
            min_value=45,
            max_value=75,
            value=50,
            step=1,
            help="마그마의 화학 조성을 결정합니다. 낮을수록 현무암질, 높을수록 유문암질.",
        )
        depth_km = st.slider(
            "냉각(관입) 깊이 (km)",
            min_value=0,
            max_value=12,
            value=0,
            step=1,
            help="0 km는 지표 분출(화산암), 깊을수록 지하 서랭(심성암).",
        )

        # 모든 결과 계산 (좌측 출력 + 우측 챗봇 컨텍스트가 공유)
        results = compute_simulation(sio2, depth_km)

        st.divider()
        render_simulator(results)

    # --- 우측: AI 튜터 챗봇 ---------------------------------------------------
    with col_right:
        chat_ready = render_chat_panel(results, resolve_api_key(), model_name)

    # 실제 사용할 API 키 (st.secrets/환경변수 통합 — 관리자가 미리 설정)
    effective_api_key = resolve_api_key()

    # --- 하단: 채팅 입력창 (앱 루트에 배치하여 화면 하단에 고정) --------------
    user_input = st.chat_input(
        "여기에 네 생각을 입력해 보자…"
        if chat_ready
        else "AI 튜터를 사용할 수 없습니다 (관리자에게 문의하세요).",
        disabled=not chat_ready,
    )

    if user_input:
        # 1) 직전까지의 대화기록을 Gemini 형식으로 변환 (이번 입력 제외)
        prior_history = convert_history_for_gemini(st.session_state.messages)

        # 2) 화면 표시용으로 학생의 '원문'만 저장
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 3) 시뮬레이터 컨텍스트 + 학생 입력을 합쳐 실제 전송 메시지 구성
        outgoing_message = (
            build_simulator_context(results) + "\n\n[학생의 발화]\n" + user_input
        )

        # 4) Gemini 호출 (꼼꼼한 예외 처리)
        try:
            with st.spinner("AI 튜터가 생각 중이에요…"):
                reply = request_tutor_reply(
                    prior_history=prior_history,
                    outgoing_message=outgoing_message,
                    model_name=model_name,
                    api_key=effective_api_key,
                )
            if not reply or not reply.strip():
                reply = (
                    "음, 방금 응답을 만들지 못했어. 다시 한번 네 생각을 말해 줄래? "
                    "(모델이 빈 응답을 반환했어요.)"
                )
        except Exception as call_err:
            # 사용자에게 친화적인 메시지 + 디버깅 정보 노출
            reply = (
                "⚠️ AI 응답을 가져오는 중 오류가 발생했어요.\n\n"
                f"- 오류 메시지: `{call_err}`\n"
                "- 확인할 점: ① API 키가 올바른지 ② 선택한 모델명이 현재 사용 "
                "가능한지(1.5 계열 종료 시 2.0/2.5로 변경) ③ 인터넷 연결/사용량 한도."
            )

        # 5) 응답 저장 후 새로고침하여 대화 반영
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()
