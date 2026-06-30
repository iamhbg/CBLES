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
# 0-3. 결정 조직(텍스처) 시각화용 numpy/matplotlib 임포트  ─  누락 시 안전하게 처리
#      (그림 없이도 기존 텍스트 결과는 그대로 동작해야 한다)
# ------------------------------------------------------------------------------
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Streamlit 서버에는 GUI 백엔드가 없으므로 비대화형 백엔드 고정
    import matplotlib.pyplot as plt
    VISUAL_AVAILABLE = True
    VISUAL_IMPORT_ERROR = None
except Exception as _visual_import_err:
    np = None
    plt = None
    VISUAL_AVAILABLE = False
    VISUAL_IMPORT_ERROR = str(_visual_import_err)


# ==============================================================================
# 1. 전역 상수 및 환경 설정
# ==============================================================================

# Gemini 모델명 (요구사항: gemini-1.5-flash 또는 gemini-1.5-pro)
#  ※ Google이 구형 모델을 단계적으로 종료할 수 있습니다. 만약 아래 모델명으로
#    호출이 실패하면 'gemini-2.0-flash', 'gemini-2.5-flash' 등 현재 사용 가능한
#    모델명으로 교체하십시오. (사이드바에서도 선택할 수 있도록 구성함)
DEFAULT_MODEL = "gemini-1.5-flash"
SELECTABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

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
MAFIC_RGB = (61, 58, 58)        # 어두운색(현무암질) 대표색
FELSIC_RGB = (230, 219, 194)    # 밝은색(유문암질) 대표색

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
            "locality": "제주특별자치도(섭지코지·성산일출봉·중문대포 주상절리대), "
                        "한탄강 일대(재인폭포), 울릉도·독도(저동 해안)",
            "process": "신생대에 일어난 화산 활동으로 현무암질 마그마가 지표로 분출한 뒤 "
                       "빠르게 식어 생성되었습니다.",
            "structure": "주상 절리 ─ 용암이 빠르게 식으며 부피가 수축해 만들어진 "
                         "다각형 기둥 모양의 절리(화산암에서 잘 발달).",
            "emphasized": True,
        },
        "화강암": {
            "era": "중생대",
            "locality": "설악산(비선대), 북한산(인수봉), 월출산(구정봉) "
                        "─ 넓게 보아 쥐라기 대보 화강암, 백악기 불국사 화강암 계열",
            "process": "중생대의 지각 변동으로 생성된 유문암질 마그마가 지하 깊은 곳에서 "
                       "서서히 식어 생성되었고, 이후 융기하여 위를 덮고 있던 암석이 "
                       "제거(삭박)되면서 지표에 드러났습니다.",
            "structure": "판상 절리 ─ 지하 심부의 암석이 융기·삭박으로 압력이 감소하며 "
                         "만들어진 얇은 판 모양의 절리(심성암에서 잘 발달).",
            "emphasized": True,
        },
        "안산암": {
            "era": "중생대(주로 백악기)",
            "locality": "경상 분지 일대의 백악기 화산암류",
            "process": "섭입대 환경에서 마그마 혼합 또는 결정 분화로 생성된 안산암질 "
                       "마그마가 지표 부근에서 식어 생성됩니다.",
            "structure": "(분류표상 중성 화산암 — 교과서는 현무암·화강암을 대표 사례로 다룸)",
            "emphasized": False,
        },
        "유문암": {
            "era": "중생대",
            "locality": "중생대 화산 활동과 관련된 산성 화산암류",
            "process": "대륙 지각의 용융 등으로 만들어진 유문암질 마그마가 지표 부근에서 "
                       "식어 생성됩니다.",
            "structure": "(분류표상 산성 화산암 — 같은 유문암질 마그마가 지하에서 서서히 "
                         "식으면 화강암이 됨)",
            "emphasized": False,
        },
        "반려암": {
            "era": "—",
            "locality": "심성암 규모로 산출되는 염기성 심성암",
            "process": "현무암질 마그마가 지하 깊은 곳에서 서서히 식어 생성됩니다.",
            "structure": "(분류표상 염기성 심성암 — 같은 현무암질 마그마가 지표에서 "
                         "빠르게 식으면 현무암이 됨)",
            "emphasized": False,
        },
        "섬록암": {
            "era": "—",
            "locality": "심성암 규모로 산출되는 중성 심성암",
            "process": "안산암질 마그마가 지하 깊은 곳에서 서서히 식어 생성됩니다.",
            "structure": "(분류표상 중성 심성암 — 같은 안산암질 마그마가 지표에서 "
                         "빠르게 식으면 안산암이 됨)",
            "emphasized": False,
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
    }
    return results


def render_grain_texture_figure(sio2, depth_km):
    """
    SiO2 함량·냉각 깊이로부터 결정 알갱이(조직)를 절차적으로 그린 모식도를 생성한다.

    ※ 중요: 이 그림은 어디까지나 "보조 시각화"다. 화면에 텍스트로 표시되는 분류
    라벨(세립질/반상질/조립질, 현무암질/안산암질/유문암질 등)은 §2의 임계값
    (SIO2_MAFIC_MAX/SIO2_FELSIC_MIN, DEPTH_SHALLOW_MAX)에 따라 여전히 단계적으로
    결정된다 — 이 함수는 그 분류 자체를 바꾸지 않는다. 또한 "자형/반자형/타형"
    같은 결정형 용어 자체를 학생에게 노출하지 않는다(교육과정 범위 밖) — 다만
    실제 암석 사진과 동떨어진 단순 원으로 그리면 오개념을 만들 수 있어, 그림 자체는
    아래 원리로 알갱이 "모양"까지 더 사실적으로 표현한다.
      - 색: SiO2 값을 SIO2_SLIDER_MIN~MAX 구간에서 MAFIC_RGB↔FELSIC_RGB로 선형 보간.
      - 평균 알갱이 크기: 깊이가 깊을수록 연속적으로 굵어짐(냉각이 느릴수록 큰 결정).
      - 알갱이 크기의 편차(균일도): 천부 관입 구간(0~3km) 중간에서 가장 커지는 종형
        곡선으로, "반상질(굵은 결정+가는 바탕이 혼재)" 느낌을 연속적으로 표현한다.
      - 알갱이 모양(결정형): 원이 아니라 불규칙 다각형으로 그린다. 냉각이 느릴수록
        (깊을수록) 결정이 자랄 시간이 많아 각진 다각형(자형에 가까움) 비율이
        높아지고, 냉각이 빠를수록(지표) 모서리가 뭉개진 둥근 불규칙 다각형
        (타형에 가까움) 비율이 높아지도록 알갱이별 "각짐 정도"를 깊이에 따라
        연속적으로(개체별 무작위 편차 포함) 보간한다. 실제 암석도 한 시료 안에
        자형~타형 알갱이가 섞여 있으므로, 같은 깊이에서도 알갱이마다 편차를 둔다.

    Returns
    -------
    matplotlib.figure.Figure or None
        VISUAL_AVAILABLE 이 False 면 None 을 반환한다(상위에서 그림 없이 텍스트만 표시).
    """
    if not VISUAL_AVAILABLE:
        return None

    from matplotlib.patches import Polygon, Rectangle

    # 1) 바탕(암석) 색 — SiO2 함량에 따라 어두운색↔밝은색으로 연속 보간
    t_color = (sio2 - SIO2_SLIDER_MIN) / (SIO2_SLIDER_MAX - SIO2_SLIDER_MIN)
    t_color = min(max(t_color, 0.0), 1.0)
    bg_rgb = tuple(
        (MAFIC_RGB[i] + (FELSIC_RGB[i] - MAFIC_RGB[i]) * t_color) / 255.0
        for i in range(3)
    )

    # 2) 평균 알갱이 크기 — 깊이가 깊을수록(냉각이 느릴수록) 연속적으로 굵어짐
    t_depth = min(max(depth_km / DEPTH_SLIDER_MAX, 0.0), 1.0)
    mean_radius = 0.012 + 0.085 * (t_depth ** 0.55)

    # 3) 알갱이 크기 편차 — 천부 관입 구간(0~3km)의 중간(1.5km)에서 가장 커지는
    #    종형(가우시안) 곡선. 지표(0km)·심부(12km)에서는 편차가 작아 알갱이가 고르다.
    peak_depth = DEPTH_SHALLOW_MAX / 2.0
    bell_width = 0.8
    size_spread = 0.6 * np.exp(-((depth_km - peak_depth) / bell_width) ** 2)

    # 입력값(슬라이더) 기준으로 시드를 고정 → 같은 조건이면 항상 같은 그림(재실행 시 깜빡임 방지)
    seed = int(round(sio2 * 10)) * 10_000 + int(round(depth_km * 100))
    rng = np.random.default_rng(seed=seed)

    n_grains = 140
    # 편차는 평균 알갱이 크기에 비례(절대값 floor를 쓰면 작은 알갱이 구간에서
    # 상대적으로 과한 편차가 생겨 "세립질=균일하게 작음" 느낌이 깨짐).
    radii = rng.normal(
        loc=mean_radius, scale=mean_radius * (0.08 + size_spread), size=n_grains
    )
    radii = np.clip(radii, 0.006, 0.14)
    xs = rng.uniform(0.05, 0.95, size=n_grains)
    ys = rng.uniform(0.05, 0.95, size=n_grains)
    grain_shade = rng.uniform(-0.18, 0.18, size=n_grains)  # 알갱이별 명암 차이(입자감)

    # 4) 결정형(자형↔타형) — 깊이가 깊을수록(냉각이 느릴수록) 평균적으로 더 각진
    #    다각형(자형)에 가까워지고, 얕을수록 모서리가 뭉개진 둥근 다각형(타형)에
    #    가까워진다. 알갱이별로 무작위 편차를 두어 한 그림 안에서도 자형~타형이
    #    섞여 보이도록 한다(실제 암석의 모습과 동일).
    mean_euhedral = 0.12 + 0.7 * (t_depth ** 0.6)
    habit = np.clip(rng.normal(loc=mean_euhedral, scale=0.22, size=n_grains), 0.0, 1.0)
    irregularity = 1.0 - habit  # 0=각진 자형, 1=둥글게 뭉개진 타형

    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=110)
    fig.patch.set_alpha(0.0)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_rgb, zorder=0))
    for x, y, r, shade, irr in zip(xs, ys, radii, grain_shade, irregularity):
        grain_rgb = tuple(min(max(c + shade, 0.0), 1.0) for c in bg_rgb)
        # 자형(irr↓): 꼭짓점 4~6개 + 변 거의 일정 → 곧은 변·뚜렷한 모서리(결정면 느낌).
        # 타형(irr↑): 꼭짓점 다수(부드러운 곡선 근사) + 저주파(완만한) 반지름 변화로
        #   각진 데 없이 둥글게 뭉개진 윤곽(틈을 메운 결정 느낌)을 만든다.
        #   (개별 꼭짓점을 독립적으로 들쭉날쭉하게 흔들면 "별 모양"처럼 뾰족해져서
        #   타형의 실제 느낌과 달라지므로, 인접한 꼭짓점끼리 연속적으로 이어지는
        #   저주파 사인 곡선을 섞어 매끄러운 굴곡만 생기도록 한다.)
        n_vertices = int(round(6 + irr * 22))
        base_angle = rng.uniform(0, 2 * np.pi)
        theta = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False) + base_angle

        k1 = rng.integers(2, 4)
        k2 = k1 + rng.integers(1, 3)
        phase1 = rng.uniform(0, 2 * np.pi)
        phase2 = rng.uniform(0, 2 * np.pi)
        smooth_bulge = 0.6 * np.sin(k1 * theta + phase1) + 0.4 * np.sin(k2 * theta + phase2)
        facet_jitter = rng.uniform(-1.0, 1.0, size=n_vertices)
        fine_noise = rng.uniform(-1.0, 1.0, size=n_vertices)

        radius_factor = (
            1.0
            + irr * 0.30 * smooth_bulge          # 타형일수록: 완만하고 둥근 굴곡
            + (1.0 - irr) * 0.12 * facet_jitter   # 자형일수록: 거의 일정(곧은 변)
            + 0.04 * fine_noise                   # 항상 약간의 자연스러운 불완전함
        )
        vertex_radii = r * np.clip(radius_factor, 0.45, 1.6)
        verts = np.column_stack([
            x + vertex_radii * np.cos(theta),
            y + vertex_radii * np.sin(theta),
        ])
        ax.add_patch(
            Polygon(verts, closed=True, facecolor=grain_rgb, edgecolor=(0, 0, 0, 0.28),
                    linewidth=0.4, joinstyle="round", zorder=1)
        )
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

        model_name = st.selectbox(
            "Gemini 모델",
            options=SELECTABLE_MODELS,
            index=SELECTABLE_MODELS.index(DEFAULT_MODEL),
            help="1.5 계열이 종료되어 호출이 실패하면 2.0/2.5 계열로 바꿔 보세요.",
        )

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
