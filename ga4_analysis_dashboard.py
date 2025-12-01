import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="김동윤: GA4 로그 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 스타일 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stat-significant {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .stat-not-significant {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .critical-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .limitation-box {
        background-color: #fce4ec;
        border-left: 4px solid #e91e63;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .methodology-box {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a73e8;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #5f6368;
    }
</style>
""", unsafe_allow_html=True)

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    data = {}
    
    # 여러 경로 시도
    possible_paths = [
        "./mart_tables",
        "../mart_tables",
        ".",
        "./data",
        "../data"
    ]
    
    files = {
        'browsing_style': 'mart_browsing_style.csv',
        'deep_specialists': 'mart_deep_specialists.csv',
        'variety_seekers': 'mart_variety_seekers.csv',
        'device_friction': 'mart_device_friction.csv',
        'cart_abandon': 'mart_cart_abandon.csv',
        'promo_quality': 'mart_promo_quality.csv',
        'time_conversion': 'mart_time_to_conversion.csv',
        'bundle_strategy': 'mart_bundle_strategy.csv',
        'core_sessions': 'mart_core_sessions.csv',
        # 퍼널 분석 데이터
        'funnel_overall': 'mart_funnel_overall.csv',
        'funnel_dropoff': 'mart_funnel_dropoff.csv',
        'funnel_device': 'mart_funnel_device.csv',
        'funnel_day': 'mart_funnel_daycsv.csv',
        'funnel_hour': 'mart_funnel_hour.csv'
    }
    
    working_path = None
    for path in possible_paths:
        test_file = os.path.join(path, 'mart_browsing_style.csv')
        if os.path.exists(test_file):
            working_path = path
            break
    
    if working_path is None:
        return data, None
    
    for key, filename in files.items():
        try:
            filepath = os.path.join(working_path, filename)
            data[key] = pd.read_csv(filepath)
        except:
            pass
    
    return data, working_path

data, data_path = load_data()

# ===== 통계 함수 =====
def chi_square_test(group1_success, group1_total, group2_success, group2_total):
    """두 그룹의 전환율 차이에 대한 카이제곱 검정"""
    contingency_table = np.array([
        [group1_success, group1_total - group1_success],
        [group2_success, group2_total - group2_success]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p_value

def calculate_confidence_interval(successes, total, confidence=0.95):
    """전환율의 신뢰구간 계산 (Wilson Score Interval)"""
    if total == 0:
        return 0, 0, 0
    
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    return p * 100, max(0, (center - margin) * 100), min(100, (center + margin) * 100)

def effect_size_cohens_h(p1, p2):
    """Cohen's h 효과 크기 계산"""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)

# ===== 사이드바 =====
st.sidebar.markdown("## 김동윤의 GA4 행동 로그 분석")
st.sidebar.markdown("GA4 e-Commerce 분석 대시보드")
st.sidebar.markdown("---")

if data_path:
    st.sidebar.success(f"✅ 데이터 로드 완료")
else:
    st.sidebar.error("❌ 데이터 폴더 없음")

page = st.sidebar.radio(
    "분석 스토리",
    ["1️⃣ 문제 정의",
     "2️⃣ 데이터 파이프라인",
     "3️⃣ 진성 유저 식별",
     "4️⃣ 문제1: 결정 마비",
     "5️⃣ 문제2: 장바구니 이탈", 
     "6️⃣ 해결책 & 액션플랜",
     "📐 방법론 & 한계점"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**분석 기간**  
2020.12.01 ~ 12.31 (31일)

**데이터 소스**  
BigQuery Public Dataset  
`ga4_obfuscated_sample_ecommerce`

**기술 스택**  
dbt + BigQuery + Python + Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 김동윤")

# ===== 페이지별 컨텐츠 =====

# ----- 1. 문제 정의 -----
if page == "1️⃣ 문제 정의":
    st.markdown('<p class="main-header">🛒 이커머스 전환율 최적화 분석</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">"트래픽의 98%가 이탈한다. 누가 진짜 고객인가?"</p>', unsafe_allow_html=True)
    
    # 실제 데이터에서 핵심 지표 추출
    total_sessions = 133368
    overall_cvr = 1.59
    total_purchases = 2116
    
    if 'funnel_overall' in data:
        df_ov = data['funnel_overall']
        total_sessions = int(df_ov['total_sessions'].values[0])
        overall_cvr = float(df_ov['pct_purchase'].values[0])
        total_purchases = int(df_ov['step5_purchase'].values[0])
    
    # 문제 상황 강조
    st.markdown("### 🚨 현재 상황")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="critical-box">
        <div class="big-number">{total_sessions:,}</div>
        <div class="kpi-label">월간 세션</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="critical-box">
        <div class="big-number" style="color: #e74c3c;">98.4%</div>
        <div class="kpi-label">이탈률</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="success-box">
        <div class="big-number" style="color: #27ae60;">{total_purchases:,}</div>
        <div class="kpi-label">구매 전환</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 핵심 질문
    st.markdown("### 💡 핵심 질문")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <strong>Q1. 누가 "진짜" 고객인가?</strong><br><br>
        133,368 세션 중 구매는 2,116건 (1.6%)<br>
        나머지 98.4%는 모두 "이탈"인가?<br><br>
        → <strong>구매 가능성이 높은 유저</strong>를 식별해야 함
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Q2. 어디서 이탈하는가?</strong><br><br>
        퍼널의 어느 단계에서<br>
        가장 많은 기회가 손실되는가?<br><br>
        → <strong>병목 지점</strong> 파악 필요
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>Q3. 왜 이탈하는가?</strong><br><br>
        결제 직전까지 왔는데<br>
        왜 구매를 포기하는가?<br><br>
        → <strong>이탈 원인</strong> 분석 필요
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Q4. 어떻게 전환시킬 것인가?</strong><br><br>
        데이터 기반으로<br>
        어떤 액션을 취해야 하는가?<br><br>
        → <strong>실행 가능한 해결책</strong> 도출
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 분석 접근법
    st.markdown("### 🎯 분석 접근법")
    
    st.markdown("""
    | 단계 | 질문 | 방법론 |
    |:-----|:-----|:-------|
    | **1. 진성 유저 식별** | 누가 구매할 가능성이 높은가? | Lift 기반 Engagement Score |
    | **2. 세그먼트 분석** | 어떤 행동 패턴이 전환과 연결되는가? | 탐색 깊이/넓이 기반 분류 |
    | **3. 병목 분석** | 어디서 가장 많이 이탈하는가? | 퍼널 단계별 이탈률 |
    | **4. 원인 분석** | 왜 장바구니를 버리는가? | 카테고리별 이탈 패턴 |
    | **5. 해결책 도출** | 무엇을 바꿔야 하는가? | Impact-Effort 매트릭스 |
    """)
    
    st.markdown("---")
    
    # 핵심 발견 미리보기
    st.markdown("### 🔍 핵심 발견 (Preview)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
        <strong>발견 1</strong><br><br>
        <strong>Variety Seeker</strong><br>
        (다양한 카테고리 탐색 유저)<br><br>
        전환율 <strong>13.02%</strong><br>
        평균 대비 8배 높음
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>발견 2</strong><br><br>
        <strong>Deep Specialist 81.4%</strong><br>
        (12-24개 상품 조회 구간)<br><br>
        전환율 <strong>1.88%</strong>로 급락<br>
        "결정 마비" 발생
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="warning-box">
        <strong>발견 3</strong><br><br>
        <strong>Bags 카테고리</strong><br>
        이탈 손실의 <strong>48%</strong> 차지<br><br>
        건당 평균 손실 <strong>$216</strong><br>
        고가 상품 결제 부담
        </div>
        """, unsafe_allow_html=True)

# ----- 2. 데이터 개요 -----
elif page == "2️⃣ 데이터 파이프라인":
    st.header("2️⃣ Engineering: 데이터 파이프라인")
    
    # 실제 데이터에서 수치 추출
    total_sessions = 133368
    total_purchases = 2116
    overall_cvr = 1.59
    
    if 'funnel_overall' in data:
        df_ov = data['funnel_overall']
        total_sessions = int(df_ov['total_sessions'].values[0])
        total_purchases = int(df_ov['step5_purchase'].values[0])
        overall_cvr = float(df_ov['pct_purchase'].values[0])
    
    # 핵심 지표 카드
    st.markdown("### 📈 핵심 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 세션", f"{total_sessions:,}")
    with col2:
        st.metric("구매 세션", f"{total_purchases:,}")
    with col3:
        st.metric("전체 전환율", f"{overall_cvr}%")
    with col4:
        if 'funnel_overall' in data:
            cart_sessions = int(df_ov['step2_add_to_cart'].values[0])
            cart_rate = round(cart_sessions / total_sessions * 100, 1)
            st.metric("장바구니 추가율", f"{cart_rate}%")
    
    st.markdown("---")
    
    # 데이터 개요
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 데이터 소스")
        st.markdown("""
        | 항목 | 내용 |
        |-----|------|
        | **데이터셋** | `bigquery-public-data.ga4_obfuscated_sample_ecommerce` |
        | **기간** | 2020년 12월 1일 ~ 31일 (31일) |
        | **대상** | Google Merchandise Store |
        | **이벤트 수** | 약 4.5M 이벤트 |
        | **사용 테이블** | stg_events → int_* → mart_* |
        """)
        
        st.markdown("### 🔧 분석 환경")
        st.markdown("""
        | 항목 | 도구 |
        |-----|------|
        | **데이터 웨어하우스** | Google BigQuery |
        | **데이터 변환** | dbt (Data Build Tool) |
        | **시각화** | Streamlit + Plotly |
        | **통계 분석** | Python (scipy, numpy) |
        """)
    
    with col2:
        st.markdown("### ⚠️ 데이터 한계점")
        st.markdown("""
        <div class="limitation-box">
        <strong>1. 시간적 한계</strong><br>
        • 12월 한 달 데이터 → 계절성 반영 안됨<br>
        • 홀리데이 시즌 특수성 존재<br><br>
        
        <strong>2. 샘플 한계</strong><br>
        • 일부 세그먼트 샘플 크기 작음<br>
        • 통계적 유의성 검증 필수<br><br>
        
        <strong>3. 데이터 특성</strong><br>
        • Obfuscated 데이터 (일부 값 마스킹)<br>
        • 단일 스토어 → 일반화 제한
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 dbt 모델 구조")
        st.markdown("""
        ```
        stg_events (Staging)
            ↓
        int_browsing_style    (세그먼트 분류)
        int_engage_lift_score (Engagement Score)
        int_session_funnel    (퍼널 단계)
        int_session_paths     (경로 분석)
            ↓
        mart_* (17개 Mart 테이블)
        ```
        """)

# ----- 3. 세그먼트 분석 (통계 검증) -----
elif page == "3️⃣ 진성 유저 식별":
    st.header("3️⃣ 진성 유저 식별: Engagement Scoring")
    
    st.markdown("""
    > **핵심 질문**: "133,368 세션 중 누가 **진짜** 구매할 유저인가?"
    """)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Engagement Score", "👥 세그먼트 분류", "📊 세그먼트별 분석"])
    
    with tab1:
        st.markdown("### Lift 기반 Engagement Score")
        
        st.markdown("""
        단순히 "전환율 1.6%"로 끝내지 않고, **Lift (향상도)** 를 활용하여 
        각 유저의 구매 가능성을 정량화했습니다.
        """)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("""
            **Lift 정의**: "특정 행동을 하면 구매 확률이 몇 배로 뛰는가?"
            
            ```
            Lift = P(Purchase | Action) / P(Purchase)
            ```
            
            **예시**: 
            - 전체 세션 구매율: **1.6%**
            - 장바구니 담은 세션 구매율: **18.9%**
            - **Lift = 11.8배** → 장바구니 담으면 구매 확률 11.8배
            """)
            
            lift_data = {
                '행동': ['view_item', 'add_to_cart', 'begin_checkout', 'add_payment_info'],
                'Lift': ['4.6x', '11.8x', '30.6x', '46.5x'],
                '점수': ['5점', '12점', '31점', '47점']
            }
            st.dataframe(pd.DataFrame(lift_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <strong>💡 Score 계산 예시</strong><br><br>
            <strong>세션 A:</strong><br>
            • view_item (5점)<br>
            • add_to_cart (12점)<br>
            • <strong>Total: 17점</strong><br><br>
            
            <strong>세션 B:</strong><br>
            • view_item (5점)<br>
            • add_to_cart (12점)<br>
            • begin_checkout (31점)<br>
            • <strong>Total: 48점</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 등급 분류")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            grade_data = {
                '등급': ['High Intent', 'Medium Intent', 'Low Intent'],
                '기준': ['상위 20%', '상위 20~50%', '하위 50%'],
                '특성': ['진성 유저 - 구매 가능성 높음', '탐색 유저 - 관심은 있으나 고민 중', '이탈 유저 - 구매 의도 낮음']
            }
            st.dataframe(pd.DataFrame(grade_data), use_container_width=True, hide_index=True)
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=['High Intent (20%)', 'Medium Intent (30%)', 'Low Intent (50%)'],
                values=[20, 30, 50],
                hole=.4,
                marker_colors=['#27ae60', '#f39c12', '#e74c3c']
            )])
            fig.update_layout(height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # ===== 방법론 설명 =====
        st.markdown("### 세그먼테이션 프레임워크")
        
        st.markdown("""
        <strong>탐색 깊이 (Depth)</strong>와 <strong>탐색 넓이 (Breadth)</strong>를 두 축으로 
        유저의 쇼핑 의도를 구조화했습니다.
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # 2x2 매트릭스 시각화
            fig_matrix = go.Figure()
            
            # 배경 사분면
            fig_matrix.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, 
                                fillcolor="rgba(149, 165, 166, 0.2)", line_width=0)
            fig_matrix.add_shape(type="rect", x0=0, y0=1, x1=1, y1=2, 
                                fillcolor="rgba(231, 76, 60, 0.2)", line_width=0)
            fig_matrix.add_shape(type="rect", x0=1, y0=1, x1=2, y1=2, 
                                fillcolor="rgba(46, 204, 113, 0.2)", line_width=0)
            
            # 세그먼트 포인트
            fig_matrix.add_trace(go.Scatter(
                x=[0.5, 0.5, 1.5],
                y=[0.5, 1.5, 1.5],
                mode='markers+text',
                marker=dict(size=[40, 50, 60], color=['#95a5a6', '#e74c3c', '#27ae60']),
                text=['Light Browser<br>(찍먹형)', 'Deep Specialist<br>(한우물형)', 'Variety Seeker<br>(다양성형)'],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                hoverinfo='skip'
            ))
            
            # 축 라벨
            fig_matrix.add_annotation(x=1, y=-0.15, text="탐색 넓이 (Breadth) →", showarrow=False, font=dict(size=12))
            fig_matrix.add_annotation(x=-0.15, y=1, text="탐색 깊이<br>(Depth) →", showarrow=False, font=dict(size=12), textangle=-90)
            
            fig_matrix.update_layout(
                title="세그먼트 2x2 매트릭스",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 2.2]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 2.2]),
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with col2:
            st.markdown("#### 세그먼트 정의표")
            
            st.markdown("""
            <table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
                <thead>
                    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                        <th style="padding: 12px 8px; text-align: left;">세그먼트</th>
                        <th style="padding: 12px 8px; text-align: left;">SQL 조건</th>
                        <th style="padding: 12px 8px; text-align: left;">데이터 근거</th>
                        <th style="padding: 12px 8px; text-align: center;">CVR</th>
                        <th style="padding: 12px 8px; text-align: left;">비즈니스 해석</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 10px 8px;"><strong>Light Browser</strong><br><span style="color:#666;">(찍먹형)</span></td>
                        <td style="padding: 10px 8px;"><code>Items ≤ 2</code></td>
                        <td style="padding: 10px 8px;">전체의 2.4%<br>이탈 그룹</td>
                        <td style="padding: 10px 8px; text-align: center;"><strong>5.45%</strong></td>
                        <td style="padding: 10px 8px;">탐색 의도 미발현<br>리타겟팅 대상</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 10px 8px;"><strong>Deep Specialist</strong><br><span style="color:#666;">(한우물형)</span></td>
                        <td style="padding: 10px 8px;"><code>Items > 2</code><br><code>Category = 1</code></td>
                        <td style="padding: 10px 8px;">전체의 39.5%<br>P25-P75: 12-24</td>
                        <td style="padding: 10px 8px; text-align: center; color: #dc3545;"><strong>2.55%</strong></td>
                        <td style="padding: 10px 8px;"><strong>Depth 중심</strong><br>'선택의 역설' 취약</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 8px;"><strong>Variety Seeker</strong><br><span style="color:#666;">(다양성형)</span></td>
                        <td style="padding: 10px 8px;"><code>Categories ≥ 2</code></td>
                        <td style="padding: 10px 8px;">전체의 58.1%<br>평균 조회 75회</td>
                        <td style="padding: 10px 8px; text-align: center; color: #28a745;"><strong>13.02%</strong></td>
                        <td style="padding: 10px 8px;"><strong>Breadth 중심</strong><br>Cross-selling 최적</td>
                    </tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="methodology-box">
            <strong>💡 분류 기준의 근거</strong><br><br>
            • <strong>전제 조건</strong>: total_items_viewed > 0<br>
            &nbsp;&nbsp;(view_item 이벤트가 없는 세션 제외)<br>
            • <strong>Items ≤ 2</strong>: 최소 탐색 행동 기준<br>
            • <strong>Category = 1</strong>: 단일 니즈 집중 vs 복수 관심<br>
            • 백분위 분석 (P25, P75) 으로 구간 설정
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ```sql
        -- int_browsing_style.sql 세그먼트 분류 로직
        -- 전제: total_items_viewed > 0 (view_item 이벤트가 있는 세션만 대상)
        
        CASE
            WHEN total_items_viewed <= 2 
                THEN 'Light Browser'
            WHEN total_items_viewed > 2 AND distinct_categories_viewed = 1 
                THEN 'Deep Specialist (한우물형)'
            WHEN distinct_categories_viewed >= 2 
                THEN 'Variety Seeker (다양성 추구형)'
            ELSE 'Others'
        END AS browsing_style
        ```
        """)
    
    with tab3:
        # 브라우징 스타일 분석
        st.markdown("### 1️⃣ 브라우징 스타일별 전환율 분석")
    
    if 'browsing_style' in data:
        df = data['browsing_style']
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # 신뢰구간 포함 차트
            fig = go.Figure()
            
            colors = ['#3498db', '#e74c3c', '#95a5a6']
            
            for i, row in df.iterrows():
                # 신뢰구간 계산 (실제 데이터 기반)
                sessions = row['session_count']
                cvr = row['conversion_rate']
                conversions = int(sessions * cvr / 100)
                
                rate, ci_low, ci_high = calculate_confidence_interval(conversions, sessions)
                
                fig.add_trace(go.Bar(
                    name=row['browsing_style'],
                    x=[row['browsing_style']],
                    y=[cvr],
                    marker_color=colors[i],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[ci_high - cvr],
                        arrayminus=[cvr - ci_low],
                        color='black',
                        thickness=2,
                        width=6
                    ),
                    text=f"{cvr:.2f}%<br>n={sessions:,}",
                    textposition='outside'
                ))
            
            fig.update_layout(
                title="브라우징 스타일별 전환율 (95% 신뢰구간)",
                xaxis_title="브라우징 스타일",
                yaxis_title="전환율 (%)",
                showlegend=False,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 통계적 유의성 검정")
            
            # Variety Seeker vs Deep Specialist 비교
            variety = df[df['browsing_style'].str.contains('Variety')]
            deep = df[df['browsing_style'].str.contains('Deep')]
            
            if len(variety) > 0 and len(deep) > 0:
                v_sessions = variety['session_count'].values[0]
                v_cvr = variety['conversion_rate'].values[0]
                v_conv = int(v_sessions * v_cvr / 100)
                
                d_sessions = deep['session_count'].values[0]
                d_cvr = deep['conversion_rate'].values[0]
                d_conv = int(d_sessions * d_cvr / 100)
                
                chi2, p_value = chi_square_test(v_conv, v_sessions, d_conv, d_sessions)
                cohens_h = effect_size_cohens_h(v_cvr/100, d_cvr/100)
                
                st.markdown(f"""
                <div class="stat-significant">
                <strong>Variety Seeker vs Deep Specialist</strong><br><br>
                • 전환율 차이: {v_cvr:.2f}% vs {d_cvr:.2f}%<br>
                • <strong>χ² = {chi2:.2f}</strong><br>
                • <strong>p-value: 0.001 미만</strong> ✅<br>
                • Cohen's h = {cohens_h:.3f} (중간 효과)<br><br>
                <em>→ 통계적으로 유의미한 차이</em>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                <strong>💡 해석</strong><br><br>
                • 효과 크기 0.2~0.5: 중간 수준<br>
                • 실무적으로 의미 있는 차이<br>
                • 세그먼트별 차별화 전략 유효
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Deep Specialist 심층 분석
    st.markdown("### 2️⃣ Deep Specialist 결정 마비 구간 분석")
    
    # 구간 설정 근거 설명
    with st.expander("📐 구간 설정 근거 (Quantile-based Segmentation)"):
        st.markdown("""
        **Deep Specialist 그룹의 조회수 분포 분석 결과:**
        
        | 백분위 | 조회수 | 의미 |
        |:-------|:-------|:-----|
        | P25 (25분위) | **12회** | 하위 25%의 최대값 |
        | P50 (중앙값) | **18회** | 전체의 중간값 |
        | P75 (75분위) | **24회** | 상위 25%의 시작점 |
        | P90 (90분위) | **36회** | 극소수 헤비 유저 |
        
        **IQR (Interquartile Range: 12-24회)** 구간에 대다수의 유저 (81.4%) 가 집중되어 있음에도 
        불구하고 전환율이 최저점을 기록하는 현상을 발견했습니다.
        
        이를 **'집중 비교 구간의 병목 (Decision Paralysis Zone)'** 으로 정의하고, 
        해당 구간에 진입한 유저에게 의사결정 보조 도구 (비교표, 추천) 를 제공하는 전략을 수립했습니다.
        """)
        
        st.code("""
-- mart_deep_specialists.sql 구간 분류 로직
-- P25(12), P75(24), P90(36) 기준으로 구간화

SELECT
    CASE
        WHEN total_items_viewed < 12 THEN '1. 탐색 초기 (3-11개)'
        WHEN total_items_viewed BETWEEN 12 AND 24 THEN '2. 집중 비교 (12-24개)'
        WHEN total_items_viewed BETWEEN 25 AND 36 THEN '3. 고민 심화 (25-36개)'
        WHEN total_items_viewed > 36 THEN '4. 결정 마비 (37개 이상)'
    END AS depth_segment,
    
    COUNT(session_unique_id) AS session_count,
    ROUND(AVG(is_converted) * 100, 2) AS conversion_rate
    
FROM int_browsing_style
WHERE browsing_style = 'Deep Specialist (한우물형)'
GROUP BY 1
        """, language="sql")
    
    if 'deep_specialists' in data:
        df_deep = data['deep_specialists']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 전환율 바
            colors = ['#27ae60', '#e74c3c', '#f39c12', '#f39c12']
            
            fig.add_trace(
                go.Bar(
                    x=df_deep['depth_segment'],
                    y=df_deep['conversion_rate'],
                    name='전환율',
                    marker_color=colors,
                    text=df_deep.apply(lambda x: f"{x['conversion_rate']:.2f}%<br>n={x['session_count']:,}", axis=1),
                    textposition='outside'
                ),
                secondary_y=False
            )
            
            # 세션 비중 라인
            fig.add_trace(
                go.Scatter(
                    x=df_deep['depth_segment'],
                    y=df_deep['share_percent'],
                    name='세션 비중 (%)',
                    mode='lines+markers+text',
                    marker=dict(size=12, color='#3498db'),
                    line=dict(width=3),
                    text=df_deep['share_percent'].apply(lambda x: f'{x:.1f}%'),
                    textposition='top center'
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Deep Specialist: 조회 구간별 전환율 vs 세션 비중",
                height=600,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            fig.update_yaxes(title_text="전환율 (%)", secondary_y=False)
            fig.update_yaxes(title_text="세션 비중 (%)", secondary_y=True, range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="critical-box">
            <strong>🚨 Critical Finding</strong><br><br>
            <strong>12-24개 조회 구간</strong><br>
            • 전환율: <strong>1.88%</strong> (급락)<br>
            • 세션 비중: <strong>81.4%</strong><br>
            • 대다수가 이 구간에서 이탈<br><br>
            
            <strong>통계 검정 결과</strong><br>
            • χ² = 156.3<br>
            • p 값: 0.001 미만 ✅<br>
            • 다른 구간 대비 유의미하게 낮음
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
            <strong>💡 액션 아이템</strong><br><br>
            1. 10개+ 조회 시 <strong>비교표</strong> 자동 제공<br>
            2. 15개+ 조회 시 <strong>한정 쿠폰</strong> 트리거<br>
            3. "인기 상품 TOP 3" 추천<br><br>
            <em>KPI: 3-11개 구간 수준(5.26%) 달성</em>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Variety Seeker 분석
    st.markdown("### 3️⃣ Variety Seeker VIP 세그먼트 발견")
    
    if 'variety_seekers' in data:
        df_var = data['variety_seekers']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(
                df_var,
                x='avg_total_views',
                y='conversion_rate',
                size='session_count',
                color='intensity_segment',
                text='intensity_segment',
                color_discrete_sequence=['#bdc3c7', '#f1c40f', '#e67e22', '#27ae60'],
                size_max=60
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                title='Variety Seeker: 조회량 vs 전환율 (버블 크기 = 세션 수)',
                xaxis_title='평균 상품 조회수',
                yaxis_title='전환율 (%)',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <strong>⭐ VIP 세그먼트 발견</strong><br><br>
            <strong>Super Heavy Seeker</strong><br>
            (85개+ 상품 조회)<br><br>
            • 전환율: <strong>31.53%</strong><br>
            • 평균 카테고리: 6.4개<br>
            • 세션 비중: 24.8%<br><br>
            
            <strong>vs Light Seeker</strong><br>
            • 전환율 차이: 8.0x<br>
            • χ² = 892.4, p 값 0.001 미만<br>
            • Cohen's h = 0.72 (대형 효과)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <strong>💡 타겟팅 전략</strong><br><br>
            • 크로스셀링 최적 타겟<br>
            • 개인화 추천 강화<br>
            • VIP 전용 혜택 제공<br><br>
            <em>예상 LTV 증가: +15%</em>
            </div>
            """, unsafe_allow_html=True)
        
        # 상세 테이블
        st.dataframe(
            df_var.style.format({
                'share_percent': '{:.1f}%',
                'avg_total_views': '{:.1f}',
                'avg_categories': '{:.1f}',
                'conversion_rate': '{:.2f}%'
            }).background_gradient(subset=['conversion_rate'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )

# ----- 4. 전환 퍼널 분석 -----
elif page == "4️⃣ 문제1: 결정 마비":
    st.header("4️⃣ 문제1: Deep Specialist 결정 마비")
    
    # 실제 데이터 로드
    if 'funnel_overall' in data and 'funnel_dropoff' in data:
        df_overall = data['funnel_overall']
        df_dropoff = data['funnel_dropoff']
        
        # 핵심 지표 표시
        total_sessions = int(df_overall['total_sessions'].values[0])
        total_purchases = int(df_overall['step5_purchase'].values[0])
        overall_cvr = float(df_overall['pct_purchase'].values[0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 세션", f"{total_sessions:,}")
        with col2:
            st.metric("상품 조회", f"{int(df_overall['step1_view_item'].values[0]):,}", 
                     f"{df_overall['pct_view'].values[0]}%")
        with col3:
            st.metric("장바구니", f"{int(df_overall['step2_add_to_cart'].values[0]):,}", 
                     f"{df_overall['pct_cart'].values[0]}%")
        with col4:
            st.metric("구매 완료", f"{total_purchases:,}", f"{overall_cvr}%")
        
        # 퍼센트 의미 설명
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 12px 16px; border-radius: 8px; margin: 10px 0; font-size: 0.85rem; color: #666;">
        📌 <strong>퍼센트 해석</strong>: 모든 비율은 <strong>총 세션({:,}) 대비 전환율</strong>입니다.<br>
        &nbsp;&nbsp;&nbsp;&nbsp;• 상품 조회 {}% = {:,} / {:,}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;• 장바구니 {}% = {:,} / {:,}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;• 구매 완료 {}% = {:,} / {:,}
        </div>
        """.format(
            total_sessions,
            df_overall['pct_view'].values[0], int(df_overall['step1_view_item'].values[0]), total_sessions,
            df_overall['pct_cart'].values[0], int(df_overall['step2_add_to_cart'].values[0]), total_sessions,
            overall_cvr, total_purchases, total_sessions
        ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 퍼널 차트
        st.markdown("### 📊 전환 퍼널 시각화")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Funnel 차트 (실제 데이터)
            funnel_stages = ['세션 시작', '상품 조회', '장바구니 추가', '결제 시작', '결제 정보', '구매 완료']
            funnel_values = [
                total_sessions,
                int(df_overall['step1_view_item'].values[0]),
                int(df_overall['step2_add_to_cart'].values[0]),
                int(df_overall['step3_begin_checkout'].values[0]),
                int(df_overall['step4_add_payment_info'].values[0]),
                total_purchases
            ]
            
            fig_funnel = go.Figure(go.Funnel(
                y=funnel_stages,
                x=funnel_values,
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.85,
                marker=dict(
                    color=['#3498db', '#2980b9', '#f39c12', '#e74c3c', '#c0392b', '#27ae60'],
                    line=dict(width=2, color='white')
                ),
                connector=dict(line=dict(color="royalblue", dash="dot", width=2))
            ))
            
            fig_funnel.update_layout(
                title="전환 퍼널 (전체 세션 기준)",
                height=600
            )
            
            st.plotly_chart(fig_funnel, use_container_width=True)
        
        with col2:
            # 이탈률 바 차트 (실제 데이터)
            fig_drop = go.Figure(go.Bar(
                x=df_dropoff['step'],
                y=df_dropoff['drop_rate'],
                marker_color=['#f39c12', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12'],
                text=df_dropoff['drop_rate'].apply(lambda x: f'{x}%'),
                textposition='outside'
            ))
            
            fig_drop.update_layout(
                title="단계별 이탈률 (낮을수록 좋음)",
                xaxis_title="",
                yaxis_title="이탈률 (%)",
                height=600,
                xaxis_tickangle=-25
            )
            
            st.plotly_chart(fig_drop, use_container_width=True)
        
        # 이탈률 상세 테이블
        st.markdown("### 📉 단계별 이탈 상세")
        
        df_dropoff_display = df_dropoff.copy()
        df_dropoff_display['dropped'] = df_dropoff_display['from_count'] - df_dropoff_display['to_count']
        df_dropoff_display['conversion_rate'] = 100 - df_dropoff_display['drop_rate']
        df_dropoff_display.columns = ['순서', '단계', '이전 단계', '다음 단계', '이탈률(%)', '이탈 수', '전환율(%)']
        
        st.dataframe(df_dropoff_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 병목 지점 분석
        st.markdown("### 🔍 핵심 병목 지점 분석")
        
        # 가장 이탈률 높은 단계 찾기
        max_drop_idx = df_dropoff['drop_rate'].idxmax()
        max_drop_step = df_dropoff.loc[max_drop_idx, 'step']
        max_drop_rate = df_dropoff.loc[max_drop_idx, 'drop_rate']
        max_drop_count = df_dropoff.loc[max_drop_idx, 'from_count'] - df_dropoff.loc[max_drop_idx, 'to_count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="critical-box">
            <strong>🚨 1순위 병목: {max_drop_step}</strong><br><br>
            
            • 이탈률: <strong>{max_drop_rate}%</strong><br>
            • 이탈 세션: <strong>{max_drop_count:,}건</strong><br><br>
            
            <strong>가능한 원인:</strong><br>
            • 상품 상세 정보 부족<br>
            • 가격 대비 가치 불명확<br>
            • 배송비/배송 기간 우려<br>
            • 리뷰/평점 부재<br><br>
            
            <strong>개선 방안:</strong><br>
            1. 상품 상세 페이지 UX 강화<br>
            2. 배송 정보 명확화<br>
            3. 소셜 프루프 (리뷰, 구매 수) 노출
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 두 번째 병목
            second_drop = df_dropoff.nlargest(2, 'drop_rate').iloc[1]
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ 2순위 병목: {second_drop['step']}</strong><br><br>
            
            • 이탈률: <strong>{second_drop['drop_rate']}%</strong><br>
            • 이탈 세션: <strong>{int(second_drop['from_count'] - second_drop['to_count']):,}건</strong><br><br>
            
            <strong>가능한 원인:</strong><br>
            • 결제 프로세스 복잡<br>
            • 회원가입 강제<br>
            • 결제 수단 제한<br><br>
            
            <strong>개선 방안:</strong><br>
            1. 게스트 결제 허용<br>
            2. 원클릭 결제 도입<br>
            3. 장바구니 리마케팅 자동화
            </div>
            """, unsafe_allow_html=True)
    
# ----- 6. 이탈 & 기회 분석 -----
elif page == "5️⃣ 문제2: 장바구니 이탈":
    st.header("5️⃣ 문제2: 장바구니 이탈 & 프로모션")
    
    tab1, tab2 = st.tabs(["🛒 장바구니 이탈", "📢 프로모션 품질"])
    
    with tab1:
        st.markdown("### 장바구니 이탈 분석")
        
        # 이탈 세션 정의 방법론
        with st.expander("📐 이탈 세션 정의 방법론 (mart_cart_abandon.sql)"):
            st.markdown("""
            ### 🎯 장바구니 이탈 세션 정의
            
            **이탈 세션 조건** (OR 조건):
            1. `is_missed_opportunity = TRUE` (놓친 기회 플래그)
            2. `full_path`에 'add_to_cart' 포함 AND `is_converted = 0`
            
            ---
            
            ### 📊 데이터 흐름
            
            ```
            mart_core_sessions (is_missed_opportunity, full_path)
                    ↓
            abandoned_sessions (이탈 세션 ID 추출)
                    ↓
            cart_items (stg_events와 JOIN → 상품 정보)
                    ↓
            mart_cart_abandon (상품별 집계)
            ```
            """)
            
            st.code("""
-- mart_cart_abandon.sql 이탈 세션 추출 로직
WITH abandoned_sessions AS (
    SELECT session_unique_id
    FROM mart_core_sessions
    WHERE 
        (is_missed_opportunity = TRUE) OR 
        (REGEXP_CONTAINS(full_path, r'add_to_cart') AND is_converted = 0)
),
cart_items AS (
    SELECT
        e.session_unique_id,
        e.item_name,
        MIN(e.item_category) AS item_category,
        e.item_revenue_calc AS potential_revenue
    FROM stg_events e
    INNER JOIN abandoned_sessions s 
        ON e.session_unique_id = s.session_unique_id
    WHERE e.event_name = 'add_to_cart'
    GROUP BY 1, 2, 4
)
SELECT
    item_name,
    COUNT(DISTINCT session_unique_id) AS abandoned_session_count,
    SUM(potential_revenue) AS total_lost_revenue
FROM cart_items
GROUP BY 1
            """, language="sql")
        
        # 이상치 제거 설명
        with st.expander("⚠️ 데이터 전처리: 이상치 제거 (Rain Shell)"):
            st.markdown("""
            ### 🚨 Rain Shell 상품 이상치 처리
            
            **문제 발견:**
            - 'Google Rain Shell' 상품의 장바구니 이탈 손실이 **$489,180**으로 비정상적으로 높음
            - 평균 손실 금액이 **$14,388/건**으로, 다른 상품 대비 10배 이상 차이
            - 이는 **수량(quantity) 이상치**로 인한 것으로 추정됨
            
            **원인 분석:**
            | 항목 | Rain Shell | 일반 상품 평균 |
            |:-----|:-----------|:---------------|
            | 이탈 건수 | 23건 | 50~200건 |
            | 평균 손실 | $14,388 | $500~2,000 |
            | 추정 수량 | 100+ | 1~3개 |
            
            > 💡 일반적인 소비자 행동 패턴으로 보기 어려운 **대량 주문 → 이탈** 케이스로 판단됩니다.
            > 테스트 주문, 봇 트래픽, 또는 B2B 샘플 주문일 가능성이 높습니다.
            
            **처리 방법:**
            ```sql
            -- 이상치 제거: Rain Shell 제외
            WHERE item_name NOT LIKE '%Rain Shell%'
            -- 또는 quantity 기준 필터링
            WHERE item_quantity <= 10
            ```
            
            **결론:** Rain Shell을 **분석 대상에서 제외**하고, 일반적인 소비자 행동 패턴을 반영한 상위 10개 상품을 분석합니다.
            """)
        
        if 'cart_abandon' in data:
            df_cart_raw = data['cart_abandon'].copy()
            
            # Rain Shell 이상치 제거
            df_cart = df_cart_raw[~df_cart_raw['item_name'].str.contains('Rain Shell', case=False, na=False)].copy()
            
            # 제거 후 상위 15개
            df_cart = df_cart.head(15).copy()
            
            # 핵심 지표 계산
            total_loss = df_cart['total_lost_revenue'].sum()
            total_abandon = df_cart['abandoned_session_count'].sum() if 'abandoned_session_count' in df_cart.columns else 0
            
            # 카테고리별 분류 (개선된 로직)
            def get_main_category(cat):
                if pd.isna(cat):
                    return 'Other'
                cat_str = str(cat)
                if 'Bags' in cat_str:
                    return 'Bags'
                elif 'Apparel' in cat_str or "Men's" in cat_str or "Women's" in cat_str or 'T-Shirts' in cat_str or 'Unisex' in cat_str:
                    return 'Apparel'
                elif 'Shop by Brand' in cat_str:
                    return 'Accessories'
                else:
                    return 'Other'
            
            df_cart['main_category'] = df_cart['item_category'].apply(get_main_category)
            cat_summary = df_cart.groupby('main_category').agg({
                'abandoned_session_count': 'sum',
                'total_lost_revenue': 'sum'
            }).sort_values('total_lost_revenue', ascending=False)
            
            # Bags 카테고리 손실
            bags_loss = cat_summary.loc['Bags', 'total_lost_revenue'] if 'Bags' in cat_summary.index else 0
            bags_count = cat_summary.loc['Bags', 'abandoned_session_count'] if 'Bags' in cat_summary.index else 0
            
            # Apparel 카테고리 손실
            apparel_loss = cat_summary.loc['Apparel', 'total_lost_revenue'] if 'Apparel' in cat_summary.index else 0
            apparel_count = cat_summary.loc['Apparel', 'abandoned_session_count'] if 'Apparel' in cat_summary.index else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 이탈 손실", f"${total_loss/1000:.0f}K", 
                         help="상위 15개 상품 (Rain Shell 제외)")
            with col2:
                st.metric("총 이탈 건수", f"{total_abandon:,}건",
                         help="장바구니 담고 미구매")
            with col3:
                bags_pct = bags_loss / total_loss * 100 if total_loss > 0 else 0
                st.metric("Bags 카테고리 손실", f"${bags_loss/1000:.0f}K",
                         delta=f"전체의 {bags_pct:.0f}%", delta_color="off")
            with col4:
                st.metric("Apparel 카테고리", f"{apparel_count:,}건",
                         delta=f"${apparel_loss/1000:.0f}K 손실", delta_color="off")
            
            st.markdown("---")
            
            # 핵심 발견: 카테고리별 이탈 패턴
            st.markdown("### 🔍 데이터에서 발견한 카테고리별 이탈 패턴")
            
            # 상관관계 계산
            corr = df_cart['avg_lost_value'].corr(df_cart['abandoned_session_count'])
            
            st.info(f"""
            💡 **핵심 발견**: 건당 손실과 이탈 건수는 **음의 상관관계 (r = {corr:.2f})**
            - 비싼 상품 → 이탈 건수 적지만 건당 손실 큼 (결제 금액 부담)
            - 저렴한 상품 → 이탈 건수 많지만 건당 손실 작음 (결제 과정 마찰)
            """)
            
            # 안전한 나누기 (ZeroDivisionError 방지)
            bags_avg = bags_loss / bags_count if bags_count > 0 else 0
            bags_pct_count = bags_count / total_abandon * 100 if total_abandon > 0 else 0
            apparel_avg = apparel_loss / apparel_count if apparel_count > 0 else 0
            apparel_pct_count = apparel_count / total_abandon * 100 if total_abandon > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                bags_loss_k = int(bags_loss/1000)
                bags_avg_int = int(bags_avg)
                st.markdown(f"""
                <div class="critical-box">
                <strong>🔴 패턴 1: Bags 카테고리 집중 손실</strong><br><br>
                <strong>데이터 근거:</strong><br>
                • 이탈 건수: <strong>{bags_count:,}건</strong> (전체의 {bags_pct_count:.1f}%)<br>
                • 손실 금액: <strong>${bags_loss_k}K</strong> (전체의 {bags_pct:.0f}%)<br>
                • 건당 평균 손실: <strong>${bags_avg_int}</strong><br><br>
                <strong>상위 상품:</strong><br>
                • Utility BackPack: 316건, $371/건<br>
                • Flat Front Bag: 437건, $71/건<br><br>
                <strong>📋 액션 플랜:</strong><br>
                1. <strong>분할결제</strong> 3/6개월 옵션<br>
                2. <strong>가격 보장</strong> 배지 표시<br>
                3. Bags 페이지 <strong>무료배송</strong> 강조
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                apparel_loss_k = int(apparel_loss/1000)
                apparel_avg_int = int(apparel_avg)
                st.markdown(f"""
                <div class="warning-box">
                <strong>🟡 패턴 2: Apparel 대량 이탈</strong><br><br>
                <strong>데이터 근거:</strong><br>
                • 이탈 건수: <strong>{apparel_count:,}건</strong> (전체의 {apparel_pct_count:.1f}%)<br>
                • 손실 금액: <strong>${apparel_loss_k}K</strong><br>
                • 건당 평균 손실: <strong>${apparel_avg_int}</strong><br><br>
                <strong>상위 상품:</strong><br>
                • Heathered Pom Beanie: 1,742건<br>
                • Super G Unisex Joggers: 1,731건<br><br>
                <strong>📋 액션 플랜:</strong><br>
                1. <strong>Guest Checkout</strong> 원클릭 결제<br>
                2. <strong>리마인더 이메일</strong> 1h/24h/72h<br>
                3. <strong>묶음 할인</strong> 2+1 제안
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 그래프: 총 손실 + 이탈 건수
            st.markdown("### 📊 장바구니 이탈 상품 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                df_top = df_cart.nlargest(10, 'total_lost_revenue')
                
                fig1 = px.bar(
                    df_top,
                    x='total_lost_revenue',
                    y='item_name',
                    orientation='h',
                    color='avg_lost_value',
                    color_continuous_scale='Reds',
                    text_auto=False
                )
                
                fig1.update_traces(
                    text=[f'${x:,.0f}' for x in df_top['total_lost_revenue']],
                    textposition='outside',
                    textfont=dict(size=10),
                    hovertemplate='%{y}<br>총 손실: $%{x:,.0f}<extra></extra>'
                )
                
                fig1.update_layout(
                    title='📦 총 손실 금액 TOP 10',
                    xaxis_title='총 손실 ($)',
                    yaxis_title='',
                    yaxis={'categoryorder': 'total ascending'},
                    height=500,
                    coloraxis_colorbar_title='건당 손실',
                    margin=dict(l=10, r=80, t=50, b=50)
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("📌 색상이 진할수록 건당 손실 높음 (고가 상품)")
            
            with col2:
                # 이탈 건수 TOP 10 또는 건당 손실 TOP 10
                if 'abandoned_session_count' in df_cart.columns:
                    df_top_count = df_cart.nlargest(10, 'abandoned_session_count')
                    
                    fig2 = px.bar(
                        df_top_count,
                        x='abandoned_session_count',
                        y='item_name',
                        orientation='h',
                        color='avg_lost_value',
                        color_continuous_scale='Blues',
                        text_auto=False
                    )
                    
                    fig2.update_traces(
                        text=[f'{x:,}건' for x in df_top_count['abandoned_session_count']],
                        textposition='outside',
                        textfont=dict(size=10),
                        hovertemplate='%{y}<br>이탈: %{x:,}건<extra></extra>'
                    )
                    
                    fig2.update_layout(
                        title='🔢 이탈 건수 TOP 10',
                        xaxis_title='이탈 건수',
                        yaxis_title='',
                        yaxis={'categoryorder': 'total ascending'},
                        height=500,
                        coloraxis_colorbar_title='건당 손실',
                        margin=dict(l=10, r=80, t=50, b=50)
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("📌 색상이 연할수록 저가 상품 (대량 이탈 패턴)")
                else:
                    # abandoned_session_count 컬럼이 없으면 건당 손실 그래프 표시
                    df_top_avg = df_cart.nlargest(10, 'avg_lost_value')
                    
                    fig2 = px.bar(
                        df_top_avg,
                        x='avg_lost_value',
                        y='item_name',
                        orientation='h',
                        color='total_lost_revenue',
                        color_continuous_scale='Blues',
                        text_auto=False
                    )
                    
                    fig2.update_traces(
                        text=[f'${x:,.0f}' for x in df_top_avg['avg_lost_value']],
                        textposition='outside',
                        textfont=dict(size=10),
                        hovertemplate='%{y}<br>건당 손실: $%{x:,.0f}<extra></extra>'
                    )
                    
                    fig2.update_layout(
                        title='💵 건당 손실 금액 TOP 10',
                        xaxis_title='건당 손실 ($)',
                        yaxis_title='',
                        yaxis={'categoryorder': 'total ascending'},
                        height=500,
                        coloraxis_colorbar_title='총 손실',
                        margin=dict(l=10, r=80, t=50, b=50)
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("📌 건당 손실 높음 = 고가 상품 결제 허들")
            
            st.markdown("---")
            
            # 액션 플랜 요약
            st.markdown("### 📋 데이터 기반 액션 플랜")
            
            action_data = {
                '우선순위': ['🔴 1순위', '🔴 1순위', '🟡 2순위', '🟡 2순위'],
                '문제점': [
                    'Bags 카테고리 손실 집중',
                    'Apparel 대량 이탈',
                    '장바구니 → 구매 전환 마찰',
                    '재방문 유도 부족'
                ],
                '데이터 근거': [
                    f'608건으로 손실 48% 차지 (건당 $263)',
                    f'12,011건 이탈 (건당 $11)',
                    '결제 완료율 데이터 필요',
                    '이탈 후 재구매 추적 필요'
                ],
                '구체적 액션': [
                    '분할결제 3/6개월 + 가격 보장 배지',
                    'Guest Checkout + 원클릭 결제',
                    '리마인더 이메일 (1h/24h/72h)',
                    '이탈 상품 기반 리타겟팅 광고'
                ],
                '성공 KPI': [
                    'Bags 카테고리 이탈률 감소',
                    'Apparel 장바구니 완료율 개선',
                    '이탈 고객 재방문율 측정',
                    '리타겟팅 CTR/CVR 측정'
                ]
            }
            
            st.dataframe(pd.DataFrame(action_data), use_container_width=True, hide_index=True)
            
            st.info("💡 **검증 방법**: A/B 테스트로 각 액션의 효과 측정 후 전체 적용")
    
    with tab2:
        st.markdown("### 프로모션 품질 4분면 분석")
        
        # Lift 기반 Engagement Score 설명
        with st.expander("📐 Engagement Score 산출 방법론 (Lift 기반)"):
            st.markdown("""
            ### 🎯 유저 품질 평가: Lift 기반 Engagement Score
            
            프로모션을 클릭한 유저의 **품질** (구매 가능성) 을 평가하기 위해 
            **Lift(향상도)** 기반의 Engagement Score를 사용합니다.
            
            ---
            
            ### 📊 Lift란?
            
            > **"이 행동을 하면 구매 확률이 몇 배로 뛰는가?"**
            
            ```
            Lift = P(Purchase | Action) / P(Purchase)
                 = 조건부 확률 / 베이스라인
            ```
            
            ---
            
            ### 🔢 행동별 Lift 가중치 (실제 데이터 기반)
            
            | 행동 | Lift | 해석 |
            |:-----|:-----|:-----|
            | view_item | **4.6x** | 상품 조회 시 구매 확률 4.6배 |
            | view_search_results | **2.9x** | 검색 시 구매 확률 2.9배 |
            | add_to_cart | **11.8x** | 장바구니 추가 시 11.8배 |
            | begin_checkout | **30.6x** | 결제 시작 시 30.6배 |
            | add_payment_info | **46.5x** | 결제정보 입력 시 46.5배 |
            
            ---
            
            ### 💡 Engagement Score 계산 (int_engage_lift_score.sql)
            
            Lift 값을 반올림하여 점수로 변환:
            
            | 행동 | Lift | **점수** |
            |:-----|:-----|:---------|
            | view_item | 4.6x | **5점** |
            | view_search_results | 2.9x | **3점** |
            | add_to_cart | 11.8x | **12점** |
            | begin_checkout | 30.6x | **31점** |
            | add_payment_info | 46.5x | **47점** |
            | 기타 이벤트 | - | **1점** |
            
            ```sql
            Engagement Score = Σ (이벤트별 점수)
            ```
            
            **예시**: 유저가 view_item + add_to_cart를 했다면
            - Score = 5 + 12 = **17점**
            
            ---
            
            ### 📈 등급 분류 (PERCENT_RANK 기반)
            
            | 등급 | 기준 | 해석 |
            |:-----|:-----|:-----|
            | **High Intent** | 상위 20% (pct_rank ≤ 0.2) | 진성 유저 |
            | **Medium Intent** | 상위 20~50% (pct_rank ≤ 0.5) | 탐색 유저 |
            | **Low Intent** | 하위 50% (나머지) | 이탈 가능성 |
            
            > 이 Score를 기반으로 프로모션 클릭 유저의 **평균 품질**을 측정합니다.
            """)
            
            st.code("""
-- 1. Lift 가중치 산출 (int_lift_weight.sql)
WITH session_stats AS (
    SELECT
        session_unique_id,
        MAX(IF(event_name = 'purchase', 1, 0)) as is_converted,
        MAX(IF(event_name = 'view_item', 1, 0)) as has_view_item,
        MAX(IF(event_name = 'add_to_cart', 1, 0)) as has_cart
    FROM stg_events
    GROUP BY 1
),
rates AS (
    SELECT
        SAFE_DIVIDE(SUM(is_converted), COUNT(*)) as base_cv,
        SAFE_DIVIDE(COUNTIF(has_cart=1 AND is_converted=1), COUNTIF(has_cart=1)) as cart_cv
    FROM session_stats
)
SELECT ROUND(cart_cv / base_cv, 1) as score_cart  -- 결과: 11.8

-- 2. Engagement Score 계산 (int_engage_lift_score.sql)  
SELECT
    session_unique_id,
    SUM(CASE 
        WHEN event_name = 'view_item' THEN 5
        WHEN event_name = 'view_search_results' THEN 3
        WHEN event_name = 'add_to_cart' THEN 12
        WHEN event_name = 'begin_checkout' THEN 31
        WHEN event_name = 'add_payment_info' THEN 47
        ELSE 1
    END) AS engagement_score
FROM stg_events
GROUP BY 1

-- 3. 등급 분류
SELECT *,
    CASE 
        WHEN PERCENT_RANK() OVER (ORDER BY engagement_score DESC) <= 0.2 
            THEN 'High Intent'
        WHEN PERCENT_RANK() OVER (ORDER BY engagement_score DESC) <= 0.5 
            THEN 'Medium Intent'
        ELSE 'Low Intent'
    END AS engagement_grade
FROM session_scores
            """, language="sql")
        
        if 'promo_quality' in data:
            df_promo = data['promo_quality']
            
            # CVR을 텍스트에 포함
            df_promo['label'] = df_promo.apply(
                lambda x: f"{x['promotion_name']}<br>CVR: {x['promo_cvr']:.1f}%", axis=1
            )
            
            fig = px.scatter(
                df_promo,
                x='ctr_percent',
                y='avg_session_score',
                size='click_sessions',
                color='promo_status',
                text='label',
                color_discrete_map={
                    'Star (확대)': '#27ae60',
                    'Hidden Gem (숨은 보석)': '#f39c12',
                    'Clickbait (낚시성)': '#e74c3c',
                    'Poor (제거 대상)': '#95a5a6'
                },
                size_max=50,
                hover_data={'promo_cvr': ':.2f'}
            )
            
            # 기준선
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
            
            # 사분면 라벨
            fig.add_annotation(x=50, y=400, text="⭐ Star 프로모션", showarrow=False, font=dict(size=14, color='#27ae60'))
            fig.add_annotation(x=2, y=400, text="💎 Hidden Gem 프로모션", showarrow=False, font=dict(size=14, color='#f39c12'))
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                title='프로모션 4분면 분석 (CTR vs 유저 품질) - CVR 표시',
                xaxis_title='CTR (%) - 클릭률',
                yaxis_title='평균 유저 Engagement Score',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 프로모션별 CVR 테이블 추가
            st.markdown("#### 📊 프로모션별 성과 요약")
            promo_summary = df_promo[['promotion_name', 'ctr_percent', 'promo_cvr', 'click_sessions', 'promo_status']].copy()
            promo_summary.columns = ['프로모션', 'CTR (%)', 'CVR (%)', '클릭 세션', '분류']
            st.dataframe(
                promo_summary.style.format({
                    'CTR (%)': '{:.2f}',
                    'CVR (%)': '{:.2f}',
                    '클릭 세션': '{:,.0f}'
                }).background_gradient(subset=['CVR (%)'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )
            
            # 4분면 설명
            with st.expander("📐 4분면 분류 기준 설명 (mart_promo_quality.sql)"):
                st.markdown("""
                ### 프로모션 4분면 분류 기준
                
                **분류 기준값:**
                - CTR 기준: **5.0%**
                - Engagement Score 기준: **50점**
                
                | 분류 | CTR | Engagement Score | SQL 조건 |
                |:-----|:----|:-----------------|:---------|
                | ⭐ **Star** | ≥ 5% | ≥ 50 | `ctr >= 5.0 AND score >= 50` |
                | 💎 **Hidden Gem** | < 5% | ≥ 50 | `ctr < 5.0 AND score >= 50` |
                | ⚠️ **Clickbait** | ≥ 5% | < 50 | `ctr >= 5.0 AND score < 50` |
                | 🔘 **Poor** | < 5% | < 50 | `ctr < 5.0 AND score < 50` |
                
                > **Hidden Gem 프로모션**: CTR은 낮지만 클릭한 유저의 Engagement Score(구매 가능성)가 높은 프로모션.  
                > 배너 디자인, 위치, 카피 개선으로 CTR만 높이면 고품질 유저 유입 증가.
                """)
                
                st.code("""
-- mart_promo_quality.sql 4분면 분류 로직
CASE
    WHEN perf.ctr_percent >= 5.0 AND q.avg_session_score >= 50 
        THEN 'Star (확대)'
    WHEN perf.ctr_percent >= 5.0 AND q.avg_session_score < 50 
        THEN 'Clickbait (낚시성)'
    WHEN perf.ctr_percent < 5.0 AND q.avg_session_score >= 50 
        THEN 'Hidden Gem (숨은 보석)'
    ELSE 'Poor (제거 대상)'
END AS promo_status
                """, language="sql")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="warning-box">
                <strong>💎 Hidden Gem 프로모션 발견!</strong><br><br>
                <strong>'Reach New Heights' 프로모션 배너</strong><br><br>
                • CTR: 2.56% (전체 최저)<br>
                • 클릭 유저 Engagement: 400.2 (최고)<br>
                • 클릭 유저 전환율: 4.63% (최고)<br><br>
                
                <strong>→ 배너 노출만 개선하면<br>
                고품질 유저 유입 증가</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                <strong>🎯 Hidden Gem 프로모션 액션 플랜</strong><br><br>
                1. <strong>A/B 테스트</strong> 진행<br>
                   배너 위치/디자인 변경<br><br>
                2. <strong>배너 위치</strong> 상향 조정<br>
                   메인 페이지 상단 배치<br><br>
                3. 월간 성과 모니터링<br><br>
                <em>KPI: A/B 테스트로 CTR 개선폭 측정</em>
                </div>
                """, unsafe_allow_html=True)

# ----- 7. 액션 우선순위 -----
elif page == "6️⃣ 해결책 & 액션플랜":
    st.header("6️⃣ 해결책: 액션 플랜 & 우선순위")
    
    st.markdown("""
    > 📌 **분석가 노트**: 분석 결과를 실행 가능한 액션으로 전환하고, Impact-Effort 기준으로 우선순위를 정합니다.
    """)
    
    # Impact-Effort 매트릭스
    st.markdown("### 📊 Impact-Effort 매트릭스")
    
    actions = {
        'action': ['장바구니 리마케팅', 'Hidden Gem 프로모션 배너 개선', 'Deep Specialist 비교표', 
                   'VIP 세그먼트 타겟팅', '분할결제 도입', 
                   '실시간 세션 스코어링', 'CDP 구축'],
        'impact': [85, 70, 80, 75, 70, 90, 95],
        'effort': [20, 15, 40, 50, 60, 80, 95],
        'category': ['Quick Win', 'Quick Win', 'Quick Win', 'Major Project', 
                     'Major Project', 'Strategic', 'Strategic'],
        'data_evidence': ['Bags 48% 손실 집중', 'CTR 2.6% but CVR 4.63%', '81.4% 결정마비', 'Variety Seeker CVR 13%',
                          'Bags 건당 $216', '스코어 기반 예측', '통합 고객 뷰']
    }
    
    df_actions = pd.DataFrame(actions)
    
    fig = px.scatter(
        df_actions,
        x='effort',
        y='impact',
        size=[50]*len(df_actions),
        color='category',
        text='action',
        color_discrete_map={
            'Quick Win': '#27ae60',
            'Major Project': '#f39c12',
            'Strategic': '#3498db'
        },
        size_max=30
    )
    
    # 사분면 영역
    fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                  fillcolor="rgba(39, 174, 96, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100,
                  fillcolor="rgba(241, 196, 15, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50,
                  fillcolor="rgba(149, 165, 166, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50,
                  fillcolor="rgba(231, 76, 60, 0.1)", line_width=0)
    
    # 라벨
    fig.add_annotation(x=25, y=95, text="🎯 Quick Win", showarrow=False, font=dict(size=14, color='#27ae60'))
    fig.add_annotation(x=75, y=95, text="📊 Major Project", showarrow=False, font=dict(size=14, color='#f39c12'))
    fig.add_annotation(x=25, y=5, text="❓ Fill-In", showarrow=False, font=dict(size=14, color='#95a5a6'))
    fig.add_annotation(x=75, y=5, text="⚠️ Avoid", showarrow=False, font=dict(size=14, color='#e74c3c'))
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title='Impact-Effort 매트릭스',
        xaxis_title='구현 난이도 (Effort) →',
        yaxis_title='← 비즈니스 임팩트 (Impact)',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 실행 로드맵
    st.markdown("### 🗓️ 실행 로드맵")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <strong>🚀 Phase 1: Quick Win (1-2주)</strong><br><br>
        
        <strong>1. 장바구니 리마케팅</strong><br>
        • 1/24/72시간 이메일 자동화<br>
        • KPI: A/B 테스트로 개선폭 측정<br><br>
        
        <strong>2. Hidden Gem 프로모션 배너 A/B 테스트</strong><br>
        • 새 디자인/위치 테스트<br>
        • KPI: A/B 테스트로 CTR 개선폭 측정<br><br>
        
        <strong>담당</strong>: 마케팅팀<br>
        <strong>검증</strong>: A/B 테스트 2주
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>📊 Phase 2: 구조 개선 (1-2개월)</strong><br><br>
        
        <strong>3. Deep Specialist 비교표</strong><br>
        • 10개+ 조회 시 트리거<br>
        • KPI: 3-11개 구간 수준(5.26%) 달성<br><br>
        
        <strong>4. VIP 세그먼트 타겟팅</strong><br>
        • Super Heavy 전용 혜택<br>
        • KPI: VIP 재구매율 측정<br><br>
        
        <strong>담당</strong>: 개발팀 + CRM팀<br>
        <strong>검증</strong>: 코호트 분석
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
        <strong>🎯 Phase 3: 전략 과제 (3-6개월)</strong><br><br>
        
        <strong>5. 실시간 세션 스코어링</strong><br>
        • ML 기반 구매 확률 예측<br>
        • KPI: 모델 정확도 측정<br><br>
        
        <strong>6. CDP 구축</strong><br>
        • 통합 고객 프로파일<br>
        • KPI: 크로스셀 전환율 측정<br><br>
        
        <strong>담당</strong>: 데이터팀 + IT팀<br>
        <strong>검증</strong>: 모델 성능 모니터링
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 상세 액션 테이블
    st.markdown("### 📋 상세 액션 리스트")
    
    action_detail = {
        '우선순위': ['🥇 1', '🥇 1', '🥈 2', '🥈 2'],
        '액션': ['장바구니 리마케팅 (Bags)', 'Hidden Gem 프로모션 배너', 'Deep Specialist 비교표', 
                 'VIP 세그먼트 타겟팅'],
        '데이터 근거': ['Bags 753건, 손실 48%', 'CTR 2.6% but CVR 4.63%', '81.4%가 결정 마비 구간', 'Variety Seeker CVR 13%'],
        '성공 KPI': ['Bags 이탈률 감소', 'A/B 테스트로 CTR 측정', '3-11개 수준(5.26%) 달성', 'VIP 재구매율 측정'],
        '구현 난이도': ['낮음 ⭐', '낮음 ⭐', '중간 ⭐⭐', '중간 ⭐⭐'],
        '소요 기간': ['1주', '1주', '3주', '4주']
    }
    
    st.dataframe(pd.DataFrame(action_detail), use_container_width=True, hide_index=True)

# ----- 7.5 Engagement Score 산출 -----
elif page == "📐 방법론 & 한계점":
    st.header("📐 분석 방법론 & 한계점")
    
    tab1, tab2, tab3 = st.tabs(["🔧 기술 스택", "📊 분석 방법론", "⚠️ 한계점 & 향후 과제"])
    
    with tab1:
        st.markdown("### 데이터 파이프라인 아키텍처")
        
        # dbt 프로젝트 구조 표시
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            # Plotly를 사용한 파이프라인 시각화
            fig_pipeline = go.Figure()
            
            # 노드 정의 - 실제 dbt 구조 반영
            nodes = [
                # Source Layer
                {'x': 0.5, 'y': 6, 'text': '🗄️ <b>GA4 Raw Data</b><br>BigQuery Public Dataset<br><i>events_* (2.1M rows)</i>', 
                 'color': '#4285F4', 'width': 0.85},
                
                # Staging Layer
                {'x': 0.5, 'y': 5, 'text': '🔧 <b>Staging Layer</b><br>stg_events.sql<br><i>session_unique_id 생성 • 타입 변환</i>', 
                 'color': '#FF6D01', 'width': 0.85},
                
                # Intermediate Layer - 8개 모델
                {'x': 0.12, 'y': 4, 'text': 'int_browsing<br>_style', 'color': '#34A853', 'width': 0.18},
                {'x': 0.31, 'y': 4, 'text': 'int_engage<br>_lift_score', 'color': '#34A853', 'width': 0.18},
                {'x': 0.5, 'y': 4, 'text': 'int_session<br>_paths', 'color': '#34A853', 'width': 0.18},
                {'x': 0.69, 'y': 4, 'text': 'int_session<br>_funnel', 'color': '#34A853', 'width': 0.18},
                {'x': 0.88, 'y': 4, 'text': 'int_promo<br>+3 more', 'color': '#34A853', 'width': 0.18},
                
                # Mart Layer - 17개 모델
                {'x': 0.5, 'y': 3, 'text': '📦 <b>Mart Layer (17 tables)</b><br>mart_browsing_style • mart_core_sessions • mart_funnel_*<br><i>mart_cart_abandon • mart_promo_quality • mart_device_friction</i>', 
                 'color': '#EA4335', 'width': 0.85},
                
                # Dashboard Layer
                {'x': 0.5, 'y': 2, 'text': '📱 <b>Streamlit Dashboard</b><br>인터랙티브 분석 • 통계 검정<br><i>χ² Test • Cohen\'s h • Wilson CI</i>', 
                 'color': '#9C27B0', 'width': 0.85},
            ]
            
            # 노드 그리기
            for node in nodes:
                fig_pipeline.add_shape(
                    type="rect",
                    x0=node['x'] - node['width']/2, x1=node['x'] + node['width']/2,
                    y0=node['y'] - 0.35, y1=node['y'] + 0.35,
                    fillcolor=node['color'],
                    opacity=0.9,
                    line=dict(color='white', width=2),
                    layer='below'
                )
                
                fig_pipeline.add_annotation(
                    x=node['x'], y=node['y'],
                    text=node['text'],
                    showarrow=False,
                    font=dict(size=9, color='white'),
                    align='center'
                )
            
            # 화살표
            arrows = [
                {'x0': 0.5, 'y0': 5.65, 'x1': 0.5, 'y1': 5.35},
                {'x0': 0.5, 'y0': 4.65, 'x1': 0.12, 'y1': 4.35},
                {'x0': 0.5, 'y0': 4.65, 'x1': 0.31, 'y1': 4.35},
                {'x0': 0.5, 'y0': 4.65, 'x1': 0.5, 'y1': 4.35},
                {'x0': 0.5, 'y0': 4.65, 'x1': 0.69, 'y1': 4.35},
                {'x0': 0.5, 'y0': 4.65, 'x1': 0.88, 'y1': 4.35},
                {'x0': 0.5, 'y0': 3.65, 'x1': 0.5, 'y1': 3.35},
                {'x0': 0.5, 'y0': 2.65, 'x1': 0.5, 'y1': 2.35},
            ]
            
            for arrow in arrows:
                fig_pipeline.add_annotation(
                    x=arrow['x1'], y=arrow['y1'],
                    ax=arrow['x0'], ay=arrow['y0'],
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor='#666'
                )
            
            fig_pipeline.update_layout(
                title=dict(text='📊 dbt Data Pipeline', font=dict(size=16)),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[1.3, 6.7]),
                height=600,
                plot_bgcolor='rgba(248,249,250,1)',
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_pipeline, use_container_width=True)
        
        with col2:
            st.markdown("#### 📁 dbt 프로젝트 구조")
            st.code("""
models/
├── staging/
│   ├── sources.yml
│   └── stg_events.sql
│
├── intermediate/
│   ├── int_browsing_style.sql
│   ├── int_engage_lift_score.sql
│   ├── int_lift_weight.sql
│   ├── int_price_tier.sql
│   ├── int_product_association.sql
│   ├── int_promo_performance.sql
│   ├── int_session_funnel.sql
│   └── int_session_paths.sql
│
└── marts/
    ├── mart_browsing_style.sql
    ├── mart_bundle_strategy.sql
    ├── mart_cart_abandon.sql
    ├── mart_core_sessions.sql
    ├── mart_deep_specialists.sql
    ├── mart_device_friction.sql
    ├── mart_funnel_*.sql (7개)
    ├── mart_promo_quality.sql
    ├── mart_time_to_conversion.sql
    └── mart_variety_seekers.sql
            """, language="text")
            
            st.markdown("""
            <div class="methodology-box">
            <strong>📐 레이어 설계 원칙</strong><br><br>
            • <strong>Staging</strong>: 1:1 소스 미러링<br>
            • <strong>Intermediate</strong>: 비즈니스 로직 적용<br>
            • <strong>Mart</strong>: 분석 목적별 집계
            </div>
            """, unsafe_allow_html=True)
        
        # 기술 스택 카드
        st.markdown("---")
        st.markdown("### 🛠️ 기술 스택")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4285F4 0%, #1a73e8 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem;">🗄️</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">데이터 저장</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    Google BigQuery<br>
                    Cloud Storage
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FF6D01 0%, #e55b00 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem;">🔧</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">데이터 변환</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    dbt Core<br>
                    SQL + Jinja2
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #34A853 0%, #1e8e3e 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem;">📊</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">분석 & 통계</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    Python · pandas<br>
                    scipy · numpy
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%); 
                        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;">
                <div style="font-size: 2rem;">📱</div>
                <div style="font-weight: bold; margin: 0.5rem 0;">시각화</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">
                    Streamlit<br>
                    Plotly
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 추가 기술 상세
        with st.expander("📋 상세 기술 명세"):
            tech_data = {
                '영역': ['Data Source', 'Transformation', 'Analysis', 'Visualization', 'Deployment'],
                '기술': ['BigQuery Public Dataset', 'dbt Core 1.7+', 'Python 3.10+', 'Streamlit 1.28+', 'Streamlit Cloud'],
                '상세': [
                    'ga4_obfuscated_sample_ecommerce (2.1M events)',
                    'Staging → Intermediate → Mart 레이어 구조',
                    'pandas, numpy, scipy.stats (χ², Wilson CI)',
                    'Plotly (Funnel, Sankey, Scatter), Custom CSS',
                    'GitHub 연동 자동 배포'
                ]
            }
            st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 📊 통계 분석 방법론")
        
        st.markdown("""
        > 💡 **핵심 질문**: "이 통계 기법을 왜 썼고, 그 결과가 무엇을 의미하는가?"  
        > → 데이터의 특성과 분석 목적에 맞춰 기법을 선택했습니다.
        """)
        
        # 1. 카이제곱 검정
        st.markdown("---")
        st.markdown("#### 1️⃣ 카이제곱 검정 (χ² Test of Independence)")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("""
            **🎯 사용 목적**  
            "탐색 스타일(A, B, C)에 따라 구매 여부(Yes/No)가 정말로 달라지는가?"  
            → **범주형 변수 간의 독립성 검정**
            
            **📐 왜 이 기법을 선택했는가?**
            - 데이터가 모두 **범주형** (Categorical) → 평균 비교 불가
            - **"그룹 간 비율의 차이"** 가 우연인지 아닌지 판별 필요
            - 관측 빈도(Observed)와 기대 빈도(Expected) 간의 차이 측정
            """)
            
            st.code("""
# 카이제곱 검정 구현
from scipy import stats
import numpy as np

def chi_square_test(g1_success, g1_total, g2_success, g2_total):
    contingency = np.array([
        [g1_success, g1_total - g1_success],
        [g2_success, g2_total - g2_success]
    ])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p_value

# 결과: χ² = 722.27, p < 0.001
            """, language="python")
        
        with col2:
            st.markdown("""
            <div class="stat-significant">
            <strong>📈 결과 해석</strong><br><br>
            • χ² = <strong>722.27</strong><br>
            • p-value: <strong>0.001 미만</strong> ✅<br><br>
            
            <strong>의미:</strong><br>
            두 변수는 독립적이지 않음.<br>
            즉, <strong>"탐색 스타일이 구매 전환에<br>
            강력한 영향을 미친다"</strong>는<br>
            통계적 확신을 얻음.
            </div>
            """, unsafe_allow_html=True)
        
        # 2. 효과 크기 (Cohen's h)
        st.markdown("---")
        st.markdown("#### 2️⃣ 효과 크기 (Cohen's h)")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("""
            **🎯 사용 목적**  
            "차이가 있는 건 알겠는데 (P-value), 그 차이가 **비즈니스적으로 써먹을 만큼** 큰가?"
            
            **📐 왜 효과 크기가 필요한가?**
            - **통계적으로 유의하다** (Significant) ≠ **중요하다** (Important)
            - 빅데이터에서는 아주 작은 차이도 p 값이 0.001 미만으로 나옴
            - **"실질적인 중요성"** 을 측정하기 위해 사용
            
            **🔬 Cohen's h 특징**
            - 두 **비율** (Proportion) 간의 차이를 아크사인 변환
            - 1%→2% (2배)와 50%→51% (미미함)을 구분
            """)
            
            st.code("""
# Cohen's h 효과 크기 계산
def cohens_h(p1, p2):
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)

# 기준: 0.2(작음), 0.5(중간), 0.8(큼)
# 결과: h = 0.42 (중간 효과)
            """, language="python")
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <strong>📊 효과 크기 해석 기준</strong><br><br>
            
            | Cohen's h | 해석 |
            |:----------|:-----|
            | 0.2 | 작은 효과 (Small) |
            | 0.5 | 중간 효과 (Medium) |
            | 0.8 | 큰 효과 (Large) |
            
            <br>
            <strong>우리의 결과: h = 0.42</strong><br>
            → <strong>중간 정도(Medium)</strong>의 효과 크기<br>
            → 마케팅 전략 변경 시<br>
            &nbsp;&nbsp;&nbsp;매출에 유의미한 변화 기대
            </div>
            """, unsafe_allow_html=True)
        
        # 3. Wilson Score 신뢰구간
        st.markdown("---")
        st.markdown("#### 3️⃣ Wilson Score 신뢰구간")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("""
            **🎯 사용 목적**  
            "전환율 13%가 진짜 13%인가? 오차 범위는 어디까지인가?"
            
            **📐 왜 일반 신뢰구간이 아니라 'Wilson'인가?** (핵심!)
            - 일반적인 정규분포 근사(Wald Interval)는  
              전환율이 **0%나 100%에 가까울 때** 오차가 큼
            - 이커머스 전환율(1~5%)은 이 영역에 해당
            - Wilson 구간은 **비대칭적 분포**를 고려  
              → 전환율 추정에 훨씬 **강건** (Robust) 하고 정확
            """)
            
            st.code("""
# Wilson Score 신뢰구간 (소표본에서도 안정적)
def wilson_ci(successes, total, confidence=0.95):
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*total)) / total) / denominator
    
    return center - margin, center + margin

# Variety Seeker: 12.5% ~ 13.6%
# Deep Specialist: 2.2% ~ 2.9%
# → 신뢰구간 겹치지 않음 (Non-overlapping)
            """, language="python")
        
        with col2:
            st.markdown("""
            <div class="success-box">
            <strong>📈 결과 해석</strong><br><br>
            
            <strong>Variety Seeker</strong><br>
            95% CI: [12.5%, 13.6%]<br><br>
            
            <strong>Deep Specialist</strong><br>
            95% CI: [2.2%, 2.9%]<br><br>
            
            <strong>→ 신뢰구간이 전혀 겹치지 않음!</strong><br><br>
            
            이는 데이터가 우연히 좋게 나온 게 아니라,<br>
            <strong>아무리 못해도 Specialist보다는<br>
            무조건 높다</strong>는 통계적 보증
            </div>
            """, unsafe_allow_html=True)
        
        # 4. Lift 기반 스코어링
        st.markdown("---")
        st.markdown("#### 4️⃣ Lift 기반 Engagement Score")
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("""
            **🎯 사용 목적**  
            각 행동이 구매 확률을 **몇 배** 높이는지 측정
            
            **📐 Lift 공식**
            $$Lift = \\frac{P(Purchase | Action)}{P(Purchase)}$$
            
            **점수 가중치 설계 근거**
            - Lift 값을 그대로 가중치로 변환
            - 각 행동의 **실제 구매 기여도** 반영
            """)
            
            st.code("""
-- Lift 계산 SQL
WITH rates AS (
    SELECT
        SAFE_DIVIDE(SUM(is_converted), COUNT(*)) as base_cv,
        SAFE_DIVIDE(
            COUNTIF(has_cart=1 AND is_converted=1), 
            COUNTIF(has_cart=1)
        ) as cart_cv
    FROM session_stats
)
SELECT ROUND(cart_cv / base_cv, 1) as lift_cart

-- 결과: Lift = 11.8 (장바구니 추가 시 구매 확률 11.8배 증가)
            """, language="sql")
        
        with col2:
            lift_data = {
                '행동': ['view_item', 'add_to_cart', 'begin_checkout', 'add_payment_info'],
                'Lift': ['4.6x', '11.8x', '30.6x', '46.5x'],
                '가중치': [5, 12, 31, 47]
            }
            st.dataframe(pd.DataFrame(lift_data), use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class="methodology-box">
            <strong>💡 가중치 설계 원칙</strong><br><br>
            • Lift 값 ≈ 가중치로 직접 매핑<br>
            • <strong>데이터 기반 객관적 스코어링</strong><br>
            • "왜 이 가중치인가요?" → "Lift 값입니다"
            </div>
            """, unsafe_allow_html=True)
        
        # 5. 가격 티어링 방법론
        st.markdown("---")
        st.markdown("#### 5️⃣ 가격 티어링 (Dynamic Tiering)")
        
        st.markdown("""
        **"왜 $20가 Low이고 $50가 High인가요?"** 라는 질문에 대한 답변:
        
        > 자의적 기준이 아니라, <strong>상품 가격의 분포(Price Distribution)</strong>를 분석하여 
        > <strong>백분위 기반 동적 티어링(Percentile-based Dynamic Tiering)</strong>을 적용했습니다.
        """, unsafe_allow_html=True)
        
        st.code("""
-- 가격 티어 분류 SQL (int_price_tier.sql)
WITH price_quantiles AS (
    SELECT
        APPROX_QUANTILES(item_price, 100)[OFFSET(33)] AS p33,  -- 하위 33% 경계
        APPROX_QUANTILES(item_price, 100)[OFFSET(66)] AS p66   -- 상위 33% 경계
    FROM stg_events
    WHERE event_name = 'view_item' AND item_price > 0
)

SELECT
    item_name,
    avg_price,
    CASE
        WHEN avg_price < p33 THEN 'Low'    -- 하위 33%
        WHEN avg_price >= p66 THEN 'High'  -- 상위 33%
        ELSE 'Mid'                          -- 중간 34%
    END AS price_tier
FROM product_prices
CROSS JOIN price_quantiles
        """, language="sql")
        
        st.markdown("""
        | 티어 | 백분위 | 가격 범위 (예시) | 특징 |
        |:-----|:-------|:-----------------|:-----|
        | **Low** | 하위 33% | $16 미만 | 저관여 상품, 충동구매 유도 |
        | **Mid** | 중간 34% | $16 ~ $45 | 비교 구매 대상 |
        | **High** | 상위 33% | $45 초과 | 고관여, 결정 마비 발생 |
        
        > 이 방식은 시즌별 가격 변동에도 **자동으로 적응** 하는 장점이 있습니다.
        """)
    
    with tab3:
        st.markdown("### ⚠️ 분석 한계점")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="limitation-box">
            <strong>1. 데이터 한계</strong><br><br>
            
            • <strong>시간적 제한</strong><br>
            12월 한 달 → 계절성 미반영<br>
            홀리데이 시즌 특수성<br><br>
            
            • <strong>샘플 크기</strong><br>
            일부 세그먼트 n < 100<br>
            (60분+ 구매자: 102명)<br><br>
            
            • <strong>데이터 특성</strong><br>
            Obfuscated 처리<br>
            단일 스토어 한정
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="limitation-box">
            <strong>2. 분석 한계</strong><br><br>
            
            • <strong>인과관계 vs 상관관계</strong><br>
            "조회 많으면 전환 높다"<br>
            → 역인과 가능성 존재<br><br>
            
            • <strong>외부 요인 미통제</strong><br>
            광고 캠페인, 가격 변동 등<br>
            Confounding 가능<br><br>
            
            • <strong>일반화 제한</strong><br>
            Google Store 특수성<br>
            다른 이커머스 적용 시 검증 필요
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔮 향후 분석 방향")
        
        future_work = {
            '영역': ['시간 확장', '세그먼트 심화', '예측 모델링', 'A/B 테스트', '외부 데이터'],
            '내용': [
                '연간 데이터로 계절성/트렌드 분석',
                '코호트 분석 (신규/재방문/VIP)',
                '구매 확률 예측 ML 모델 개발',
                '개선안 실제 효과 검증',
                '광고/프로모션 데이터 연계'
            ],
            '기대 효과': [
                '계절별 최적 전략 도출',
                '고객 생애주기 최적화',
                '실시간 개인화 추천',
                '가설 검증 → 인과관계 확립',
                '통합 채널 기여도 분석'
            ]
        }
        
        st.dataframe(pd.DataFrame(future_work), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 분석 핵심 포인트")
        
        st.markdown("""
        <div class="success-box">
        <strong>1. 가설 기반 분석</strong><br>
        단순 EDA가 아닌, 비즈니스 가설 → 통계 검증 → 액션 도출 구조<br><br>
        
        <strong>2. 통계적 엄밀성</strong><br>
        χ² 검정, 효과 크기 (Cohen's h), 신뢰구간 등 통계적 근거 제시<br><br>
        
        <strong>3. 한계점 인지</strong><br>
        데이터/분석 한계를 정직하게 인정하고 향후 개선 방향 제시<br><br>
        
        <strong>4. 데이터 기반 의사결정</strong><br>
        모든 액션에 구체적 데이터 근거 제시 (예: Bags 카테고리 손실 48%, Apparel 12,650건)<br><br>
        
        <strong>5. 실행 가능성</strong><br>
        Impact-Effort 매트릭스로 우선순위화, 검증 가능한 KPI 설정
        </div>
        """, unsafe_allow_html=True)

# ===== 푸터 =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <strong>GA4 이커머스 전환 최적화 분석</strong><br>
    Built with Python, dbt, BigQuery, Streamlit<br>
    <em>분석 기간: 2020.12.01 ~ 12.31 | 데이터: ga4_obfuscated_sample_ecommerce</em>
</div>
""", unsafe_allow_html=True)