import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="GA4 이커머스 세션 행동 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 스타일 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .critical-box {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .sql-box {
        background-color: #263238;
        color: #80CBC4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    .methodology-box {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== 데이터 로드 =====
@st.cache_data
def load_data():
    data = {}
    
    # 로컬 PC OneDrive 경로
    BASE_PATH = "./mart_tables/"
    
    # 대체 경로들 (다른 환경 대비)
    alt_paths = [
        BASE_PATH,
        r".\mart_tables",
        r"mart_tables",
        r"/mnt/user-data/uploads"
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
        'core_sessions': 'mart_core_sessions.csv'
    }
    
    import os
    
    # 작동하는 경로 찾기
    working_path = None
    for path in alt_paths:
        test_file = os.path.join(path, 'mart_browsing_style.csv')
        if os.path.exists(test_file):
            working_path = path
            break
    
    if working_path is None:
        st.error(f"❌ 데이터 폴더를 찾을 수 없습니다. 다음 경로를 확인해주세요:\n{BASE_PATH}")
        return data
    
    for key, filename in files.items():
        try:
            filepath = os.path.join(working_path, filename)
            data[key] = pd.read_csv(filepath)
        except Exception as e:
            if key != 'core_sessions':  # core_sessions는 선택적
                st.warning(f"⚠️ {filename} 로드 실패: {e}")
    
    return data

data = load_data()

# ===== 사이드바 =====
st.sidebar.image("https://www.gstatic.com/analytics-suite/header/suite/v2/ic_analytics.svg", width=50)
st.sidebar.title("📊 GA4 분석 대시보드")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "분석 페이지 선택",
    ["🏠 Executive Summary", 
     "🔍 브라우징 스타일 분석", 
     "📱 디바이스 마찰 분석",
     "🛒 장바구니 이탈 분석",
     "📢 프로모션 품질 분석",
     "⏱️ 구매 소요 시간 분석",
     "🎁 번들 전략 분석",
     "📐 분석 방법론 (SQL)"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**분석 기간**: 2020년 12월  
**데이터 소스**: BigQuery GA4 Public Dataset  
**분석 도구**: dbt + BigQuery + Streamlit
""")

# ===== 페이지별 컨텐츠 =====

# ----- Executive Summary -----
if page == "🏠 Executive Summary":
    st.markdown('<p class="main-header">GA4 이커머스 세션 행동 분석</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Google Merchandise Store 전환율 최적화 프로젝트</p>', unsafe_allow_html=True)
    
    # 핵심 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 분석 세션", "22,521", help="High/Medium Intent 세션만 포함")
    with col2:
        st.metric("브라우징 유형", "3 Types", help="Variety Seeker, Deep Specialist, Light Browser")
    with col3:
        st.metric("최고 전환율", "31.5%", "Super Heavy Seeker", help="85개+ 상품 조회 고객")
    with col4:
        st.metric("잠재 손실", "$795K+", help="장바구니 이탈 상위 10개 상품")
    
    st.markdown("---")
    
    # 가설 검증 요약
    st.subheader("🔬 가설 검증 결과")
    
    hypothesis_data = {
        '가설': [
            'H1: 다양한 탐색 = 높은 전환율',
            'H2: 특정 구간에서 결정 마비 발생',
            'H3: Mobile/Tablet에서 UX 마찰',
            'H4: 구매 시간 ↑ = 객단가 ↑',
            'H5: CTR과 실제 전환은 별개',
            'H6: 고가 상품에서 이탈 집중'
        ],
        '결과': ['✅ 검증됨', '✅ 검증됨', '⚠️ 부분 검증', '✅ 강하게 검증', '✅ 검증됨', '✅ 검증됨'],
        '근거': [
            'Variety Seeker 13% vs Deep Specialist 2.5%',
            '12-24개 구간 전환율 1.88% 급락',
            'Tablet만 10% 저조, Mobile은 오히려 높음',
            '60분+ 고객 AOV $1,847 (즉시구매의 7.7배)',
            'Hidden Gem 발견 (CTR 2.6% but CVR 4.6%)',
            'Rain Shell 23건 이탈에 $489K 손실'
        ]
    }
    
    df_hypothesis = pd.DataFrame(hypothesis_data)
    st.dataframe(df_hypothesis, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 핵심 발견 사항
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Critical Findings")
        st.markdown("""
        <div class="critical-box">
        <strong>Deep Specialist 결정 마비 구간 발견</strong><br>
        12-24개 상품 조회 시 전환율 <strong>1.88%</strong>로 급락<br>
        (전체의 81.4%가 이 구간에 집중)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Hidden Gem 프로모션 발굴</strong><br>
        'Reach New Heights' 배너: CTR 2.6%로 낮지만<br>
        클릭 유저의 평균 점수 <strong>400.2</strong>, 전환율 <strong>4.63%</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 Quick Win 기회")
        st.markdown("""
        <div class="insight-box">
        <strong>장바구니 리마케팅 자동화</strong><br>
        5% 회수율 달성 시 월 <strong>$39,700</strong> 추가 매출
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>비교표 기능 추가</strong><br>
        Deep Specialist 10개+ 조회 시 제공<br>
        예상 추가 전환: <strong>+361건/월</strong>
        </div>
        """, unsafe_allow_html=True)

# ----- 브라우징 스타일 분석 -----
elif page == "🔍 브라우징 스타일 분석":
    st.header("🔍 브라우징 스타일별 전환 분석")
    
    # 방법론 설명
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.markdown("""
        ### 브라우징 스타일 분류 기준 (int_browsing_style.sql)
        """)
        
        st.code("""
-- 브라우징 스타일 정의 로직
CASE
    WHEN total_items_viewed <= 2 THEN 'Light Browser'
    WHEN total_items_viewed > 2 AND distinct_categories_viewed = 1 
        THEN 'Deep Specialist (한우물형)'
    WHEN distinct_categories_viewed >= 2 
        THEN 'Variety Seeker (다양성 추구형)'
    ELSE 'Others'
END AS browsing_style
        """, language="sql")
        
        st.markdown("""
        **분류 기준 설명:**
        - **Light Browser**: 2개 이하 상품 조회 → 단순 방문자
        - **Deep Specialist**: 3개+ 상품 조회, 1개 카테고리만 → 특정 제품에 집중
        - **Variety Seeker**: 2개+ 카테고리 탐색 → 비교 쇼핑 성향
        """)
    
    st.markdown("---")
    
    if 'browsing_style' in data:
        df = data['browsing_style']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("세션 분포")
            fig1 = px.pie(
                df, 
                values='session_count', 
                names='browsing_style',
                color_discrete_sequence=['#3498DB', '#E74C3C', '#95A5A6'],
                hole=0.4
            )
            fig1.update_traces(textposition='outside', textinfo='percent+label')
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("전환율 비교")
            fig2 = px.bar(
                df.sort_values('conversion_rate', ascending=True),
                x='conversion_rate',
                y='browsing_style',
                orientation='h',
                color='conversion_rate',
                color_continuous_scale='RdYlGn',
                text='conversion_rate'
            )
            fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig2.update_layout(
                xaxis_title="전환율 (%)",
                yaxis_title="",
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # 상세 테이블
        st.subheader("📊 상세 데이터")
        st.dataframe(
            df.style.format({
                'session_share_percent': '{:.1f}%',
                'avg_items_viewed': '{:.1f}',
                'conversion_rate': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Deep Specialist 심층 분석
        st.subheader("🔬 Deep Specialist 심층 분석")
        
        with st.expander("📐 구간 분류 기준", expanded=False):
            st.code("""
-- Deep Specialist Depth 구간화 (백분위 기준)
-- P25(12), P75(24), P90(36) 기준으로 구간 설정
CASE
    WHEN total_items_viewed < 12 THEN '1. 탐색 초기 (3-11개)'
    WHEN total_items_viewed BETWEEN 12 AND 24 THEN '2. 집중 비교 (12-24개)'
    WHEN total_items_viewed BETWEEN 25 AND 36 THEN '3. 고민 심화 (25-36개)'
    WHEN total_items_viewed > 36 THEN '4. 결정 마비 (37개 이상)'
END AS depth_segment
            """, language="sql")
        
        if 'deep_specialists' in data:
            df_deep = data['deep_specialists']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_deep = go.Figure()
                
                # 전환율 막대
                fig_deep.add_trace(go.Bar(
                    x=df_deep['depth_segment'],
                    y=df_deep['conversion_rate'],
                    name='전환율',
                    marker_color=['#27AE60', '#E74C3C', '#F39C12', '#F39C12'],
                    text=df_deep['conversion_rate'].apply(lambda x: f'{x:.2f}%'),
                    textposition='outside'
                ))
                
                # 세션 비중 라인
                fig_deep.add_trace(go.Scatter(
                    x=df_deep['depth_segment'],
                    y=df_deep['share_percent'],
                    name='세션 비중 (%)',
                    yaxis='y2',
                    mode='lines+markers+text',
                    marker=dict(size=12, color='#3498DB'),
                    line=dict(width=3),
                    text=df_deep['share_percent'].apply(lambda x: f'{x:.1f}%'),
                    textposition='top center'
                ))
                
                fig_deep.update_layout(
                    title='Deep Specialist: 조회 구간별 전환율 vs 세션 비중',
                    xaxis_title='조회 구간',
                    yaxis=dict(title='전환율 (%)', side='left'),
                    yaxis2=dict(title='세션 비중 (%)', side='right', overlaying='y', range=[0, 100]),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    height=450
                )
                st.plotly_chart(fig_deep, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="critical-box">
                <strong>🚨 Critical Finding</strong><br><br>
                <strong>12-24개 조회 구간</strong>에서<br>
                전환율이 <strong style="color:#E74C3C;">1.88%</strong>로 급락<br><br>
                이 구간에 전체의<br>
                <strong>81.4%</strong>가 집중되어 있음
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="insight-box">
                <strong>💡 해결 방안</strong><br><br>
                • 10개+ 조회 시 <strong>비교표 제공</strong><br>
                • 15개+ 조회 시 <strong>한정 쿠폰</strong><br>
                • "가장 많이 선택된 상품" 추천
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Variety Seeker 심층 분석
        st.subheader("🌈 Variety Seeker 심층 분석")
        
        if 'variety_seekers' in data:
            df_variety = data['variety_seekers']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_variety = px.scatter(
                    df_variety,
                    x='avg_total_views',
                    y='conversion_rate',
                    size='session_count',
                    color='intensity_segment',
                    text='intensity_segment',
                    color_discrete_sequence=['#BDC3C7', '#F1C40F', '#E67E22', '#27AE60'],
                    size_max=60
                )
                fig_variety.update_traces(textposition='top center')
                fig_variety.update_layout(
                    title='Variety Seeker: 조회량 vs 전환율 (버블 크기 = 세션 수)',
                    xaxis_title='평균 상품 조회수',
                    yaxis_title='전환율 (%)',
                    height=450
                )
                st.plotly_chart(fig_variety, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                <strong>⭐ VIP 세그먼트 발견</strong><br><br>
                <strong>Super Heavy Seeker</strong><br>
                (85개+ 조회)<br><br>
                전환율: <strong style="color:#27AE60;">31.53%</strong><br>
                평균 카테고리: <strong>6.4개</strong><br><br>
                → 크로스셀링 최적 타겟
                </div>
                """, unsafe_allow_html=True)
            
            st.dataframe(
                df_variety.style.format({
                    'share_percent': '{:.1f}%',
                    'avg_total_views': '{:.1f}',
                    'avg_categories': '{:.1f}',
                    'conversion_rate': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

# ----- 디바이스 마찰 분석 -----
elif page == "📱 디바이스 마찰 분석":
    st.header("📱 디바이스별 전환 효율 분석")
    
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.markdown("""
        ### Engagement Score 계산 로직 (int_engage_lift_score.sql)
        
        **Lift 기반 점수 산정**: 각 행동이 구매 확률을 얼마나 높이는지 분석
        """)
        
        st.code("""
-- Lift 기반 점수 산정
SUM(CASE 
    WHEN event_name = 'view_item' THEN 5          -- Lift 4.6 → 5점
    WHEN event_name = 'view_search_results' THEN 3 -- Lift 2.9 → 3점
    WHEN event_name = 'add_to_cart' THEN 12        -- Lift 11.8 → 12점
    WHEN event_name = 'begin_checkout' THEN 31     -- Lift 30.6 → 31점
    WHEN event_name = 'add_payment_info' THEN 47   -- Lift 46.5 → 47점
    ELSE 1
END) AS engagement_score

-- 등급 부여 (백분위 기준)
CASE 
    WHEN pct_rank <= 0.2 THEN 'High Intent'   -- 상위 20%
    WHEN pct_rank <= 0.5 THEN 'Medium Intent' -- 상위 20~50%
    ELSE 'Low Intent'                         -- 하위 50%
END AS engagement_grade
        """, language="sql")
        
        st.markdown("""
        ### Lift 값 산출 방식 (int_lift_weight.sql)
        """)
        
        st.code("""
-- Lift(향상도) = 조건부 확률 / 베이스라인
-- "이 행동을 하면 구매 확률이 몇 배로 뛰는가?"
ROUND(view_cv / base_cv, 1) as score_view,      -- 결과: 4.6
ROUND(cart_cv / base_cv, 1) as score_cart,      -- 결과: 11.8
ROUND(checkout_cv / base_cv, 1) as score_checkout, -- 결과: 30.6
ROUND(payment_cv / base_cv, 1) as score_payment  -- 결과: 46.5
        """, language="sql")
    
    st.markdown("---")
    
    if 'device_friction' in data:
        df_device = data['device_friction']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mobile_eff = df_device[df_device['device_category'] == 'mobile']['efficiency_index_vs_pc'].values[0]
            st.metric("📱 Mobile 효율", f"{mobile_eff}", "+2 vs PC", delta_color="normal")
        
        with col2:
            desktop_eff = 100
            st.metric("🖥️ Desktop 효율", f"{desktop_eff}", "기준값")
        
        with col3:
            tablet_eff = df_device[df_device['device_category'] == 'tablet']['efficiency_index_vs_pc'].values[0]
            st.metric("📟 Tablet 효율", f"{tablet_eff}", "-10 vs PC", delta_color="inverse")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_device = go.Figure()
            
            fig_device.add_trace(go.Bar(
                x=df_device['device_category'],
                y=df_device['high_intent_cvr_percent'],
                name='High Intent 전환율',
                marker_color=['#3498DB', '#27AE60', '#E74C3C'],
                text=df_device['high_intent_cvr_percent'].apply(lambda x: f'{x}%'),
                textposition='outside'
            ))
            
            fig_device.update_layout(
                title='디바이스별 High Intent 유저 전환율',
                xaxis_title='디바이스',
                yaxis_title='전환율 (%)',
                height=400
            )
            st.plotly_chart(fig_device, use_container_width=True)
        
        with col2:
            fig_eff = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=tablet_eff,
                delta={'reference': 100, 'relative': False},
                title={'text': "Tablet 효율지수 (PC=100 기준)"},
                gauge={
                    'axis': {'range': [70, 110]},
                    'bar': {'color': "#E74C3C"},
                    'steps': [
                        {'range': [70, 90], 'color': "#FFEBEE"},
                        {'range': [90, 100], 'color': "#FFF3E0"},
                        {'range': [100, 110], 'color': "#E8F5E9"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig_eff.update_layout(height=400)
            st.plotly_chart(fig_eff, use_container_width=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Tablet UX 개선 필요</strong><br><br>
        • Tablet 효율지수 90 (PC 대비 10% 저조)<br>
        • <strong>개선 방안</strong>: 반응형 레이아웃 최적화, 터치 영역 확대, 원클릭 결제 도입<br>
        • <strong>예상 효과</strong>: 효율지수 90 → 98 개선 시 전환율 +2.5%p
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>💡 반직관적 발견: Mobile > Desktop</strong><br><br>
        모바일 전환 효율이 데스크탑보다 2% 높음<br>
        → 기존 가설(H3)과 반대 결과. 모바일 UX 최적화가 이미 잘 되어 있거나,<br>
        모바일 사용자의 구매 의도가 더 명확할 가능성
        </div>
        """, unsafe_allow_html=True)

# ----- 장바구니 이탈 분석 -----
elif page == "🛒 장바구니 이탈 분석":
    st.header("🛒 장바구니 이탈 분석")
    
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.markdown("""
        ### 장바구니 이탈 정의 (mart_cart_abandon.sql)
        """)
        
        st.code("""
-- 이탈 세션 정의: add_to_cart 했지만 purchase 없음
WHERE REGEXP_CONTAINS(full_path, r'add_to_cart') 
  AND is_converted = 0

-- 손실 매출 계산
SUM(item_revenue_calc) AS total_lost_revenue,
ROUND(AVG(item_revenue_calc), 0) AS avg_lost_value
        """, language="sql")
    
    st.markdown("---")
    
    if 'cart_abandon' in data:
        df_cart = data['cart_abandon'].head(20)
        
        # 요약 메트릭
        top10_loss = df_cart.head(10)['total_lost_revenue'].sum()
        total_abandon = df_cart['abandoned_count'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("상위 10개 상품 손실", f"${top10_loss:,.0f}")
        with col2:
            st.metric("총 이탈 건수", f"{total_abandon:,}")
        with col3:
            st.metric("5% 회수 시 예상 매출", f"${top10_loss * 0.05:,.0f}/월")
        
        st.markdown("---")
        
        # 차트
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("손실 매출 TOP 10")
            df_top10 = df_cart.nlargest(10, 'total_lost_revenue')
            
            fig_loss = px.bar(
                df_top10,
                x='total_lost_revenue',
                y='item_name',
                orientation='h',
                color='avg_lost_value',
                color_continuous_scale='Reds',
                text='total_lost_revenue'
            )
            fig_loss.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_loss.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='손실 매출 ($)',
                yaxis_title='',
                height=500
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with col2:
            st.subheader("이탈 빈도 vs 평균 금액")
            
            fig_scatter = px.scatter(
                df_cart.head(15),
                x='abandoned_count',
                y='avg_lost_value',
                size='total_lost_revenue',
                color='item_category',
                hover_name='item_name',
                size_max=50
            )
            fig_scatter.update_layout(
                xaxis_title='이탈 횟수',
                yaxis_title='평균 이탈 금액 ($)',
                height=500
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        <div class="critical-box">
        <strong>🚨 Rain Shell 상품 집중 분석</strong><br><br>
        • 이탈 23건에 <strong>$489,180 손실</strong> (평균 $14,388/건)<br>
        • 고가 상품 특성상 결제 허들 높음<br>
        • <strong>해결책</strong>: 분할결제 옵션, 상세 사이즈 가이드, 반품 정책 강조
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>💡 리마케팅 자동화 권장</strong><br><br>
        • 이탈 1시간/24시간/72시간 3단계 이메일 발송<br>
        • 고가 상품($100+)은 분할결제 옵션 별도 안내<br>
        • "장바구니 상품이 품절될 수 있습니다" 긴급성 알림
        </div>
        """, unsafe_allow_html=True)

# ----- 프로모션 품질 분석 -----
elif page == "📢 프로모션 품질 분석":
    st.header("📢 프로모션 품질 4분면 분석")
    
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.markdown("""
        ### 프로모션 품질 평가 로직 (mart_promo_quality.sql)
        """)
        
        st.code("""
-- 4분면 분석: CTR과 유저 품질(점수)로 평가
CASE
    WHEN ctr_percent >= 5.0 AND avg_session_score >= 50 
        THEN 'Star (확대)'      -- 클릭도 많고, 좋은 고객이 클릭
    WHEN ctr_percent >= 5.0 AND avg_session_score < 50 
        THEN 'Clickbait (낚시성)' -- 클릭은 많지만, 이탈 고객이 클릭
    WHEN ctr_percent < 5.0 AND avg_session_score >= 50 
        THEN 'Hidden Gem (숨은 보석)' -- 클릭은 적지만, 좋은 고객이 클릭
    ELSE 'Poor (제거 대상)'
END AS promo_status
        """, language="sql")
        
        st.markdown("""
        **평가 기준:**
        - **CTR 5% 기준**: 프로모션 노출 대비 클릭률
        - **점수 50 기준**: 클릭한 유저의 평균 Engagement Score
        """)
    
    st.markdown("---")
    
    if 'promo_quality' in data:
        df_promo = data['promo_quality']
        
        # 4분면 차트
        fig_quad = px.scatter(
            df_promo,
            x='ctr_percent',
            y='avg_session_score',
            size='click_sessions',
            color='promo_status',
            text='promotion_name',
            color_discrete_map={
                'Star (확대)': '#27AE60',
                'Hidden Gem (숨은 보석)': '#F39C12',
                'Clickbait (낚시성)': '#E74C3C',
                'Poor (제거 대상)': '#95A5A6'
            },
            size_max=50
        )
        
        # 기준선 추가
        fig_quad.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_quad.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_quad.update_traces(textposition='top center')
        fig_quad.update_layout(
            title='프로모션 4분면 분석 (버블 크기 = 클릭 세션 수)',
            xaxis_title='CTR (%)',
            yaxis_title='평균 유저 점수',
            height=500
        )
        st.plotly_chart(fig_quad, use_container_width=True)
        
        # 상세 테이블
        st.dataframe(
            df_promo.style.format({
                'ctr_percent': '{:.2f}%',
                'avg_session_score': '{:.1f}',
                'promo_cvr': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="warning-box">
            <strong>💎 Hidden Gem 발견!</strong><br><br>
            <strong>Reach New Heights</strong> 배너<br><br>
            • CTR: <strong>2.56%</strong> (낮음)<br>
            • 클릭 유저 점수: <strong>400.2</strong> (최고)<br>
            • 전환율: <strong>4.63%</strong> (최고 수준)<br><br>
            → 배너 디자인 개선 시 높은 ROI 기대
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <strong>🎯 액션 플랜</strong><br><br>
            1. <strong>Hidden Gem 활성화</strong><br>
               - A/B 테스트로 CTR 2.6% → 10% 목표<br><br>
            2. <strong>Star 프로모션 확대</strong><br>
               - 메인 배너 위치 배정<br><br>
            3. <strong>Clickbait 검토</strong><br>
               - 현재 데이터에는 없지만 주기적 모니터링
            </div>
            """, unsafe_allow_html=True)

# ----- 구매 소요 시간 분석 -----
elif page == "⏱️ 구매 소요 시간 분석":
    st.header("⏱️ 구매 소요 시간별 객단가 분석")
    
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.code("""
-- 구매 소요 시간 계산
TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) AS minutes_to_buy

-- 시간 구간 버케팅
CASE
    WHEN minutes < 5 THEN '0-5분 (즉시 구매)'
    WHEN minutes < 15 THEN '5-15분 (단기 탐색)'
    WHEN minutes < 30 THEN '15-30분 (중기 탐색)'
    WHEN minutes < 60 THEN '30-60분 (장기 고민)'
    ELSE '60분 이상'
END AS time_bucket
        """, language="sql")
    
    st.markdown("---")
    
    if 'time_conversion' in data:
        df_time = data['time_conversion']
        
        # 버킷별 집계
        bucket_summary = df_time.groupby('time_bucket').agg({
            'session_count': 'sum',
            'avg_order_value': 'mean'
        }).reset_index()
        
        # 순서 정렬
        bucket_order = ['0-5분 (즉시 구매)', '5-15분 (단기 탐색)', '15-30분 (중기 탐색)', '30-60분 (장기 고민)', '60분 이상']
        bucket_summary['time_bucket'] = pd.Categorical(bucket_summary['time_bucket'], categories=bucket_order, ordered=True)
        bucket_summary = bucket_summary.sort_values('time_bucket')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_time = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_time.add_trace(
                go.Bar(
                    x=bucket_summary['time_bucket'],
                    y=bucket_summary['avg_order_value'],
                    name='평균 객단가',
                    marker_color='#3498DB',
                    text=bucket_summary['avg_order_value'].apply(lambda x: f'${x:.0f}'),
                    textposition='outside'
                ),
                secondary_y=False
            )
            
            fig_time.add_trace(
                go.Scatter(
                    x=bucket_summary['time_bucket'],
                    y=bucket_summary['session_count'],
                    name='세션 수',
                    mode='lines+markers',
                    marker=dict(size=10, color='#E74C3C'),
                    line=dict(width=3)
                ),
                secondary_y=True
            )
            
            fig_time.update_layout(
                title='구매 소요 시간별 객단가 & 세션 수',
                xaxis_title='시간 구간',
                height=450
            )
            fig_time.update_yaxes(title_text="평균 객단가 ($)", secondary_y=False)
            fig_time.update_yaxes(title_text="세션 수", secondary_y=True)
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            instant_aov = bucket_summary[bucket_summary['time_bucket'] == '0-5분 (즉시 구매)']['avg_order_value'].values[0]
            long_aov = bucket_summary[bucket_summary['time_bucket'] == '60분 이상']['avg_order_value'].values[0]
            
            st.metric("즉시 구매 AOV", f"${instant_aov:.0f}")
            st.metric("60분+ AOV", f"${long_aov:.0f}", f"+{((long_aov/instant_aov)-1)*100:.0f}%")
            st.metric("배율", f"{long_aov/instant_aov:.1f}x")
            
            st.markdown("""
            <div class="insight-box">
            <strong>💡 H4 가설 강하게 검증</strong><br><br>
            구매 시간이 길어질수록<br>
            객단가가 선형적으로 증가<br><br>
            60분+ 고객은 VIP 세그먼트로<br>
            프리미엄 서비스 제공 권장
            </div>
            """, unsafe_allow_html=True)
        
        # 분 단위 상세 분포
        st.subheader("📊 분 단위 상세 분포")
        
        fig_detail = px.scatter(
            df_time[df_time['minutes_to_buy'] <= 100],
            x='minutes_to_buy',
            y='avg_order_value',
            size='session_count',
            color='time_bucket',
            hover_data=['session_count'],
            size_max=30
        )
        fig_detail.update_layout(
            xaxis_title='구매 소요 시간 (분)',
            yaxis_title='평균 객단가 ($)',
            height=400
        )
        st.plotly_chart(fig_detail, use_container_width=True)

# ----- 번들 전략 분석 -----
elif page == "🎁 번들 전략 분석":
    st.header("🎁 상품 번들 전략 분석")
    
    with st.expander("📐 분석 방법론 보기", expanded=False):
        st.markdown("""
        ### 상품 연관 분석 로직 (int_product_association.sql)
        """)
        
        st.code("""
-- 동일 거래에서 함께 구매된 상품 쌍 추출
SELECT
    a.item_name AS product_A,
    b.item_name AS product_B
FROM purchase_items a
JOIN purchase_items b
  ON a.transaction_id = b.transaction_id
 AND a.item_name < b.item_name  -- 중복 제거 (A-B만 남김)
        """, language="sql")
        
        st.markdown("""
        ### 가격 티어 분류 (int_price_tier.sql)
        """)
        
        st.code("""
-- 백분위 기반 가격 등급
CASE
    WHEN avg_price >= p66_cutoff THEN 'High' -- 상위 33%
    WHEN avg_price >= p33_cutoff THEN 'Mid'  -- 중간 33%
    ELSE 'Low'                               -- 하위 33%
END AS price_tier
        """, language="sql")
        
        st.markdown("""
        ### 번들 전략 유형 정의
        """)
        
        st.code("""
CASE
    WHEN (tier_A = 'High' AND tier_B = 'Low') OR 
         (tier_A = 'Low' AND tier_B = 'High')
        THEN 'Add-on Strategy (업셀링)'
    WHEN tier_A = 'High' AND tier_B = 'High' 
        THEN 'Premium Set (VIP 타겟)'
    WHEN tier_A = 'Mid' AND tier_B = 'Mid'
        THEN 'Volume Builder (크로스셀링)'
    ELSE 'General Bundle'
END AS bundle_strategy_type
        """, language="sql")
    
    st.markdown("---")
    
    if 'bundle_strategy' in data:
        df_bundle = data['bundle_strategy'].head(30)
        
        # 전략 유형별 분포
        strategy_counts = df_bundle['bundle_strategy_type'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("번들 전략 유형 분포")
            fig_strategy = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                color_discrete_sequence=['#3498DB', '#E74C3C', '#27AE60', '#9B59B6'],
                hole=0.4
            )
            fig_strategy.update_layout(height=400)
            st.plotly_chart(fig_strategy, use_container_width=True)
        
        with col2:
            st.subheader("전략별 평균 구매자 점수")
            avg_score_by_strategy = df_bundle.groupby('bundle_strategy_type')['avg_buyer_score'].mean().sort_values(ascending=True)
            
            fig_score = px.bar(
                x=avg_score_by_strategy.values,
                y=avg_score_by_strategy.index,
                orientation='h',
                color=avg_score_by_strategy.values,
                color_continuous_scale='Viridis'
            )
            fig_score.update_layout(
                xaxis_title='평균 구매자 점수',
                yaxis_title='',
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig_score, use_container_width=True)
        
        # TOP 번들 조합
        st.subheader("🏆 TOP 번들 조합")
        
        top_bundles = df_bundle.nlargest(10, 'pair_sales_count')
        
        st.dataframe(
            top_bundles[['product_A', 'product_B', 'pair_sales_count', 'avg_buyer_score', 'bundle_strategy_type']].style.format({
                'avg_buyer_score': '{:.1f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <strong>💡 Add-on Strategy 활용</strong><br><br>
            • Camp Mug + Flat Front Bag (17건)<br>
            • 저가 상품으로 유인 → 고가 상품 추가 제안<br>
            • "함께 구매하면 10% 할인" 프로모션 적용
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <strong>💡 Premium Set 활용</strong><br><br>
            • Crewneck Grey + Navy (10건)<br>
            • VIP 고객 대상 세트 상품 구성<br>
            • 별도 패키지 할인 제공
            </div>
            """, unsafe_allow_html=True)

# ----- 분석 방법론 (SQL) -----
elif page == "📐 분석 방법론 (SQL)":
    st.header("📐 분석 방법론 및 SQL 로직")
    
    st.markdown("""
    ### 데이터 파이프라인 구조
    
    본 프로젝트는 **dbt (Data Build Tool)**를 활용하여 계층적 데이터 모델링을 구현했습니다.
    """)
    
    # 파이프라인 구조도
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        GA4 Raw Data                             │
    │                    (BigQuery Public Dataset)                     │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Staging Layer                               │
    │                       stg_events                                 │
    │              (이벤트 정제, 세션 ID 생성)                           │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │  int_browsing_    │ │  int_engage_      │ │  int_session_     │
    │  style            │ │  lift_score       │ │  paths            │
    │                   │ │                   │ │                   │
    │  • 카테고리 수      │ │  • Lift 기반 점수  │ │  • 행동 경로       │
    │  • 조회 상품 수     │ │  • Intent 등급    │ │  • 전환 여부       │
    │  • 스타일 분류     │ │                   │ │                   │
    └───────────────────┘ └───────────────────┘ └───────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                         Mart Layer                               │
    │   mart_browsing_style  │  mart_deep_specialists                  │
    │   mart_device_friction │  mart_cart_abandon                      │
    │   mart_promo_quality   │  mart_bundle_strategy                   │
    │   mart_time_conversion │  mart_variety_seekers                   │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.markdown("---")
    
    # SQL 탭
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Lift Score 계산", 
        "🔍 브라우징 스타일", 
        "🛣️ 세션 경로",
        "💰 가격 티어",
        "🔗 상품 연관"
    ])
    
    with tab1:
        st.subheader("Lift Score 계산 로직")
        st.markdown("""
        **Lift(향상도)**는 특정 행동이 구매 확률을 얼마나 높이는지 측정합니다.
        
        **공식**: `Lift = P(Purchase | Action) / P(Purchase)`
        """)
        
        st.code("""
-- int_lift_weight.sql
-- 1단계: 세션별 행동 여부와 구매 여부 집계
WITH session_stats AS (
    SELECT
        session_unique_id,
        MAX(IF(event_name = 'purchase', 1, 0)) as is_converted,
        MAX(IF(event_name = 'view_item', 1, 0)) as has_view_item,
        MAX(IF(event_name = 'add_to_cart', 1, 0)) as has_cart,
        MAX(IF(event_name = 'begin_checkout', 1, 0)) as has_checkout,
        MAX(IF(event_name = 'add_payment_info', 1, 0)) as has_payment
    FROM stg_events
    GROUP BY 1
),

-- 2단계: 베이스라인 및 조건부 확률 계산
rates AS (
    SELECT
        SAFE_DIVIDE(SUM(is_converted), COUNT(*)) as base_cv,  -- 전체 전환율
        SAFE_DIVIDE(
            COUNTIF(has_view_item=1 AND is_converted=1), 
            COUNTIF(has_view_item=1)
        ) as view_cv,  -- view_item 한 세션의 전환율
        -- ... 각 행동별 전환율
    FROM session_stats
)

-- 3단계: Lift 계산
SELECT
    ROUND(view_cv / base_cv, 1) as lift_view,      -- 결과: 4.6
    ROUND(cart_cv / base_cv, 1) as lift_cart,      -- 결과: 11.8
    ROUND(checkout_cv / base_cv, 1) as lift_checkout, -- 결과: 30.6
    ROUND(payment_cv / base_cv, 1) as lift_payment  -- 결과: 46.5
FROM rates
        """, language="sql")
        
        st.markdown("""
        **결과 해석:**
        - `view_item` → 구매 확률 **4.6배** 증가
        - `add_to_cart` → 구매 확률 **11.8배** 증가
        - `begin_checkout` → 구매 확률 **30.6배** 증가
        - `add_payment_info` → 구매 확률 **46.5배** 증가
        
        이 Lift 값을 가중치로 사용하여 세션별 Engagement Score를 계산합니다.
        """)
    
    with tab2:
        st.subheader("브라우징 스타일 분류 로직")
        
        st.code("""
-- int_browsing_style.sql
WITH category_counts AS (
    SELECT
        session_unique_id,
        -- 조회한 카테고리 수 (Distinct)
        COUNT(DISTINCT CASE 
            WHEN event_name = 'view_item' THEN item_category 
        END) AS distinct_categories_viewed,
        
        -- 총 상품 조회 수
        COUNT(CASE 
            WHEN event_name = 'view_item' THEN item_name 
        END) AS total_items_viewed,
        
        -- 구매 여부
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS is_converted
    FROM stg_events 
    GROUP BY 1
    HAVING total_items_viewed > 0
)

SELECT
    *,
    -- 브라우징 스타일 분류
    CASE
        WHEN total_items_viewed <= 2 
            THEN 'Light Browser'
        WHEN total_items_viewed > 2 AND distinct_categories_viewed = 1 
            THEN 'Deep Specialist (한우물형)'
        WHEN distinct_categories_viewed >= 2 
            THEN 'Variety Seeker (다양성 추구형)'
        ELSE 'Others'
    END AS browsing_style
FROM category_counts
        """, language="sql")
        
        st.markdown("""
        **분류 기준:**
        | 유형 | 조건 | 특성 |
        |-----|------|-----|
        | Light Browser | 조회 ≤ 2개 | 단순 방문자 |
        | Deep Specialist | 조회 > 2개 & 카테고리 = 1개 | 특정 제품 집중 탐색 |
        | Variety Seeker | 카테고리 ≥ 2개 | 비교 쇼핑, 다양한 관심 |
        """)
    
    with tab3:
        st.subheader("세션 경로 분석 로직")
        
        st.code("""
-- int_session_paths.sql
SELECT
    session_unique_id,
    
    -- 행동 순서를 문자열로 연결
    -- 예: "page_view > view_item > add_to_cart > purchase"
    STRING_AGG(
        event_name, 
        ' > ' 
        ORDER BY event_timestamp ASC
    ) AS full_path,
    
    -- 경로 길이 (총 이벤트 수)
    COUNT(*) AS path_length,
    
    -- 구매 전환 여부
    MAX(CASE 
        WHEN action_type = 'Purchase' THEN 1 
        ELSE 0 
    END) AS is_converted
    
FROM stg_events
GROUP BY 1
        """, language="sql")
    
    with tab4:
        st.subheader("가격 티어 분류 로직")
        
        st.code("""
-- int_price_tier.sql
WITH price_stats AS (
    -- 전체 상품 가격 분포 계산
    SELECT
        APPROX_QUANTILES(item_price, 100)[OFFSET(33)] AS p33_cutoff,
        APPROX_QUANTILES(item_price, 100)[OFFSET(66)] AS p66_cutoff
    FROM stg_events
    WHERE event_name = 'view_item' AND item_price > 0
),

product_avg_prices AS (
    -- 상품별 평균 가격
    SELECT
        item_name,
        AVG(item_price) AS avg_price
    FROM stg_events
    WHERE event_name = 'view_item' AND item_price > 0
    GROUP BY 1
)

SELECT
    p.item_name,
    p.avg_price,
    -- 백분위 기준 등급 분류
    CASE
        WHEN p.avg_price >= s.p66_cutoff THEN 'High'  -- 상위 33%
        WHEN p.avg_price >= s.p33_cutoff THEN 'Mid'   -- 중간 33%
        ELSE 'Low'                                    -- 하위 33%
    END AS price_tier
FROM product_avg_prices p
CROSS JOIN price_stats s
        """, language="sql")
    
    with tab5:
        st.subheader("상품 연관 분석 로직")
        
        st.code("""
-- int_product_association.sql
-- Market Basket Analysis의 기초: 동시 구매 상품 쌍 추출

WITH purchase_items AS (
    SELECT
        transaction_id,
        session_unique_id,
        item_name
    FROM stg_events
    WHERE event_name = 'purchase' 
      AND transaction_id IS NOT NULL
)

-- Self Join으로 상품 쌍 생성
SELECT
    a.session_unique_id,
    a.transaction_id,
    a.item_name AS product_A,
    b.item_name AS product_B
FROM purchase_items a
JOIN purchase_items b
  ON a.transaction_id = b.transaction_id
 AND a.item_name < b.item_name  -- 중복 제거 (A-B와 B-A 중 하나만)
        """, language="sql")
        
        st.markdown("""
        **Self Join 로직 설명:**
        - 동일한 `transaction_id`를 가진 상품들을 페어링
        - `a.item_name < b.item_name` 조건으로 중복 제거
        - (A, B)와 (B, A)는 같은 조합이므로 하나만 유지
        """)
    
    st.markdown("---")
    
    st.subheader("📊 사용 기술 스택")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **데이터 처리**
        - Google BigQuery
        - dbt (Data Build Tool)
        - Python (pandas)
        """)
    
    with col2:
        st.markdown("""
        **분석 기법**
        - Lift 기반 행동 스코어링
        - 백분위 기반 세그멘테이션
        - Market Basket Analysis
        """)
    
    with col3:
        st.markdown("""
        **시각화**
        - Streamlit
        - Plotly
        - Custom CSS
        """)

# ===== 푸터 =====
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.85rem;">
    GA4 이커머스 세션 행동 분석 | Built with Streamlit & Plotly<br>
    데이터 분석가 포트폴리오 프로젝트
</div>
""", unsafe_allow_html=True)