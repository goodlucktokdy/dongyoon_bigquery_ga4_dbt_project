{{ config(materialized='table') }}

WITH device_stats AS (
    -- 1. 디바이스별 기초 통계 집계
    SELECT
        e.device_category,
        COUNT(DISTINCT e.session_unique_id) AS total_sessions,
        
        -- High Intent 유저 수
        COUNTIF(s.engagement_grade = 'High Intent') AS high_intent_users,
        
        -- High Intent 유저 중 구매자 수
        COUNTIF(s.engagement_grade = 'High Intent' AND p.is_converted = 1) AS high_intent_converters
    FROM {{ ref('stg_events') }} e
    JOIN {{ ref('int_engage_lift_score') }} s ON e.session_unique_id = s.session_unique_id
    JOIN {{ ref('int_session_paths') }} p ON e.session_unique_id = p.session_unique_id
    GROUP BY 1
),

cvr_calculation AS (
    -- 2. 전환율(CVR) 계산
    SELECT
        *,
        -- 진성 유저 전환율 (CVR)
        SAFE_DIVIDE(high_intent_converters, high_intent_users) AS high_intent_cvr
    FROM device_stats
    WHERE high_intent_users > 0 -- 모수 0인 경우 제외
)

SELECT
    device_category,
    total_sessions,
    high_intent_users,
    
    -- 진성 유저 비중 (%)
    ROUND(SAFE_DIVIDE(high_intent_users, total_sessions) * 100, 1) AS high_intent_ratio,
    
    -- 전환율 (%)
    ROUND(high_intent_cvr * 100, 1) AS high_intent_cvr_percent,
    
    -- [핵심] PC 대비 상대 효율 (Relative Efficiency Index)
    -- 공식: (내 CVR / 데스크탑 CVR) * 100
    ROUND(
        SAFE_DIVIDE(
            high_intent_cvr, 
            MAX(CASE WHEN device_category = 'desktop' THEN high_intent_cvr END) OVER()
        ) * 100, 
    0) AS efficiency_index_vs_pc

FROM cvr_calculation
ORDER BY high_intent_cvr DESC