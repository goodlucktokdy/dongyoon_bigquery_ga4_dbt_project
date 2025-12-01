{{ config(materialized='table') }}

WITH promo_clicks AS (
    -- 1. 배너를 클릭한 세션 정보 가져오기
    SELECT
        e.promotion_name,
        e.session_unique_id
    FROM {{ ref('stg_events') }} e
    WHERE e.event_name = 'select_promotion'
      AND e.promotion_name != '(not set)'
),

promo_quality AS (
    -- 2. 클릭한 유저들의 점수와 구매 여부 결합
    SELECT
        p.promotion_name,
        COUNT(DISTINCT p.session_unique_id) AS click_sessions,
        ROUND(AVG(s.engagement_score), 1) AS avg_session_score, -- 클릭한 사람들의 평균 점수
        COUNTIF(s.engagement_grade = 'High Intent') AS high_intent_session_count,
        -- 배너 클릭 후 구매 전환율 (Conversion)
        ROUND(COUNTIF(path.is_converted = 1) / COUNT(DISTINCT p.session_unique_id) * 100, 2) AS promo_cvr
    FROM promo_clicks p
    LEFT JOIN {{ ref('int_engage_lift_score') }} s ON p.session_unique_id = s.session_unique_id
    LEFT JOIN {{ ref('int_session_paths') }} path ON p.session_unique_id = path.session_unique_id
    GROUP BY 1
)

SELECT
    q.promotion_name,
    -- 기존 CTR 정보 (int_promo_performance에서 가져옴)
    perf.ctr_percent,
    q.click_sessions,
    q.avg_session_score,
    q.high_intent_session_count,
    q.promo_cvr,
    
    -- [종합 평가] 4분면 분석을 위한 태그 생성
    CASE
        WHEN perf.ctr_percent >= 5.0 AND q.avg_session_score >= 50 THEN 'Star (확대)'
        WHEN perf.ctr_percent >= 5.0 AND q.avg_session_score < 50 THEN 'Clickbait (낚시성)'
        WHEN perf.ctr_percent < 5.0 AND q.avg_session_score >= 50 THEN 'Hidden Gem (숨은 보석)'
        ELSE 'Poor (제거 대상)'
    END AS promo_status
    
FROM promo_quality q
LEFT JOIN {{ ref('int_promo_performance') }} perf ON q.promotion_name = perf.promotion_name
ORDER BY q.click_sessions DESC