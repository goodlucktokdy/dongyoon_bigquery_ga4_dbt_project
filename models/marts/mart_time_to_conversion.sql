{{ config(materialized='table') }}

WITH purchase_sessions AS (
    SELECT
        session_unique_id,
        MIN(event_timestamp) AS session_start_at,
        MAX(CASE WHEN event_name = 'purchase' THEN event_timestamp END) AS purchased_at,
        SUM(item_revenue_calc) AS total_revenue
    FROM {{ ref('stg_events') }}
    GROUP BY 1
    HAVING MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) = 1 -- 구매 세션만
)

SELECT
    -- 구매 소요 시간 (분 단위)
    TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) AS minutes_to_buy,
    
    -- 시간대별 그룹핑 (Bucketing)
    CASE
        WHEN TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) < 5 THEN '0-5분 (즉시 구매)'
        WHEN TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) < 15 THEN '5-15분 (단기 탐색)'
        WHEN TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) < 30 THEN '15-30분 (중기 탐색)'
        WHEN TIMESTAMP_DIFF(purchased_at, session_start_at, MINUTE) < 60 THEN '30-60분 (장기 고민)'
        ELSE '60분 이상'
    END AS time_bucket,

    COUNT(*) AS session_count,
    ROUND(AVG(total_revenue), 1) AS avg_order_value -- 객단가

FROM purchase_sessions
GROUP BY 1, 2
ORDER BY 1 ASC