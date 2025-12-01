{{ config(materialized='table') }}

SELECT
    browsing_style,
    
    -- 1. 규모 파악
    COUNT(session_unique_id) AS session_count,
    ROUND(COUNT(session_unique_id) * 100.0 / SUM(COUNT(session_unique_id)) OVER(), 1) AS session_share_percent,
    
    -- 2. 활동성 파악 (평균 조회수)
    ROUND(AVG(total_items_viewed), 3) AS avg_items_viewed,
    ROUND(APPROX_QUANTILES(total_items_viewed,100)[OFFSET(25)], 3) AS item_viewed_p25,
    ROUND(APPROX_QUANTILES(total_items_viewed,100)[OFFSET(50)], 3) AS item_viewed_p50,
    ROUND(APPROX_QUANTILES(total_items_viewed,100)[OFFSET(75)], 3) AS item_viewed_p75,
    ROUND(APPROX_QUANTILES(total_items_viewed,100)[OFFSET(90)], 3) AS item_viewed_p90,
    ROUND(APPROX_QUANTILES(total_items_viewed,100)[OFFSET(100)], 3) AS item_viewed_p100,
    
    -- 3. 성과 파악 (전환율)
    ROUND(AVG(is_converted) * 100, 3) AS conversion_rate

FROM {{ ref('int_browsing_style') }}
GROUP BY 1
ORDER BY conversion_rate DESC