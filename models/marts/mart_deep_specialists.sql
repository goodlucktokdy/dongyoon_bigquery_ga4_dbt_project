{{ config(materialized='table') }}

WITH specialist_data AS (
    SELECT 
        session_unique_id,
        total_items_viewed,
        is_converted
    FROM {{ ref('int_browsing_style') }}
    WHERE browsing_style = 'Deep Specialist (한우물형)'
)

SELECT
    -- [Depth 구간화] P25(12), P75(24), P90(36) 기준
    CASE
        WHEN total_items_viewed < 12 THEN '1. 탐색 초기 (3-11개)'
        WHEN total_items_viewed BETWEEN 12 AND 24 THEN '2. 집중 비교 (12-24개)'
        WHEN total_items_viewed BETWEEN 25 AND 36 THEN '3. 고민 심화 (25-36개)'
        WHEN total_items_viewed > 36 THEN '4. 결정 마비 (37개 이상)'
        ELSE 'Others'
    END AS depth_segment,

    -- 1. 세션(방문) 규모
    COUNT(session_unique_id) AS session_count,
    ROUND(COUNT(session_unique_id) * 100.0 / SUM(COUNT(session_unique_id)) OVER(), 1) AS share_percent,

    -- 2. 평균 조회수 (강도)
    ROUND(AVG(total_items_viewed), 1) AS avg_views,

    -- 3. [핵심] 구간별 전환율 (어디서 급락하는지 확인!)
    ROUND(AVG(is_converted) * 100, 2) AS conversion_rate

FROM specialist_data
GROUP BY 1
ORDER BY avg_views ASC