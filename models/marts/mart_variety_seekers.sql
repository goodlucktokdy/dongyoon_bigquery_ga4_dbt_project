{{ config(materialized='table') }}

WITH seeker_data AS (
    SELECT 
        session_unique_id,
        distinct_categories_viewed, -- 카테고리 수
        total_items_viewed,         -- 상품 수 (이걸 기준으로 나눔)
        is_converted
    FROM {{ ref('int_browsing_style') }}
    WHERE browsing_style = 'Variety Seeker (다양성 추구형)'
)

SELECT
    -- [Intensity 구간화] P25(24), P50(36), P75(84) 기준
    CASE
        WHEN total_items_viewed <= 24 THEN '1. Light Seeker (24개 이하)'
        WHEN total_items_viewed BETWEEN 25 AND 36 THEN '2. Moderate Seeker (25-36개)'
        WHEN total_items_viewed BETWEEN 37 AND 84 THEN '3. Heavy Seeker (37-84개)'
        WHEN total_items_viewed > 84 THEN '4. Super Heavy Seeker (85개 이상)'
        ELSE 'Others'
    END AS intensity_segment,

    -- 1. 세션 규모
    COUNT(session_unique_id) AS session_count,
    ROUND(COUNT(session_unique_id) * 100.0 / SUM(COUNT(session_unique_id)) OVER(), 1) AS share_percent,
    
    -- 2. 활동성 지표
    ROUND(AVG(total_items_viewed), 1) AS avg_total_views,
    ROUND(AVG(distinct_categories_viewed), 1) AS avg_categories, -- 평균 몇 카테고리나 보는지 확인

    -- 3. [핵심] 구간별 세션 전환율
    ROUND(AVG(is_converted) * 100, 2) AS conversion_rate

FROM seeker_data
GROUP BY 1
ORDER BY avg_total_views ASC