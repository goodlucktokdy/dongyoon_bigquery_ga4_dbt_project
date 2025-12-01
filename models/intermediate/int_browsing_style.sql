{{ config(materialized='table') }}

WITH category_counts AS (
    SELECT
        session_unique_id,
        
        -- 조건부 집계 
        COUNT(DISTINCT CASE WHEN event_name = 'view_item' THEN item_category END) AS distinct_categories_viewed,
        COUNT(CASE WHEN event_name = 'view_item' THEN item_name END) AS total_items_viewed,
        
        -- 구매 여부
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS is_converted

    FROM {{ ref('stg_events') }} 
    GROUP BY 1
    HAVING total_items_viewed > 0 -- 최소 1개 이상 상품 본 세션만
)

SELECT
    session_unique_id, -- 세션 ID 유지
    
    -- 분석에 필요한 원본 수치들도 남겨둠
    distinct_categories_viewed,
    total_items_viewed,
    is_converted,

    -- 스타일 정의 
    CASE
        WHEN total_items_viewed <= 2 THEN 'Light Browser'
        WHEN total_items_viewed > 2 AND distinct_categories_viewed = 1 THEN 'Deep Specialist (한우물형)'
        WHEN distinct_categories_viewed >= 2 THEN 'Variety Seeker (다양성 추구형)'
        ELSE 'Others'
    END AS browsing_style

FROM category_counts