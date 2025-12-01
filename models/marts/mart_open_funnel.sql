{{ config(materialized='table') }}

WITH funnel_counts AS (
    SELECT
        -- 1. 전체 세션 수
        COUNT(DISTINCT session_unique_id) AS step_1_session_start,
        
        -- 2. 상품 상세 조회 세션 수
        COUNT(DISTINCT CASE WHEN event_name = 'view_item' THEN session_unique_id END) AS step_2_view_product,
        
        -- 3. 장바구니 담기 세션 수
        COUNT(DISTINCT CASE WHEN event_name = 'add_to_cart' THEN session_unique_id END) AS step_3_add_to_cart,
        
        -- 4. 구매 완료 세션 수
        COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN session_unique_id END) AS step_4_purchase
    FROM {{ ref('stg_events') }}
)

SELECT
    '1. Session Start' AS step_name, 
    1 AS step_order, 
    step_1_session_start AS user_count, 
    1.0 AS conversion_rate, 
    0.0 AS drop_off_rate 
FROM funnel_counts

UNION ALL

SELECT
    '2. View Product', 
    2, 
    step_2_view_product, 
    ROUND(step_2_view_product / step_1_session_start, 4), -- 전체 대비 전환율
    ROUND(1 - (step_2_view_product / step_1_session_start), 4) -- 이전 단계 대비 이탈률
FROM funnel_counts

UNION ALL

SELECT
    '3. Add to Cart', 
    3, 
    step_3_add_to_cart, 
    ROUND(step_3_add_to_cart / step_2_view_product, 4), -- 상세 -> 장바구니 전환율
    ROUND(1 - (step_3_add_to_cart / step_2_view_product), 4)
FROM funnel_counts

UNION ALL

SELECT
    '4. Purchase', 
    4, 
    step_4_purchase, 
    ROUND(step_4_purchase / step_3_add_to_cart, 4), -- 장바구니 -> 구매 전환율
    ROUND(1 - (step_4_purchase / step_3_add_to_cart), 4)
FROM funnel_counts

ORDER BY step_order