{{ config(materialized='table') }}

WITH abandoned_sessions AS (
    -- 1. 장바구니에는 담았으나(add_to_cart), 구매하지 않은(Converted=0) 세션 추출
    SELECT 
        session_unique_id,
        user_pseudo_id,
        engagement_grade -- 등급 정보도 가져와서 볼 수 있게 함
    FROM {{ ref('mart_core_sessions') }}
    WHERE REGEXP_CONTAINS(full_path, r'add_to_cart') AND is_converted = 0
),

cart_items AS (
    -- 2. 해당 세션들이 담았던 상품 정보 추출
    SELECT
        e.session_unique_id,
        e.item_name,
        e.item_category,
        e.item_price,
        e.item_quantity,
        e.item_revenue_calc AS potential_revenue
    FROM {{ ref('stg_events') }} e
    INNER JOIN abandoned_sessions s ON e.session_unique_id = s.session_unique_id
    WHERE e.event_name = 'add_to_cart'
)

SELECT
    c.item_name,
    c.item_category,
    -- 이탈된 총 횟수
    COUNT(DISTINCT c.session_unique_id) AS abandoned_count,
    -- 총 손실 금액 (Lost Revenue)
    SUM(c.potential_revenue) AS total_lost_revenue,
    -- 평균 이탈 금액
    ROUND(AVG(c.potential_revenue), 0) AS avg_lost_value
FROM cart_items c
GROUP BY 1, 2
HAVING abandoned_count > 0
ORDER BY total_lost_revenue DESC