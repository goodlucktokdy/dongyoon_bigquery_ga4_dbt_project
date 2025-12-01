{{ config(materialized='table') }}

WITH abandoned_sessions AS (
    -- 1. 이탈 세션 추출 (기존과 동일)
    SELECT 
        session_unique_id
    FROM {{ ref('mart_core_sessions') }}
    WHERE 
        (is_missed_opportunity = TRUE) OR 
        (REGEXP_CONTAINS(full_path, r'add_to_cart') AND is_converted = 0)
),

cart_items AS (
    -- 2. 상품 정보 추출
    SELECT
        e.session_unique_id,
        e.item_name,
        -- [핵심] 카테고리가 여러 개일 경우, 알파벳 순서상 첫 번째 것 하나만 가져옴 (대표 카테고리)
        -- 또는 SPLIT(e.item_category, '/')[SAFE_OFFSET(0)] 처럼 대분류만 쓸 수도 있음
        MIN(e.item_category) AS item_category, 
        e.item_revenue_calc AS potential_revenue
    FROM {{ ref('stg_events') }} e
    INNER JOIN abandoned_sessions s ON e.session_unique_id = s.session_unique_id
    WHERE e.event_name = 'add_to_cart'
    GROUP BY 1, 2, 4 -- item_category는 집계함수(MIN)를 썼으므로 그룹핑에서 제외하거나 조정
)

SELECT
    item_name,
    -- 대표 카테고리 하나만 남김
    MAX(item_category) AS item_category,
    
    -- 이탈된 총 세션 수 (중복 제거된 상품명 기준)
    COUNT(DISTINCT session_unique_id) AS abandoned_session_count,
    
    -- 총 손실 금액 (합산)
    SUM(potential_revenue) AS total_lost_revenue,
    
    -- 평균 이탈 금액
    ROUND(AVG(potential_revenue), 0) AS avg_lost_value

FROM cart_items
GROUP BY 1 -- item_name 기준으로만 그룹핑!
HAVING abandoned_session_count > 0
ORDER BY total_lost_revenue DESC