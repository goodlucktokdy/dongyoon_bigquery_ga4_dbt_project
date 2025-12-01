{{ config(materialized='table') }}

WITH product_tiers AS (
    SELECT item_name, price_tier, avg_price 
    FROM {{ ref('int_price_tier') }}
),

pair_stats AS (
    -- 2. 상품 조합 + 유저 점수 + 이동 경로
    SELECT
        p.product_A,
        p.product_B,
        p.transaction_id,
        s.engagement_score,
        s.engagement_grade,
        ph.path_length
    FROM {{ ref('int_product_association') }} p
    LEFT JOIN {{ ref('int_engage_lift_score') }} s ON p.session_unique_id = s.session_unique_id
    LEFT JOIN {{ ref('int_session_paths') }} ph ON p.session_unique_id = ph.session_unique_id
)

SELECT
    -- A. 상품 조합 정보
    ps.product_A,
    ROUND(pr_a.avg_price,3) AS price_A,
    pr_a.price_tier AS tier_A, 
    ps.product_B,
    ROUND(pr_b.avg_price,3) AS price_B,
    pr_b.price_tier AS tier_B, 
    
    -- B. 판매 성과
    COUNT(DISTINCT ps.transaction_id) AS pair_sales_count,
    
    -- C. 구매자 특성
    ROUND(AVG(ps.engagement_score), 1) AS avg_buyer_score,
    ROUND(COUNTIF(ps.engagement_grade = 'High Intent') / COUNT(*) * 100, 1) AS high_intent_ratio,
    
    -- D. 번들 전략
    CASE
        WHEN (pr_a.price_tier = 'High' AND pr_b.price_tier = 'Low') OR 
             (pr_a.price_tier = 'Low' AND pr_b.price_tier = 'High') OR 
             (pr_a.price_tier = 'Medium' AND pr_b.price_tier = 'High') OR 
             (pr_a.price_tier = 'High' AND pr_b.price_tier = 'Medium')
             THEN 'Add-on Strategy (업셀링)'
             
        WHEN pr_a.price_tier = 'High' AND pr_b.price_tier = 'High' 
             THEN 'Premium Set (VIP 타겟)'
             
        WHEN pr_a.price_tier = 'Mid' AND pr_b.price_tier = 'Mid'
             THEN 'Volume Builder (크로스셀링)'
             
        ELSE 'General Bundle'
    END AS bundle_strategy_type

FROM pair_stats ps

LEFT JOIN product_tiers pr_a ON ps.product_A = pr_a.item_name
LEFT JOIN product_tiers pr_b ON ps.product_B = pr_b.item_name

GROUP BY 1, 2, 3, 4, 5, 6, bundle_strategy_type
HAVING pair_sales_count >= 3
ORDER BY pair_sales_count DESC