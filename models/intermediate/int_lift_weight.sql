{{ config(materialized = 'table') }}

WITH session_stats AS (
    SELECT
        session_unique_id,
        -- 1. 목표(Goal): 구매 여부 (0 or 1)
        MAX(IF(event_name = 'purchase', 1, 0)) as is_converted,
        
        -- 2. 신호(Signal): 각 행동을 했는지 여부 (0 or 1)
        MAX(IF(event_name = 'view_item', 1, 0)) as has_view_item,
        MAX(IF(event_name = 'view_search_results', 1, 0)) as has_search,
        MAX(IF(event_name = 'add_to_cart', 1, 0)) as has_cart,
        MAX(IF(event_name = 'begin_checkout', 1, 0)) as has_checkout,
        MAX(IF(event_name = 'add_payment_info', 1, 0)) as has_payment
    FROM {{ ref('stg_events') }} -- (혹은 stg_ga4_events)
    GROUP BY 1
),

rates AS (
    SELECT
        -- A. 베이스라인: 전체 세션의 평균 구매율 (Base Probability)
        SAFE_DIVIDE(SUM(is_converted), COUNT(*)) as base_cv,
        
        -- B. 조건부 확률: 각 행동을 한 세션의 구매율 (Conditional Probability)
        -- P(Purchase | Action)
        SAFE_DIVIDE(COUNTIF(has_view_item=1 AND is_converted=1), COUNTIF(has_view_item=1)) as view_cv,
        SAFE_DIVIDE(COUNTIF(has_search=1 AND is_converted=1), COUNTIF(has_search=1)) as search_cv,
        SAFE_DIVIDE(COUNTIF(has_cart=1 AND is_converted=1), COUNTIF(has_cart=1)) as cart_cv,
        SAFE_DIVIDE(COUNTIF(has_checkout=1 AND is_converted=1), COUNTIF(has_checkout=1)) as checkout_cv,
        SAFE_DIVIDE(COUNTIF(has_payment=1 AND is_converted=1), COUNTIF(has_payment=1)) as payment_cv
    FROM session_stats
)

SELECT
    -- C. Lift(향상도) 계산: 조건부 확률 / 베이스라인
    -- "이 행동을 하면 구매 확률이 몇 배(X)로 뛰는가?"
    ROUND(view_cv / base_cv, 1) as score_view,       -- 결과: 4.6
    ROUND(search_cv / base_cv, 1) as score_search,   -- 결과: 2.9
    ROUND(cart_cv / base_cv, 1) as score_cart,       -- 결과: 11.8
    ROUND(checkout_cv / base_cv, 1) as score_checkout, -- 결과: 30.6
    ROUND(payment_cv / base_cv, 1) as score_payment  -- 결과: 46.5
FROM rates