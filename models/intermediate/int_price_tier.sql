{{ config(materialized='table') }}

WITH price_stats AS (
    -- 1. 전체 상품의 가격 분포를 계산하여 동적 기준점
    SELECT
        APPROX_QUANTILES(item_price, 100)[OFFSET(33)] AS p33_cutoff, -- 하위 33% 지점
        APPROX_QUANTILES(item_price, 100)[OFFSET(66)] AS p66_cutoff  -- 상위 33% 지점
    FROM {{ ref('stg_events') }}
    WHERE event_name = 'view_item'
      AND item_price > 0
),

product_avg_prices AS (
    -- 각 상품별 평균 가격을 계산
    SELECT
        item_name,
        AVG(item_price) AS avg_price
    FROM {{ ref('stg_events') }}
    WHERE event_name = 'view_item'
      AND item_price > 0
    GROUP BY 1
)

SELECT
    p.item_name,
    p.avg_price,
    -- 3. 기준점(stats)과 비교하여 등급을 매깁니다.
    CASE
        WHEN p.avg_price >= s.p66_cutoff THEN 'High' -- 상위 33% 이상
        WHEN p.avg_price >= s.p33_cutoff THEN 'Mid'  -- 중간
        ELSE 'Low'                                   -- 하위 33% 미만
    END AS price_tier,
    s.p33_cutoff,
    s.p66_cutoff
FROM product_avg_prices p
CROSS JOIN price_stats s -- 통계값(1줄)을 모든 상품에 붙임