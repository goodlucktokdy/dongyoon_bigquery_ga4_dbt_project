{{ config(materialized='table') }}

WITH promo_stats AS (
    SELECT
        promotion_name,
        COUNTIF(event_name = 'view_promotion') AS impressions,
        COUNTIF(event_name = 'select_promotion') AS clicks
    FROM {{ ref('stg_events') }}
    WHERE promotion_name != '(not set)'
    GROUP BY 1
)
SELECT
    promotion_name,
    impressions,
    clicks,
    -- CTR 계산 (백분율)
    ROUND(SAFE_DIVIDE(clicks, impressions) * 100, 2) AS ctr_percent
FROM promo_stats
WHERE impressions > 50 -- 노출이 너무 적은 건 제외
ORDER BY ctr_percent DESC