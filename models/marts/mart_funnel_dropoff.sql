{{ config(materialized='table') }}

WITH funnel_counts AS (
    SELECT
        COUNT(*) AS total_sessions,
        SUM(has_view_item) AS viewed,
        SUM(has_add_to_cart) AS carted,
        SUM(has_begin_checkout) AS checkout,
        SUM(has_add_payment_info) AS payment,
        SUM(has_purchase) AS purchased
    FROM {{ ref('int_session_funnel') }}
)

SELECT 1 AS step_order, 'Session → View Item' AS step, total_sessions AS from_count, viewed AS to_count,
       ROUND((total_sessions - viewed) / total_sessions * 100, 2) AS drop_rate
FROM funnel_counts
UNION ALL
SELECT 2, 'View Item → Add to Cart', viewed, carted, ROUND((viewed - carted) / NULLIF(viewed, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 3, 'Add to Cart → Begin Checkout', carted, checkout, ROUND((carted - checkout) / NULLIF(carted, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 4, 'Begin Checkout → Add Payment', checkout, payment, ROUND((checkout - payment) / NULLIF(checkout, 0) * 100, 2) FROM funnel_counts
UNION ALL
SELECT 5, 'Add Payment → Purchase', payment, purchased, ROUND((payment - purchased) / NULLIF(payment, 0) * 100, 2) FROM funnel_counts
ORDER BY step_order