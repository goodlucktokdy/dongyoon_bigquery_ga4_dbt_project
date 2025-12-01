{{ config(materialized='table') }}

SELECT
    device_category,
    COUNT(*) AS sessions,
    SUM(has_view_item) AS viewed,
    SUM(has_add_to_cart) AS carted,
    SUM(has_purchase) AS purchased,
    
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS overall_cvr,
    ROUND(SUM(has_add_to_cart) / NULLIF(SUM(has_view_item), 0) * 100, 2) AS view_to_cart
FROM {{ ref('int_session_funnel') }}
GROUP BY device_category
ORDER BY sessions DESC