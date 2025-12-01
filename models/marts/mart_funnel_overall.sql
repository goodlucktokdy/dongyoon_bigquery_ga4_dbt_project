{{ config(materialized='table') }}

SELECT
    COUNT(*) AS total_sessions,
    SUM(has_view_item) AS step1_view_item,
    SUM(has_add_to_cart) AS step2_add_to_cart,
    SUM(has_begin_checkout) AS step3_begin_checkout,
    SUM(has_add_payment_info) AS step4_add_payment_info,
    SUM(has_purchase) AS step5_purchase,
    
    ROUND(SUM(has_view_item) / COUNT(*) * 100, 2) AS pct_view,
    ROUND(SUM(has_add_to_cart) / COUNT(*) * 100, 2) AS pct_cart,
    ROUND(SUM(has_purchase) / COUNT(*) * 100, 2) AS pct_purchase
FROM {{ ref('int_session_funnel') }}