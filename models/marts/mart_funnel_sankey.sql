{{ config(materialized='table') }}

SELECT
    CASE 
        WHEN has_purchase = 1 THEN '5_Purchased'
        WHEN has_add_payment_info = 1 THEN '4_Dropped_Payment'
        WHEN has_begin_checkout = 1 THEN '3_Dropped_Checkout'
        WHEN has_add_to_cart = 1 THEN '2_Dropped_Cart'
        WHEN has_view_item = 1 THEN '1_Dropped_View'
        ELSE '0_No_View'
    END AS final_stage,
    COUNT(*) AS session_count
FROM {{ ref('int_session_funnel') }}
GROUP BY 1
ORDER BY 1